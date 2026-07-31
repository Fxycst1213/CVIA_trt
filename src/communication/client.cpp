#include "client.h"
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fstream>
#include <experimental/filesystem>
#include <iomanip>

namespace
{
    struct EncodedImage
    {
        std::vector<uchar> bytes;
        float scale_x = 1.0f;
        float scale_y = 1.0f;
    };

    EncodedImage encode_image(const cv::Mat &src, const cv::Size &send_size, int jpeg_quality)
    {
        EncodedImage result;
        if (src.empty())
        {
            return result;
        }

        cv::Mat send_img;
        if (src.cols != send_size.width || src.rows != send_size.height)
        {
            cv::resize(src, send_img, send_size, 0, 0, cv::INTER_AREA);
        }
        else
        {
            send_img = src;
        }

        result.scale_x = static_cast<float>(send_img.cols) / static_cast<float>(src.cols);
        result.scale_y = static_cast<float>(send_img.rows) / static_cast<float>(src.rows);

        if (send_img.isContinuous())
        {
            const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
            cv::imencode(".jpg", send_img, result.bytes, params);
        }

        return result;
    }

    void write_image_packet(char *&ptr_curr, const std::vector<uchar> &encoded_img)
    {
        uint32_t img_len = static_cast<uint32_t>(encoded_img.size());
        uint32_t net_img_len = htonl(img_len);
        memcpy(ptr_curr, &net_img_len, sizeof(uint32_t));
        ptr_curr += sizeof(uint32_t);

        if (img_len > 0)
        {
            memcpy(ptr_curr, encoded_img.data(), img_len);
            ptr_curr += img_len;
        }
    }

    void publish_preview_pair(const std::string &directory,
                              const std::vector<uchar> &primary,
                              const std::vector<uchar> &secondary)
    {
        if (directory.empty() || primary.empty() || secondary.empty()) return;
        const std::string final_path = directory + "/preview_pair.bin";
        const std::string temporary_path = final_path + ".tmp";
        std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
        if (!output) return;

        // 一个原子文件同时携带两张 JPEG，避免网页恰好读到“主图新帧、副图旧帧”。
        // 格式：magic[4] + primary_len(u32, network order)
        //      + secondary_len(u32, network order) + primary + secondary。
        static const char kMagic[4] = {'C', 'V', 'P', '1'};
        const uint32_t primary_length = htonl(static_cast<uint32_t>(primary.size()));
        const uint32_t secondary_length = htonl(static_cast<uint32_t>(secondary.size()));
        output.write(kMagic, sizeof(kMagic));
        output.write(reinterpret_cast<const char *>(&primary_length), sizeof(primary_length));
        output.write(reinterpret_cast<const char *>(&secondary_length), sizeof(secondary_length));
        output.write(reinterpret_cast<const char *>(primary.data()), primary.size());
        output.write(reinterpret_cast<const char *>(secondary.data()), secondary.size());
        output.close();
        std::rename(temporary_path.c_str(), final_path.c_str());
    }

    void publish_reprojection(const std::string &directory, const Resultframe &frame)
    {
        if (directory.empty()) return;
        const std::string final_path = directory + "/reprojection.json";
        const std::string temporary_path = final_path + ".tmp";
        std::ofstream output(temporary_path, std::ios::trunc);
        if (!output) return;
        const bool valid = frame.pose_valid && frame.pose_result.size() >= 6;
        output << "{\"timestamp\":" << frame.timestamp
               << ",\"valid\":" << (valid ? "true" : "false")
               << ",\"pose\":[";
        if (frame.pose_result.size() >= 6)
        {
            // 网页和 CSV 顺序统一为 x,y,z,rx,ry,rz。
            output << std::setprecision(9)
                   << frame.pose_result[3] << ',' << frame.pose_result[4] << ',' << frame.pose_result[5] << ','
                   << frame.pose_result[0] << ',' << frame.pose_result[1] << ',' << frame.pose_result[2];
        }
        output << "],\"points\":[";
        for (size_t index = 0; index < frame.reprojected_points.size(); ++index)
        {
            if (index > 0) output << ',';
            output << '[' << frame.reprojected_points[index].x << ','
                   << frame.reprojected_points[index].y << ']';
        }
        output << "]}\n";
        output.close();
        std::rename(temporary_path.c_str(), final_path.c_str());
    }

}

void client::init(const prj_params &p_params)
{
    // 1. 保存参数到成员变量，方便 pack_and_send 使用
    _socket_mode = p_params.socket_mode;
    _img_size_bytes = p_params.t_params.IMG_SIZE;
    _kpt_size_bytes = p_params.t_params.KEYPOINTS_BUFSIZE;
    _pose_size_bytes = p_params.t_params.POSE_BUFSIZE;
    _resolution = p_params.detect_camera.resolution;
    _keyPoint_box = p_params.t_params.KeyPoint_box;
    _udp_enabled = p_params.enable_udp;
    _tcp_enabled = p_params.enable_tcp;
    _save_pnp_results = p_params.save_pnp_results;
    _preview_dir = p_params.preview_dir;
    // PnP 先记录到网页运行目录；停止推理后由浏览器另存到笔记本。
    _pnp_result_dir = p_params.preview_dir;
    _preview_interval = std::chrono::milliseconds(1000 / std::max(1, p_params.preview_fps));

    // 2. 清理旧资源
    if (_buffer)
    {
        delete[] _buffer;
        _buffer = nullptr;
    }
    if (_udp_fd != -1)
    {
        close(_udp_fd);
        _udp_fd = -1;
    }
    char header[3];
    header[0] = 'I';
    // TCP 默认关闭；位姿先写入网页运行目录中的临时会话，停止后再由浏览器保存。
    if (_tcp_enabled)
    {
        if (_socket_mode == 0)
        {
            header[1] = 'H'; header[2] = 'V';
            _total_send_size = sizeof(uint32_t) + _img_size_bytes + _kpt_size_bytes + _pose_size_bytes;
        }
        else if (_socket_mode == 2)
        {
            header[1] = '2'; header[2] = 'V';
            _total_send_size = sizeof(uint32_t) * 2 +
                               (_img_size_bytes * tcp_params::IMAGE_COUNT_DUAL) +
                               _kpt_size_bytes + _pose_size_bytes;
        }
        else if (_socket_mode == 1)
        {
            header[1] = 'H'; header[2] = 'D';
            _total_send_size = _pose_size_bytes;
        }
        else
        {
            LOGE("socket_mode error");
        }
        _buffer = new char[_total_send_size]();

        memset(&_remoteAddress, 0, sizeof(_remoteAddress));
        _remoteAddress.sin_family = AF_INET;
        _remoteAddress.sin_addr.s_addr = inet_addr(p_params.ip.c_str());
        _remoteAddress.sin_port = htons(p_params.port);

        if ((_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
            LOGE("create socket error");
        if (_fd != -1 && connect(_fd, (struct sockaddr *)&_remoteAddress, sizeof(struct sockaddr)) < 0)
        {
            LOGW("TCP target unavailable; local PNP recording and web preview remain active");
            close(_fd);
            _fd = -1;
        }
        if (_fd != -1) SendAll(header, 3);
    }
    else
    {
        LOG("TCP disabled; temporary PNP session: %s/pnp_session.csv", _pnp_result_dir.c_str());
    }

    if (!_pnp_result_dir.empty())
    {
        std::error_code error;
        std::experimental::filesystem::create_directories(_pnp_result_dir, error);
        const std::string csv_path = _pnp_result_dir + "/pnp_session.csv";
        if (_save_pnp_results)
        {
            // 每次启动都是独立会话；旧临时会话会在新推理开始时被替换。
            _pnp_csv.open(csv_path, std::ios::trunc);
            if (_pnp_csv)
            {
                _pnp_csv << "timestamp,x,y,z,rx,ry,rz\n";
                _pnp_csv.flush();
            }
        }
        else
            std::remove(csv_path.c_str());
    }

    if (_udp_enabled)
    {
        const std::string udp_ip = p_params.udp_ip.empty() ? p_params.ip : p_params.udp_ip;
        const int udp_port = p_params.udp_port > 0 ? p_params.udp_port : p_params.port;

        if (udp_ip.empty() || udp_port <= 0)
        {
            LOGW("UDP result upload disabled: invalid target %s:%d", udp_ip.c_str(), udp_port);
            _udp_enabled = false;
        }
        else
        {
            memset(&_udpAddress, 0, sizeof(_udpAddress));
            _udpAddress.sin_family = AF_INET;
            _udpAddress.sin_addr.s_addr = inet_addr(udp_ip.c_str());
            _udpAddress.sin_port = htons(udp_port);

            if ((_udp_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
            {
                LOGW("create UDP socket error");
                _udp_enabled = false;
            }
            else
            {
                LOG("UDP result upload target: %s:%d", udp_ip.c_str(), udp_port);
            }
        }
    }
}

void client::save_pnp_pose(const Resultframe &frame)
{
    // 旧 TCP 位姿段的顺序是 rx, ry, rz, x, y, z；CSV 按用户要求重排为
    // timestamp, x, y, z, rx, ry, rz。
    if (!_save_pnp_results || _pnp_result_dir.empty() || !frame.pose_valid || frame.pose_result.size() < 6) return;

    if (!_pnp_csv)
    {
        return;
    }
    _pnp_csv << frame.timestamp << ',' << std::setprecision(9)
             << frame.pose_result[3] << ',' << frame.pose_result[4] << ',' << frame.pose_result[5] << ','
             << frame.pose_result[0] << ',' << frame.pose_result[1] << ',' << frame.pose_result[2] << '\n';
    // 每约半秒刷新一次，显著减少每帧 flush 的存储抖动；正常退出时析构函数会再次刷新。
    if (++_csv_rows_since_flush >= 30)
    {
        _pnp_csv.flush();
        _csv_rows_since_flush = 0;
    }
}

void client::record_pose(const Resultframe &frame)
{
    save_pnp_pose(frame);
}

// --- 新增函数的实现 ---
bool client::pack_and_send(const Resultframe &frame)
{
    static const cv::Size kTcpImageSize(1280, 720);
    static const cv::Size kPreviewImageSize(960, 540);
    static const int kTcpJpegQuality = 30;
    static const int kPreviewJpegQuality = 45;

    const auto now = std::chrono::steady_clock::now();
    bool refresh_preview = false;
    if (_last_preview_at.time_since_epoch().count() == 0)
    {
        refresh_preview = true;
        _last_preview_at = now;
    }
    else if (now - _last_preview_at >= _preview_interval)
    {
        refresh_preview = true;
        // 保留不足一个预览周期的余量。原先直接赋值 now，会把推理帧周期
        // 向上取整：约 44 ms 的处理帧配 15 FPS 时会退化成约 11 FPS。
        const auto elapsed_intervals = (now - _last_preview_at) / _preview_interval;
        _last_preview_at += _preview_interval * elapsed_intervals;
    }
    const bool tcp_ready = _tcp_enabled && _fd != -1 && _buffer != nullptr;
    if (!tcp_ready && !refresh_preview) return true;

    const bool tcp_needs_primary = tcp_ready && (_socket_mode == 0 || _socket_mode == 2);
    const bool tcp_needs_secondary = tcp_ready && _socket_mode == 2;
    EncodedImage primary_image;
    EncodedImage secondary_image;

    if (refresh_preview || tcp_needs_primary)
    {
        primary_image = encode_image(
            frame.rgb,
            tcp_needs_primary ? kTcpImageSize : kPreviewImageSize,
            tcp_needs_primary ? kTcpJpegQuality : kPreviewJpegQuality);
    }

    if (refresh_preview || tcp_needs_secondary)
    {
        const bool same_source = !frame.rgb.empty() && !frame.rgb_secondary.empty() &&
                                 frame.rgb.data == frame.rgb_secondary.data &&
                                 frame.rgb.size() == frame.rgb_secondary.size() &&
                                 frame.rgb.type() == frame.rgb_secondary.type();
        const bool same_encoding_profile = tcp_needs_primary == tcp_needs_secondary;
        if (same_source && same_encoding_profile)
        {
            // 文件夹模式下副图就是主图，直接复用 JPEG，避免每帧重复编码。
            secondary_image = primary_image;
        }
        else
        {
            secondary_image = encode_image(
                frame.rgb_secondary,
                tcp_needs_secondary ? kTcpImageSize : kPreviewImageSize,
                tcp_needs_secondary ? kTcpJpegQuality : kPreviewJpegQuality);
        }
    }

    if (refresh_preview)
    {
        publish_preview_pair(_preview_dir, primary_image.bytes, secondary_image.bytes);
        publish_reprojection(_preview_dir, frame);
    }

    if (!tcp_ready) return true;

    int current_packet_size = 0;

    if (_socket_mode == 0)
    {
        char *ptr_curr = _buffer;
        write_image_packet(ptr_curr, primary_image.bytes);

        memset(ptr_curr, 0, _kpt_size_bytes);
        if (!frame.bboxes.empty())
        {
            float *tmp = new float[_keyPoint_box]();
            int size = (frame.bboxes[0].keypoints).size();
            for (int j = 0; j < size; j++)
            {
                auto &keypoint = frame.bboxes[0].keypoints[j];
                tmp[3 * j] = keypoint.x * primary_image.scale_x;
                tmp[3 * j + 1] = keypoint.y * primary_image.scale_y;
                tmp[3 * j + 2] = keypoint.conf;
            }

            tmp[_keyPoint_box - 5] = frame.bboxes[0].x0 * primary_image.scale_x;
            tmp[_keyPoint_box - 4] = frame.bboxes[0].y0 * primary_image.scale_y;
            tmp[_keyPoint_box - 3] = frame.bboxes[0].x1 * primary_image.scale_x;
            tmp[_keyPoint_box - 2] = frame.bboxes[0].y1 * primary_image.scale_y;
            tmp[_keyPoint_box - 1] = frame.bboxes[0].confidence;

            // copy data
            memcpy(ptr_curr, tmp, _kpt_size_bytes);
            delete[] tmp;
        }
        ptr_curr += _kpt_size_bytes;

        memset(ptr_curr, 0, _pose_size_bytes);

        if (!frame.pose_result.empty())
        {
            size_t data_len = frame.pose_result.size() * sizeof(float);
            if (data_len <= _pose_size_bytes)
            {
                memcpy(ptr_curr, frame.pose_result.data(), data_len);
            }
        }

        memcpy(ptr_curr + _pose_size_bytes - 8, &frame.timestamp, sizeof(uint64_t));
        ptr_curr += _pose_size_bytes;

        current_packet_size = ptr_curr - _buffer;
    }
    else if (_socket_mode == 2)
    {
        char *ptr_curr = _buffer;
        write_image_packet(ptr_curr, primary_image.bytes);
        write_image_packet(ptr_curr, secondary_image.bytes);

        memset(ptr_curr, 0, _kpt_size_bytes);
        if (!frame.bboxes.empty())
        {
            float *tmp = new float[_keyPoint_box]();
            int size = (frame.bboxes[0].keypoints).size();
            for (int j = 0; j < size; j++)
            {
                auto &keypoint = frame.bboxes[0].keypoints[j];
                tmp[3 * j] = keypoint.x * primary_image.scale_x;
                tmp[3 * j + 1] = keypoint.y * primary_image.scale_y;
                tmp[3 * j + 2] = keypoint.conf;
            }

            tmp[_keyPoint_box - 5] = frame.bboxes[0].x0 * primary_image.scale_x;
            tmp[_keyPoint_box - 4] = frame.bboxes[0].y0 * primary_image.scale_y;
            tmp[_keyPoint_box - 3] = frame.bboxes[0].x1 * primary_image.scale_x;
            tmp[_keyPoint_box - 2] = frame.bboxes[0].y1 * primary_image.scale_y;
            tmp[_keyPoint_box - 1] = frame.bboxes[0].confidence;

            memcpy(ptr_curr, tmp, _kpt_size_bytes);
            delete[] tmp;
        }
        ptr_curr += _kpt_size_bytes;

        memset(ptr_curr, 0, _pose_size_bytes);

        if (!frame.pose_result.empty())
        {
            size_t data_len = frame.pose_result.size() * sizeof(float);
            if (data_len <= _pose_size_bytes)
            {
                memcpy(ptr_curr, frame.pose_result.data(), data_len);
            }
        }

        memcpy(ptr_curr + _pose_size_bytes - 8, &frame.timestamp, sizeof(uint64_t));
        ptr_curr += _pose_size_bytes;

        current_packet_size = ptr_curr - _buffer;
    }
    else if (_socket_mode == 1)
    {
        // Mode 1 (纯数据模式) 保持不变，使用定长
        current_packet_size = _total_send_size;
        memset(_buffer, 0, _total_send_size);
        if (!frame.pose_result.empty())
        {
            size_t data_len = frame.pose_result.size() * sizeof(float);
            if (data_len <= _total_send_size)
            {
                memcpy(_buffer, frame.pose_result.data(), data_len);
            }
        }
        memcpy(_buffer + _total_send_size - 8, &frame.timestamp, sizeof(uint64_t));
    }

    // 统一发送
    return SendAll(_buffer, current_packet_size);
}

bool client::send_udp_result(const std::vector<float> &result, uint64_t timestamp)
{
    static const size_t kPose6DCount = 6;
    if (!_udp_enabled || _udp_fd == -1 || result.size() < kPose6DCount)
    {
        return false;
    }

    char buffer[tcp_params::POSE_BUFSIZE] = {0};
    const size_t pose_bytes = kPose6DCount * sizeof(float);
    memcpy(buffer, result.data(), pose_bytes);
    memcpy(buffer + tcp_params::POSE_BUFSIZE - sizeof(uint64_t), &timestamp, sizeof(uint64_t));

    const ssize_t bytes_sent = sendto(_udp_fd, buffer, sizeof(buffer), 0,
                                      (struct sockaddr *)&_udpAddress,
                                      sizeof(_udpAddress));
    if (bytes_sent != static_cast<ssize_t>(sizeof(buffer)))
    {
        LOGW("UDP result send failed, expected %zu bytes, sent %zd bytes", sizeof(buffer), bytes_sent);
        return false;
    }
    return true;
}

bool client::SendAll(char *buffer, int size)
{
    while (size > 0)
    {
        int SendSize = send(_fd, buffer, size, MSG_NOSIGNAL);
        if (-1 == SendSize)
            return false;
        size = size - SendSize; // 用于循环发送且退出功能
        buffer += SendSize;     // 用于计算已发buffer的偏移量
    }
    return true;
}

client::~client()
{
    if (_pnp_csv.is_open())
    {
        _pnp_csv.flush();
        _pnp_csv.close();
    }
    if (_buffer)
    {
        delete[] _buffer;
        _buffer = nullptr;
    }

    if (_fd != -1)
    {
        close(_fd);
        _fd = -1;
    }

    if (_udp_fd != -1)
    {
        close(_udp_fd);
        _udp_fd = -1;
    }
}

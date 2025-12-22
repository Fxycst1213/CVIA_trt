#include "client.h"
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

void client::init(const prj_params &p_params)
{
    // 1. 保存参数到成员变量，方便 pack_and_send 使用
    _socket_mode = p_params.socket_mode;
    _img_size_bytes = p_params.t_params.IMG_SIZE;
    _kpt_size_bytes = p_params.t_params.KEYPOINTS_BUFSIZE;
    _pose_size_bytes = p_params.t_params.POSE_BUFSIZE;
    _resolution = p_params.resolution;
    _keyPoint_box = p_params.t_params.KeyPoint_box;

    // 2. 清理旧内存
    if (_buffer)
    {
        delete[] _buffer;
        _buffer = nullptr;
    }
    char header[3];
    header[0] = 'I';
    // 3. 分配内存并计算总大小
    if (_socket_mode == 0)
    {
        header[1] = 'H';
        header[2] = 'V';
        _total_send_size = _img_size_bytes + _kpt_size_bytes + _pose_size_bytes;
        _buffer = new char[_total_send_size];
        memset(_buffer, 0, _total_send_size);
    }
    else if (_socket_mode == 1)
    {
        header[1] = 'H';
        header[2] = 'D';
        _total_send_size = _pose_size_bytes;
        _buffer = new char[_total_send_size];
        memset(_buffer, 0, _total_send_size);
    }
    else
    {
        LOGE("socket_mode error");
    }
    // 4. 连接服务器 (保持原样)
    memset(&_remoteAddress, 0, sizeof(_remoteAddress));
    _remoteAddress.sin_family = AF_INET;
    _remoteAddress.sin_addr.s_addr = inet_addr(p_params.ip.c_str());
    _remoteAddress.sin_port = htons(p_params.port);

    if ((_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
    {
        LOGE("create socket error");
    }
    if (connect(_fd, (struct sockaddr *)&_remoteAddress, sizeof(struct sockaddr)) < 0)
    {
        LOGE("connect error");
    }
    SendAll(header, 3);
}

// --- 新增函数的实现 ---
bool client::pack_and_send(const Resultframe &frame)
{
    if (!_buffer || _fd == -1)
        return false;

    int current_packet_size = 0;

    // 根据模式进行打包
    if (_socket_mode == 0)
    {
        // --- 修复1: 声明局部变量 ---
        std::vector<uchar> encoded_img;
        uint32_t img_len = 0;

        // 1. 压缩图片
        auto start = std::chrono::high_resolution_clock::now();
        if (!frame.rgb.empty() && frame.rgb.isContinuous())
        {
            std::vector<int> params;
            params.push_back(cv::IMWRITE_JPEG_QUALITY);
            params.push_back(50); // 质量设为 50

            cv::imencode(".jpg", frame.rgb, encoded_img, params);
            img_len = static_cast<uint32_t>(encoded_img.size());
        }
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();

        // 2. 如果你想看日志，直接打印出来，或者手动添加到 timer 的列表里（如果 timer 有 addLog 接口）
        LOG("\tJPEG uses %.6lf ms", duration);

        // 使用 ptr_curr 作为移动指针，全程跟踪写入位置
        char *ptr_curr = _buffer;

        // 2. 写入图片大小头 (4字节)
        uint32_t net_img_len = htonl(img_len); // 转网络字节序
        memcpy(ptr_curr, &net_img_len, sizeof(uint32_t));
        ptr_curr += sizeof(uint32_t); // --- 指针移动 ---

        // 3. 写入图片数据
        if (img_len > 0)
        {
            memcpy(ptr_curr, encoded_img.data(), img_len);
            ptr_curr += img_len; // --- 指针移动 ---
        }

        // 4. 写入 KeyPoints
        // 直接向 ptr_curr 写入，省去中间变量 ptr_kpt 防止算错
        memset(ptr_curr, 0, _kpt_size_bytes);

        if (!frame.bboxes.empty())
        {
            float *tmp = new float[_keyPoint_box];
            int size = (frame.bboxes[0].keypoints).size();
            for (int j = 0; j < size; j++)
            {
                auto &keypoint = frame.bboxes[0].keypoints[j];
                tmp[3 * j] = keypoint.x;
                tmp[3 * j + 1] = keypoint.y;
                tmp[3 * j + 2] = keypoint.conf;
            }

            tmp[_keyPoint_box - 5] = frame.bboxes[0].x0;
            tmp[_keyPoint_box - 4] = frame.bboxes[0].y0;
            tmp[_keyPoint_box - 3] = frame.bboxes[0].x1;
            tmp[_keyPoint_box - 2] = frame.bboxes[0].y1;
            tmp[_keyPoint_box - 1] = frame.bboxes[0].confidence;

            // copy data
            memcpy(ptr_curr, tmp, _kpt_size_bytes);
            delete[] tmp;
        }
        // --- 修复2: 写入完 KPT 后必须移动指针 ---
        ptr_curr += _kpt_size_bytes;

        // 5. 写入 Pose
        memset(ptr_curr, 0, _pose_size_bytes);

        if (!frame.pose_result.empty())
        {
            size_t data_len = frame.pose_result.size() * sizeof(double);
            if (data_len <= _pose_size_bytes)
            {
                memcpy(ptr_curr, frame.pose_result.data(), data_len);
            }
        }
        // 写入时间戳 (在 Pose 区域的末尾)
        memcpy(ptr_curr + _pose_size_bytes - 8, &frame.timestamp, sizeof(uint64_t));

        // --- 修复3: 写入完 Pose 后必须移动指针 ---
        ptr_curr += _pose_size_bytes;

        // 6. 计算最终总包大小
        // 因为 ptr_curr 一路都在累加，现在减去起始地址就是总有效长度
        current_packet_size = ptr_curr - _buffer;
    }
    else if (_socket_mode == 1)
    {
        // Mode 1 (纯数据模式) 保持不变，使用定长
        current_packet_size = _total_send_size;
        memset(_buffer, 0, _total_send_size);
        if (!frame.pose_result.empty())
        {
            size_t data_len = frame.pose_result.size() * sizeof(double);
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

bool client::SendAll(char *buffer, int size)
{
    while (size > 0)
    {
        int SendSize = send(_fd, buffer, size, 0);
        if (-1 == SendSize)
            return false;
        size = size - SendSize; // 用于循环发送且退出功能
        buffer += SendSize;     // 用于计算已发buffer的偏移量
    }
    return true;
}

client::~client()
{
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
}

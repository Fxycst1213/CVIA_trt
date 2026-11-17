#include "client.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

void client::init(const prj_params &params)
{
    disconnect();
    _socket_mode = params.socket_mode;
    _jpeg_width = params.t_params.jpeg_width;
    _jpeg_quality = params.t_params.jpeg_quality;
    _send_timeout_ms = params.t_params.send_timeout_ms;
    if ((_socket_mode < 0 || _socket_mode > 4) || _jpeg_width < 0 ||
        _jpeg_quality < 1 || _jpeg_quality > 100 || _send_timeout_ms <= 0 ||
        params.port < 1 || params.port > 65535)
        throw std::invalid_argument("Invalid TCP preview settings");
    _remoteAddress = {};
    _remoteAddress.sin_family = AF_INET;
    _remoteAddress.sin_port = htons(params.port);
    if (inet_pton(AF_INET, params.ip.c_str(), &_remoteAddress.sin_addr) != 1)
        throw std::invalid_argument("--tcp-ip must be an IPv4 address");
    _next_connect = {};
    // Connect in the TCP worker: an unavailable viewer must not stop inference.
}

bool client::connect_to_server()
{
    if (_fd >= 0)
        return true;
    if (std::chrono::steady_clock::now() < _next_connect)
        return false;
    _next_connect = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    _fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (_fd < 0)
        return false;
    int flags = fcntl(_fd, F_GETFL, 0);
    int enabled = 1;
    int send_buffer = 64 * 1024;
    setsockopt(_fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
    setsockopt(_fd, SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
    if (flags < 0 || fcntl(_fd, F_SETFL, flags | O_NONBLOCK) < 0)
    {
        disconnect();
        return false;
    }
    int result = connect(_fd, reinterpret_cast<sockaddr *>(&_remoteAddress), sizeof(_remoteAddress));
    if (result < 0 && errno == EINPROGRESS)
    {
        pollfd socket_poll{_fd, POLLOUT, 0};
        result = poll(&socket_poll, 1, _send_timeout_ms);
        int error = 0;
        socklen_t size = sizeof(error);
        if (result <= 0 || getsockopt(_fd, SOL_SOCKET, SO_ERROR, &error, &size) < 0 || error)
            result = -1;
        else
            result = 0;
    }
    const char modes[] = {'V', 'D', 'X', 'C', 'B'};
    const char header[] = {'I', 'H', modes[_socket_mode]};
    if (result < 0 || !SendAll(header, sizeof(header)))
    {
        disconnect();
        return false;
    }
    std::cerr << "[tcp] Viewer connected\n";
    return true;
}

bool client::pack_and_send(const Resultframe &frame)
{
    if (_socket_mode < 0 || !connect_to_server())
        return false;
    size_t image_bytes = 0;
    double scale_x = 1.0, scale_y = 1.0;
    uint32_t encoded_width = 0, encoded_height = 0;
    if (_socket_mode != 1)
    {
        if (frame.rgb.empty())
            return false;
        cv::Mat image = frame.rgb;
        if (_jpeg_width > 0 && image.cols > _jpeg_width)
        {
            const int height = std::max(1, cvRound(double(image.rows) * _jpeg_width / image.cols));
            cv::resize(frame.rgb, image, cv::Size(_jpeg_width, height), 0, 0, cv::INTER_AREA);
            scale_x = double(image.cols) / frame.rgb.cols;
            scale_y = double(image.rows) / frame.rgb.rows;
        }
        encoded_width = image.cols;
        encoded_height = image.rows;
        if (!cv::imencode(".jpg", image, _jpeg, {cv::IMWRITE_JPEG_QUALITY, _jpeg_quality}))
            return false;
        image_bytes = _jpeg.size();
        if (image_bytes > 16 * 1024 * 1024)
            return false;
    }

    // IHV: [JPEG length, big endian][JPEG][3*N+5 floats][6 floats + uint64].
    // x_plane1 uses N=9, so the keypoint/bbox payload contains 32 floats.
    // IHD: [6 floats + uint64]. Keep pose units/order and timestamp unchanged.
    // IHX appends [uint32 valid][3 float rvec radians][3 float tvec mm], little endian.
    // IHC also appends <III16d>: JPEG width/height, right-eye flag, fx/fy/cx/cy, 12 distortion values.
    // IHB appends <I3f> to IHC: initialized flag and robot-base XYZ in mm, before serial axis/offset mapping.
    const size_t prefix = _socket_mode != 1 ? sizeof(uint32_t) + image_bytes + tcp_params::KEYPOINTS_BUFSIZE : 0;
    const size_t extra = _socket_mode >= 2 ? sizeof(uint32_t) + 6 * sizeof(float) : 0;
    const size_t calibration_bytes = _socket_mode >= 3 ? 3 * sizeof(uint32_t) + 16 * sizeof(double) : 0;
    const size_t base_bytes = _socket_mode == 4 ? sizeof(uint32_t) + 3 * sizeof(float) : 0;
    _packet.assign(prefix + tcp_params::POSE_BUFSIZE + extra + calibration_bytes + base_bytes, 0);
    if (_socket_mode != 1)
    {
        uint32_t length = htonl(static_cast<uint32_t>(image_bytes));
        memcpy(_packet.data(), &length, sizeof(length));
        memcpy(_packet.data() + sizeof(length), _jpeg.data(), image_bytes);
        std::array<float, tcp_params::KeyPoint_box> points{};
        if (!frame.bboxes.empty())
        {
            const auto &box = frame.bboxes.front();
            const size_t count = std::min(box.keypoints.size(), size_t(tcp_params::Point_num));
            for (size_t i = 0; i < count; ++i)
            {
                points[i * 3] = box.keypoints[i].x * scale_x;
                points[i * 3 + 1] = box.keypoints[i].y * scale_y;
                points[i * 3 + 2] = box.keypoints[i].conf;
            }
            const size_t offset = tcp_params::Point_num * 3;
            points[offset] = box.x0 * scale_x;
            points[offset + 1] = box.y0 * scale_y;
            points[offset + 2] = box.x1 * scale_x;
            points[offset + 3] = box.y1 * scale_y;
            points[offset + 4] = box.confidence;
        }
        memcpy(_packet.data() + sizeof(length) + image_bytes, points.data(), sizeof(points));
    }
    const size_t pose_bytes = std::min(frame.pose_result.size(), size_t(6)) * sizeof(float);
    if (pose_bytes)
        memcpy(_packet.data() + prefix, frame.pose_result.data(), pose_bytes);
    memcpy(_packet.data() + prefix + tcp_params::POSE_BUFSIZE - sizeof(frame.timestamp), &frame.timestamp, sizeof(frame.timestamp));
    if (extra)
    {
        auto *extension = _packet.data() + prefix + tcp_params::POSE_BUFSIZE;
        const uint32_t valid = frame.pnp_valid ? 1 : 0;
        memcpy(extension, &valid, sizeof(valid));
        if (valid)
        {
            memcpy(extension + sizeof(valid), frame.pnp_rvec.data(), 3 * sizeof(float));
            memcpy(extension + sizeof(valid) + 3 * sizeof(float), frame.pnp_tvec.data(), 3 * sizeof(float));
        }
    }
    if (calibration_bytes)
    {
        const auto &calibration = frame.calibration;
        const auto finite = [](double value) { return std::isfinite(value); };
        if (calibration.width != static_cast<uint32_t>(frame.rgb.cols) ||
            calibration.height != static_cast<uint32_t>(frame.rgb.rows) ||
            calibration.intrinsics[0] <= 0 || calibration.intrinsics[1] <= 0 ||
            !std::all_of(calibration.intrinsics.begin(), calibration.intrinsics.end(), finite) ||
            !std::all_of(calibration.distortion.begin(), calibration.distortion.end(), finite))
            throw std::invalid_argument("Camera calibration does not match source frame");
        auto *destination = _packet.data() + prefix + tcp_params::POSE_BUFSIZE + extra;
        const std::array<uint32_t, 3> description{{encoded_width, encoded_height, calibration.right_eye ? 1U : 0U}};
        std::array<double, 16> parameters{};
        parameters[0] = calibration.intrinsics[0] * scale_x;
        parameters[1] = calibration.intrinsics[1] * scale_y;
        parameters[2] = calibration.intrinsics[2] * scale_x;
        parameters[3] = calibration.intrinsics[3] * scale_y;
        std::copy(calibration.distortion.begin(), calibration.distortion.end(), parameters.begin() + 4);
        memcpy(destination, description.data(), sizeof(description));
        memcpy(destination + sizeof(description), parameters.data(), sizeof(parameters));
    }
    if (base_bytes)
    {
        auto *destination = _packet.data() + prefix + tcp_params::POSE_BUFSIZE + extra + calibration_bytes;
        const uint32_t valid = frame.robot_base_valid ? 1 : 0;
        memcpy(destination, &valid, sizeof(valid));
        if (valid)
        {
            if (!std::all_of(frame.robot_base_xyz.begin(), frame.robot_base_xyz.end(),
                             [](float value) { return std::isfinite(value); }))
                throw std::invalid_argument("Non-finite robot-base coordinates");
            memcpy(destination + sizeof(valid), frame.robot_base_xyz.data(), 3 * sizeof(float));
        }
    }
    if (!SendAll(_packet.data(), _packet.size()))
    {
        // A partial frame must never be followed by another frame on this connection.
        disconnect();
        return false;
    }
    return true;
}

bool client::SendAll(const void *data, size_t size)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(_send_timeout_ms);
    auto buffer = static_cast<const unsigned char *>(data);
    while (size)
    {
        if (std::chrono::steady_clock::now() >= deadline)
            return false;
        const ssize_t sent = send(_fd, buffer, size, MSG_NOSIGNAL);
        if (sent > 0)
        {
            buffer += sent;
            size -= sent;
            continue;
        }
        if (sent < 0 && errno == EINTR)
            continue;
        if (sent == 0 || (errno != EAGAIN && errno != EWOULDBLOCK))
            return false;
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0)
            return false;
        pollfd socket_poll{_fd, POLLOUT, 0};
        const int result = poll(&socket_poll, 1, static_cast<int>(left));
        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0 || (socket_poll.revents & (POLLERR | POLLHUP | POLLNVAL)))
            return false;
    }
    return true;
}

void client::disconnect()
{
    if (_fd >= 0)
        close(_fd);
    _fd = -1;
    _next_connect = std::chrono::steady_clock::now() + std::chrono::seconds(1);
}

client::~client()
{
    disconnect();
}

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

    // 根据模式进行打包
    if (_socket_mode == 0)
    {
        if (!frame.rgb.empty() && frame.rgb.isContinuous())
        {
            memcpy(_buffer, frame.rgb.data, _img_size_bytes);
        }
        char *ptr_kpt = _buffer + _img_size_bytes;
        memset(ptr_kpt, 0, _kpt_size_bytes);
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
            memcpy(ptr_kpt, tmp, _kpt_size_bytes);
            delete[] tmp;
        }

        char *ptr_pose = ptr_kpt + _kpt_size_bytes;
        memset(ptr_pose, 0, _pose_size_bytes);

        if (!frame.pose_result.empty())
        {
            size_t data_len = frame.pose_result.size() * sizeof(double);
            if (data_len <= _pose_size_bytes)
            {
                memcpy(ptr_pose, frame.pose_result.data(), data_len);
            }
        }
        memcpy(ptr_pose + _pose_size_bytes - 8, &frame.timestamp, sizeof(uint64_t));
    }
    else if (_socket_mode == 1)
    {
        memset(_buffer, 0, _total_send_size); // 清零
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
    return SendAll(_buffer, _total_send_size);
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

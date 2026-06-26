#include "RS485.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include "../TensorRT/logger/logger.hpp"

using namespace std;

const unsigned char RS485::FRAME_HEAD_1;
const unsigned char RS485::TX_FRAME_HEAD_2;
const unsigned char RS485::RX_FRAME_HEAD_2;
const std::size_t RS485::TX_PAYLOAD_SIZE;
const std::size_t RS485::RX_PAYLOAD_SIZE;
const std::size_t RS485::TX_FIELD_COUNT;

RS485::RS485()
{
    _debug = false;
}

RS485::~RS485()
{
    closePort();
}

int RS485::init(const prj_params &p_params)
{
    _fd = open(p_params.rs485_port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (_fd < 0)
    {
        cerr << "无法打开串口设备: " << p_params.rs485_port << endl;
        return -1;
    }

    struct termios options;
    if (tcgetattr(_fd, &options) != 0)
    {
        LOGW("获取串口属性失败: %s", strerror(errno));
        closePort();
        return -1;
    }

    cfsetispeed(&options, p_params.rs485_baudrate);
    cfsetospeed(&options, p_params.rs485_baudrate);

    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CRTSCTS;
    options.c_cflag |= (CLOCAL | CREAD);

    options.c_iflag &= ~(IXON | IXOFF | IXANY | ICRNL | INLCR | IGNCR | INPCK);
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;

    options.c_cc[VTIME] = 1;
    options.c_cc[VMIN] = 0;

    tcflush(_fd, TCIOFLUSH);
    if (tcsetattr(_fd, TCSANOW, &options) != 0)
    {
        LOGW("设置串口属性失败: %s", strerror(errno));
        closePort();
        return -1;
    }

    int flags = fcntl(_fd, F_GETFL, 0);
    if (flags >= 0)
    {
        fcntl(_fd, F_SETFL, flags & ~O_NONBLOCK);
    }

    LOG("RS422/RS485串口初始化成功: %s", p_params.rs485_port.c_str());
    return 0;
}

bool RS485::sendVisionFrame(const std::vector<float> &values)
{
    if (_fd < 0)
    {
        LOGW("串口未初始化，无法发送数据");
        return false;
    }

    std::vector<unsigned char> frame;
    frame.reserve(2 + 1 + TX_PAYLOAD_SIZE + 1);
    frame.push_back(FRAME_HEAD_1);
    frame.push_back(TX_FRAME_HEAD_2);
    frame.push_back(static_cast<unsigned char>(TX_PAYLOAD_SIZE));

    for (std::size_t i = 0; i < TX_FIELD_COUNT; ++i)
    {
        const float value = i < values.size() ? values[i] : 0.0f;
        const float scale = i >= 6 ? 100.0f : 1.0f;
        appendInt16BE(frame, floatToProtocolInt16(value, scale));
    }

    const unsigned char checksum = calculateChecksum(&frame[2], 1 + TX_PAYLOAD_SIZE);
    frame.push_back(checksum);

    const bool ok = writeAll(frame.data(), frame.size());
    if (ok && _debug)
    {
        printf("[RS485 TX] raw %zu bytes:", frame.size());
        for (unsigned char byte : frame)
        {
            printf(" %02X", byte);
        }
        printf("\n");
        fflush(stdout);
    }
    if (ok && _send_interval_us > 0)
    {
        usleep(_send_interval_us);
    }
    return ok;
}

bool RS485::sendFloatArray(const float arr[3])
{
    std::vector<float> values(TX_FIELD_COUNT, 0.0f);
    values[0] = arr[0];
    values[1] = arr[1];
    values[2] = arr[2];
    return sendVisionFrame(values);
}

bool RS485::sendDoubleArray(const double arr[3])
{
    std::vector<float> values(TX_FIELD_COUNT, 0.0f);
    values[0] = static_cast<float>(arr[0]);
    values[1] = static_cast<float>(arr[1]);
    values[2] = static_cast<float>(arr[2]);
    return sendVisionFrame(values);
}

bool RS485::receiveAndPrintAvailable(int timeout_ms)
{
    if (_fd < 0)
    {
        return false;
    }

    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(_fd, &readfds);

    struct timeval timeout;
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;

    int ret = select(_fd + 1, &readfds, nullptr, nullptr, &timeout);
    if (ret < 0)
    {
        if (errno != EINTR)
        {
            LOGW("串口接收select失败: %s", strerror(errno));
        }
        return false;
    }
    if (ret == 0 || !FD_ISSET(_fd, &readfds))
    {
        return false;
    }

    unsigned char buffer[256];
    ssize_t bytes_read = read(_fd, buffer, sizeof(buffer));
    if (bytes_read < 0)
    {
        if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
        {
            LOGW("串口读取失败: %s", strerror(errno));
        }
        return false;
    }
    if (bytes_read == 0)
    {
        return false;
    }

    printRawBytes(buffer, static_cast<std::size_t>(bytes_read));
    _rx_buffer.insert(_rx_buffer.end(), buffer, buffer + bytes_read);
    parseReceiveBuffer();
    return true;
}

void RS485::closePort()
{
    if (_fd >= 0)
    {
        close(_fd);
        _fd = -1;
        LOG("RS422/RS485串口已关闭");
    }
}

void RS485::setDebug(bool enable)
{
    _debug = enable;
}

void RS485::setSendIntervalUs(unsigned int interval_us)
{
    _send_interval_us = interval_us;
}

unsigned char RS485::calculateChecksum(const unsigned char *data, std::size_t length) const
{
    unsigned int sum = 0;
    for (std::size_t i = 0; i < length; ++i)
    {
        sum += data[i];
    }
    return static_cast<unsigned char>(sum & 0xFF);
}

void RS485::appendInt16BE(std::vector<unsigned char> &buffer, int value) const
{
    const int clamped = std::max<int>(std::numeric_limits<int16_t>::min(),
                                      std::min<int>(std::numeric_limits<int16_t>::max(), value));
    const uint16_t encoded = static_cast<uint16_t>(static_cast<int16_t>(clamped));
    buffer.push_back(static_cast<unsigned char>((encoded >> 8) & 0xFF));
    buffer.push_back(static_cast<unsigned char>(encoded & 0xFF));
}

int RS485::floatToProtocolInt16(float value, float scale) const
{
    if (!std::isfinite(value))
    {
        return 0;
    }
    const long rounded = std::lround(value * scale);
    if (rounded > std::numeric_limits<int16_t>::max())
    {
        return std::numeric_limits<int16_t>::max();
    }
    if (rounded < std::numeric_limits<int16_t>::min())
    {
        return std::numeric_limits<int16_t>::min();
    }
    return static_cast<int>(rounded);
}

bool RS485::writeAll(const unsigned char *data, std::size_t length)
{
    std::size_t total_written = 0;
    while (total_written < length)
    {
        ssize_t bytes_written = write(_fd, data + total_written, length - total_written);
        if (bytes_written < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            LOGW("串口发送失败: %s", strerror(errno));
            return false;
        }
        if (bytes_written == 0)
        {
            LOGW("串口发送失败: write返回0");
            return false;
        }
        total_written += static_cast<std::size_t>(bytes_written);
    }

    tcdrain(_fd);
    return true;
}

void RS485::parseReceiveBuffer()
{
    while (_rx_buffer.size() >= 3)
    {
        if (_rx_buffer[0] != FRAME_HEAD_1)
        {
            auto next_header = std::find(_rx_buffer.begin() + 1, _rx_buffer.end(), FRAME_HEAD_1);
            _rx_buffer.erase(_rx_buffer.begin(), next_header);
            continue;
        }

        if (_rx_buffer.size() < 2)
        {
            return;
        }

        if (_rx_buffer[1] != RX_FRAME_HEAD_2)
        {
            _rx_buffer.erase(_rx_buffer.begin());
            continue;
        }

        const unsigned char payload_length = _rx_buffer[2];
        const std::size_t total_length = 2 + 1 + payload_length + 1;
        if (payload_length != RX_PAYLOAD_SIZE)
        {
            LOGW("收到未知串口帧长度: %u", payload_length);
            _rx_buffer.erase(_rx_buffer.begin());
            continue;
        }

        if (_rx_buffer.size() < total_length)
        {
            return;
        }

        const unsigned char expected_checksum = calculateChecksum(&_rx_buffer[2], 1 + RX_PAYLOAD_SIZE);
        const unsigned char received_checksum = _rx_buffer[total_length - 1];
        if (expected_checksum != received_checksum)
        {
            LOGW("串口接收校验失败: expected=0x%02X received=0x%02X",
                 expected_checksum, received_checksum);
            _rx_buffer.erase(_rx_buffer.begin());
            continue;
        }

        printReceiveFrame(&_rx_buffer[3], received_checksum);
        _rx_buffer.erase(_rx_buffer.begin(), _rx_buffer.begin() + total_length);
    }
}

void RS485::printRawBytes(const unsigned char *data, std::size_t length) const
{
    printf("[RS485 RX] raw %zu bytes:", length);
    for (std::size_t i = 0; i < length; ++i)
    {
        printf(" %02X", data[i]);
    }
    printf("\n");
    fflush(stdout);
}

void RS485::printReceiveFrame(const unsigned char *payload, unsigned char checksum) const
{
    const int pitch_ch1_raw = readInt32BE(payload);
    const int pitch_ch2_raw = readInt32BE(payload + 4);
    const int roll_raw = readInt32BE(payload + 8);
    const unsigned int rod_length = readUInt16BE(payload + 12);

    printf("[RS485 RX] frame EB 91 len=%zu checksum=0x%02X ok\n", RX_PAYLOAD_SIZE, checksum);
    printf("[RS485 RX] 俯仰通道1=%.3f deg, 俯仰通道2=%.3f deg, 横滚通道=%.3f deg, 伸缩杆长度=%u mm\n",
           pitch_ch1_raw * 0.001,
           pitch_ch2_raw * 0.001,
           roll_raw * 0.001,
           rod_length);
    fflush(stdout);
}

int RS485::readInt32BE(const unsigned char *data) const
{
    const uint32_t value = (static_cast<uint32_t>(data[0]) << 24) |
                           (static_cast<uint32_t>(data[1]) << 16) |
                           (static_cast<uint32_t>(data[2]) << 8) |
                           static_cast<uint32_t>(data[3]);
    const int64_t signed_value = (value & 0x80000000u)
                                     ? static_cast<int64_t>(value) - 0x100000000LL
                                     : static_cast<int64_t>(value);
    return static_cast<int>(signed_value);
}

unsigned int RS485::readUInt16BE(const unsigned char *data) const
{
    return (static_cast<unsigned int>(data[0]) << 8) |
           static_cast<unsigned int>(data[1]);
}

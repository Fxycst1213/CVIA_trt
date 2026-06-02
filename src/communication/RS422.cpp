#include "RS422.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include "../logger/logger.hpp"

RS422::RS422()
{
    _debug = false;
}

RS422::~RS422()
{
    closePort();
}

int RS422::init(const prj_params &p_params)
{
    return init(p_params.rs442_port, static_cast<speed_t>(p_params.rs442_baudrate));
}

int RS422::init(const std::string &port, speed_t baudrate)
{
    closePort();

    _fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (_fd < 0)
    {
        std::cerr << "无法打开RS422串口设备: " << port << std::endl;
        return -1;
    }

    fcntl(_fd, F_SETFL, 0);

    if (!configurePort(baudrate))
    {
        closePort();
        return -1;
    }

    LOG("RS422串口初始化成功: %s 波特率: %d", port.c_str(), static_cast<int>(baudrate));
    return 0;
}

bool RS422::configurePort(speed_t baudrate)
{
    struct termios options;
    memset(&options, 0, sizeof(options));

    if (tcgetattr(_fd, &options) != 0)
    {
        LOGE("获取RS422串口属性失败");
        return false;
    }

    cfsetispeed(&options, baudrate);
    cfsetospeed(&options, baudrate);

    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag &= ~PARENB;
    options.c_iflag &= ~INPCK;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CRTSCTS;
    options.c_cflag |= (CLOCAL | CREAD);

    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);

    options.c_cc[VTIME] = 10;
    options.c_cc[VMIN] = 0;

    tcflush(_fd, TCIOFLUSH);
    if (tcsetattr(_fd, TCSANOW, &options) != 0)
    {
        LOGE("设置RS422串口属性失败");
        return false;
    }

    return true;
}

bool RS422::sendFrame(const TxData &data)
{
    if (_fd < 0)
    {
        LOGE("RS422串口未初始化");
        return false;
    }

    uint8_t buffer[TX_FRAME_SIZE];
    packTxFrame(data, buffer);

    if (!writeAll(buffer, TX_FRAME_SIZE))
    {
        LOGE("RS422发送失败");
        return false;
    }

    if (_debug)
    {
        LOG("RS422发送%d字节", TX_FRAME_SIZE);
    }
    return true;
}

bool RS422::sendFrame(const int16_t absolute[3], const int16_t relative[3], const int16_t attitude[3])
{
    TxData data;
    memcpy(data.absolute, absolute, sizeof(data.absolute));
    memcpy(data.relative, relative, sizeof(data.relative));
    memcpy(data.attitude, attitude, sizeof(data.attitude));
    return sendFrame(data);
}

bool RS422::sendFloatArray(const float arr[9])
{
    TxData data;
    data.absolute[0] = floatToInt16(arr[0]);
    data.absolute[1] = floatToInt16(arr[1]);
    data.absolute[2] = floatToInt16(arr[2]);
    data.relative[0] = floatToInt16(arr[3]);
    data.relative[1] = floatToInt16(arr[4]);
    data.relative[2] = floatToInt16(arr[5]);
    data.attitude[0] = floatToInt16(arr[6], 100.0f);
    data.attitude[1] = floatToInt16(arr[7], 100.0f);
    data.attitude[2] = floatToInt16(arr[8], 100.0f);
    return sendFrame(data);
}

bool RS422::sendDoubleArray(const double arr[9])
{
    TxData data;
    data.absolute[0] = doubleToInt16(arr[0]);
    data.absolute[1] = doubleToInt16(arr[1]);
    data.absolute[2] = doubleToInt16(arr[2]);
    data.relative[0] = doubleToInt16(arr[3]);
    data.relative[1] = doubleToInt16(arr[4]);
    data.relative[2] = doubleToInt16(arr[5]);
    data.attitude[0] = doubleToInt16(arr[6], 100.0);
    data.attitude[1] = doubleToInt16(arr[7], 100.0);
    data.attitude[2] = doubleToInt16(arr[8], 100.0);
    return sendFrame(data);
}

bool RS422::readFrame(RxData &data)
{
    if (_fd < 0)
    {
        LOGE("RS422串口未初始化");
        return false;
    }

    uint8_t frame[RX_FRAME_SIZE];
    int matched = 0;

    while (matched < 2)
    {
        uint8_t byte = 0;
        if (!readExact(&byte, 1))
        {
            return false;
        }

        if (matched == 0)
        {
            matched = (byte == RX_HEAD_1) ? 1 : 0;
        }
        else
        {
            if (byte == RX_HEAD_2)
            {
                matched = 2;
            }
            else
            {
                matched = (byte == RX_HEAD_1) ? 1 : 0;
            }
        }
    }

    frame[0] = RX_HEAD_1;
    frame[1] = RX_HEAD_2;
    if (!readExact(frame + 2, RX_FRAME_SIZE - 2))
    {
        return false;
    }

    return parseRxFrame(frame, data);
}

void RS422::closePort()
{
    if (_fd >= 0)
    {
        close(_fd);
        _fd = -1;
        LOG("RS422串口已关闭");
    }
}

void RS422::setDebug(bool enable)
{
    _debug = enable;
}

bool RS422::writeAll(const uint8_t *buffer, int size)
{
    int offset = 0;
    while (offset < size)
    {
        int written = write(_fd, buffer + offset, size - offset);
        if (written < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return false;
        }
        if (written == 0)
        {
            return false;
        }
        offset += written;
    }

    tcdrain(_fd);
    return true;
}

bool RS422::readExact(uint8_t *buffer, int size)
{
    int offset = 0;
    while (offset < size)
    {
        int bytes_read = read(_fd, buffer + offset, size - offset);
        if (bytes_read < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return false;
        }
        if (bytes_read == 0)
        {
            return false;
        }
        offset += bytes_read;
    }
    return true;
}

void RS422::packTxFrame(const TxData &data, uint8_t buffer[TX_FRAME_SIZE])
{
    memset(buffer, 0, TX_FRAME_SIZE);
    buffer[0] = TX_HEAD_1;
    buffer[1] = TX_HEAD_2;
    buffer[2] = TX_PAYLOAD_LEN;

    int offset = 3;
    for (int i = 0; i < 3; ++i)
    {
        putInt16BE(buffer + offset, data.absolute[i]);
        offset += 2;
    }
    for (int i = 0; i < 3; ++i)
    {
        putInt16BE(buffer + offset, data.relative[i]);
        offset += 2;
    }
    for (int i = 0; i < 3; ++i)
    {
        putInt16BE(buffer + offset, data.attitude[i]);
        offset += 2;
    }

    buffer[21] = checksum(buffer + 2, 1 + TX_PAYLOAD_LEN);
}

bool RS422::parseRxFrame(const uint8_t buffer[RX_FRAME_SIZE], RxData &data)
{
    if (buffer[0] != RX_HEAD_1 || buffer[1] != RX_HEAD_2 || buffer[2] != RX_PAYLOAD_LEN)
    {
        LOGE("RS422接收帧头或长度错误");
        return false;
    }

    uint8_t expected = checksum(buffer + 2, 1 + RX_PAYLOAD_LEN);
    if (buffer[17] != expected)
    {
        LOGE("RS422接收校验失败");
        return false;
    }

    const uint8_t *payload = buffer + 3;
    data.pitch_channel_1 = getInt32BE(payload);
    data.pitch_channel_2 = getInt32BE(payload + 4);
    data.roll_channel = getInt32BE(payload + 8);
    data.telescopic_length = getUInt16BE(payload + 12);
    return true;
}

uint8_t RS422::checksum(const uint8_t *data, int length)
{
    uint8_t sum = 0;
    for (int i = 0; i < length; ++i)
    {
        sum = static_cast<uint8_t>(sum + data[i]);
    }
    return sum;
}

void RS422::putInt16BE(uint8_t *buffer, int16_t value)
{
    uint16_t raw = static_cast<uint16_t>(value);
    buffer[0] = static_cast<uint8_t>((raw >> 8) & 0xFF);
    buffer[1] = static_cast<uint8_t>(raw & 0xFF);
}

int16_t RS422::floatToInt16(float value)
{
    return floatToInt16(value, 1.0f);
}

int16_t RS422::floatToInt16(float value, float scale)
{
    if (value > 32767.0f)
    {
        return 32767;
    }
    if (value < -32768.0f)
    {
        return -32768;
    }
    float scaled = value * scale;
    if (scaled > 32767.0f)
    {
        return 32767;
    }
    if (scaled < -32768.0f)
    {
        return -32768;
    }
    return static_cast<int16_t>(std::lround(scaled));
}

int16_t RS422::doubleToInt16(double value)
{
    return doubleToInt16(value, 1.0);
}

int16_t RS422::doubleToInt16(double value, double scale)
{
    if (value > 32767.0)
    {
        return 32767;
    }
    if (value < -32768.0)
    {
        return -32768;
    }
    double scaled = value * scale;
    if (scaled > 32767.0)
    {
        return 32767;
    }
    if (scaled < -32768.0)
    {
        return -32768;
    }
    return static_cast<int16_t>(std::lround(scaled));
}

int32_t RS422::getInt32BE(const uint8_t *buffer)
{
    uint32_t raw = (static_cast<uint32_t>(buffer[0]) << 24) |
                   (static_cast<uint32_t>(buffer[1]) << 16) |
                   (static_cast<uint32_t>(buffer[2]) << 8) |
                   static_cast<uint32_t>(buffer[3]);
    return static_cast<int32_t>(raw);
}

uint16_t RS422::getUInt16BE(const uint8_t *buffer)
{
    return static_cast<uint16_t>((static_cast<uint16_t>(buffer[0]) << 8) |
                                 static_cast<uint16_t>(buffer[1]));
}

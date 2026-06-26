#ifndef TENSORRT_PRO_YOLOV8_MAIN_RS485_H
#define TENSORRT_PRO_YOLOV8_MAIN_RS485_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "../params/params.hpp"

class RS485
{
public:
    RS485();
    ~RS485();

    // 初始化RS485串口
    int init(const prj_params &p_params);

    // 按 2026-4-7 协议发送视觉数据:
    // EB 90 12 + 9 个 Big-Endian Int16 + 1 Byte 累加和
    // values[0..5] 单位 mm，values[6..8] 单位 deg，会转换为 0.01 deg。
    // 如果调用方暂时只提供 3 个位移值，其余协议字段自动填 0。
    bool sendVisionFrame(const std::vector<float> &values);

    // 旧接口兼容: 只发送前三个绝对位移字段，其余协议字段置 0。
    bool sendDoubleArray(const double arr[3]);
    bool sendFloatArray(const float arr[3]);

    // 读取串口收到的数据，打印原始字节；若收到 EB 91 协议帧，同时打印解析后的字段。
    bool receiveAndPrintAvailable(int timeout_ms);

    // 设置每次串口发送完成后的休眠时间，单位 us。
    void setSendIntervalUs(unsigned int interval_us);

    // 关闭串口
    void closePort();

    // 启用/禁用调试模式
    void setDebug(bool enable);

private:
    // 串口文件描述符
    int _fd = -1;
    // 调试模式
    bool _debug = false;
    unsigned int _send_interval_us = 0;

    std::vector<unsigned char> _rx_buffer;

    static const unsigned char FRAME_HEAD_1 = 0xEB;
    static const unsigned char TX_FRAME_HEAD_2 = 0x90;
    static const unsigned char RX_FRAME_HEAD_2 = 0x91;
    static const std::size_t TX_PAYLOAD_SIZE = 18;
    static const std::size_t RX_PAYLOAD_SIZE = 14;
    static const std::size_t TX_FIELD_COUNT = 9;

    unsigned char calculateChecksum(const unsigned char *data, std::size_t length) const;
    void appendInt16BE(std::vector<unsigned char> &buffer, int value) const;
    int floatToProtocolInt16(float value, float scale) const;
    bool writeAll(const unsigned char *data, std::size_t length);
    void parseReceiveBuffer();
    void printRawBytes(const unsigned char *data, std::size_t length) const;
    void printReceiveFrame(const unsigned char *payload, unsigned char checksum) const;
    int readInt32BE(const unsigned char *data) const;
    unsigned int readUInt16BE(const unsigned char *data) const;
};

#endif // TENSORRT_PRO_YOLOV8_MAIN_RS485_H

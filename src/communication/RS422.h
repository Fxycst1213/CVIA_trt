#ifndef TENSORRT_PRO_YOLOV8_MAIN_RS422_H
#define TENSORRT_PRO_YOLOV8_MAIN_RS422_H

#include <cstdint>
#include <string>
#include <termios.h>
#include "../params/params.hpp"

class RS422
{
public:
    struct TxData
    {
        // 单位: mm
        int16_t absolute[3] = {0, 0, 0};
        // 单位: mm
        int16_t relative[3] = {0, 0, 0};
        // 单位: 0.01 degree
        int16_t attitude[3] = {0, 0, 0};
    };

    struct RxData
    {
        // 单位: 0.001 degree
        int32_t pitch_channel_1 = 0;
        int32_t pitch_channel_2 = 0;
        int32_t roll_channel = 0;
        // 单位: mm
        uint16_t telescopic_length = 0;
    };

    RS422();
    ~RS422();

    // 使用当前工程参数初始化。
    int init(const prj_params &p_params);

    // 直接指定串口设备和波特率，推荐 RS422 新功能接入时使用。
    int init(const std::string &port, speed_t baudrate = B115200);

    bool sendFrame(const TxData &data);
    bool sendFrame(const int16_t absolute[3], const int16_t relative[3], const int16_t attitude[3]);

    // 按文档顺序发送9个字段：
    // absolute xyz(mm), relative xyz(mm), attitude xyz(degree, encoded as 0.01 degree).
    bool sendFloatArray(const float arr[9]);
    bool sendDoubleArray(const double arr[9]);

    // 读取并解析视觉接收帧: EB 91 0E payload(14) checksum。
    bool readFrame(RxData &data);

    void closePort();
    void setDebug(bool enable);

private:
    static const uint8_t TX_HEAD_1 = 0xEB;
    static const uint8_t TX_HEAD_2 = 0x90;
    static const uint8_t RX_HEAD_1 = 0xEB;
    static const uint8_t RX_HEAD_2 = 0x91;
    static const uint8_t TX_PAYLOAD_LEN = 18;
    static const uint8_t RX_PAYLOAD_LEN = 14;
    static const int TX_FRAME_SIZE = 22;
    static const int RX_FRAME_SIZE = 18;

    int _fd = -1;
    bool _debug = false;

    bool configurePort(speed_t baudrate);
    bool writeAll(const uint8_t *buffer, int size);
    bool readExact(uint8_t *buffer, int size);

    void packTxFrame(const TxData &data, uint8_t buffer[TX_FRAME_SIZE]);
    bool parseRxFrame(const uint8_t buffer[RX_FRAME_SIZE], RxData &data);

    static uint8_t checksum(const uint8_t *data, int length);
    static void putInt16BE(uint8_t *buffer, int16_t value);
    static int16_t floatToInt16(float value);
    static int16_t floatToInt16(float value, float scale);
    static int16_t doubleToInt16(double value);
    static int16_t doubleToInt16(double value, double scale);
    static int32_t getInt32BE(const uint8_t *buffer);
    static uint16_t getUInt16BE(const uint8_t *buffer);
};

#endif // TENSORRT_PRO_YOLOV8_MAIN_RS422_H

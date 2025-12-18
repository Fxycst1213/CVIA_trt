#ifndef TENSORRT_PRO_YOLOV8_MAIN_RS485_H
#define TENSORRT_PRO_YOLOV8_MAIN_RS485_H

#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <string>
#include "logger.hpp"

class RS485
{
public:
    // 初始化RS485串口
    int init(const std::string &port, int baud_rate = B115200);
    // 发送双精度浮点数数组
    bool sendDoubleArray(const double arr[3]);
    // 发送单精度浮点数数组（保持向后兼容）
    bool sendFloatArray(const float arr[3]);
    // 关闭串口
    void closePort();
    // 启用/禁用调试模式
    void setDebug(bool enable);
    // 构造函数和析构函数
    RS485();
    ~RS485();

private:
    // 双精度数据帧打包函数
    void packDoubleDataFrame(const double arr[3], unsigned char *buffer, int *frame_size);
    // 单精度数据帧打包函数
    void packFloatDataFrame(const float arr[3], unsigned char *buffer, int *frame_size);
    // CRC校验计算
    unsigned char calculateCRC(const unsigned char *data, int length);
    // 串口文件描述符
    int _fd = -1;
    // 帧头帧尾定义
    static const unsigned char FRAME_HEADER[2];
    static const unsigned char FRAME_FOOTER[2];
    // 数据标识符
    static const unsigned char DATA_TYPE_FLOAT = 0xF0;
    static const unsigned char DATA_TYPE_DOUBLE = 0xD0;
    // 调试模式
    bool _debug = false;
};

#endif // TENSORRT_PRO_YOLOV8_MAIN_RS485_H
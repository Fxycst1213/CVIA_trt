#ifndef TENSORRT_PRO_YOLOV8_MAIN_RS485_H
#define TENSORRT_PRO_YOLOV8_MAIN_RS485_H

#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <string>
#include <cstdio> // 建议添加，用于标准的输入输出定义
#include "../params/params.hpp"
#include "../logger/logger.hpp"

class RS485
{
public:
    RS485();
    ~RS485();

    // 初始化RS485串口
    int init(const prj_params &p_params);

    // 发送双精度浮点数数组
    // 格式: "x.xxx,y.yyy,z.zzz\n" (文本模式)
    bool sendDoubleArray(const double arr[3]);

    // 发送单精度浮点数数组
    // 格式: "x.xxx,y.yyy,z.zzz\n" (文本模式)
    bool sendFloatArray(const float arr[3]);

    // 关闭串口
    void closePort();

    // 启用/禁用调试模式
    void setDebug(bool enable);

private:
    // 串口文件描述符
    int _fd = -1;
    // 调试模式
    bool _debug = false;

    // ==========================================
    // 旧协议保留字段 (Legacy)
    // 注意：以下函数和变量在当前的文本发送逻辑中不再被调用
    // 保留它们是为了匹配 .cpp 文件中未删除的旧函数定义，
    // 如果你在 .cpp 中删除了旧函数，这里也可以删除。
    // ==========================================
    
    // 双精度数据帧打包函数 (未使用)
    void packDoubleDataFrame(const double arr[3], unsigned char *buffer, int *frame_size);
    // 单精度数据帧打包函数 (未使用)
    void packFloatDataFrame(const float arr[3], unsigned char *buffer, int *frame_size);
    // CRC校验计算 (未使用)
    unsigned char calculateCRC(const unsigned char *data, int length);

    // 帧头帧尾定义 (未使用)
    static const unsigned char FRAME_HEADER[2];
    static const unsigned char FRAME_FOOTER[2];
    
    // 数据标识符 (未使用)
    static const unsigned char DATA_TYPE_FLOAT = 0xF0;
    static const unsigned char DATA_TYPE_DOUBLE = 0xD0;
};

#endif // TENSORRT_PRO_YOLOV8_MAIN_RS485_H
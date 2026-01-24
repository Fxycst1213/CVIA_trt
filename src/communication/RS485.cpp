//
// Created by assistant on 2024/3/10.
//

#include <iostream>
#include <cstring>
#include <cstdio>   // 新增：用于 snprintf 格式化
#include <unistd.h> // 新增：用于 write, close
#include <fcntl.h>  // 用于 open 等
#include <termios.h> // 用于串口设置
#include "RS485.h"

using namespace std;

// 帧头定义: 0xAA 0x55 (注：新协议如果不使用，可保留定义防止编译错误，但逻辑中不再引用)
const unsigned char RS485::FRAME_HEADER[2] = {0xAA, 0x55};
// 帧尾定义: 0x0D 0x0A (回车换行)
const unsigned char RS485::FRAME_FOOTER[2] = {0x0D, 0x0A};

/*
    @brief: RS485构造函数
*/
RS485::RS485()
{
    // 可以在此初始化默认设置
    _debug = false; // 默认关闭调试模式
}

/*
    @brief: RS485析构函数
*/
RS485::~RS485()
{
    closePort();
}

/*
    @brief: 初始化RS485串口
    @param:
    1.const string& port : 串口设备路径，如 "/dev/ttyUSB0" 或 "/dev/ttyS0"
    2.int baud_rate : 波特率，默认9600
    @return: 成功返回0，失败返回-1
*/
int RS485::init(const prj_params &p_params)
{
    // 打开串口设备
    _fd = open(p_params.rs485_port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (_fd < 0)
    {
        cerr << "无法打开串口设备: " << p_params.rs485_port << endl;
        return -1;
    }
    // 恢复串口为阻塞状态
    fcntl(_fd, F_SETFL, 0);
    // 获取串口当前配置
    struct termios options;
    tcgetattr(_fd, &options);
    // 设置波特率
    cfsetispeed(&options, p_params.rs485_baudrate);
    cfsetospeed(&options, p_params.rs485_baudrate);
    // 设置数据位: 8位
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    // 设置奇偶校验: 无校验
    options.c_cflag &= ~PARENB;
    options.c_iflag &= ~INPCK;
    // 设置停止位: 1位
    options.c_cflag &= ~CSTOPB;
    // 设置流控制: 无
    options.c_cflag &= ~CRTSCTS;
    // 设置本地模式: 使能接收
    options.c_cflag |= (CLOCAL | CREAD);
    // 设置输入模式: 原始模式
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    // 设置输出模式: 原始模式
    options.c_oflag &= ~OPOST;
    // 设置等待时间和最小接收字符
    options.c_cc[VTIME] = 10; // 等待时间(单位: 0.1秒)
    options.c_cc[VMIN] = 0;   // 最小接收字符
    // 清空输入输出缓冲区
    tcflush(_fd, TCIOFLUSH);
    // 应用配置
    if (tcsetattr(_fd, TCSANOW, &options) != 0)
    {
        LOGE("设置串口属性失败");
        close(_fd);
        _fd = -1;
        return -1;
    }

    LOG("RS485串口初始化成功: %s 波特率: %d", p_params.rs485_port.c_str(), p_params.rs485_baudrate);
    return 0;
}

/*
    @brief: CRC8校验计算
    @note: 新协议已不再使用CRC，保留此函数以防其他模块调用或编译报错
*/
unsigned char RS485::calculateCRC(const unsigned char *data, int length)
{
    unsigned char crc = 0x00;
    unsigned char polynomial = 0x07; // CRC-8多项式

    for (int i = 0; i < length; i++)
    {
        crc ^= data[i];
        for (int j = 0; j < 8; j++)
        {
            if (crc & 0x80)
            {
                crc = (crc << 1) ^ polynomial;
            }
            else
            {
                crc <<= 1;
            }
        }
    }

    return crc;
}

/*
    @brief: 单精度数据帧打包
    @note: 已被新的文本协议取代，保留定义以兼容头文件
*/
void RS485::packFloatDataFrame(const float arr[3], unsigned char *buffer, int *frame_size)
{
    // 旧协议实现，新逻辑中不再调用
    int index = 0;
    buffer[index++] = FRAME_HEADER[0];
    buffer[index++] = FRAME_HEADER[1];
    buffer[index++] = DATA_TYPE_FLOAT;
    unsigned char data_length = 12;
    buffer[index++] = data_length;
    unsigned char *float_ptr = (unsigned char *)arr;
    for (int i = 0; i < data_length; i++)
    {
        buffer[index++] = float_ptr[i];
    }
    unsigned char crc = calculateCRC(&buffer[2], data_length + 2);
    buffer[index++] = crc;
    buffer[index++] = FRAME_FOOTER[0];
    buffer[index++] = FRAME_FOOTER[1];
    *frame_size = index;
}

/*
    @brief: 双精度数据帧打包
    @note: 已被新的文本协议取代，保留定义以兼容头文件
*/
void RS485::packDoubleDataFrame(const double arr[3], unsigned char *buffer, int *frame_size)
{
    // 旧协议实现，新逻辑中不再调用
    int index = 0;
    buffer[index++] = FRAME_HEADER[0];
    buffer[index++] = FRAME_HEADER[1];
    buffer[index++] = DATA_TYPE_DOUBLE;
    unsigned char data_length = 24;
    buffer[index++] = data_length;
    unsigned char *double_ptr = (unsigned char *)arr;
    for (int i = 0; i < data_length; i++)
    {
        buffer[index++] = double_ptr[i];
    }
    unsigned char crc = calculateCRC(&buffer[2], data_length + 2);
    buffer[index++] = crc;
    buffer[index++] = FRAME_FOOTER[0];
    buffer[index++] = FRAME_FOOTER[1];
    *frame_size = index;
}

/*
    @brief: 发送单精度浮点数数组
    @brief: 修改为ASCII格式发送，格式: "x.xxx,y.yyy,z.zzz\n"
    @param:
    1.const float arr[3] : 要发送的浮点数数组
    @return: 成功返回true，失败返回false
*/
bool RS485::sendFloatArray(const float arr[3])
{
    if (_fd < 0)
    {
        LOGE("串口未初始化");
        return false;
    }

    // 定义发送缓冲区 (文本格式)
    char buffer[128]; 
    memset(buffer, 0, sizeof(buffer));

    // 使用snprintf格式化字符串
    // %.3f: 保留三位小数
    // ,: 逗号分隔
    // \n: 换行符
    int len = snprintf(buffer, sizeof(buffer), "%.3f,%.3f,%.3f", arr[0], arr[1], arr[2]);

    // 发送数据帧
    int bytes_written = write(_fd, buffer, len);

    // 等待数据发送完成
    // tcdrain(_fd);

    if (bytes_written == len)
    {
        if (_debug)
        {
            LOG("成功发送文本数据%d字节", bytes_written);
            // 此时buffer本身就是可读字符串
            LOG("发送的数据: %s", buffer); 
        }
        return true;
    }
    else
    {
        LOGE("发送失败，预期%d字节，实际发送%d字节", len, bytes_written);
        return false;
    }
}

/*
    @brief: 发送双精度浮点数数组
    @brief: 修改为ASCII格式发送，格式: "x.xxx,y.yyy,z.zzz\n"
    @param:
    1.const double arr[3] : 要发送的双精度浮点数数组
    @return: 成功返回true，失败返回false
*/
bool RS485::sendDoubleArray(const double arr[3])
{
    if (_fd < 0)
    {
        LOGE("串口未初始化");
        return false;
    }

    // 定义发送缓冲区 (double可能比float长，给大一点空间)
    char buffer[256]; 
    memset(buffer, 0, sizeof(buffer));

    // 使用snprintf格式化字符串
    // %.3lf: double类型保留三位小数
    int len = snprintf(buffer, sizeof(buffer), "%.3lf,%.3lf,%.3lf\n", arr[0], arr[1], arr[2]);

    // 发送数据帧
    int bytes_written = write(_fd, buffer, len);

    // 等待数据发送完成
    // tcdrain(_fd); 
    
    if (bytes_written == len)
    {
        if (_debug)
        {
            LOG("成功发送双精度文本数据%d字节", bytes_written);
            LOG("发送的数据: %s", buffer);
        }
        return true;
    }
    else
    {
        LOGE("发送失败，预期%d字节，实际发送%d字节", len, bytes_written);
        return false;
    }
}

/*
    @brief: 启用/禁用调试模式
    @param:
    1.bool enable : true启用调试模式，false禁用
*/
void RS485::setDebug(bool enable)
{
    _debug = enable;
}

/*
    @brief: 关闭串口
*/
void RS485::closePort()
{
    if (_fd >= 0)
    {
        close(_fd);
        _fd = -1;
        LOG("RS485串口已关闭");
    }
}
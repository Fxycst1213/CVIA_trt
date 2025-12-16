//
// Created by assistant on 2024/3/10.
//

#include <iostream>
#include <cstring>
#include "RS485.h"

using namespace std;

// 帧头定义: 0xAA 0x55
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
int RS485::init(const string &port, int baud_rate)
{
    // 打开串口设备
    _fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (_fd < 0)
    {
        cerr << "无法打开串口设备: " << port << endl;
        return -1;
    }
    // 恢复串口为阻塞状态
    fcntl(_fd, F_SETFL, 0);
    // 获取串口当前配置
    struct termios options;
    tcgetattr(_fd, &options);
    // 设置波特率
    cfsetispeed(&options, baud_rate);
    cfsetospeed(&options, baud_rate);
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
        cerr << "串口配置失败" << endl;
        close(_fd);
        _fd = -1;
        return -1;
    }

    cout << "RS485串口初始化成功: " << port << " 波特率: " << baud_rate << endl;
    return 0;
}

/*
    @brief: CRC8校验计算
    @param:
    1.const unsigned char* data : 数据指针
    2.int length : 数据长度
    @return: CRC校验值
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
    @param:
    1.const float arr[3] : 要发送的浮点数数组
    2.unsigned char* buffer : 输出缓冲区
    3.int* frame_size : 输出帧大小
*/
void RS485::packFloatDataFrame(const float arr[3], unsigned char *buffer, int *frame_size)
{
    int index = 0;
    // 1. 添加帧头
    buffer[index++] = FRAME_HEADER[0];
    buffer[index++] = FRAME_HEADER[1];
    // 2. 添加数据类型标识
    buffer[index++] = DATA_TYPE_FLOAT;
    // 3. 添加数据长度信息 (3个float = 12字节)
    unsigned char data_length = 12;
    buffer[index++] = data_length;
    // 4. 添加浮点数数据
    unsigned char *float_ptr = (unsigned char *)arr;
    for (int i = 0; i < data_length; i++)
    {
        buffer[index++] = float_ptr[i];
    }
    // 5. 计算并添加CRC校验 (对数据部分进行校验: 类型+长度+数据)
    unsigned char crc = calculateCRC(&buffer[2], data_length + 2); // +2是包含类型和长度字节
    buffer[index++] = crc;
    // 6. 添加帧尾
    buffer[index++] = FRAME_FOOTER[0];
    buffer[index++] = FRAME_FOOTER[1];
    *frame_size = index;
    // 调试输出
    if (_debug)
    {
        cout << "单精度数据帧(" << *frame_size << "字节): ";
        for (int i = 0; i < *frame_size; i++)
        {
            printf("%02X ", buffer[i]);
        }
        cout << endl;
    }
}

/*
    @brief: 双精度数据帧打包
    @param:
    1.const double arr[3] : 要发送的双精度浮点数数组
    2.unsigned char* buffer : 输出缓冲区
    3.int* frame_size : 输出帧大小
*/
void RS485::packDoubleDataFrame(const double arr[3], unsigned char *buffer, int *frame_size)
{
    int index = 0;

    // 1. 添加帧头
    buffer[index++] = FRAME_HEADER[0];
    buffer[index++] = FRAME_HEADER[1];

    // 2. 添加数据类型标识
    buffer[index++] = DATA_TYPE_DOUBLE;

    // 3. 添加数据长度信息 (3个double = 24字节)
    unsigned char data_length = 24;
    buffer[index++] = data_length;

    // 4. 添加双精度浮点数数据
    unsigned char *double_ptr = (unsigned char *)arr;
    for (int i = 0; i < data_length; i++)
    {
        buffer[index++] = double_ptr[i];
    }

    // 5. 计算并添加CRC校验 (对数据部分进行校验: 类型+长度+数据)
    unsigned char crc = calculateCRC(&buffer[2], data_length + 2); // +2是包含类型和长度字节
    buffer[index++] = crc;

    // 6. 添加帧尾
    buffer[index++] = FRAME_FOOTER[0];
    buffer[index++] = FRAME_FOOTER[1];

    *frame_size = index;

    // 调试输出
    if (_debug)
    {
        cout << "双精度数据帧(" << *frame_size << "字节): ";
        for (int i = 0; i < *frame_size; i++)
        {
            printf("%02X ", buffer[i]);
        }
        cout << endl;
    }
}

/*
    @brief: 发送单精度浮点数数组
    @param:
    1.const float arr[3] : 要发送的浮点数数组
    @return: 成功返回true，失败返回false
*/
bool RS485::sendFloatArray(const float arr[3])
{
    if (_fd < 0)
    {
        cerr << "串口未初始化" << endl;
        return false;
    }

    // 打包数据帧
    unsigned char frame_buffer[22]; // 帧头2 + 类型1 + 长度1 + 数据12 + CRC1 + 帧尾2 = 19字节
    int frame_size = 0;
    packFloatDataFrame(arr, frame_buffer, &frame_size);

    // 发送数据帧
    int bytes_written = write(_fd, frame_buffer, frame_size);

    // 等待数据发送完成
    tcdrain(_fd);

    if (bytes_written == frame_size)
    {
        if (_debug)
        {
            cout << "成功发送单精度数据" << bytes_written << "字节" << endl;
            cout << "发送的数据: [" << arr[0] << ", " << arr[1] << ", " << arr[2] << "]" << endl;
        }
        return true;
    }
    else
    {
        cerr << "发送失败，预期" << frame_size << "字节，实际发送" << bytes_written << "字节" << endl;
        return false;
    }
}

/*
    @brief: 发送双精度浮点数数组
    @param:
    1.const double arr[3] : 要发送的双精度浮点数数组
    @return: 成功返回true，失败返回false
*/
bool RS485::sendDoubleArray(const double arr[3])
{
    if (_fd < 0)
    {
        cerr << "串口未初始化" << endl;
        return false;
    }

    // 打包数据帧
    unsigned char frame_buffer[32]; // 帧头2 + 类型1 + 长度1 + 数据24 + CRC1 + 帧尾2 = 31字节
    int frame_size = 0;
    packDoubleDataFrame(arr, frame_buffer, &frame_size);

    // 发送数据帧
    int bytes_written = write(_fd, frame_buffer, frame_size);

    // 等待数据发送完成
    // tcdrain(_fd);
    // std::cout<<"等待fd改变"<<std::endl;
    if (bytes_written == frame_size)
    {
        if (_debug)
        {
            cout << "成功发送双精度数据" << bytes_written << "字节" << endl;
            cout << "发送的数据: [" << arr[0] << ", " << arr[1] << ", " << arr[2] << "]" << endl;
        }
        return true;
    }
    else
    {
        cerr << "发送失败，预期" << frame_size << "字节，实际发送" << bytes_written << "字节" << endl;
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
        cout << "RS485串口已关闭" << endl;
    }
}
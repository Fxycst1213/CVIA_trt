

#include <iostream>
#include "client.h"

/*
    @brief: socket init
    @param:
    1.const string& ip : server ip address
    2.const int& port : which port of the server to use
*/
int client::init(const string &ip, const int &port)
{
    memset(&_remoteAddress, 0, sizeof(_remoteAddress));
    _remoteAddress.sin_family = AF_INET;                    // 设置为IP通信
    _remoteAddress.sin_addr.s_addr = inet_addr(ip.c_str()); // 服务器IP地址
    _remoteAddress.sin_port = htons(port);                  // 服务器端口号

    // 创建socket通信文件描述符，设置协议
    if ((_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
    {
        std::cout << "socket error" << std::endl;
        return -1;
    }

    // 尝试进行tcp连接至服务器
    if (connect(_fd, (struct sockaddr *)&_remoteAddress, sizeof(struct sockaddr)) < 0)
    {
        std::cout << "connect error" << std::endl;
        return -1;
    }
    return 0;
}

/*
    @brief: send data(block mode)
    @param:
    1.char *buffer : ptr to the first address of the data that to be sent
    2.int size : how many bytes you want to send
*/
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
    close(_fd);
}

//
// Created by wwwindows on 2024/3/9.
//

#ifndef CLIENT_H
#define CLIENT_H

#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using namespace std;

class client
{
public:
    bool SendAll(char *buffer, int size);
    int init(const std::string &ip, const int &port);
    client() {};
    ~client();

private:
    struct sockaddr_in _remoteAddress;
    int _fd = -1;
};

#endif // TENSORRT_PRO_YOLOV8_MAIN_CLIENT_H

#pragma once
#include <netinet/in.h>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "../params/params.hpp"
#include <chrono>
#include <ratio>
using namespace std;

class client
{
public:
    client() = default;
    ~client();
    client(const client &) = delete;
    client &operator=(const client &) = delete;

    void init(const prj_params &p_params);
    bool pack_and_send(const Resultframe &frame);

private:
    bool connect_to_server();
    bool SendAll(const void *buffer, size_t size);
    void disconnect();

    struct sockaddr_in _remoteAddress;
    int _fd = -1;
    int _socket_mode = -1;
    int _jpeg_width = 0;
    int _jpeg_quality = 50;
    int _send_timeout_ms = 250;
    std::vector<uchar> _jpeg;
    std::vector<unsigned char> _packet;
    std::chrono::steady_clock::time_point _next_connect{};
};

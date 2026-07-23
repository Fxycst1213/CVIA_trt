#pragma once
#include <netinet/in.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp> // 需要 cv::Mat
#include "../params/params.hpp"
// 假设 bbox 定义在这里，如果在其他地方请 include 对应头文件
#include "../yolov_pose/pose.hpp" // (假设你的 bbox 定义在这里)
#include <chrono>
#include <ratio>
#include <fstream>
using namespace std;

class client
{
public:
    client() {};
    ~client();

    void init(const prj_params &p_params);
    bool pack_and_send(const Resultframe &frame);
    bool send_udp_result(const std::vector<float> &result, uint64_t timestamp);
    void record_pose(const Resultframe &frame);

private:
    // 内部使用的发送函数，设为 private 也可以，或者 public 也可以
    bool SendAll(char *buffer, int size);
    void save_pnp_pose(const Resultframe &frame);

    struct sockaddr_in _remoteAddress;
    struct sockaddr_in _udpAddress;
    int _fd = -1;
    int _udp_fd = -1;
    char *_buffer = nullptr;

    // --- 新增：记录每一部分的尺寸 ---
    int _socket_mode = -1;
    int _total_send_size = 0;
    int _img_size_bytes = 0;
    int _kpt_size_bytes = 0;
    int _pose_size_bytes = 0;
    int _keyPoint_box = 0;
    std::string _resolution;
    bool _udp_enabled = false;
    bool _tcp_enabled = false;
    bool _save_pnp_results = true;
    std::string _preview_dir;
    std::string _pnp_result_dir;
    std::ofstream _pnp_csv;
    size_t _csv_rows_since_flush = 0;
    std::chrono::steady_clock::time_point _last_preview_at;
    std::chrono::milliseconds _preview_interval{100};
};

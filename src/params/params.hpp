#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "pose_params.hpp"

struct Resultframe
{
    cv::Mat rgb;
    cv::Mat rgb_secondary;
    std::vector<model::pose::bbox> bboxes;
    std::vector<float> pose_result;
    std::vector<float> udp_result;
    uint64_t timestamp;
    uint64_t secondary_timestamp = 0;
};

struct camera_params
{
    std::string name = "IR Camera";
    int cameraID = 0;
    int cameraframe = 30;
    std::string resolution = "HD1080";
    double auto_exposure_mode = 1.0;
    bool apply_exposure = true;
    double exposure = 78.0;
    bool auto_white_balance = false;
    bool apply_white_balance_temperature = true;
    double white_balance_temperature = 4600.0;
    double brightness = -64.0;
    double contrast = 100.0;
    double sharpness = 100.0;
};

struct tcp_params
{
    // 4. 将常量改为 static constexpr
    // 这样它们只占用一份内存，且可以用作数组长度声明
    static constexpr int Point_num = 10;
    static constexpr int KeyPoint_box = (Point_num * 3 + 5);
    static constexpr int KEYPOINTS_BUFSIZE = KeyPoint_box * 4;
    static constexpr int POSE_BUFSIZE = 8 * 4;
    static constexpr int POSE_DATE_NUM = 10;
    static constexpr int IMAGE_COUNT_DUAL = 2;

    // 这两个是变量，不加 const
    int IMG_SIZE = 1920 * 1080 * 3;
    int DATA_SIZE = 0;
    int SOCKETSEND_SIZE = 0;
};

struct prj_params
{
    // 2. 给 int 类型赋默认值，防止随机数
    int H = 0;
    int W = 0;
    camera_params detect_camera;
    camera_params photo_camera;
    std::string ip; // 默认为空字符串
    int port = 0;   // 赋默认值
    std::string udp_ip;
    int udp_port = 0;
    bool enable_udp = true;
    int socket_mode = 0;
    tcp_params t_params;
    // 串口参数保留给 RS485 模块；当前主流程已停用串口发送，改用 UDP 上传结果。
    std::string rs485_port = "/dev/ttyUSB0";
    int rs485_baudrate = 57600;
};

#endif // PARAMS_HPP

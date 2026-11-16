#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <vector>
#include <array>
#include <cstdint>
#include <string>
#include <termios.h>
#include <opencv2/opencv.hpp>
#include "pose_params.hpp"

struct CameraCalibration
{
    uint32_t width = 0;
    uint32_t height = 0;
    bool right_eye = true;
    std::array<double, 4> intrinsics{}; // fx, fy, cx, cy, for the raw source image.
    std::array<double, 12> distortion{}; // OpenCV: k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4.
};

struct Resultframe
{
    cv::Mat rgb;
    std::vector<model::pose::bbox> bboxes;
    std::vector<float> pose_result;
    std::vector<float> rs485_result;
    uint64_t timestamp;
    bool pnp_valid = false;
    std::array<float, 3> pnp_rvec{};  // Rodrigues vector: body to camera, radians.
    std::array<float, 3> pnp_tvec{};  // Unfiltered body origin in camera coordinates, mm.
    CameraCalibration calibration;
};

struct tcp_params
{
    // 4. 将常量改为 static constexpr
    // 这样它们只占用一份内存，且可以用作数组长度声明
    static constexpr int Point_num = model::pose::NUM_KEYPOINTS;
    static constexpr int KeyPoint_box = (Point_num * 3 + 5);
    static constexpr int KEYPOINTS_BUFSIZE = KeyPoint_box * 4;
    static constexpr int POSE_BUFSIZE = 8 * 4;
    static constexpr int POSE_DATE_NUM = 7;

    // 这两个是变量，不加 const
    int IMG_SIZE = 1920 * 1080 * 3;
    int DATA_SIZE = 0;
    int SOCKETSEND_SIZE = 0;
    int jpeg_width = 0;  // 0: original size; keypoints follow the JPEG dimensions.
    int jpeg_quality = 50;
    int max_fps = 30;
    int send_timeout_ms = 250;
};

struct prj_params
{
    // 2. 给 int 类型赋默认值，防止随机数
    int H = 0;
    int W = 0;
    std::string resolution = "HD1080"; // 3. 去掉 using namespace std，改为 std::string
    int cameraID = 0;
    std::string camera_eye = "right";
    std::string ip; // 默认为空字符串
    int port = 0;   // 赋默认值
    int socket_mode = 0;
    tcp_params t_params;
    std::string rs485_port = "/dev/ttyUSB0";
    speed_t rs485_baudrate = B115200;
    bool verbose = false;
    bool enable_serial = true;
};

#endif // PARAMS_HPP

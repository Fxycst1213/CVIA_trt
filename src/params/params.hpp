#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <array>
#include "pose_params.hpp"

struct Resultframe
{
    cv::Mat rgb;
    cv::Mat rgb_secondary;
    std::vector<model::pose::bbox> bboxes;
    std::vector<float> pose_result;
    std::vector<float> udp_result;
    // 刚体坐标系原点 (0,0,0) 与当前 PnP 模型特征点在红外原图上的重投影。
    cv::Point2d reprojected_origin;
    bool reprojected_origin_valid = false;
    std::vector<cv::Point2d> reprojected_points;
    bool pose_valid = false;
    // timestamp 保留毫秒协议；capture_timestamp_ns 用于 PTP 时间域精确配对。
    uint64_t timestamp;
    uint64_t capture_timestamp_ns = 0;
    uint64_t dequeue_timestamp_ns = 0;
    uint64_t publish_timestamp_ns = 0;
    bool driver_timestamp = false;
    bool timestamp_start_of_exposure = false;
    uint64_t secondary_timestamp = 0;
};

struct camera_params
{
    std::string name = "IR Camera";
    int cameraID = 0;
    int cameraframe = 60;
    std::string resolution = "HD1080";
    // false 时只设置采集格式、分辨率和 FPS，不写曝光/白平衡/画质控制。
    bool apply_image_controls = true;
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
    // 输入源与网页预览。配置保存后在下次启动 trt 时生效。
    std::string input_mode = "camera"; // camera | folder
    std::string folder_path = "data/source";
    bool folder_loop = true;
    int folder_interval_ms = 17;
    std::string preview_dir = "web_monitor/runtime";
    int preview_fps = 30;
    bool enable_tcp = false;
    bool save_pnp_results = true;
    std::string pnp_result_dir = "./PNP_result";

    // PnP 相机标定参数，以及 T_M_C（相机坐标系到 NOKOV 动捕坐标系）的 4x4 外参。
    std::array<double, 9> camera_matrix = {{1078.39302318049, 0, 939.772377680377,
                                            0, 1078.53917414318, 595.175265662463,
                                            0.0, 0.0, 1.0}};
    std::array<double, 5> distortion = {{-0.0630336003089920, 0.187345652299030, 0, 0, -0.163375015304349}};
    std::array<double, 16> extrinsic = {{-0.036791138,     0.480521384,     0.87621094,  -1908.919479898,
                        0.999094004,     0.036455454,     0.021958388,  -388.940632775,
                        -0.021391192313944075,0.8762249705686078,-0.48142727160333376,647.8513827001716,
                        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00}};
    // 串口参数保留给 RS485 模块；当前主流程已停用串口发送，改用 UDP 上传结果。
    std::string rs485_port = "/dev/ttyUSB0";
    int rs485_baudrate = 57600;
};

#endif // PARAMS_HPP

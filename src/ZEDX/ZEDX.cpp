#include "ZEDX.h"
#include <string>
#include <iostream>

using namespace sl;

/*
    @brief: zed init setting and open the camera.
    @param:
    1.int ID : which camera will be used
    2.std::string resolution : picture resolution
    3. int frame: fps
    4. bool enable_fill_mode: enable depth fill mode or not
*/
int ZEDX::init(int ID, std::string resolution, int frame, bool enable_fill_mode)
{
    if (resolution == "HD1080")
        _init_parameters.camera_resolution = RESOLUTION::HD1080;
    else
        _init_parameters.camera_resolution = RESOLUTION::SVGA;
    _init_parameters.camera_fps = frame;
    _init_parameters.depth_mode = sl::DEPTH_MODE::NONE;
    // _init_parameters.grab_compute_capping_fps = 30;
    _init_parameters.input.setFromCameraID(ID);
    _runtime_parameters.enable_fill_mode = true;

    auto returned_state = _zedx.open(_init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS)
    {
        std::cout << "Error:" << returned_state << "." << std::endl;
        return -1;
    }

    setIntrinsic();

    return 0;
}

/*
    @brief: lazy mode to get the only instance of camera

*/

ZEDX *ZEDX::GetInstance()
{
    static ZEDX zedx;
    return &zedx;
}

cv::Mat ZEDX::getK() const
{
    return _K;
}

/*
    @brief: get the image data. meanwhile turn rgb data into cv::Mat format
*/
int ZEDX::grab_frame(ZEDframe *frame)
{
    sl::Mat img;

    if (_zedx.grab(_runtime_parameters) == ERROR_CODE::SUCCESS)
    {
        _zedx.retrieveImage(img, VIEW::LEFT);
        frame->timestamp = img.timestamp.getMilliseconds();
        cv::Mat tmp = ZEDX::slMat2cvMat(img);
        cv::cvtColor(tmp, *(frame->rgb_ptr), cv::COLOR_BGRA2BGR);
        return 0;
    }
    else
        return -1;
}

cv::Mat ZEDX::slMat2cvMat(sl::Mat &input)
{
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

ZEDX::~ZEDX()
{
    _zedx.close();
}

/*
    @brief: set the intrinsic matrix
*/
void ZEDX::setIntrinsic()
{
    CalibrationParameters calibration_params = _zedx.getCameraInformation().camera_configuration.calibration_parameters;
    // Focal length of the left eye in pixels
    double fx = (double)calibration_params.left_cam.fx;
    double fy = (double)calibration_params.left_cam.fy;
    double cx = (double)calibration_params.left_cam.cx;
    double cy = (double)calibration_params.left_cam.cy;
    _K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
          0.0, fy, cy,
          0.0, 0.0, 1.0);
}

// #include "ZEDX.h"
// #include <string>
// #include <iostream>
// #include <vector>
// #include <opencv2/opencv.hpp>
// #include <thread> // 用于 sleep
// #include <chrono> // 用于时间戳获取

// using namespace sl;

// static std::vector<cv::String> g_image_files;                        // 存储所有图片路径
// static int g_current_img_index = 0;                                  // 当前读取到了第几张
// static const std::string IMAGE_FOLDER = "/home/cvia/yifei/images3/"; // 图片文件夹路径

// /*
//     @brief: zed init setting (修改版：离线模式)
//     不再打开真实相机，而是加载文件夹内的图片列表
// */
// int ZEDX::init(int ID, std::string resolution, int frame, bool enable_fill_mode)
// {
//     std::cout << "[Offline Mode] Initializing Image Loader..." << std::endl;

//     // 1. 读取文件夹下所有的 jpg 图片 (如果是png，请改为 *.png)
//     // cv::glob 会自动按文件名排序，这对回放序列很重要
//     cv::glob(IMAGE_FOLDER + "*.png", g_image_files);

//     if (g_image_files.empty())
//     {
//         std::cerr << "[Error] No .png images found in: " << IMAGE_FOLDER << std::endl;
//         return -1;
//     }

//     // 统计数量
//     std::cout << "[Success] Total images found: " << g_image_files.size() << std::endl;
//     g_current_img_index = 0;

//     return 0;
// }

// /*
//     @brief: lazy mode to get the only instance of camera
// */
// ZEDX *ZEDX::GetInstance()
// {
//     static ZEDX zedx;
//     return &zedx;
// }

// /*
//     @brief: get the image data (修改版：从本地读取)
// */
// int ZEDX::grab_frame(ZEDframe *frame)
// {
//     // 1. 模拟 40ms 的帧间隔 (25 FPS)
//     std::this_thread::sleep_for(std::chrono::milliseconds(10));
//     // 2. 检查索引是否越界，如果越界则从头开始（循环播放）
//     if (g_current_img_index >= g_image_files.size())
//     {
//         // 读完了，打印一次提示
//         static bool has_printed_end = false;
//         if (!has_printed_end)
//         {
//             std::cout << "============================================" << std::endl;
//             std::cout << "[End of Sequence] All " << g_image_files.size() << " images processed." << std::endl;
//             std::cout << "[Info] Stopping data stream." << std::endl;
//             std::cout << "============================================" << std::endl;
//             has_printed_end = true;
//         }

//         // 为了防止你的主程序 while(1) 疯狂刷屏 "get image failed"，
//         // 这里让线程稍微睡久一点，减轻CPU负担，或者你可以选择在这里直接 exit(0) 退出程序
//         std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//         return -1; // 返回错误码，告诉主程序没有取到数据
//     }

//     // 3. 读取图片
//     // cv::imread 读出来的直接就是 BGR 格式的 cv::Mat，不需要做复杂的转换
//     std::string current_file = g_image_files[g_current_img_index];
//     cv::Mat img_bgr = cv::imread(current_file);

//     if (img_bgr.empty())
//     {
//         std::cerr << "[Error] Failed to read image: " << current_file << std::endl;
//         // 跳过这一帧或重试，这里选择跳过
//         g_current_img_index++;
//         return -1;
//     }

//     // 4. 将读取的数据拷贝到 frame->rgb_ptr
//     // air_refuel.cpp 中分配的是 CV_8UC3 (1080, 1920)
//     // 确保读取的图片尺寸一致，否则需要 resize
//     if (img_bgr.size() != frame->rgb_ptr->size())
//     {
//         cv::resize(img_bgr, *(frame->rgb_ptr), frame->rgb_ptr->size());
//     }
//     else
//     {
//         img_bgr.copyTo(*(frame->rgb_ptr));
//     }

//     // 5. 生成时间戳
//     // 这是一个系统当前时间的毫秒数，用于模拟实时流
//     auto now = std::chrono::system_clock::now();

//     // 将这个时间点转换为“自1970年以来的毫秒数” (即：时间戳)
//     uint64_t current_timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

//     // 赋值给 frame
//     frame->timestamp = current_timestamp_ms;

//     // 6. 索引递增
//     g_current_img_index++;

//     return 0; // 返回 0 代表 grab 成功
// }

// ZEDX::~ZEDX()
// {
//     // 离线模式不需要做任何释放操作
// }

// // 内参设置函数：必须要有实现，哪怕是空的
// void ZEDX::setIntrinsic()
// {
//     // 离线模式不需要读取内参
// }
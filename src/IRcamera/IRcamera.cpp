#include <iostream>
#include <chrono>
#include "IRcamera.h"

void IRCamera::init(const camera_params &params)
{
    // 1. 强制使用 V4L2 后端打开相机
    _cap.open(params.cameraID, cv::CAP_V4L2);
    if (!_cap.isOpened())
    {
        LOGE("%s INIT ERROR: Cannot open camera ID %d", params.name.c_str(), params.cameraID);
        return;
    }

    // 2. 极其重要：设置 MJPG 编码格式以支持高帧率和高分辨率
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    _cap.set(cv::CAP_PROP_FOURCC, fourcc);

    // 3. 解析并设置分辨率
    if (params.resolution == "HD1080")
    {
        _width = 1920;
        _height = 1080;
    }
    else
    {
        // 默认作为 HD720 处理
        _width = 1280;
        _height = 720;
    }

    // v4l2-ctl -d /dev/video0 --list-ctrls  查看相机参数
    _cap.set(cv::CAP_PROP_FRAME_WIDTH, _width);
    _cap.set(cv::CAP_PROP_FRAME_HEIGHT, _height);
    _cap.set(cv::CAP_PROP_FPS, params.cameraframe);

    // 4. 每路相机独立应用曝光/白平衡/成像参数
    _cap.set(cv::CAP_PROP_AUTO_EXPOSURE, params.auto_exposure_mode);
    if (params.apply_exposure)
    {
        _cap.set(cv::CAP_PROP_EXPOSURE, params.exposure);
    }

    _cap.set(cv::CAP_PROP_AUTO_WB, params.auto_white_balance ? 1.0 : 0.0);
    if (params.apply_white_balance_temperature)
    {
        _cap.set(cv::CAP_PROP_WB_TEMPERATURE, params.white_balance_temperature);
    }

    _cap.set(cv::CAP_PROP_BRIGHTNESS, params.brightness);
    _cap.set(cv::CAP_PROP_CONTRAST, params.contrast);
    _cap.set(cv::CAP_PROP_SHARPNESS, params.sharpness);

    std::cout << "--- " << params.name << " 硬件级参数锁定 ---" << std::endl;
    std::cout << "曝光模式: " << _cap.get(cv::CAP_PROP_AUTO_EXPOSURE) << " (期望: 1)" << std::endl;
    if (params.apply_exposure)
    {
        std::cout << "曝光值:   " << _cap.get(cv::CAP_PROP_EXPOSURE) << " (期望: " << params.exposure << ")" << std::endl;
    }
    std::cout << "亮度:     " << _cap.get(cv::CAP_PROP_BRIGHTNESS) << " (期望: " << params.brightness << ")" << std::endl;
    std::cout << "对比度:   " << _cap.get(cv::CAP_PROP_CONTRAST) << " (期望: " << params.contrast << ")" << std::endl;
    std::cout << "清晰度:   " << _cap.get(cv::CAP_PROP_SHARPNESS) << " (期望: " << params.sharpness << ")" << std::endl;
    std::cout << "自动白平衡: " << _cap.get(cv::CAP_PROP_AUTO_WB) << " (期望: " << (params.auto_white_balance ? 1 : 0) << ")" << std::endl;
    if (params.apply_white_balance_temperature)
    {
        std::cout << "色温:     " << _cap.get(cv::CAP_PROP_WB_TEMPERATURE) << " (期望: " << params.white_balance_temperature << ")" << std::endl;
    }

    // 5. 初始化内参矩阵和计时器
    _timer = std::make_shared<timer::Timer>(logger::Level::VERB);

    std::cout << "--- " << params.name << " 初始化成功 ---" << std::endl;
    std::cout << "生效分辨率: " << _cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x" << _cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "生效 FPS: " << _cap.get(cv::CAP_PROP_FPS) << std::endl;
}

void IRCamera::grab_frame(IRFrame *frame)
{
    // 安全拦截：确保相机打开且指针已分配内存
    if (!_cap.isOpened() || frame == nullptr || frame->rgb_ptr == nullptr)
        return;

    cv::Mat tmp;
    if (_cap.read(tmp))
    {
        // 1. 获取最精确的时间戳
        double driver_timestamp_ms = _cap.get(cv::CAP_PROP_POS_MSEC);
        if (driver_timestamp_ms > 0)
        {
            frame->timestamp = static_cast<uint64_t>(driver_timestamp_ms);
        }
        else
        {
            auto now = std::chrono::system_clock::now();
            frame->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        }
        tmp.copyTo(*(frame->rgb_ptr));
    }
    else
    {
        LOGE("IR Camera Grabframe ERROR");
    }
}

IRCamera::~IRCamera()
{
    if (_cap.isOpened())
    {
        _cap.release();
    }
}

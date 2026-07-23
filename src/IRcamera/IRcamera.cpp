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
    // 只保留驱动侧最新帧，避免推理速度低于采集速度时读取历史帧。
    _cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

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

    // 4. 只有需要调节的相机才写入成像参数；可见光相机可保持设备默认值。
    if (params.apply_image_controls)
    {
        _cap.set(cv::CAP_PROP_AUTO_EXPOSURE, params.auto_exposure_mode);
        if (params.apply_exposure)
            _cap.set(cv::CAP_PROP_EXPOSURE, params.exposure);

        _cap.set(cv::CAP_PROP_AUTO_WB, params.auto_white_balance ? 1.0 : 0.0);
        if (params.apply_white_balance_temperature)
            _cap.set(cv::CAP_PROP_WB_TEMPERATURE, params.white_balance_temperature);

        _cap.set(cv::CAP_PROP_BRIGHTNESS, params.brightness);
        _cap.set(cv::CAP_PROP_CONTRAST, params.contrast);
        _cap.set(cv::CAP_PROP_SHARPNESS, params.sharpness);
    }

    if (params.apply_image_controls)
    {
        std::cout << "--- " << params.name << " 成像参数配置 ---" << std::endl;
        std::cout << "曝光模式: " << _cap.get(cv::CAP_PROP_AUTO_EXPOSURE)
                  << " (期望: " << params.auto_exposure_mode << ")" << std::endl;
        if (params.apply_exposure)
            std::cout << "曝光值:   " << _cap.get(cv::CAP_PROP_EXPOSURE) << " (期望: " << params.exposure << ")" << std::endl;
        std::cout << "亮度:     " << _cap.get(cv::CAP_PROP_BRIGHTNESS) << " (期望: " << params.brightness << ")" << std::endl;
        std::cout << "对比度:   " << _cap.get(cv::CAP_PROP_CONTRAST) << " (期望: " << params.contrast << ")" << std::endl;
        std::cout << "清晰度:   " << _cap.get(cv::CAP_PROP_SHARPNESS) << " (期望: " << params.sharpness << ")" << std::endl;
        std::cout << "自动白平衡: " << _cap.get(cv::CAP_PROP_AUTO_WB) << " (期望: " << (params.auto_white_balance ? 1 : 0) << ")" << std::endl;
        if (params.apply_white_balance_temperature)
            std::cout << "色温:     " << _cap.get(cv::CAP_PROP_WB_TEMPERATURE) << " (期望: " << params.white_balance_temperature << ")" << std::endl;
    }
    else
    {
        std::cout << "--- " << params.name << " 使用设备默认成像参数（未写入曝光/白平衡/画质控制）---" << std::endl;
    }

    const int active_fourcc = static_cast<int>(_cap.get(cv::CAP_PROP_FOURCC));
    char active_format[5] = {
        static_cast<char>(active_fourcc & 0xff),
        static_cast<char>((active_fourcc >> 8) & 0xff),
        static_cast<char>((active_fourcc >> 16) & 0xff),
        static_cast<char>((active_fourcc >> 24) & 0xff),
        '\0'};
    const double active_width = _cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const double active_height = _cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double active_fps = _cap.get(cv::CAP_PROP_FPS);

    std::cout << "--- " << params.name << " 初始化成功 ---" << std::endl;
    std::cout << "请求模式: " << _width << "x" << _height << " @ "
              << params.cameraframe << " FPS, MJPG" << std::endl;
    std::cout << "生效模式: " << active_width << "x" << active_height << " @ "
              << active_fps << " FPS, " << active_format << std::endl;
    if (active_fourcc != fourcc || active_width != _width || active_height != _height ||
        (active_fps > 0.0 && active_fps + 0.5 < params.cameraframe))
    {
        LOGW("%s: camera driver did not accept requested %dx%d@%d MJPG mode",
             params.name.c_str(), _width, _height, params.cameraframe);
    }
}

bool IRCamera::grab_frame(IRFrame *frame)
{
    if (!_cap.isOpened() || frame == nullptr || frame->rgb_ptr == nullptr)
        return false;

    cv::Mat captured;
    if (_cap.read(captured) && !captured.empty())
    {
        const auto now = std::chrono::system_clock::now();
        frame->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        // captured 独占其像素内存，移动 Mat 头即可，避免一整帧 copyTo。
        *(frame->rgb_ptr) = std::move(captured);
        return true;
    }

    ++_failed_reads;
    if (_failed_reads == 1 || _failed_reads % 60 == 0)
    {
        LOGW("IR Camera read failed (%llu times)",
             static_cast<unsigned long long>(_failed_reads));
    }
    return false;
}

IRCamera::~IRCamera()
{
    if (_cap.isOpened())
    {
        _cap.release();
    }
}

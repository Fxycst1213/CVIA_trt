#include <iostream>
#include <chrono>
#include "IRcamera.h"

IRCamera *IRCamera::GetInstance()
{
    static IRCamera instance;
    return &instance;
}

void IRCamera::init(int ID, std::string resolution, int frame)
{
    // 1. 强制使用 V4L2 后端打开相机
    _cap.open(ID, cv::CAP_V4L2);
    if (!_cap.isOpened())
    {
        LOGE("IR Camera INIT ERROR: Cannot open camera ID %d", ID);
        return;
    }

    // 2. 极其重要：设置 MJPG 编码格式以支持高帧率和高分辨率
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    _cap.set(cv::CAP_PROP_FOURCC, fourcc);

    // 3. 解析并设置分辨率
    if (resolution == "HD1080")
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
    _cap.set(cv::CAP_PROP_FPS, frame);

    // 4. 采用我们测试成功的自动曝光模式 (V4L2 中 3 代表自动)
    _cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
    _cap.set(cv::CAP_PROP_EXPOSURE, 78);

    _cap.set(cv::CAP_PROP_AUTO_WB, 0);
    _cap.set(cv::CAP_PROP_WB_TEMPERATURE, 4600.0); // 锁定色温为 4600

    _cap.set(cv::CAP_PROP_BRIGHTNESS, -64.0); // 亮度极暗 [-64, 64]
    _cap.set(cv::CAP_PROP_CONTRAST, 100.0);   // 对比度拉满 [0, 100]
    _cap.set(cv::CAP_PROP_SHARPNESS, 100.0);  // 清晰度拉满 [0, 100]

    std::cout << "--- 硬件级参数锁定 ---" << std::endl;
    std::cout << "曝光模式: " << _cap.get(cv::CAP_PROP_AUTO_EXPOSURE) << " (期望: 1)" << std::endl;
    std::cout << "曝光值:   " << _cap.get(cv::CAP_PROP_EXPOSURE) << " (期望: -6)" << std::endl;
    std::cout << "亮度:     " << _cap.get(cv::CAP_PROP_BRIGHTNESS) << " (期望: -64)" << std::endl;
    std::cout << "对比度:   " << _cap.get(cv::CAP_PROP_CONTRAST) << " (期望: 100)" << std::endl;
    std::cout << "清晰度:   " << _cap.get(cv::CAP_PROP_SHARPNESS) << " (期望: 100)" << std::endl;
    std::cout << "自动白平衡: " << _cap.get(cv::CAP_PROP_AUTO_WB) << " (期望: 0)" << std::endl;

    // 5. 初始化内参矩阵和计时器
    _timer = std::make_shared<timer::Timer>(logger::Level::VERB);

    std::cout << "--- IR Camera 初始化成功 ---" << std::endl;
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
#ifndef IRCAMERA_H
#define IRCAMERA_H

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
// 引入你工程原有的工具类
#include "logger.hpp"
#include "time.hpp"

// 定义红外相机的帧结构，与 ZEDframe 保持高度一致
struct IRFrame
{
    cv::Mat *rgb_ptr = nullptr;
    uint64_t timestamp = 0;
};

class IRCamera
{
public:
    // 保持与 ZEDX 相同的接口传参方式，enable_fill_mode 这里保留作为兼容参数
    void init(int ID, std::string resolution, int frame = 60);
    void grab_frame(IRFrame *frame);

    static IRCamera *GetInstance();

private:
    IRCamera() {};
    ~IRCamera();

    cv::VideoCapture _cap;
    int _width = 1920;
    int _height = 1080;
    std::shared_ptr<timer::Timer> _timer;
};

#endif // IRCAMERA_H
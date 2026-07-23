#ifndef IRCAMERA_H
#define IRCAMERA_H

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include "../params/params.hpp"
// 引入你工程原有的工具类
#include "logger.hpp"

// 定义红外相机的帧结构，与 ZEDframe 保持高度一致
struct IRFrame
{
    cv::Mat *rgb_ptr = nullptr;
    uint64_t timestamp = 0;
};

class IRCamera
{
public:
    IRCamera() = default;
    ~IRCamera();

    void init(const camera_params &params);
    bool grab_frame(IRFrame *frame);

    cv::VideoCapture _cap;
    int _width = 1920;
    int _height = 1080;
    uint64_t _failed_reads = 0;
};

#endif // IRCAMERA_H

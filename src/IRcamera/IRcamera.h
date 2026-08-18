#ifndef IRCAMERA_H
#define IRCAMERA_H

#include <opencv2/opencv.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include "../params/params.hpp"
// 引入你工程原有的工具类
#include "logger.hpp"
#include "timestamp_sync.hpp"

// 定义红外相机的帧结构，与 ZEDframe 保持高度一致
struct IRFrame
{
    cv::Mat *rgb_ptr = nullptr;
    // PTP 驯服的 CLOCK_REALTIME 时间域；timestamp 保持毫秒以兼容现有协议。
    uint64_t timestamp = 0;
    uint64_t capture_timestamp_ns = 0;
    uint64_t dequeue_timestamp_ns = 0;
    bool driver_timestamp = false;
    bool timestamp_start_of_exposure = false;
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

private:
    struct MappedBuffer
    {
        void *address = nullptr;
        size_t length = 0;
    };

    bool init_v4l2(const camera_params &params);
    bool grab_v4l2(IRFrame *frame);
    void close_v4l2();
    bool init_opencv_fallback(const camera_params &params);
    bool grab_opencv_fallback(IRFrame *frame);

    int _video_fd = -1;
    std::vector<MappedBuffer> _mapped_buffers;
    bool _v4l2_streaming = false;
    bool _logged_timestamp_fallback = false;
    bool _logged_timestamp_mode = false;
    std::string _camera_name;
    uint64_t _last_capture_timestamp_ns = 0;
};

#endif // IRCAMERA_H

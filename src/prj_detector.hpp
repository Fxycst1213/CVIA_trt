#ifndef PRJ_DETECTOR_HPP
#define PRJ_DETECTOR_HPP

#include "model.hpp"
#include "logger.hpp"
#include "worker.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include "communication/client.h"
#include "params/params.hpp"
#include <opencv2/opencv.hpp>
#include <thread>
#include <queue>
#include <memory>
#include "IRcamera/IRcamera.h"

using namespace std;

class prj_v8detector
{
public:
    prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params);
    ~prj_v8detector();
    void run();
    void request_stop();
    void camera();
    void detect_camera_loop();
    void photo_camera_loop();
    void camera_foldimages();
    void output_loop();
    void udp_loop();

private:
    shared_ptr<thread::Worker> _worker;
    std::unique_ptr<IRCamera> _ir_camera_detect;
    std::unique_ptr<IRCamera> _ir_camera_photo;
    std::function<void()> _func_camera;
    std::function<void()> _func_detect_camera;
    std::function<void()> _func_photo_camera;
    std::function<void()> _func_camera_foldimages;
    std::function<void()> _func_output;
    std::function<void()> _func_udp_send;

    IRFrame *_detect_writeframe = nullptr;
    IRFrame *_photo_writeframe = nullptr;

    cv::Mat _latest_detect_rgb;
    cv::Mat _latest_photo_rgb;
    uint64_t _latest_detect_timestamp = 0;
    uint64_t _latest_detect_capture_timestamp_ns = 0;
    uint64_t _latest_detect_dequeue_timestamp_ns = 0;
    bool _latest_detect_driver_timestamp = false;
    bool _latest_detect_timestamp_start_of_exposure = false;
    uint64_t _latest_photo_timestamp = 0;
    uint64_t _latest_detect_sequence = 0;
    bool _has_detect_frame = false;
    bool _has_photo_frame = false;
    std::mutex _detect_frame_mtx;
    std::mutex _photo_frame_mtx;
    std::condition_variable _detect_frame_cv;

    queue<Resultframe> _output_queue; // 单槽最新帧队列，避免预览/CSV处理积压拖慢实时性
    std::mutex _output_mtx;
    std::condition_variable _output_cv;

    struct UdpPoseFrame
    {
        std::vector<float> pose_result;
        uint64_t timestamp = 0;
    };

    queue<UdpPoseFrame> _udp_queue; // 仅存储 6D 位姿和时间戳，减少内存开销
    std::mutex _udp_mtx;
    std::condition_variable _udp_cv;

    std::atomic<bool> _is_running;

    client _client;
    prj_params _project_params;
};

#endif // PRJ_DETECTOR_HPP

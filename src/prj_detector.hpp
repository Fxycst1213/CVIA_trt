#ifndef PRJ_DETECTOR_HPP
#define PRJ_DETECTOR_HPP

#include "model.hpp"
#include "logger.hpp"
#include "worker.hpp"
#include "utils.hpp"
#include "lock.hpp"
#include "detector.hpp"
#include "ZEDX.h"
#include "time.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <ratio>
#include "communication/client.h"
// #include "communication/RS485.h" // 串口发送已停用，结果改用 UDP 上传。
#include <chrono>
#include "params/params.hpp"
#include "params/pose_params.hpp"
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
    void camera();
    void detect_camera_loop();
    void photo_camera_loop();
    void camera_foldimages();
    void tcp_loop();
    void udp_loop();
    // void rs485_loop(); // 串口发送已停用。

private:
    shared_ptr<thread::Worker> _worker;
    // ZEDX *_zed = nullptr;
    std::unique_ptr<IRCamera> _ir_camera_detect;
    std::unique_ptr<IRCamera> _ir_camera_photo;
    std::shared_ptr<timer::Timer> _timer;
    std::shared_ptr<timer::Timer> _timer_tcp;
    std::shared_ptr<timer::Timer> _timer_detect_grab;
    std::shared_ptr<timer::Timer> _timer_photo_grab;

    std::function<void()> _func_camera;
    std::function<void()> _func_detect_camera;
    std::function<void()> _func_photo_camera;
    std::function<void()> _func_camera_foldimages;
    std::function<void()> _func_pack_and_send;
    std::function<void()> _func_udp_send;
    // std::function<void()> _func_rs485_send; // 串口发送已停用。

    // ZEDframe *_writeframe = nullptr;
    IRFrame *_detect_writeframe = nullptr;
    IRFrame *_photo_writeframe = nullptr;

    cv::Mat _latest_detect_rgb;
    cv::Mat _latest_photo_rgb;
    uint64_t _latest_detect_timestamp = 0;
    uint64_t _latest_photo_timestamp = 0;
    uint64_t _latest_detect_sequence = 0;
    bool _has_detect_frame = false;
    bool _has_photo_frame = false;
    std::mutex _detect_frame_mtx;
    std::mutex _photo_frame_mtx;
    std::condition_variable _detect_frame_cv;

    queue<Resultframe> _resultframe_queue;
    std::mutex _queue_mtx;             // 保护队列的互斥锁
    std::condition_variable _queue_cv; // 用于通知"有新数据了"

    struct UdpPoseFrame
    {
        std::vector<float> pose_result;
        uint64_t timestamp = 0;
    };

    queue<UdpPoseFrame> _udp_queue; // 仅存储 6D 位姿和时间戳，减少内存开销
    std::mutex _udp_mtx;
    std::condition_variable _udp_cv;

    // 串口发送已停用，以下 RS485 队列成员不再使用。
    // queue<std::vector<float>> _rs485_queue;
    // std::mutex _rs485_mtx;
    // std::condition_variable _rs485_cv;

    std::atomic<bool> _is_running;

    client _client;
    // RS485 _rs485; // 串口发送已停用。
    uint64_t m_time;
};

#endif // PRJ_DETECTOR_HPP

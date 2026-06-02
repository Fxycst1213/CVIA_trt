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
#include "communication/RS422.h"
#include <chrono>
#include "params/params.hpp"
#include "params/pose_params.hpp"
#include <opencv2/opencv.hpp>
#include <thread>
#include <queue>
#include "IRcamera/IRcamera.h"

using namespace std;

class prj_v8detector
{
public:
    prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params);
    ~prj_v8detector();
    void run();
    void camera();
    void camera_foldimages();
    void tcp_loop();
    void rs442_loop();

private:
    shared_ptr<thread::Worker> _worker;
    // ZEDX *_zed = nullptr;
    IRCamera *_ir_camera = nullptr;
    std::shared_ptr<timer::Timer> _timer;
    std::shared_ptr<timer::Timer> _timer_tcp;

    std::function<void()> _func_camera;
    std::function<void()> _func_camera_foldimages;
    std::function<void()> _func_pack_and_send;

    std::function<void()> _func_rs442_send;

    // ZEDframe *_writeframe = nullptr;
    IRFrame *_writeframe = nullptr;

    queue<Resultframe> _resultframe_queue;
    std::mutex _queue_mtx;             // 保护队列的互斥锁
    std::condition_variable _queue_cv; // 用于通知"有新数据了"

    // RS442 serial output
    queue<std::vector<float>> _rs442_queue;
    std::mutex _rs442_mtx;
    std::condition_variable _rs442_cv;

    std::atomic<bool> _is_running;

    client _client;
    RS422 _rs442;
    uint64_t m_time;
};

#endif // PRJ_DETECTOR_HPP

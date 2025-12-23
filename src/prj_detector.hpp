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
#include "communication/RS485.h"
#include <chrono>
#include "params/params.hpp"
#include "params/pose_params.hpp"
#include <opencv2/opencv.hpp>
#include <thread>
#include <queue>

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

private:
    shared_ptr<thread::Worker> _worker;
    ZEDX *_zed = nullptr;
    std::shared_ptr<timer::Timer> _timer;
    std::shared_ptr<timer::Timer> _timer_tcp;

    std::function<void()> _func_camera;
    std::function<void()> _func_camera_foldimages;
    std::function<void()> _func_pack_and_send;
    ZEDframe *_writeframe = nullptr;

    queue<Resultframe> _resultframe_queue;
    std::mutex _queue_mtx;             // 保护队列的互斥锁
    std::condition_variable _queue_cv; // 用于通知"有新数据了"
    std::atomic<bool> _is_running;

    client _client;
    RS485 _rs485;
    uint64_t m_time;
};

#endif // PRJ_DETECTOR_HPP
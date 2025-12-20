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

using namespace std;

class prj_v8detector
{
public:
    prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params);
    ~prj_v8detector();
    void run();
    void camera();
    void camera_foldimages();
    template <typename T>
    void swapPtr(T **a, T **b)
    {
        T *temp = *a;
        *a = *b;
        *b = temp;
    }

private:
    shared_ptr<thread::Worker> _worker;
    ZEDX *_zed = nullptr;
    std::shared_ptr<timer::Timer> _timer;
    std::function<void()> _func_camera;
    std::function<void()> _func_camera_foldimages;
    ZEDframe *_writeframe = nullptr;
    queue<Resultframe> _resultframe_queue;
    client _client;
    RS485 _rs485;
};

#endif // PRJ_DETECTOR_HPP
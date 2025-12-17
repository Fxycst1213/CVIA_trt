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
using namespace std;
struct Resultframe
{
    cv::Mat *rgb_ptr;
    vector<model::pose::bbox> bboxes;
    vector<double> result;
};
struct prj_params
{
    int H;
    int W;
    string resolution = "HD1080";
    int cameraID = 0;
    string ip;
    string port;
    int socket_mode;
};
class prj_v8detector
{
public:
    prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params);
    ~prj_v8detector();
    void run();
    void camera();
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
    ZEDframe *_writeframe = nullptr;
    queue<Resultframe> _resultframe_queue;
};

#endif // PRJ_DETECTOR_HPP
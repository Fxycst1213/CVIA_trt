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
using namespace std;

class prj_v8detector
{
public:
    prj_v8detector(string onnxPath, logger::Level level, model::Params params);
    void infer();
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
    locker _lockInfer;
    shared_ptr<thread::Worker> _worker;
    ZEDX *_zed = nullptr;
    string _resolution = "HD1080";
    std::shared_ptr<timer::Timer> _timer;
    int _cameraID = 0;
    std::function<void()> _func_infer;
    std::function<void()> _func_camera;
    ZEDframe *_readyframe = nullptr;
    ZEDframe *_writeframe = nullptr;
    ZEDframe *_inferframe = nullptr;
    std::atomic<bool> has_image{false};
    std::condition_variable _cv;
    std::mutex _cv_m;
};
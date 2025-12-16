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
struct prj_params
{
    int H;
    int W;
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
    locker _lockInfer;
    shared_ptr<thread::Worker> _worker;
    ZEDX *_zed = nullptr;
    string _resolution = "HD1080";
    std::shared_ptr<timer::Timer> _timer;
    int _cameraID = 0;
    std::function<void()> _func_camera;
    ZEDframe *_writeframe = nullptr;
};
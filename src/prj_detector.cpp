#include "prj_detector.hpp"

prj_v8detector::prj_v8detector(string onnxPath, logger::Level level, model::Params params)
{
    _worker = thread::create_worker(onnxPath, level, params);
    _zed = ZEDX::GetInstance();
    _timer = make_shared<timer::Timer>();
    _timer->init();
    if (_zed->init(_cameraID, _resolution) == -1)
    {
        std::cout << "camera init failed!" << endl;
        exit(EXIT_FAILURE);
    }
    _readyframe = new ZEDframe;
    _writeframe = new ZEDframe;
    _inferframe = new ZEDframe;
    _readyframe->rgb_ptr = new cv::Mat(1080, 1920, CV_8UC3);
    _writeframe->rgb_ptr = new cv::Mat(1080, 1920, CV_8UC3);
    _inferframe->rgb_ptr = new cv::Mat(1080, 1920, CV_8UC3);
    _func_camera = std::bind(&prj_v8detector::camera, this);
    _func_infer = std::bind(&prj_v8detector::infer, this);
}

void prj_v8detector::infer()
{
    while (1)
    {
        {
            std::unique_lock<std::mutex> lk(_cv_m);
            _cv.wait(lk, [&]
                     { return has_image.load(); });
        }

        _timer->start_cpu();
        _lockInfer.lock();
        if (_readyframe->timestamp > _inferframe->timestamp)
        {
            swapPtr(&_inferframe, &_readyframe);
        }

        _lockInfer.unlock();
        _worker->inference(*(_inferframe->rgb_ptr));
        LOG("\t%-60s uses %-6llu ms", "this frame e2e is", _inferframe->end_timestamp - _inferframe->timestamp);
        _timer->stop_cpu<timer::Timer::ms>("1 inference");
        _timer->show();
        _timer->init();
    }
}

void prj_v8detector::camera()
{
    while (1)
    {
        if (_zed->grab_frame(_writeframe) == -1)
        {
            printf("get image failed\n");
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(_cv_m);
            has_image.store(true);
        }
        _cv.notify_one();
    }
}

void prj_v8detector::run()
{
    auto t1 = std::thread(_func_camera);
    auto t2 = std::thread(_func_infer);

    t1.join();
    t2.join();
}

prj_v8detector::~prj_v8detector()
{
    delete _readyframe->rgb_ptr;
    delete _writeframe->rgb_ptr;
    delete _inferframe->rgb_ptr;
    delete _readyframe;
    delete _writeframe;
    delete _inferframe;
}

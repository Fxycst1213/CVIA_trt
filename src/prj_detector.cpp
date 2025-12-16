#include "prj_detector.hpp"

prj_v8detector::prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params)
{
    _worker = thread::create_worker(onnxPath, level, params);
    preprocess::init_process(p_params.H, p_params.W);
    _zed = ZEDX::GetInstance();
    _timer = make_shared<timer::Timer>(logger::Level::INFO);
    _timer->init();
    if (_zed->init(_cameraID, _resolution) == -1)
    {
        std::cout << "camera init failed!" << endl;
        exit(EXIT_FAILURE);
    }
    _writeframe = new ZEDframe;
    _writeframe->rgb_ptr = new cv::Mat(1080, 1920, CV_8UC3);
    _func_camera = std::bind(&prj_v8detector::camera, this);
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
        _timer->init();
        _timer->start_cpu();
        _worker->inference(*(_writeframe->rgb_ptr));
        _timer->stop_cpu<timer::Timer::ms>("inference");
        _timer->show();
    }
}

void prj_v8detector::run()
{
    auto t1 = std::thread(_func_camera);
    t1.join();
}

prj_v8detector::~prj_v8detector()
{
    delete _writeframe->rgb_ptr;
    delete _writeframe;
    preprocess::destroy_process();
}

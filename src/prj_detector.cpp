#include "prj_detector.hpp"

prj_v8detector::prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params)
{
    _worker = thread::create_worker(onnxPath, level, params);
    _timer = make_shared<timer::Timer>(logger::Level::INFO);
    _timer->init();

    _zed = ZEDX::GetInstance();
    _zed->init(p_params.cameraID, p_params.resolution);

    preprocess::init_process(p_params.H, p_params.W);

    _writeframe = new ZEDframe;
    _writeframe->rgb_ptr = new cv::Mat(p_params.H, p_params.W, CV_8UC3);
    _func_camera = std::bind(&prj_v8detector::camera, this);
    // _client.init(p_params.ip, p_params.port);
}

void prj_v8detector::camera()
{
    while (1)
    {
        _timer->init();
        _timer->start_cpu();
        // _zed->grab_frame(_writeframe);
        *(_writeframe->rgb_ptr) = cv::imread(("data/source/00590.png"));
        _timer->stop_cpu<timer::Timer::ms>("ZED Grab frame");
        _timer->start_cpu();
        _worker->inference(*(_writeframe->rgb_ptr));
        _timer->stop_cpu<timer::Timer::ms>("inference");
        _timer->show();
        _resultframe_queue.push(
            Resultframe{
                _writeframe->rgb_ptr,
                _worker->m_pose->m_bboxes,
                _worker->m_pose->m_result});
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

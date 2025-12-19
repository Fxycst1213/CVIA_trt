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
    _func_camera_foldimages = std::bind(&prj_v8detector::camera_foldimages, this);
    _client.init(p_params);
    _rs485.init(p_params);
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
        _worker->inference(*(_writeframe->rgb_ptr), _writeframe->timestamp);
        _timer->stop_cpu<timer::Timer::ms>("inference");
        _timer->show();
        _rs485.sendDoubleArray(_worker->m_pose->m_result.data());
        _client.pack_and_send(
            *(_writeframe->rgb_ptr),
            _worker->m_pose->m_bboxes,
            _worker->m_pose->m_result,
            _writeframe->timestamp);
        // _resultframe_queue.push(
        //     Resultframe{
        //         _writeframe->rgb_ptr,
        //         _worker->m_pose->m_bboxes,
        //         _worker->m_pose->m_result});
    }
}

void prj_v8detector::camera_foldimages()
{
    std::vector<cv::String> filenames;
    cv::String folder = "/home/cvia/yifei/images3/*.png";

    cv::glob(folder, filenames, false);
    std::sort(filenames.begin(), filenames.end());
    int current_idx = 0;

    while (1)
    {
        _timer->init();
        _timer->start_cpu();

        cv::Mat img = cv::imread(filenames[current_idx]);
        *(_writeframe->rgb_ptr) = img;
        auto now = std::chrono::system_clock::now();
        _writeframe->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        current_idx++;
        if (current_idx >= filenames.size())
        {
            break;
        }
        _timer->stop_cpu<timer::Timer::ms>("ZED Grab frame");
        _timer->start_cpu();
        _worker->inference(*(_writeframe->rgb_ptr), _writeframe->timestamp);
        _timer->stop_cpu<timer::Timer::ms>("inference");
        _timer->show();
        _rs485.sendDoubleArray(_worker->m_pose->m_result.data());
        _client.pack_and_send(
            *(_writeframe->rgb_ptr),
            _worker->m_pose->m_bboxes,
            _worker->m_pose->m_result,
            _writeframe->timestamp);
    }
}

void prj_v8detector::run()
{
    // auto t1 = std::thread(_func_camera);
    auto t2 = std::thread(_func_camera_foldimages);
    // t1.join();
    t2.join();
}

prj_v8detector::~prj_v8detector()
{
    delete _writeframe->rgb_ptr;
    delete _writeframe;
    preprocess::destroy_process();
}

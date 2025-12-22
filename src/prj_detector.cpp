#include "prj_detector.hpp"

prj_v8detector::prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params)
{
    _worker = thread::create_worker(onnxPath, level, params);
    _timer = make_shared<timer::Timer>(logger::Level::INFO);
    _timer_tcp = make_shared<timer::Timer>(logger::Level::INFO);

    _zed = ZEDX::GetInstance();
    _zed->init(p_params.cameraID, p_params.resolution);

    preprocess::init_process(p_params.H, p_params.W);

    _writeframe = new ZEDframe;
    _writeframe->rgb_ptr = new cv::Mat(p_params.H, p_params.W, CV_8UC3);
    _func_camera = std::bind(&prj_v8detector::camera, this);
    _func_camera_foldimages = std::bind(&prj_v8detector::camera_foldimages, this);
    _func_pack_and_send = std::bind(&prj_v8detector::tcp_loop, this);

    _client.init(p_params);
    _rs485.init(p_params);
    _is_running = true;
}

void prj_v8detector::camera()
{
    while (1)
    {
        Resultframe _resultframe;
        _timer->init();
        _timer->start_cpu();
        _zed->grab_frame(_writeframe);
        // *(_writeframe->rgb_ptr) = cv::imread(("data/source/00590.png"));
        _timer->stop_cpu<timer::Timer::ms>("ZED Grab frame");
        _resultframe.rgb = *(_writeframe->rgb_ptr);
        _resultframe.timestamp = _writeframe->timestamp;
        _timer->start_cpu();
        _worker->inference(_resultframe);
        _timer->stop_cpu<timer::Timer::ms>("inference");

        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _timer->start_cpu();
        _rs485.sendDoubleArray(_resultframe.pose_result.data());
        _timer->stop_cpu<timer::Timer::ms>("RS485");

        _timer->start_cpu();
        {
            std::lock_guard<std::mutex> lock(_queue_mtx);
            _resultframe_queue.push(_resultframe);
            _queue_cv.notify_one();
        }
        _timer->stop_cpu<timer::Timer::ms>("Load TCP");
        _timer->show();
    }
}

void prj_v8detector::camera_foldimages()
{
    std::vector<cv::String> filenames;
    cv::String folder = "/home/cvia/yifei/images3/*.png";
    cv::glob(folder, filenames, false);
    std::sort(filenames.begin(), filenames.end());
    int current_idx = 0;

    auto now = std::chrono::system_clock::now();
    long long current_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    while (1)
    {
        auto mstart = std::chrono::high_resolution_clock::now();
        Resultframe _resultframe;
        _timer->init();
        _timer->start_cpu();
        *(_writeframe->rgb_ptr) = cv::imread(filenames[current_idx]);
        _writeframe->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        current_idx++;
        if (current_idx >= filenames.size())
        {
            break;
        }
        _timer->stop_cpu<timer::Timer::ms>("ZED Grab frame");
        _resultframe.rgb = *(_writeframe->rgb_ptr);
        _resultframe.timestamp = _writeframe->timestamp;
        _timer->start_cpu();
        _worker->inference(_resultframe);
        _timer->stop_cpu<timer::Timer::ms>("inference");

        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _timer->start_cpu();
        _rs485.sendDoubleArray(_resultframe.pose_result.data());
        _timer->stop_cpu<timer::Timer::ms>("RS485");
        _timer->start_cpu();
        {
            std::lock_guard<std::mutex> lock(_queue_mtx);
            _resultframe_queue.push(_resultframe);
            _queue_cv.notify_one();
        }
        _timer->stop_cpu<timer::Timer::ms>("Load TCP");
        _timer->show();
    }
}

void prj_v8detector::tcp_loop()
{
    while (1)
    {
        Resultframe frame_to_send;
        {
            std::unique_lock<std::mutex> lock(_queue_mtx);

            // 等待条件：有数据 或者 停止运行
            _queue_cv.wait(lock, [this]
                           { return !_resultframe_queue.empty() || !_is_running; });

            frame_to_send = _resultframe_queue.front();
            _resultframe_queue.pop();
        }
        _timer_tcp->init();
        _timer_tcp->start_cpu();
        _client.pack_and_send(frame_to_send);
        _timer_tcp->stop_cpu<timer::Timer::ms>("TCP Thread Send");
        _timer_tcp->show();
    }
}

void prj_v8detector::run()
{
    _is_running = true;
    // auto t1 = std::thread(_func_camera);
    auto t2 = std::thread(_func_camera_foldimages);
    auto t3 = std::thread(_func_pack_and_send);
    // t1.join();
    if (t2.joinable())
    {
        t2.join();
    }
    _is_running = false;
    _queue_cv.notify_all(); // 唤醒 TCP 线程让它检查 _is_running 并退出
    if (t3.joinable())
    {
        t3.join();
    }
}

prj_v8detector::~prj_v8detector()
{
    _is_running = false;
    delete _writeframe->rgb_ptr;
    delete _writeframe;
    preprocess::destroy_process();
}

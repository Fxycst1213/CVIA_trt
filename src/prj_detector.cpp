#include "prj_detector.hpp"

prj_v8detector::prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params)
{
    _worker = thread::create_worker(onnxPath, level, params);
    _timer = make_shared<timer::Timer>(logger::Level::INFO);
    _timer_tcp = make_shared<timer::Timer>(logger::Level::INFO);

    // _zed = ZEDX::GetInstance();
    // _zed->init(p_params.cameraID, p_params.resolution);
    _ir_camera = IRCamera::GetInstance();
    _ir_camera->init(p_params.cameraID, p_params.resolution, p_params.cameraframe);

    preprocess::init_process(p_params.H, p_params.W);

    // _writeframe = new ZEDframe;
    _writeframe = new IRFrame;
    _writeframe->rgb_ptr = new cv::Mat(p_params.H, p_params.W, CV_8UC3);
    _func_camera = std::bind(&prj_v8detector::camera, this);
    _func_camera_foldimages = std::bind(&prj_v8detector::camera_foldimages, this);
    _func_pack_and_send = std::bind(&prj_v8detector::tcp_loop, this);
    _func_rs442_send = std::bind(&prj_v8detector::rs442_loop, this);
    _client.init(p_params);
    _rs442.init(p_params);
    _is_running = true;
}

void prj_v8detector::rs442_loop()
{
    while (true)
    {
        std::vector<float> data_to_send;
        {
            std::unique_lock<std::mutex> lock(_rs442_mtx);
            // 等待数据或停止信号
            _rs442_cv.wait(lock, [this] { 
                return !_rs442_queue.empty() || !_is_running; 
            });

            if (!_is_running && _rs442_queue.empty()) {
                break;
            }

            if (_rs442_queue.empty()) {
                continue;
            }

            data_to_send = _rs442_queue.front();
            _rs442_queue.pop();
        }

        // 发送数据 (移除了之前的 Timer 计时，如果需要可以加回)
        if (data_to_send.size() >= 9)
        {
            float float_temp_pose[9];
            for (int i = 0; i < 9; ++i)
            {
                float_temp_pose[i] = data_to_send[i];
            }

            // RS442 fixed test frame. Uncomment this block to verify byte order,
            // checksum and payload parsing with the receiver.
            // Expected frame:
            // EB 90 12 03 E8 FC 18 00 7B 01 C8 FE 38 03 15 00 7B FF 85 11 D7 8A
            //
            // float_temp_pose[0] = 1000.0f;  // absolute X, mm
            // float_temp_pose[1] = -1000.0f; // absolute Y, mm
            // float_temp_pose[2] = 123.0f;   // absolute Z, mm
            // float_temp_pose[3] = 456.0f;   // relative X, mm
            // float_temp_pose[4] = -456.0f;  // relative Y, mm
            // float_temp_pose[5] = 789.0f;   // relative Z, mm
            // float_temp_pose[6] = 1.23f;    // attitude X, degree
            // float_temp_pose[7] = -1.23f;   // attitude Y, degree
            // float_temp_pose[8] = 45.67f;   // attitude Z, degree

            _rs442.sendFloatArray(float_temp_pose);
        }

        usleep(150000);
    }
}

void prj_v8detector::camera()
{
    while (1)
    {
        Resultframe _resultframe;
        _timer->init();
        _timer->start_cpu();
        _ir_camera->grab_frame(_writeframe);
        _timer->stop_cpu<timer::Timer::ms>("IR Camera Grab frame");
        _resultframe.rgb = _writeframe->rgb_ptr->clone(); 
        _resultframe.timestamp = _writeframe->timestamp;

        _timer->start_cpu();

        _worker->inference(_resultframe);
        
        _timer->stop_cpu<timer::Timer::ms>("inference");
        
        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _resultframe.rs442_result = _worker->m_pose->uart_result;
        {
            std::lock_guard<std::mutex> lock(_rs442_mtx);
            while (_rs442_queue.size() > 2) 
            {
                _rs442_queue.pop();
            }
            _rs442_queue.push(_resultframe.rs442_result);
            _rs442_cv.notify_one();
        }
        _timer->start_cpu();
        {
            std::lock_guard<std::mutex> lock(_queue_mtx);
            while (_resultframe_queue.size() > 2)
            {
                _resultframe_queue.pop();
                LOGW("TCP queue full, dropping old frame!");
            }
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
    cv::String folder = "/home/cvia/yifei/yifei_results/*.jpg";
    cv::glob(folder, filenames, false);
    std::sort(filenames.begin(), filenames.end());
    // std::sort(filenames.rbegin(), filenames.rend());
    int current_idx = 0;
    while (1)
    {
        auto now = std::chrono::system_clock::now();
        auto mstart = std::chrono::high_resolution_clock::now();
        Resultframe _resultframe;
        _timer->init();
        _timer->start_cpu();
        // usleep(1000000);
        *(_writeframe->rgb_ptr) = cv::imread(filenames[current_idx]);
        _writeframe->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        current_idx++;
        if (current_idx >= filenames.size())
        {
            break;
        }
        // _timer->stop_cpu<timer::Timer::ms>("ZED Grab frame");
        _timer->stop_cpu<timer::Timer::ms>("IR Camera Grab frame");
        _resultframe.rgb = *(_writeframe->rgb_ptr);
        _resultframe.timestamp = _writeframe->timestamp;
        _timer->start_cpu();
        _worker->inference(_resultframe);
        _timer->stop_cpu<timer::Timer::ms>("inference");

        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _resultframe.rs442_result = _worker->m_pose->uart_result;
        {
            std::lock_guard<std::mutex> lock(_rs442_mtx);
            // 同样应用漏桶策略
            while (_rs442_queue.size() > 2) 
            {
                _rs442_queue.pop();
            }
            if (_resultframe.rs442_result.size() >= 9) {
                 _rs442_queue.push(_resultframe.rs442_result);
                 _rs442_cv.notify_one();
            }
        }

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
            if (_resultframe_queue.empty())
            {
                continue;
            }
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
    auto t_rs442 = std::thread(_func_rs442_send);
    auto t3 = std::thread(_func_pack_and_send);
    // if (t1.joinable())
    // {
    //     t1.join();
    // }

    if (t2.joinable())
    {
        t2.join();
    }
    _is_running = false;
    _queue_cv.notify_all(); // 唤醒 TCP 线程让它检查 _is_running 并退出
    _rs442_cv.notify_all();
    if (t_rs442.joinable())
    {
        t_rs442.join();
    }
    if (t3.joinable())
    {
        t3.join();
    }
}

prj_v8detector::~prj_v8detector()
{
    _is_running = false;
    _queue_cv.notify_all();
    _rs442_cv.notify_all();
    delete _writeframe->rgb_ptr;
    delete _writeframe;
    preprocess::destroy_process();
}

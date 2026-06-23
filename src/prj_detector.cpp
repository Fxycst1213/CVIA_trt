#include "prj_detector.hpp"

prj_v8detector::prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params)
{
    _worker = thread::create_worker(onnxPath, level, params);
    _timer = make_shared<timer::Timer>(logger::Level::INFO);
    _timer_tcp = make_shared<timer::Timer>(logger::Level::INFO);
    _timer_detect_grab = make_shared<timer::Timer>(logger::Level::INFO);
    _timer_photo_grab = make_shared<timer::Timer>(logger::Level::INFO);

    // _zed = ZEDX::GetInstance();
    // _zed->init(p_params.cameraID, p_params.resolution);
    _ir_camera_detect = std::make_unique<IRCamera>();
    _ir_camera_detect->init(p_params.detect_camera);
    _ir_camera_photo = std::make_unique<IRCamera>();
    _ir_camera_photo->init(p_params.photo_camera);

    preprocess::init_process(p_params.H, p_params.W);

    // _writeframe = new ZEDframe;
    _detect_writeframe = new IRFrame;
    _detect_writeframe->rgb_ptr = new cv::Mat(p_params.H, p_params.W, CV_8UC3, cv::Scalar(0, 0, 0));
    _photo_writeframe = new IRFrame;
    _photo_writeframe->rgb_ptr = new cv::Mat(p_params.H, p_params.W, CV_8UC3, cv::Scalar(0, 0, 0));
    _func_camera = std::bind(&prj_v8detector::camera, this);
    _func_detect_camera = std::bind(&prj_v8detector::detect_camera_loop, this);
    _func_photo_camera = std::bind(&prj_v8detector::photo_camera_loop, this);
    _func_camera_foldimages = std::bind(&prj_v8detector::camera_foldimages, this);
    _func_pack_and_send = std::bind(&prj_v8detector::tcp_loop, this);
    _func_udp_send = std::bind(&prj_v8detector::udp_loop, this);
    // _func_rs485_send = std::bind(&prj_v8detector::rs485_loop, this); // 串口发送已停用。
    _client.init(p_params);
    // _rs485.init(p_params); // 串口发送已停用。
    _is_running = true;
}

void prj_v8detector::detect_camera_loop()
{
    while (_is_running)
    {
        _timer_detect_grab->init();
        _timer_detect_grab->start_cpu();
        _ir_camera_detect->grab_frame(_detect_writeframe);
        _timer_detect_grab->stop_cpu<timer::Timer::ms>("Detect IR Camera Grab frame");

        if (_detect_writeframe->rgb_ptr == nullptr || _detect_writeframe->rgb_ptr->empty())
        {
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(_detect_frame_mtx);
            _latest_detect_rgb = _detect_writeframe->rgb_ptr->clone();
            _latest_detect_timestamp = _detect_writeframe->timestamp;
            ++_latest_detect_sequence;
            _has_detect_frame = true;
        }
        _detect_frame_cv.notify_one();
        _timer_detect_grab->show();
    }
}

void prj_v8detector::photo_camera_loop()
{
    while (_is_running)
    {
        _timer_photo_grab->init();
        _timer_photo_grab->start_cpu();
        _ir_camera_photo->grab_frame(_photo_writeframe);
        _timer_photo_grab->stop_cpu<timer::Timer::ms>("Photo IR Camera Grab frame");

        if (_photo_writeframe->rgb_ptr == nullptr || _photo_writeframe->rgb_ptr->empty())
        {
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(_photo_frame_mtx);
            _latest_photo_rgb = _photo_writeframe->rgb_ptr->clone();
            _latest_photo_timestamp = _photo_writeframe->timestamp;
            _has_photo_frame = true;
        }
        _timer_photo_grab->show();
    }
}

void prj_v8detector::udp_loop()
{
    while (true)
    {
        UdpPoseFrame data_to_send;
        {
            std::unique_lock<std::mutex> lock(_udp_mtx);
            // 等待数据或停止信号
            _udp_cv.wait(lock, [this] { 
                return !_udp_queue.empty() || !_is_running; 
            });

            if (!_is_running && _udp_queue.empty()) {
                break;
            }

            if (_udp_queue.empty()) {
                continue;
            }

            data_to_send = _udp_queue.front();
            _udp_queue.pop();
        }

        // UDP_TEST_DATA_BEGIN: 临时测试数据，测试完注释下面 2 行即可恢复真实 UDP 数据。
        // data_to_send.pose_result = {1.0f, 2.0f, 3.0f, 100.0f, 200.0f, 300.0f};
        // data_to_send.timestamp = 1234567890123ULL;
        // UDP_TEST_DATA_END

        // UDP 发送和 TCP 姿态段一致的数据: rx, ry, rz, x, y, z, timestamp。
        if (data_to_send.pose_result.size() >= 6)
        {
            _client.send_udp_result(data_to_send.pose_result, data_to_send.timestamp);
        }
    }
}

void prj_v8detector::camera()
{
    uint64_t last_processed_sequence = 0;
    while (_is_running)
    {
        Resultframe _resultframe;
        {
            std::unique_lock<std::mutex> lock(_detect_frame_mtx);
            _detect_frame_cv.wait(lock, [this, &last_processed_sequence]
                                  { return _latest_detect_sequence > last_processed_sequence || !_is_running; });
            if (!_is_running && _latest_detect_sequence <= last_processed_sequence)
            {
                break;
            }

            _resultframe.rgb = _latest_detect_rgb.clone();
            _resultframe.timestamp = _latest_detect_timestamp;
            last_processed_sequence = _latest_detect_sequence;
        }

        {
            std::lock_guard<std::mutex> lock(_photo_frame_mtx);
            if (_has_photo_frame)
            {
                _resultframe.rgb_secondary = _latest_photo_rgb.clone();
                _resultframe.secondary_timestamp = _latest_photo_timestamp;
            }
        }

        if (_resultframe.rgb_secondary.empty())
        {
            _resultframe.rgb_secondary = _resultframe.rgb.clone();
            _resultframe.secondary_timestamp = _resultframe.timestamp;
        }

        _timer->init();
        _timer->start_cpu();
        _worker->inference(_resultframe);
        _timer->stop_cpu<timer::Timer::ms>("inference");

        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _resultframe.udp_result = _resultframe.pose_result;
        {
            std::lock_guard<std::mutex> lock(_udp_mtx);
            // 漏桶策略：如果 UDP 发送线程处理不过来，丢弃旧数据，保证实时性。
            while (_udp_queue.size() > 2) 
            {
                _udp_queue.pop();
            }
            if (_resultframe.udp_result.size() >= 6)
            {
                UdpPoseFrame udp_frame;
                udp_frame.pose_result = _resultframe.udp_result;
                udp_frame.timestamp = _resultframe.timestamp;
                _udp_queue.push(udp_frame);
                _udp_cv.notify_one();
            }
        }

        // 4. TCP 队列处理
        _timer->start_cpu();
        {
            std::lock_guard<std::mutex> lock(_queue_mtx);
            
            // 【关键修复2】：实现“漏桶”策略，防止队列堆积
            // 如果队列里堆积超过 2 帧，说明 TCP 发不过来了，直接把最老的帧扔掉
            // 保持队列始终很短，确保发送的是较新的数据
            while (_resultframe_queue.size() > 2) 
            {
                // 注意：如果 Resultframe 很大，pop 可能会析构释放内存，这很好
                _resultframe_queue.pop(); 
                // 可选：打印个日志提示丢帧了
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
    cv::String folder = "/home/cvia/yifei/images_old2/*.png";
    cv::glob(folder, filenames, false);
    std::sort(filenames.begin(), filenames.end());
    // std::sort(filenames.rbegin(), filenames.rend());
    int current_idx = 0;
    while (1)
    {
        auto now = std::chrono::system_clock::now();
        Resultframe _resultframe;
        _timer->init();
        _timer->start_cpu();
        // usleep(200000);
        *(_detect_writeframe->rgb_ptr) = cv::imread(filenames[current_idx]);
        if (_detect_writeframe->rgb_ptr->empty())
        {
            break;
        }
        _photo_writeframe->rgb_ptr->release();
        _photo_writeframe->rgb_ptr->create(_detect_writeframe->rgb_ptr->rows, _detect_writeframe->rgb_ptr->cols, _detect_writeframe->rgb_ptr->type());
        _detect_writeframe->rgb_ptr->copyTo(*(_photo_writeframe->rgb_ptr));
        _detect_writeframe->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        _photo_writeframe->timestamp = _detect_writeframe->timestamp;
        current_idx++;
        if (current_idx >= filenames.size())
        {
            break;
        }
        _timer->stop_cpu<timer::Timer::ms>("Dual IR Camera Grab frame");
        _resultframe.rgb = _detect_writeframe->rgb_ptr->clone();
        _resultframe.rgb_secondary = _photo_writeframe->rgb_ptr->clone();
        _resultframe.timestamp = _detect_writeframe->timestamp;
        _resultframe.secondary_timestamp = _photo_writeframe->timestamp;
        _timer->start_cpu();
        _worker->inference(_resultframe);
        _timer->stop_cpu<timer::Timer::ms>("inference");

        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _resultframe.udp_result = _resultframe.pose_result;
        {
            std::lock_guard<std::mutex> lock(_udp_mtx);
            // 同样应用漏桶策略
            while (_udp_queue.size() > 2) 
            {
                _udp_queue.pop();
            }
            if (_resultframe.udp_result.size() >= 6) {
                 UdpPoseFrame udp_frame;
                 udp_frame.pose_result = _resultframe.udp_result;
                 udp_frame.timestamp = _resultframe.timestamp;
                 _udp_queue.push(udp_frame);
                 _udp_cv.notify_one();
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
            if (!_is_running && _resultframe_queue.empty())
            {
                break;
            }
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
    auto t_detect = std::thread(_func_detect_camera);
    auto t_photo = std::thread(_func_photo_camera);
    auto t1 = std::thread(_func_camera);
    // auto t2 = std::thread(_func_camera_foldimages);
    auto t_udp = std::thread(_func_udp_send);
    // auto t_rs485 = std::thread(_func_rs485_send); // 串口发送已停用。
    auto t3 = std::thread(_func_pack_and_send);
    if (t1.joinable())
    {
        t1.join();
    }

    // if (t2.joinable())
    // {
    //     t2.join();
    // }
    _is_running = false;
    _queue_cv.notify_all(); // 唤醒 TCP 线程让它检查 _is_running 并退出
    _udp_cv.notify_all();   // 唤醒 UDP
    // _rs485_cv.notify_all(); // 串口发送已停用。
    _detect_frame_cv.notify_all();
    if (t_detect.joinable())
    {
        t_detect.join();
    }
    if (t_photo.joinable())
    {
        t_photo.join();
    }
    if (t_udp.joinable())
    {
        t_udp.join();
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
    _udp_cv.notify_all();
    _detect_frame_cv.notify_all();
    if (_detect_writeframe)
    {
        delete _detect_writeframe->rgb_ptr;
        delete _detect_writeframe;
    }
    if (_photo_writeframe)
    {
        delete _photo_writeframe->rgb_ptr;
        delete _photo_writeframe;
    }
    preprocess::destroy_process();
}

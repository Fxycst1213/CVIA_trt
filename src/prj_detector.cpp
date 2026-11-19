#include "prj_detector.hpp"
#include <csignal>

namespace
{
    static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "Signal handling requires lock-free atomic bool");
    std::atomic<bool> stop_requested{false};
    void request_stop(int)
    {
        stop_requested.store(true, std::memory_order_relaxed);
    }
}

prj_v8detector::prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params)
{
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
    _loop_images = params.keypoints_only;
    _worker = thread::create_worker(onnxPath, level, params);
    _timer = make_shared<timer::Timer>(logger::Level::INFO);

    if (!params.keypoints_only)
    {
        _zed = ZEDX::GetInstance();
        _zed->init(p_params.cameraID, p_params.resolution, 60, false, p_params.camera_eye);
    }
    // _worker->m_pose->set_camera_calibration(_zed->calibration());

    preprocess::init_process(p_params.H, p_params.W);

    _writeframe = new ZEDframe;
    _writeframe->rgb_ptr = new cv::Mat(p_params.H, p_params.W, CV_8UC3);
    _func_camera = std::bind(&prj_v8detector::camera, this);
    _func_camera_foldimages = std::bind(&prj_v8detector::camera_foldimages, this);
    _func_pack_and_send = std::bind(&prj_v8detector::tcp_loop, this);
    _func_rs485_send = std::bind(&prj_v8detector::rs485_loop, this); // [新增]
    _func_rs485_receive = std::bind(&prj_v8detector::rs485_receive_loop, this);

    _client.init(p_params);
    _serial_enabled = p_params.enable_serial && _rs485.init(p_params) == 0;
    if (p_params.enable_serial)
        std::cerr << "[serial] " << p_params.rs485_port
                  << (_serial_enabled ? " opened; TX period=100 ms\n"
                                      : " unavailable; serial TX disabled\n");
    _rs485.setDebug(p_params.verbose);
    // rs485_loop paces frame starts at 10 Hz, including time spent writing.
    _rs485.setSendIntervalUs(0);
    _is_running = true;
}

// [新增] RS485 独立线程循环
void prj_v8detector::rs485_loop()
{
    using Clock = std::chrono::steady_clock;
    const auto send_period = std::chrono::milliseconds(100);
    auto next_send = Clock::now();
    while (true)
    {
        std::vector<float> data_to_send;
        {
            std::unique_lock<std::mutex> lock(_rs485_mtx);
            // 等待数据或停止信号
            _rs485_cv.wait(lock, [this] { 
                return !_rs485_queue.empty() || !_is_running; 
            });

            // Producers may replace the pending result during this wait.
            _rs485_cv.wait_until(lock, next_send, [this] { return !_is_running; });
            if (!_is_running) {
                break;
            }

            data_to_send = std::move(_rs485_queue.back());
            while (!_rs485_queue.empty())
                _rs485_queue.pop();
        }

        // 发送 2026-4-7 协议二进制帧。
        if (!data_to_send.empty())
        {
            // Schedule from the actual start: no extra wire-time delay or catch-up bursts.
            next_send = Clock::now() + send_period;
            _rs485.sendVisionFrame(data_to_send);
        }

    }
}

void prj_v8detector::rs485_receive_loop()
{
    while (_is_running)
    {
        if (!_rs485.receiveAndPrintAvailable(100))
        {
            usleep(10000);
        }
    }
}

void prj_v8detector::camera()
{
    while (_is_running && !stop_requested.load(std::memory_order_relaxed))
    {
        Resultframe _resultframe;
        _timer->init();
        _timer->start_cpu();

        // 1. 获取 ZED 图像
        _zed->grab_frame(_writeframe);
        // ++cam_index;
        
        _timer->stop_cpu<timer::Timer::ms>("ZED Grab frame");

        // 【关键修复1】：使用 clone() 进行深拷贝
        // 这样 _resultframe 拥有一块独立的内存，不会被下一帧 grab_frame 覆盖
        _resultframe.rgb = _writeframe->rgb_ptr->clone(); 
        _resultframe.timestamp = _writeframe->timestamp;

        _timer->start_cpu();
        
        // 2. 推理 (使用独立内存的图片)
        _worker->inference(_resultframe);
        
        _timer->stop_cpu<timer::Timer::ms>("inference");
        
        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _resultframe.rs485_result = _worker->m_pose->uart_result;
        _worker->m_pose->copy_pnp_pose(_resultframe);
        if (_zed)
            _resultframe.calibration = _zed->calibration();
        // 3. 独立串口线程按 10 Hz 发送最新结果。
        // _timer->start_cpu();
        // // _rs485.sendDoubleArray(_resultframe.ose_result.data());
        // float float_temp_pose[3];
        // float_temp_pose[0] = _resultframe.rs485_result[0];
        // float_temp_pose[1] = _resultframe.rs485_result[1];
        // float_temp_pose[2] = _resultframe.rs485_result[2];
        // _rs485.sendFloatArray(float_temp_pose);
        // // usleep(600000);
        // _timer->stop_cpu<timer::Timer::ms>("RS485");
        // _timer->start_cpu(); // 如果不需要统计入队耗时，可以注释掉计时
        if (_serial_enabled)
        {
            std::lock_guard<std::mutex> lock(_rs485_mtx);
            // 仅保留最新结果，避免 100 ms 发送周期内积压旧数据。
            while (!_rs485_queue.empty())
            {
                _rs485_queue.pop();
            }
            _rs485_queue.push(_resultframe.rs485_result);
            _rs485_cv.notify_one();
        }

        // 4. TCP 队列处理
        _timer->start_cpu();
        submit_tcp_frame(std::move(_resultframe));
        _timer->stop_cpu<timer::Timer::ms>("Load TCP");
        _timer->show();
    }
}

void prj_v8detector::camera_foldimages()
{
    std::vector<cv::String> filenames;
    cv::String folder = "/home/cvia/nuaa/CVIA_trt/data/val/*.jpg";
    cv::glob(folder, filenames, false);
    std::sort(filenames.begin(), filenames.end());
    // std::sort(filenames.rbegin(), filenames.rend());
    if (filenames.empty())
    {
        LOGW("No preview images found: %s", folder.c_str());
        return;
    }
    size_t current_idx = 0;
    while (_is_running && !stop_requested.load(std::memory_order_relaxed))
    {
        usleep(20000); // 仅控制照片读取间隔（微秒）；设为 0 可取消，不影响 TCP/相机线程。
        auto now = std::chrono::system_clock::now();
        Resultframe _resultframe;
        _timer->init();
        _timer->start_cpu();
        *(_writeframe->rgb_ptr) = cv::imread(filenames[current_idx]);
        _writeframe->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        if (_writeframe->rgb_ptr->empty())
        {
            LOGW("Cannot read preview image: %s", filenames[current_idx].c_str());
            return;
        }
        _timer->stop_cpu<timer::Timer::ms>("ZED Grab frame");
        _resultframe.rgb = *(_writeframe->rgb_ptr);
        _resultframe.timestamp = _writeframe->timestamp;
        _timer->start_cpu();
        _worker->inference(_resultframe);
        _timer->stop_cpu<timer::Timer::ms>("inference");

        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _resultframe.rs485_result = _worker->m_pose->uart_result;
        _worker->m_pose->copy_pnp_pose(_resultframe);
        if (_zed)
            _resultframe.calibration = _zed->calibration();
        // _timer->start_cpu();
        // //
        // // std::vector<double> partial_result(_resultframe.pose_result.begin(), _resultframe.pose_result.begin() + 3);
        // // _rs485.sendDoubleArray(partial_result.data());
        // if (_resultframe.pose_result.size() >= 3) {
        //     // _rs485.sendDoubleArray(_resultframe.rs485_result.data());
        //     float float_temp_pose[3];
        //     float_temp_pose[0] = (float)_resultframe.rs485_result[0];
        //     float_temp_pose[1] = (float)_resultframe.rs485_result[1];
        //     float_temp_pose[2] = (float)_resultframe.rs485_result[2];
        //     _rs485.sendFloatArray(float_temp_pose);
        // } else {
        //     std::cerr << "错误：数据不足，无法发送串口数据" << std::endl;
        // }
        // //
        // // _rs485.sendDoubleArray(_resultframe.pose_result.data());
        // _timer->stop_cpu<timer::Timer::ms>("RS485");
        if (_serial_enabled)
        {
            std::lock_guard<std::mutex> lock(_rs485_mtx);
            // 同样仅保留最新结果，由串口线程按 10 Hz 发送。
            while (!_rs485_queue.empty())
            {
                _rs485_queue.pop();
            }
            if (_resultframe.rs485_result.size() >= 3) {
                 _rs485_queue.push(_resultframe.rs485_result);
                 _rs485_cv.notify_one();
            }
        }

        _timer->start_cpu();
        submit_tcp_frame(std::move(_resultframe));
        _timer->stop_cpu<timer::Timer::ms>("Load TCP");
        _timer->show();

        if (++current_idx >= filenames.size())
        {
            if (!_loop_images)
                break;
            current_idx = 0;
            LOGV("Restarting folder preview: %zu images", filenames.size());
        }
    }
}

void prj_v8detector::submit_tcp_frame(Resultframe frame)
{
    {
        std::lock_guard<std::mutex> lock(_queue_mtx);
        if (_has_pending_frame)
            ++_tcp_replaced;
        _pending_frame = std::move(frame);
        _has_pending_frame = true;
    }
    _queue_cv.notify_one();
}

void prj_v8detector::tcp_loop()
{
    using Clock = std::chrono::steady_clock;
    auto last_report = Clock::now();
    uint64_t sent = 0, failed = 0;
    while (true)
    {
        Resultframe frame_to_send;
        {
            std::unique_lock<std::mutex> lock(_queue_mtx);

            // 等待条件：有数据 或者 停止运行
            _queue_cv.wait(lock, [this]
                           { return _has_pending_frame || !_is_running; });
            if (!_is_running && !_has_pending_frame)
            {
                break;
            }

            if (!_has_pending_frame)
            {
                continue;
            }
            frame_to_send = std::move(_pending_frame);
            _has_pending_frame = false;
        }
        try
        {
            if (_client.pack_and_send(frame_to_send))
                ++sent;
            else
                ++failed;
        }
        catch (const std::exception &error)
        {
            ++failed;
            LOGW("TCP frame skipped: %s", error.what());
        }
        if (Clock::now() - last_report >= std::chrono::seconds(1))
        {
            uint64_t replaced;
            {
                std::lock_guard<std::mutex> lock(_queue_mtx);
                replaced = _tcp_replaced;
                _tcp_replaced = 0;
            }
            LOGV("TCP preview: sent=%llu, skipped=%llu, replaced=%llu",
                 static_cast<unsigned long long>(sent), static_cast<unsigned long long>(failed),
                 static_cast<unsigned long long>(replaced));
            sent = failed = 0;
            last_report = Clock::now();
        }
    }
}

void prj_v8detector::run()
{
    _is_running = true;
   auto t1 = std::thread(_func_camera);
    // auto t2 = std::thread(_func_camera_foldimages);
    std::thread t_rs485, t_rs485_rx;
    if (_serial_enabled)
    {
        t_rs485 = std::thread(_func_rs485_send);
        t_rs485_rx = std::thread(_func_rs485_receive);
    }
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
    _rs485_cv.notify_all(); // 唤醒 RS485
    if (t_rs485.joinable())
    {
        t_rs485.join();
    }
    if (t_rs485_rx.joinable())
    {
        t_rs485_rx.join();
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
    _rs485_cv.notify_all();
    if (_zed)
        _zed->close();
    delete _writeframe->rgb_ptr;
    delete _writeframe;
    preprocess::destroy_process();
}

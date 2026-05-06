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
    _func_rs485_send = std::bind(&prj_v8detector::rs485_loop, this); // [新增]

    _client.init(p_params);
    _rs485.init(p_params);
    _is_running = true;
}

// [新增] RS485 独立线程循环
void prj_v8detector::rs485_loop()
{
    while (true)
    {
        std::vector<float> data_to_send;
        {
            std::unique_lock<std::mutex> lock(_rs485_mtx);
            // 等待数据或停止信号
            _rs485_cv.wait(lock, [this] { 
                return !_rs485_queue.empty() || !_is_running; 
            });

            if (!_is_running && _rs485_queue.empty()) {
                break;
            }

            if (_rs485_queue.empty()) {
                continue;
            }

            data_to_send = _rs485_queue.front();
            _rs485_queue.pop();
        }

        // 发送数据 (移除了之前的 Timer 计时，如果需要可以加回)
        if (data_to_send.size() >= 3)
        {
            float float_temp_pose[3];
            float_temp_pose[0] = data_to_send[0];
            float_temp_pose[1] = data_to_send[1];
            float_temp_pose[2] = data_to_send[2];
            _rs485.sendFloatArray(float_temp_pose);
        }

        usleep(150000);
    }
}


// void prj_v8detector::camera()
// {
//     while (1)
//     {
//         Resultframe _resultframe;
//         _timer->init();
//         _timer->start_cpu();
//         _zed->grab_frame(_writeframe);
//         // *(_writeframe->rgb_ptr) = cv::imread(("data/source/00590.png"));
//         _timer->stop_cpu<timer::Timer::ms>("ZED Grab frame");
//         // _resultframe.rgb = *(_writeframe->rgb_ptr);
//         _resultframe.rgb = _writeframe->rgb_ptr->clone();
//         _resultframe.timestamp = _writeframe->timestamp;
//         _timer->start_cpu();
//         _worker->inference(_resultframe);
//         _timer->stop_cpu<timer::Timer::ms>("inference");
//         _resultframe.bboxes = _worker->m_pose->m_bboxes;
//         _resultframe.pose_result = _worker->m_pose->m_result;
//         _timer->start_cpu();
//         _rs485.sendDoubleArray(_resultframe.pose_result.data());
//         _timer->stop_cpu<timer::Timer::ms>("RS485");

//         _timer->start_cpu();
//         {
//             std::lock_guard<std::mutex> lock(_queue_mtx);
//             _resultframe_queue.push(_resultframe);
//             _queue_cv.notify_one();
//         }
//         _timer->stop_cpu<timer::Timer::ms>("Load TCP");
//         _timer->show();
//     }
// }

void prj_v8detector::camera()
{
    while (1)
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
        // 3. 串口发送 (保持不变)
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
        {
            std::lock_guard<std::mutex> lock(_rs485_mtx);
            // 漏桶策略：如果串口发送太慢，丢弃旧数据，保证实时性
            while (_rs485_queue.size() > 2) 
            {
                _rs485_queue.pop();
            }
            _rs485_queue.push(_resultframe.rs485_result);
            _rs485_cv.notify_one();
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
    cv::String folder = "/home/cvia/yifei/images10/*.png";
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
        // usleep(200000);
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
        _resultframe.rs485_result = _worker->m_pose->uart_result;
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
        {
            std::lock_guard<std::mutex> lock(_rs485_mtx);
            // 同样应用漏桶策略
            while (_rs485_queue.size() > 2) 
            {
                _rs485_queue.pop();
            }
            if (_resultframe.rs485_result.size() >= 3) {
                 _rs485_queue.push(_resultframe.rs485_result);
                 _rs485_cv.notify_one();
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
    auto t_rs485 = std::thread(_func_rs485_send);
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
    _rs485_cv.notify_all(); // 唤醒 RS485
    if (t_rs485.joinable())
    {
        t_rs485.join();
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
    delete _writeframe->rgb_ptr;
    delete _writeframe;
    preprocess::destroy_process();
}

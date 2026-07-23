#include "prj_detector.hpp"

prj_v8detector::prj_v8detector(string onnxPath, logger::Level level, model::Params params, prj_params p_params)
{
    _project_params = p_params;
    _worker = thread::create_worker(onnxPath, level, params);
    if (_worker && _worker->m_pose)
        _worker->m_pose->set_calibration(p_params.camera_matrix, p_params.distortion, p_params.extrinsic);
    if (p_params.input_mode == "camera")
    {
        _ir_camera_detect = std::make_unique<IRCamera>();
        _ir_camera_detect->init(p_params.detect_camera);
        _ir_camera_photo = std::make_unique<IRCamera>();
        _ir_camera_photo->init(p_params.photo_camera);
    }

    preprocess::init_process(p_params.H, p_params.W);

    _detect_writeframe = new IRFrame;
    _detect_writeframe->rgb_ptr = new cv::Mat();
    _photo_writeframe = new IRFrame;
    _photo_writeframe->rgb_ptr = new cv::Mat();
    _func_camera = std::bind(&prj_v8detector::camera, this);
    _func_detect_camera = std::bind(&prj_v8detector::detect_camera_loop, this);
    _func_photo_camera = std::bind(&prj_v8detector::photo_camera_loop, this);
    _func_camera_foldimages = std::bind(&prj_v8detector::camera_foldimages, this);
    _func_output = std::bind(&prj_v8detector::output_loop, this);
    _func_udp_send = std::bind(&prj_v8detector::udp_loop, this);
    _client.init(p_params);
    _is_running = true;
}

void prj_v8detector::detect_camera_loop()
{
    uint64_t frames = 0;
    auto report_at = std::chrono::steady_clock::now();
    while (_is_running)
    {
        if (!_ir_camera_detect->grab_frame(_detect_writeframe)) continue;

        {
            std::lock_guard<std::mutex> lock(_detect_frame_mtx);
            // 交换 Mat 所有权，不复制 1920x1080 像素。
            std::swap(_latest_detect_rgb, *_detect_writeframe->rgb_ptr);
            _latest_detect_timestamp = _detect_writeframe->timestamp;
            ++_latest_detect_sequence;
            _has_detect_frame = true;
        }
        _detect_frame_cv.notify_one();
        ++frames;
        const auto now = std::chrono::steady_clock::now();
        if (now - report_at >= std::chrono::seconds(5))
        {
            const double seconds = std::chrono::duration<double>(now - report_at).count();
            const double measured_fps = frames / seconds;
            LOG("Detect camera capture: %.1f FPS", measured_fps);
            if (measured_fps < _project_params.detect_camera.cameraframe * 0.85)
                LOGW("Detect camera requested %d FPS but delivers %.1f FPS; check the active MJPG mode, exposure and USB bus bandwidth",
                     _project_params.detect_camera.cameraframe, measured_fps);
            frames = 0; report_at = now;
        }
    }
}

void prj_v8detector::photo_camera_loop()
{
    uint64_t frames = 0;
    auto report_at = std::chrono::steady_clock::now();
    while (_is_running)
    {
        if (!_ir_camera_photo->grab_frame(_photo_writeframe)) continue;

        {
            std::lock_guard<std::mutex> lock(_photo_frame_mtx);
            std::swap(_latest_photo_rgb, *_photo_writeframe->rgb_ptr);
            _latest_photo_timestamp = _photo_writeframe->timestamp;
            _has_photo_frame = true;
        }
        ++frames;
        const auto now = std::chrono::steady_clock::now();
        if (now - report_at >= std::chrono::seconds(5))
        {
            const double seconds = std::chrono::duration<double>(now - report_at).count();
            const double measured_fps = frames / seconds;
            LOG("Photo camera capture: %.1f FPS", measured_fps);
            if (measured_fps < _project_params.photo_camera.cameraframe * 0.85)
                LOGW("Photo camera requested %d FPS but delivers %.1f FPS; check the active MJPG mode, exposure and USB bus bandwidth",
                     _project_params.photo_camera.cameraframe, measured_fps);
            frames = 0; report_at = now;
        }
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
    uint64_t frames = 0;
    auto report_at = std::chrono::steady_clock::now();
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

            // cv::Mat 浅拷贝只增加引用计数；采集线程通过 swap 发布下一帧，不会覆写该内存。
            _resultframe.rgb = _latest_detect_rgb;
            _resultframe.timestamp = _latest_detect_timestamp;
            last_processed_sequence = _latest_detect_sequence;
        }

        {
            std::lock_guard<std::mutex> lock(_photo_frame_mtx);
            if (_has_photo_frame)
            {
                _resultframe.rgb_secondary = _latest_photo_rgb;
                _resultframe.secondary_timestamp = _latest_photo_timestamp;
            }
        }

        if (_resultframe.rgb_secondary.empty())
        {
            _resultframe.rgb_secondary = _resultframe.rgb;
            _resultframe.secondary_timestamp = _resultframe.timestamp;
        }

        _worker->inference(_resultframe);

        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _resultframe.reprojected_points = _worker->m_pose->m_reprojected_points;
        _resultframe.pose_valid = _worker->m_pose->is_current_frame_good;
        _resultframe.udp_result = _resultframe.pose_result;
        // CSV 在推理线程直接顺序追加，避免预览限频/丢帧影响位姿记录完整性。
        _client.record_pose(_resultframe);
        {
            std::lock_guard<std::mutex> lock(_udp_mtx);
            // 漏桶策略：如果 UDP 发送线程处理不过来，丢弃旧数据，保证实时性。
            while (!_udp_queue.empty()) _udp_queue.pop();
            if (_resultframe.udp_result.size() >= 6)
            {
                UdpPoseFrame udp_frame;
                udp_frame.pose_result = _resultframe.udp_result;
                udp_frame.timestamp = _resultframe.timestamp;
                _udp_queue.push(std::move(udp_frame));
                _udp_cv.notify_one();
            }
        }

        // 输出线程只处理最新结果；旧预览帧没有继续排队的价值。
        {
            std::lock_guard<std::mutex> lock(_output_mtx);
            while (!_output_queue.empty()) _output_queue.pop();
            _output_queue.push(std::move(_resultframe));
            _output_cv.notify_one();
        }
        ++frames;
        const auto now = std::chrono::steady_clock::now();
        if (now - report_at >= std::chrono::seconds(5))
        {
            const double seconds = std::chrono::duration<double>(now - report_at).count();
            LOG("Inference pipeline: %.1f FPS", frames / seconds);
            frames = 0; report_at = now;
        }
    }
}

void prj_v8detector::camera_foldimages()
{
    std::vector<cv::String> filenames;
    cv::String folder = _project_params.folder_path;
    if (!folder.empty() && folder.back() != '/') folder += "/";
    std::vector<cv::String> extensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp"};
    for (const auto &extension : extensions)
    {
        std::vector<cv::String> matches;
        cv::glob(folder + extension, matches, false);
        filenames.insert(filenames.end(), matches.begin(), matches.end());
    }
    std::sort(filenames.begin(), filenames.end());
    if (filenames.empty())
    {
        LOGE("No images found in folder: %s", _project_params.folder_path.c_str());
        _is_running = false;
        return;
    }
    size_t current_idx = 0;
    while (_is_running)
    {
        auto now = std::chrono::system_clock::now();
        Resultframe _resultframe;
        _resultframe.rgb = cv::imread(filenames[current_idx]);
        if (_resultframe.rgb.empty())
        {
            break;
        }
        _resultframe.rgb_secondary = _resultframe.rgb;
        _resultframe.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        _resultframe.secondary_timestamp = _resultframe.timestamp;
        _worker->inference(_resultframe);

        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _resultframe.reprojected_points = _worker->m_pose->m_reprojected_points;
        _resultframe.pose_valid = _worker->m_pose->is_current_frame_good;
        _resultframe.udp_result = _resultframe.pose_result;
        _client.record_pose(_resultframe);
        {
            std::lock_guard<std::mutex> lock(_udp_mtx);
            // 同样应用漏桶策略
            while (!_udp_queue.empty()) _udp_queue.pop();
            if (_resultframe.udp_result.size() >= 6) {
                 UdpPoseFrame udp_frame;
                 udp_frame.pose_result = _resultframe.udp_result;
                 udp_frame.timestamp = _resultframe.timestamp;
                 _udp_queue.push(std::move(udp_frame));
                 _udp_cv.notify_one();
            }
        }

        {
            std::lock_guard<std::mutex> lock(_output_mtx);
            while (!_output_queue.empty()) _output_queue.pop();
            _output_queue.push(std::move(_resultframe));
            _output_cv.notify_one();
        }
        current_idx++;
        if (current_idx >= filenames.size())
        {
            if (_project_params.folder_loop) current_idx = 0;
            else break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(_project_params.folder_interval_ms));
    }
    _is_running = false;
    _output_cv.notify_all();
    _udp_cv.notify_all();
}

void prj_v8detector::output_loop()
{
    while (1)
    {
        Resultframe frame_to_send;
        {
            std::unique_lock<std::mutex> lock(_output_mtx);

            // 等待条件：有数据 或者 停止运行
            _output_cv.wait(lock, [this]
                           { return !_output_queue.empty() || !_is_running; });
            if (!_is_running && _output_queue.empty())
            {
                break;
            }
            if (_output_queue.empty())
            {
                continue;
            }
            frame_to_send = std::move(_output_queue.front());
            _output_queue.pop();
        }
        _client.pack_and_send(frame_to_send);
    }
}

void prj_v8detector::run()
{
    _is_running = true;
    std::thread t_detect;
    std::thread t_photo;
    std::thread t1;
    std::thread t2;
    if (_project_params.input_mode == "folder")
        t2 = std::thread(_func_camera_foldimages);
    else
    {
        t_detect = std::thread(_func_detect_camera);
        t_photo = std::thread(_func_photo_camera);
        t1 = std::thread(_func_camera);
    }
    auto t_udp = std::thread(_func_udp_send);
    auto t3 = std::thread(_func_output);
    if (t1.joinable())
    {
        t1.join();
    }

    if (t2.joinable())
        t2.join();

    _is_running = false;
    _output_cv.notify_all();
    _udp_cv.notify_all();   // 唤醒 UDP
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

void prj_v8detector::request_stop()
{
    _is_running = false;
    _output_cv.notify_all();
    _udp_cv.notify_all();
    _detect_frame_cv.notify_all();
}

prj_v8detector::~prj_v8detector()
{
    _is_running = false;
    _output_cv.notify_all();
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

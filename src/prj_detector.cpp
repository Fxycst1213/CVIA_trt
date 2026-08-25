#include "prj_detector.hpp"
#include "params/config.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace
{
bool same_camera_params(const camera_params &left, const camera_params &right)
{
    return left.name == right.name &&
           left.cameraID == right.cameraID &&
           left.cameraframe == right.cameraframe &&
           left.resolution == right.resolution &&
           left.apply_image_controls == right.apply_image_controls &&
           left.auto_exposure_mode == right.auto_exposure_mode &&
           left.apply_exposure == right.apply_exposure &&
           left.exposure == right.exposure &&
           left.auto_white_balance == right.auto_white_balance &&
           left.apply_white_balance_temperature == right.apply_white_balance_temperature &&
           left.white_balance_temperature == right.white_balance_temperature &&
           left.brightness == right.brightness &&
           left.contrast == right.contrast &&
           left.sharpness == right.sharpness;
}

bool restart_required_for(const prj_params &running, const prj_params &candidate)
{
    return running.H != candidate.H || running.W != candidate.W ||
           !same_camera_params(running.detect_camera, candidate.detect_camera) ||
           !same_camera_params(running.photo_camera, candidate.photo_camera) ||
           running.ip != candidate.ip || running.port != candidate.port ||
           running.udp_ip != candidate.udp_ip || running.udp_port != candidate.udp_port ||
           running.enable_udp != candidate.enable_udp ||
           running.enable_tcp != candidate.enable_tcp ||
           running.socket_mode != candidate.socket_mode ||
           running.onnx_model_path != candidate.onnx_model_path ||
           running.model_keypoints_3d.size() != candidate.model_keypoints_3d.size() ||
           running.input_mode != candidate.input_mode ||
           running.folder_path != candidate.folder_path ||
           running.folder_loop != candidate.folder_loop ||
           running.folder_interval_ms != candidate.folder_interval_ms ||
           running.preview_dir != candidate.preview_dir ||
           running.preview_fps != candidate.preview_fps ||
           running.save_pnp_results != candidate.save_pnp_results ||
           running.pnp_result_dir != candidate.pnp_result_dir;
}

std::string json_escape(const std::string &value)
{
    std::ostringstream output;
    for (const char character : value)
    {
        switch (character)
        {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default: output << character; break;
        }
    }
    return output.str();
}
}

prj_v8detector::prj_v8detector(string onnxPath, logger::Level level, model::Params params,
                               prj_params p_params, const std::string &config_path)
{
    _project_params = p_params;
    _config_path = config_path;
    _runtime_config_status_path = p_params.preview_dir + "/runtime_config_status.json";
    _worker = thread::create_worker(onnxPath, level, params);
    if (!_worker || !_worker->ready())
    {
        LOGW("Runtime initialization stopped: pose model output does not match the configured keypoint geometry");
        _is_running = false;
        return;
    }
    if (_worker && _worker->m_pose)
        _worker->m_pose->set_calibration(p_params.camera_matrix, p_params.distortion,
                                         p_params.extrinsic, p_params.model_keypoints_3d);
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
    read_config_file_stamp(_last_config_stamp);
    write_runtime_config_status(_last_config_stamp, "ok", "startup configuration loaded",
                                false, false);
    _next_config_check = std::chrono::steady_clock::now();
    _is_running = true;
    _ready = true;
}

bool prj_v8detector::read_config_file_stamp(ConfigFileStamp &stamp) const
{
    struct stat info{};
    if (stat(_config_path.c_str(), &info) != 0)
    {
        stamp = ConfigFileStamp{};
        return false;
    }
    stamp.modified_ns = static_cast<uint64_t>(info.st_mtim.tv_sec) * 1000000000ULL +
                        static_cast<uint64_t>(info.st_mtim.tv_nsec);
    stamp.size = static_cast<uint64_t>(info.st_size);
    stamp.inode = static_cast<uint64_t>(info.st_ino);
    stamp.valid = true;
    return true;
}

void prj_v8detector::write_runtime_config_status(const ConfigFileStamp &stamp,
                                                 const std::string &result,
                                                 const std::string &message,
                                                 bool calibration_changed,
                                                 bool restart_required)
{
    if (_runtime_config_status_path.empty()) return;
    const std::string temporary_path = _runtime_config_status_path + ".tmp";
    const uint64_t applied_at_unix_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    std::ofstream output(temporary_path, std::ios::trunc);
    if (!output) return;
    output << "{\n"
           << "  \"result\": \"" << json_escape(result) << "\",\n"
           << "  \"message\": \"" << json_escape(message) << "\",\n"
           << "  \"observed_config_mtime_ns\": \"" << stamp.modified_ns << "\",\n"
           << "  \"applied_at_unix_ns\": \"" << applied_at_unix_ns << "\",\n"
           << "  \"calibration_revision\": " << _calibration_revision << ",\n"
           << "  \"calibration_changed\": " << (calibration_changed ? "true" : "false") << ",\n"
           << "  \"restart_required\": " << (restart_required ? "true" : "false") << "\n"
           << "}\n";
    output.flush();
    output.close();
    if (std::rename(temporary_path.c_str(), _runtime_config_status_path.c_str()) != 0)
        std::remove(temporary_path.c_str());
}

void prj_v8detector::maybe_reload_runtime_config()
{
    const auto now = std::chrono::steady_clock::now();
    if (now < _next_config_check) return;
    _next_config_check = now + std::chrono::milliseconds(250);

    ConfigFileStamp current_stamp;
    if (!read_config_file_stamp(current_stamp))
        return;
    const bool unchanged = current_stamp.valid == _last_config_stamp.valid &&
        (!current_stamp.valid ||
         (current_stamp.modified_ns == _last_config_stamp.modified_ns &&
          current_stamp.size == _last_config_stamp.size &&
          current_stamp.inode == _last_config_stamp.inode));
    if (unchanged) return;

    prj_params candidate = _project_params;
    std::string error;
    if (!load_project_config(_config_path, candidate, error))
    {
        _last_config_stamp = current_stamp;
        write_runtime_config_status(current_stamp, "error", error, false, false);
        LOGW("Runtime config reload failed: %s", error.c_str());
        return;
    }

    const bool intrinsics_changed =
        candidate.camera_matrix != _project_params.camera_matrix ||
        candidate.distortion != _project_params.distortion;
    const bool extrinsic_changed = candidate.extrinsic != _project_params.extrinsic;
    const bool model_keypoint_count_changed =
        candidate.model_keypoints_3d.size() != _project_params.model_keypoints_3d.size();
    const bool model_keypoints_changed = !model_keypoint_count_changed &&
        candidate.model_keypoints_3d != _project_params.model_keypoints_3d;
    const bool calibration_changed =
        intrinsics_changed || extrinsic_changed || model_keypoints_changed;
    const bool restart_required = restart_required_for(_project_params, candidate);

    if (calibration_changed && _worker && _worker->m_pose)
    {
        const model::pose::ModelKeypoints3D &points_to_apply =
            model_keypoint_count_changed
                ? _project_params.model_keypoints_3d
                : candidate.model_keypoints_3d;
        _worker->m_pose->set_calibration(candidate.camera_matrix, candidate.distortion,
                                         candidate.extrinsic, points_to_apply,
                                         intrinsics_changed || model_keypoints_changed);
        _project_params.camera_matrix = candidate.camera_matrix;
        _project_params.distortion = candidate.distortion;
        _project_params.extrinsic = candidate.extrinsic;
        if (!model_keypoint_count_changed)
            _project_params.model_keypoints_3d = candidate.model_keypoints_3d;
        ++_calibration_revision;
    }

    _last_config_stamp = current_stamp;
    const std::string message = calibration_changed
        ? (restart_required ? "calibration applied; other changes require restart"
                            : "calibration applied")
        : (restart_required ? "configuration observed; changes require restart"
                            : "configuration observed; no runtime change");
    write_runtime_config_status(current_stamp, "ok", message,
                                calibration_changed, restart_required);
    LOG("Runtime config observed: calibration=%s, restart_required=%s, revision=%llu",
        calibration_changed ? "applied" : "unchanged",
        restart_required ? "true" : "false",
        static_cast<unsigned long long>(_calibration_revision));
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
            _latest_detect_capture_timestamp_ns = _detect_writeframe->capture_timestamp_ns;
            _latest_detect_dequeue_timestamp_ns = _detect_writeframe->dequeue_timestamp_ns;
            _latest_detect_driver_timestamp = _detect_writeframe->driver_timestamp;
            _latest_detect_timestamp_start_of_exposure =
                _detect_writeframe->timestamp_start_of_exposure;
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
            _resultframe.capture_timestamp_ns = _latest_detect_capture_timestamp_ns;
            _resultframe.dequeue_timestamp_ns = _latest_detect_dequeue_timestamp_ns;
            _resultframe.driver_timestamp = _latest_detect_driver_timestamp;
            _resultframe.timestamp_start_of_exposure =
                _latest_detect_timestamp_start_of_exposure;
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

        maybe_reload_runtime_config();
        _worker->inference(_resultframe);

        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _resultframe.reprojected_origin = _worker->m_pose->m_reprojected_origin;
        _resultframe.reprojected_origin_valid = _worker->m_pose->m_reprojected_origin_valid;
        _resultframe.reprojected_points = _worker->m_pose->m_reprojected_points;
        _resultframe.pose_valid = _worker->m_pose->is_current_frame_good;
        _resultframe.udp_result = _resultframe.pose_result;
        _resultframe.publish_timestamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
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
        const auto frame_started_at = std::chrono::steady_clock::now();
        auto now = std::chrono::system_clock::now();
        Resultframe _resultframe;
        _resultframe.rgb = cv::imread(filenames[current_idx]);
        if (_resultframe.rgb.empty())
        {
            break;
        }
        _resultframe.rgb_secondary = _resultframe.rgb;
        _resultframe.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        _resultframe.capture_timestamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
        _resultframe.dequeue_timestamp_ns = _resultframe.capture_timestamp_ns;
        _resultframe.secondary_timestamp = _resultframe.timestamp;
        maybe_reload_runtime_config();
        _worker->inference(_resultframe);

        _resultframe.bboxes = _worker->m_pose->m_bboxes;
        _resultframe.pose_result = _worker->m_pose->m_result;
        _resultframe.reprojected_origin = _worker->m_pose->m_reprojected_origin;
        _resultframe.reprojected_origin_valid = _worker->m_pose->m_reprojected_origin_valid;
        _resultframe.reprojected_points = _worker->m_pose->m_reprojected_points;
        _resultframe.pose_valid = _worker->m_pose->is_current_frame_good;
        _resultframe.udp_result = _resultframe.pose_result;
        _resultframe.publish_timestamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
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
        // interval_ms 表示相邻输入帧的起始间隔。原先在推理结束后再完整
        // sleep，会把解码和推理耗时额外叠加，造成文件夹播放节奏偏慢且抖动。
        std::this_thread::sleep_until(
            frame_started_at + std::chrono::milliseconds(_project_params.folder_interval_ms));
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
    if (!_ready) return;
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

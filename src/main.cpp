#include "TensorRT/model/model.hpp"
#include "logger.hpp"
#include "worker/worker.hpp"
#include "utils.hpp"
#include "prj_detector.hpp"
#include "params/params.hpp"
#include "params/config.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>
using namespace std;

namespace
{
volatile std::sig_atomic_t g_stop_requested = 0;

void handle_stop_signal(int)
{
    g_stop_requested = 1;
}
}

int main(int argc, char const *argv[])
{
    LOG("CVIA runtime revision: v4l2-ptp-timestamp-v4");
    // 尽早接管停止信号；即使模型或相机仍在初始化，也会在构造完成后立即走优雅停止流程。
    std::signal(SIGINT, handle_stop_signal);
    std::signal(SIGTERM, handle_stop_signal);
    // pose
    // string onnxPath = "models/onnx/last_rebest_1203.onnx";
    // string onnxPath = "models/onnx/0128last.onnx";

    string onnxPath = "models/onnx/qdy0815.onnx";
    // INFO 会保留启动/告警日志，但关闭每帧 VERB 计时输出，避免终端 I/O 拖慢双 60 FPS。
    auto level = logger::Level::INFO;
    auto params = model::Params();
    params.img = {640, 640, 3};
    params.task = model::task_type::POSE;
    params.dev = model::device::GPU;
    params.prec = model::precision::FP16;
    auto p_params = prj_params();

    p_params.t_params = tcp_params();

    p_params.H = 1080;
    p_params.W = 1920;
    p_params.detect_camera.name = "Detect IR Camera";
    p_params.detect_camera.cameraID = 0;
    p_params.detect_camera.cameraframe = 60;
    p_params.detect_camera.resolution = "HD1080";
    p_params.detect_camera.apply_image_controls = true;
    p_params.detect_camera.auto_exposure_mode = 1.0;
    p_params.detect_camera.apply_exposure = true;
    p_params.detect_camera.exposure = 78.0;
    p_params.detect_camera.auto_white_balance = false;
    p_params.detect_camera.apply_white_balance_temperature = true;
    p_params.detect_camera.white_balance_temperature = 4600.0;
    p_params.detect_camera.brightness = -64.0;
    p_params.detect_camera.contrast = 100.0;
    p_params.detect_camera.sharpness = 100.0;

    p_params.photo_camera.name = "Visible Light Camera";
    p_params.photo_camera.cameraID = 2;
    p_params.photo_camera.cameraframe = 60;
    p_params.photo_camera.resolution = "HD1080";
    // 可见光相机保持设备默认成像效果，不写曝光、白平衡、亮度、对比度和清晰度。
    p_params.photo_camera.apply_image_controls = false;
    p_params.ip = "192.168.137.1";
    p_params.port = 1234;
    p_params.udp_ip = "10.128.85.15";
    p_params.udp_port = 1234;
    p_params.enable_udp = true;
    p_params.socket_mode = 2;
    // p_params.rs485_port = "/dev/ttyUSB0"; // 串口发送已停用，结果改用 UDP 上传。
    // p_params.rs485_baudrate = B57600;

    const bool check_assets = argc > 1 && string(argv[1]) == "--check-assets";
    const std::string config_path = check_assets
        ? (argc > 2 ? argv[2] : "web_monitor/config.json")
        : (argc > 1 ? argv[1] : "web_monitor/config.json");
    std::string config_error;
    if (load_project_config(config_path, p_params, config_error))
        LOG("Loaded project config: %s", config_path.c_str());
    else
        LOGW("Using compiled defaults (%s)", config_error.c_str());
    params.pose_keypoint_count = static_cast<int>(p_params.model_keypoints_3d.size() / 3);
    onnxPath = p_params.onnx_model_path;
    const string enginePath = changePath(onnxPath, "../engine", ".engine", "fp16");
    if (check_assets)
    {
        if (!fileExists(enginePath) && !fileExists(onnxPath))
        {
            fprintf(stderr,
                    "[CVIA] Model unavailable; engine: %s; ONNX fallback: %s\n",
                    enginePath.c_str(), onnxPath.c_str());
            return 4;
        }
        fprintf(stdout, "[CVIA] Runtime model OK: onnx=%s, engine=%s, source=%s, configured_keypoints=%d\n",
                onnxPath.c_str(), enginePath.c_str(),
                fileExists(enginePath) ? "engine" : "onnx",
                params.pose_keypoint_count);
        return 0;
    }

    // 根据worker中的task类型进行推理
    prj_v8detector prj(onnxPath, level, params, p_params, config_path);
    if (!prj.ready())
    {
        fprintf(stderr,
                "[CVIA] Runtime did not start; verify model output keypoint count "
                "and calibration.model_keypoints_3d.\n");
        return 6;
    }
    std::atomic<bool> run_finished{false};
    std::thread stop_watcher([&]() {
        while (!run_finished && !g_stop_requested)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (g_stop_requested)
        {
            LOG("Stop requested; finalizing temporary PNP session...");
            prj.request_stop();
        }
    });
    prj.run();
    run_finished = true;
    stop_watcher.join();

    return 0;
}

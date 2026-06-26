#include "TensorRT/model/model.hpp"
#include "logger.hpp"
#include "worker/worker.hpp"
#include "utils.hpp"
#include "prj_detector.hpp"
#include "params/params.hpp"
using namespace std;

int main(int argc, char const *argv[])
{
    // pose
    // string onnxPath = "models/onnx/last_rebest_1203.onnx";
    // string onnxPath = "models/onnx/0128last.onnx";

    string onnxPath = "models/onnx/qdy_last.onnx";
    auto level = logger::Level::VERB;
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
    p_params.detect_camera.auto_exposure_mode = 1.0;
    p_params.detect_camera.apply_exposure = true;
    p_params.detect_camera.exposure = 78.0;
    p_params.detect_camera.auto_white_balance = false;
    p_params.detect_camera.apply_white_balance_temperature = true;
    p_params.detect_camera.white_balance_temperature = 4600.0;
    p_params.detect_camera.brightness = -64.0;
    p_params.detect_camera.contrast = 100.0;
    p_params.detect_camera.sharpness = 100.0;

    p_params.photo_camera.name = "Photo IR Camera";
    p_params.photo_camera.cameraID = 2;
    p_params.photo_camera.cameraframe = 30;
    p_params.photo_camera.resolution = "HD1080";
    // 这一路用于看清周遭环境，默认打开自动曝光和自动白平衡
    p_params.photo_camera.auto_exposure_mode = 3.0;
    p_params.photo_camera.apply_exposure = false;
    p_params.photo_camera.auto_white_balance = true;
    p_params.photo_camera.apply_white_balance_temperature = false;
    p_params.photo_camera.brightness = 0.0;
    p_params.photo_camera.contrast = 50.0;
    p_params.photo_camera.sharpness = 50.0;
    p_params.ip = "192.168.137.1";
    p_params.port = 1234;
    p_params.udp_ip = "192.168.137.1";
    p_params.udp_port = 1234;
    p_params.enable_udp = true;
    p_params.socket_mode = 2;
    // p_params.rs485_port = "/dev/ttyUSB0"; // 串口发送已停用，结果改用 UDP 上传。
    // p_params.rs485_baudrate = B57600;

    // // 根据worker中的task类型进行推理
    prj_v8detector prj(onnxPath, level, params, p_params);
    prj.run();

    return 0;
}

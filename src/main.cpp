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
    // sudo date -s "20260402 20:32:00"
    string onnxPath = "models/onnx/0128last.onnx";
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
    p_params.resolution = "HD1080";
    p_params.cameraID = 0;
    p_params.ip = "192.168.137.1";
    p_params.port = 1234;
    p_params.socket_mode = 0;
    p_params.rs485_port = "/dev/ttyUSB0";
    p_params.rs485_baudrate = B9600;

    // // 根据worker中的task类型进行推理
    prj_v8detector prj(onnxPath, level, params, p_params);
    prj.run();

    return 0;
}

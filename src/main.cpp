#include "TensorRT/model/model.hpp"
#include "logger.hpp"
#include "worker/worker.hpp"
#include "utils.hpp"
#include "prj_detector.hpp"
using namespace std;

int main(int argc, char const *argv[])
{
    // pose
    string onnxPath = "models/onnx/last_rebest_1203.onnx";
    auto level = logger::Level::VERB;
    auto params = model::Params();
    params.img = {640, 640, 3};
    params.task = model::task_type::POSE;
    params.dev = model::device::GPU;
    params.prec = model::precision::FP16;
    auto p_params = prj_params();

    p_params.H = 1080;
    p_params.W = 1920;
    p_params.resolution = "HD1080";
    p_params.cameraID = 0;
    p_params.ip = "10.1.1.195";
    p_params.port = 1234;
    p_params.socket_mode = 0;

    // // 根据worker中的task类型进行推理
    prj_v8detector prj(onnxPath, level, params, p_params);
    prj.run();

    // auto worker = thread::create_worker(onnxPath, level, params);
    // preprocess::init_process(p_params.H, p_params.W);
    // cv::Mat Image = cv::imread("data/source/00590.png");
    // for (int i = 0; i < 2; i++)
    // {
    //     worker->inference(Image);
    // }
    return 0;
}

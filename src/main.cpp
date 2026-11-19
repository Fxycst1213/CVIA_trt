#include "TensorRT/model/model.hpp"
#include "logger.hpp"
#include "worker/worker.hpp"
#include "utils.hpp"
#include "prj_detector.hpp"
#include "params/params.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <arpa/inet.h>
using namespace std;

int main(int argc, char const *argv[])
{
    // White aircraft: nine keypoints, ordered to match Pose's CAD points.
    string onnxPath = "models/onnx/X_1118.onnx";
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
    p_params.ip = "172.23.127.15";
    p_params.port = 1234;
    p_params.socket_mode = 0;
    p_params.rs485_port = "/dev/ttyUSB0";
    p_params.rs485_baudrate = B9600;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--help")
            {
                std::cout << "Usage: ./trt [--preview] [--tcp-ip IPv4] [--tcp-port PORT]\n"
                             "             [--jpeg-width WIDTH] [--jpeg-quality 1..100]\n"
                             "             [--extended-pose] [--calibrated-pose]\n"
                             "             [--coordinate-frames] [--2d-only]\n"
                             "             [--camera-eye right|left] [--no-serial] [--verbose]\n"
                             "--preview: send a 1280px-wide preview to 127.0.0.1:1234.\n"
                             "--2d-only: 2D keypoints; skip PnP, LSTM and serial output.\n"
                             "--coordinate-frames: include camera calibration and robot-base XYZ (IHB).\n"
                             "--jpeg-width 0: original image size. Inference always uses full size.\n";
                return 0;
            }
            if (arg == "--preview")
            {
                p_params.ip = "127.0.0.1";
                p_params.t_params.jpeg_width = 1280;
                continue;
            }
            if (arg == "--verbose")
            {
                p_params.verbose = true;
                continue;
            }
            if (arg == "--no-serial")
            {
                p_params.enable_serial = false;
                continue;
            }
            if (arg == "--2d-only")
            {
                params.keypoints_only = true;
                p_params.enable_serial = false;
                continue;
            }
            if (arg == "--extended-pose")
            {
                p_params.socket_mode = 2;
                continue;
            }
            if (arg == "--calibrated-pose")
            {
                p_params.socket_mode = 3;
                continue;
            }
            if (arg == "--coordinate-frames")
            {
                p_params.socket_mode = 4;
                continue;
            }
            if (i + 1 >= argc)
                throw std::invalid_argument("Missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--tcp-ip")
                p_params.ip = value;
            else if (arg == "--camera-eye")
                p_params.camera_eye = value;
            else
            {
                size_t end = 0;
                int number = std::stoi(value, &end);
                if (end != value.size())
                    throw std::invalid_argument("Invalid number for " + arg);
                if (arg == "--tcp-port") p_params.port = number;
                else if (arg == "--jpeg-width") p_params.t_params.jpeg_width = number;
                else if (arg == "--jpeg-quality") p_params.t_params.jpeg_quality = number;
                else throw std::invalid_argument("Unknown option: " + arg);
            }
        }
        if (p_params.port < 1 || p_params.port > 65535 ||
            (p_params.camera_eye != "left" && p_params.camera_eye != "right") ||
            p_params.t_params.jpeg_width < 0 || p_params.t_params.jpeg_width > 8192 ||
            p_params.t_params.jpeg_quality < 1 || p_params.t_params.jpeg_quality > 100)
            throw std::invalid_argument("Preview option outside supported range");
        in_addr address{};
        if (inet_pton(AF_INET, p_params.ip.c_str(), &address) != 1)
            throw std::invalid_argument("--tcp-ip must be an IPv4 address");
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << "; see ./trt --help\n";
        return EXIT_FAILURE;
    }
    if (params.keypoints_only)
        p_params.socket_mode = 0; // Image/keypoint packets do not need camera or 3D calibration.
    auto level = p_params.verbose ? logger::Level::VERB : logger::Level::WARN;
    // // 根据worker中的task类型进行推理
    prj_v8detector prj(onnxPath, level, params, p_params);
    prj.run();

    return 0;
}

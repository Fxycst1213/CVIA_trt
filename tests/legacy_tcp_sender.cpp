// Exercise the production TCP client without camera, CUDA, TensorRT, or serial I/O.
#include "communication/client.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

int main(int argc, char **argv)
{
    if (argc != 7)
        return 2;
    prj_params params;
    params.ip = "127.0.0.1";
    params.port = std::atoi(argv[1]);
    params.t_params.jpeg_width = std::atoi(argv[2]);
    params.socket_mode = std::atoi(argv[3]);
    const int count = std::atoi(argv[4]);
    const int delay_ms = std::atoi(argv[5]);
    const bool pressure = std::atoi(argv[6]);
    params.t_params.jpeg_quality = pressure ? 95 : 50;
    client sender;
    sender.init(params);
    Resultframe frame;
    frame.rgb = cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(25, 35, 45));
    if (pressure)
        cv::randu(frame.rgb, 0, 256);
    model::pose::bbox box(480, 270, 1440, 810, 0.9f, 0);
    box.keypoints.assign(model::pose::NUM_KEYPOINTS, model::pose::keypoint(960, 540, 0.95f));
    frame.bboxes.push_back(box);
    frame.pose_result = {1, 2, 3, 4, 5, 6, 999, 999};
    frame.pnp_rvec = {{0.1f, 0.2f, 0.3f}};
    frame.pnp_tvec = {{400, 500, 600}};
    frame.calibration.width = 1920;
    frame.calibration.height = 1080;
    frame.calibration.right_eye = true;
    frame.calibration.intrinsics = {{1000, 1100, 960, 540}};
    frame.calibration.distortion = {{-.05, .01, .001, -.002, .003, .004, .005, .006, .007, .008, .009, .01}};
    int sent = 0;
    double maximum_ms = 0;
    for (int i = 0; i < count; ++i)
    {
        frame.timestamp = 1234567890000ULL + i;
        frame.pnp_valid = i % 2 == 0;
        frame.robot_base_valid = i % 3 != 2;
        frame.robot_base_xyz = {{100.25f + i, -200.5f - i, 300.75f + i}};
        const auto start = std::chrono::steady_clock::now();
        if (sender.pack_and_send(frame))
            ++sent;
        maximum_ms = std::max(maximum_ms, std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
    std::cout << "{\"sent\":" << sent << ",\"maximum_ms\":" << maximum_ms << "}\n";
}

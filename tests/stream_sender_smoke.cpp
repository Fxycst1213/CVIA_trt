#include "frame_streamer.hpp"
#include <opencv2/core.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

int main(int argc, char **argv)
{
    if (argc < 3)
        return 2;
    const int port = std::atoi(argv[1]);
    const int duration_ms = std::atoi(argv[2]);
    const bool pressure = argc > 3;
    streaming::FrameStreamer sender("127.0.0.1", port, 90);
    sender.start();
    cv::Mat image(pressure ? 720 : 120, pressure ? 1280 : 160, CV_8UC3, cv::Scalar(0, 0, 0));
    if (pressure)
        cv::randu(image, 0, 256);
    const auto start = std::chrono::steady_clock::now();
    uint64_t sequence = 0;
    while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(duration_ms))
    {
        streaming::Frame frame;
        frame.image = image.clone();
        frame.timestamp_ms = 1234567890000ULL + ++sequence;
        frame.num_keypoints = 17;
        streaming::Detection detection;
        detection.label = 2;
        detection.confidence = 0.9f;
        detection.bbox = {{10, 10, 110, 90}};
        detection.keypoints.assign(17, {{40, 30, 0.95f}});
        frame.detections.push_back(detection);
        sender.submit(std::move(frame));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    const auto stop_started = std::chrono::steady_clock::now();
    sender.stop();
    const auto stop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - stop_started)
                             .count();
    std::cout << "STOP_MS=" << stop_ms << '\n';
    return stop_ms < 2000 ? 0 : 1;
}

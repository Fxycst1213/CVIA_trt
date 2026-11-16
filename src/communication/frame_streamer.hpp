#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace streaming
{
    struct Detection
    {
        int label = 0;
        float confidence = 0;
        std::array<float, 4> bbox; // x0, y0, x1, y1 in original image pixels.
        std::vector<std::array<float, 3>> keypoints; // x, y, confidence.
    };

    struct Frame
    {
        uint64_t frame_id = 0;
        uint64_t timestamp_ms = 0;
        int num_keypoints = 0;
        cv::Mat image; // Owned snapshot; annotated by the sending thread.
        std::vector<Detection> detections;
    };

    class FrameStreamer
    {
    public:
        FrameStreamer(std::string ip, int port, int jpeg_quality = 80);
        ~FrameStreamer();
        void start();
        void stop();
        void submit(Frame frame);

    private:
        void send_loop();

        std::string _ip;
        int _port;
        int _jpeg_quality;
        uint64_t _next_frame_id = 0;
        std::atomic<bool> _running{false};
        std::mutex _mutex;
        std::condition_variable _cv;
        Frame _pending;
        bool _has_pending = false;
        std::thread _thread;
    };
}

#include "frame_streamer.hpp"
#include "client.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <arpa/inet.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace streaming
{
    namespace
    {
        float finite(float value)
        {
            return std::isfinite(value) ? value : 0.0f;
        }

        std::string metadata(const Frame &frame)
        {
            std::ostringstream json;
            json.imbue(std::locale::classic());
            json << std::setprecision(7)
                 << "{\"frame_id\":" << frame.frame_id
                 << ",\"timestamp_ms\":" << frame.timestamp_ms
                 << ",\"width\":" << frame.image.cols
                 << ",\"height\":" << frame.image.rows
                 << ",\"num_keypoints\":" << frame.num_keypoints
                 << ",\"coordinate_system\":\"original_image_pixels\",\"detections\":[";
            bool first_box = true;
            for (const auto &box : frame.detections)
            {
                if (!first_box)
                    json << ',';
                first_box = false;
                json << "{\"label\":" << box.label
                     << ",\"confidence\":" << finite(box.confidence) << ",\"bbox\":[";
                for (std::size_t i = 0; i < box.bbox.size(); ++i)
                {
                    if (i)
                        json << ',';
                    json << finite(box.bbox[i]);
                }
                json << "],\"keypoints\":[";
                bool first_point = true;
                for (const auto &point : box.keypoints)
                {
                    if (!first_point)
                        json << ',';
                    first_point = false;
                    json << '[' << finite(point[0]) << ',' << finite(point[1])
                         << ',' << finite(point[2]) << ']';
                }
                json << "]}";
            }
            json << "]}";
            return json.str();
        }

        cv::Point image_point(float x, float y, const cv::Mat &image)
        {
            return cv::Point(static_cast<int>(std::max(0.0f, std::min(finite(x), float(image.cols - 1)))),
                             static_cast<int>(std::max(0.0f, std::min(finite(y), float(image.rows - 1)))));
        }

        std::vector<unsigned char> encode_packet(Frame &frame, int quality)
        {
            const std::string json = metadata(frame);
            for (const auto &box : frame.detections)
            {
                const auto top_left = image_point(box.bbox[0], box.bbox[1], frame.image);
                const auto bottom_right = image_point(box.bbox[2], box.bbox[3], frame.image);
                cv::rectangle(frame.image, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);
                std::ostringstream label;
                label << box.label << ' ' << std::fixed << std::setprecision(2) << box.confidence;
                cv::putText(frame.image, label.str(), top_left, cv::FONT_HERSHEY_SIMPLEX,
                            0.5, cv::Scalar(0, 255, 0), 1);
                for (const auto &point : box.keypoints)
                {
                    if (point[2] >= 0.5f && std::isfinite(point[0]) && std::isfinite(point[1]))
                        cv::circle(frame.image, image_point(point[0], point[1], frame.image),
                                   3, cv::Scalar(0, 0, 255), -1);
                }
            }
            std::vector<unsigned char> jpeg;
            if (!cv::imencode(".jpg", frame.image, jpeg, {cv::IMWRITE_JPEG_QUALITY, quality}))
                throw std::runtime_error("JPEG encoding failed");
            if (json.size() > 4 * 1024 * 1024 || jpeg.size() > 16 * 1024 * 1024)
                throw std::runtime_error("TCP frame exceeds protocol size limits");

            // 16-byte header: magic CVIA, version 1, JSON length, JPEG length.
            // All four uint32 values are in network byte order. JSON and JPEG
            // follow immediately and belong to the same frame_id/timestamp_ms.
            const std::array<uint32_t, 4> header{{htonl(0x43564941), htonl(1),
                                                htonl(static_cast<uint32_t>(json.size())),
                                                htonl(static_cast<uint32_t>(jpeg.size()))}};
            std::vector<unsigned char> packet(sizeof(header) + json.size() + jpeg.size());
            std::memcpy(packet.data(), header.data(), sizeof(header));
            std::memcpy(packet.data() + sizeof(header), json.data(), json.size());
            std::memcpy(packet.data() + sizeof(header) + json.size(), jpeg.data(), jpeg.size());
            return packet;
        }
    }

    FrameStreamer::FrameStreamer(std::string ip, int port, int jpeg_quality)
        : _ip(std::move(ip)), _port(port), _jpeg_quality(jpeg_quality)
    {
        if (_port <= 0 || _port > 65535 || jpeg_quality < 1 || jpeg_quality > 100)
            throw std::invalid_argument("Invalid TCP port or JPEG quality");
    }

    FrameStreamer::~FrameStreamer()
    {
        stop();
    }

    void FrameStreamer::start()
    {
        if (_running.exchange(true))
            return;
        try
        {
            _thread = std::thread(&FrameStreamer::send_loop, this);
        }
        catch (...)
        {
            _running.store(false);
            throw;
        }
    }

    void FrameStreamer::stop()
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _running.store(false);
        }
        _cv.notify_all();
        if (_thread.joinable())
            _thread.join();
    }

    void FrameStreamer::submit(Frame frame)
    {
        if (frame.image.empty())
            return;
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_running.load())
            return;
        frame.frame_id = ++_next_frame_id;
        _pending = std::move(frame);
        _has_pending = true;
        _cv.notify_one();
    }

    void FrameStreamer::send_loop()
    {
        client connection;
        while (_running.load())
        {
            if (!connection.connected())
            {
                if (connection.init(_ip, _port) != 0)
                {
                    std::cerr << "[tcp] Cannot connect to " << _ip << ':' << _port
                              << ": " << std::strerror(errno) << "; retrying.\n";
                    std::unique_lock<std::mutex> lock(_mutex);
                    _cv.wait_for(lock, std::chrono::seconds(1), [this]
                                 { return !_running.load(); });
                    continue;
                }
                std::cerr << "[tcp] Connected to " << _ip << ':' << _port << '\n';
            }
            Frame frame;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _cv.wait(lock, [this] { return _has_pending || !_running.load(); });
                if (!_running.load())
                    break;
                frame = std::move(_pending);
                _has_pending = false;
            }
            try
            {
                const auto packet = encode_packet(frame, _jpeg_quality);
                if (!connection.SendAll(packet.data(), packet.size()))
                {
                    std::cerr << "[tcp] Send failed: " << std::strerror(errno)
                              << "; reconnecting.\n";
                    connection.disconnect();
                }
            }
            catch (const std::exception &error)
            {
                std::cerr << "[tcp] Dropped frame: " << error.what() << '\n';
            }
        }
    }
}

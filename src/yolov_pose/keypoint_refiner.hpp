#pragma once

#include "../params/pose_params.hpp"
#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>

namespace model
{
namespace pose
{
    struct KeypointRefinementStats
    {
        int color_matches = 0;
        int flow_matches = 0;
        int rejected_jumps = 0;
    };

    class KeypointRefiner
    {
    public:
        void reset();
        void refine(const cv::Mat &image, std::vector<keypoint> &points,
                    const cv::Rect2f &box, uint64_t timestamp_ms);
        const KeypointRefinementStats &stats() const { return stats_; }

    private:
        cv::Mat previous_gray_;
        cv::Size previous_size_;
        cv::Rect2f previous_box_;
        uint64_t previous_timestamp_ = 0;
        std::vector<cv::Point2f> previous_points_;
        std::vector<unsigned char> previous_valid_;
        std::vector<float> marker_areas_;
        std::vector<int> rejected_frames_;
        std::vector<cv::Point2f> pending_points_;
        KeypointRefinementStats stats_;
    };
}
}

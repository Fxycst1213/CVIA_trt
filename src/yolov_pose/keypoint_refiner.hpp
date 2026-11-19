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
        // Compatibility with the offline CSV evaluator: these now count
        // validated subpixel ellipse-edge measurements, not color centroids.
        int color_matches = 0;
        int flow_matches = 0;   // Always zero; optical flow no longer supplies centers.
        int rejected_jumps = 0; // Always zero; measurements are not temporally filtered.
        int ambiguous_matches = 0;
        int unrefined = 0;
    };

    struct KeypointRefinementMeasurement
    {
        bool valid = false;
        cv::RotatedRect ellipse; // Full-resolution coordinates; axes are diameters.
        float coverage = 0;
        float inlier_fraction = 0;
        float edge_rms_px = 0;
        float contrast = 0;
        float center_spread_px = 0;
    };

    class KeypointRefiner
    {
    public:
        void reset();

        // YOLO supplies identity/search seeds, never the fitted center.
        // Only validated current-image measurements replace x/y. On failure,
        // retain raw coordinates/confidence; measurements()[i].valid is false.
        // Non-finite inputs are cleared to avoid invalid downstream PnP input.
        // Confidence is never promoted. No prediction/smoothing is returned
        // as a measurement. Frames are independent, including unordered photos;
        // timestamp_ms is accepted only for source compatibility.
        void refine(const cv::Mat &image, std::vector<keypoint> &points,
                    const cv::Rect2f &box, uint64_t timestamp_ms);
        const KeypointRefinementStats &stats() const { return stats_; }
        const std::vector<KeypointRefinementMeasurement> &measurements() const { return measurements_; }

    private:
        std::vector<KeypointRefinementMeasurement> measurements_;
        KeypointRefinementStats stats_;
    };
}
}

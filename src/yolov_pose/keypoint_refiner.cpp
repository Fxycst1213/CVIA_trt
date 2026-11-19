#include "keypoint_refiner.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace model
{
namespace pose
{
namespace
{
    // Sunlit black stickers can be gray/cyan with V well above 100. Use local
    // contrast for segmentation; V only exempts genuinely dark, tinted pixels
    // from the saturation check.
    constexpr int kBlackMarkerMaxValue = 100;
    constexpr int kMarkerMaxSaturation = 140;
    constexpr float kSearchRadiusScale = 1.6f;
    constexpr float kSearchRadiusMax = 48.0f;
    constexpr float kCorrectionLimitMax = 24.0f;

    float bounded(float value, float low, float high)
    {
        return std::max(low, std::min(value, high));
    }

    bool inside(const cv::Point2f &point, const cv::Size &size)
    {
        return std::isfinite(point.x) && std::isfinite(point.y) &&
               point.x >= 0 && point.y >= 0 && point.x < size.width && point.y < size.height;
    }

    float median(std::vector<float> values)
    {
        const auto middle = values.begin() + values.size() / 2;
        std::nth_element(values.begin(), middle, values.end());
        return *middle;
    }

    bool marker_center(const cv::Mat &image, const cv::Mat &gray, const cv::Point2f &raw,
                       const cv::Point2f &prediction, bool tracked,
                       float diameter, float previous_area,
                       cv::Point2f &center, float &selected_area)
    {
        // Leave room for the entire ellipse when the network lands on its rim.
        const float radius = bounded(diameter * kSearchRadiusScale, 12.0f, kSearchRadiusMax);
        const float correction_limit = bounded(diameter * 1.05f, 6.0f, kCorrectionLimitMax);
        const cv::Point2f anchor = tracked ? prediction * 0.6f + raw * 0.4f : raw;
        const cv::Rect requested(cvFloor(anchor.x - radius), cvFloor(anchor.y - radius),
                                 cvCeil(radius * 2 + 1), cvCeil(radius * 2 + 1));
        const cv::Rect roi = requested & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width < 5 || roi.height < 5)
            return false;
        cv::Mat hsv, smooth, contrast, eligible, dark;
        cv::cvtColor(image(roi), hsv, cv::COLOR_BGR2HSV);
        cv::Mat channels[3];
        cv::split(image(roi), channels);
        cv::min(channels[0], channels[1], smooth);
        cv::min(smooth, channels[2], smooth);
        // Cyan reflections are still dark in the red channel. Blending the
        // darkest channel with luminance retains that contrast without relying
        // on one particular marker hue.
        cv::addWeighted(gray(roi), 0.5, smooth, 0.5, 0, smooth);
        cv::GaussianBlur(smooth, smooth, cv::Size(3, 3), 0);
        // Closing estimates the surrounding body brightness without assuming a
        // white exposure level. Subtraction preserves compact dark stickers but
        // removes broad illumination gradients. A rectangular kernel is separable.
        const int background_size = cvCeil(bounded(diameter * 1.8f, 9.0f, 65.0f)) | 1;
        cv::morphologyEx(smooth, contrast, cv::MORPH_BLACKHAT,
                        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(background_size, background_size)));
        cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(179, kMarkerMaxSaturation, 255), eligible);
        cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(179, 255, kBlackMarkerMaxValue), dark);
        cv::bitwise_or(eligible, dark, eligible);
        const cv::Mat cleanup = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        const float expected_area = previous_area > 0 ? previous_area :
                                    static_cast<float>(CV_PI) * diameter * diameter * 0.25f;
        float best_score = std::numeric_limits<float>::infinity();
        // A single threshold either loses reflective stickers or merges them
        // with a nearby shadow. Compare complete ellipses at several contrasts.
        for (int threshold : {18, 30, 48, 72})
        {
            cv::Mat mask;
            cv::compare(contrast, threshold, mask, cv::CMP_GE);
            cv::bitwise_and(mask, eligible, mask);
            // Disconnect thin cables/shadow edges, but retain tiny distant marks.
            if (diameter >= 12.0f)
                cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cleanup);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cleanup);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            for (const auto &contour : contours)
            {
                const double area = cv::contourArea(contour);
                if (contour.size() < 5 || area < std::max(3.0f, expected_area * 0.08f) || area > expected_area * 4.0f)
                    continue;
                const cv::Rect bounds = cv::boundingRect(contour);
                // A clipped dark background/stripe is not a complete marker.
                if (bounds.x <= 0 || bounds.y <= 0 || bounds.br().x >= roi.width || bounds.br().y >= roi.height)
                    continue;
                const cv::RotatedRect ellipse = cv::fitEllipse(contour);
                const float major = std::max(ellipse.size.width, ellipse.size.height);
                const float minor = std::min(ellipse.size.width, ellipse.size.height);
                const double ellipse_area = CV_PI * major * minor * 0.25;
                if (!std::isfinite(ellipse_area) || minor < 2.0f || major < diameter * 0.6f ||
                    major > diameter * 1.8f || major > minor * 6.0f ||
                    area < ellipse_area * 0.72 || area > ellipse_area * 1.15)
                    continue;
                // Penalize irregular contours while allowing strongly
                // foreshortened circular stickers at any image rotation.
                const float angle = ellipse.angle * static_cast<float>(CV_PI) / 180.0f;
                const float cosine = std::cos(angle), sine = std::sin(angle);
                float shape_error = 0;
                for (const auto &pixel : contour)
                {
                    const cv::Point2f p = cv::Point2f(pixel) - ellipse.center;
                    const float x = (p.x * cosine + p.y * sine) * 2 / ellipse.size.width;
                    const float y = (-p.x * sine + p.y * cosine) * 2 / ellipse.size.height;
                    shape_error += std::abs(std::sqrt(x * x + y * y) - 1.0f);
                }
                shape_error /= contour.size();
                if (shape_error > 0.18f)
                    continue;
                const cv::Point2f candidate = ellipse.center + cv::Point2f(roi.x, roi.y);
                if (!inside(candidate, image.size()))
                    continue;
                const float raw_distance = cv::norm(candidate - raw);
                const float predicted_distance = cv::norm(candidate - prediction);
                if (raw_distance > correction_limit && (!tracked || predicted_distance > correction_limit * 0.5f))
                    continue;
                cv::Mat filled = cv::Mat::zeros(mask.size(), CV_8U);
                cv::drawContours(filled, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), cv::FILLED);
                const cv::Scalar color = cv::mean(hsv, filled);
                // Do not accept the low-saturation antialiased rim of a bright
                // colored object as a dark marker with a hole in its middle.
                if (color[1] > kMarkerMaxSaturation && color[2] > kBlackMarkerMaxValue)
                    continue;
                const float score = std::min(4.0f, raw_distance / correction_limit) +
                                    (tracked ? 0.7f * predicted_distance / correction_limit : 0.0f) +
                                    0.2f * std::abs(std::log(area / expected_area)) +
                                    0.4f * std::abs(std::log(major / diameter)) + 3.0f * shape_error;
                if (score < best_score)
                {
                    best_score = score;
                    center = candidate;
                    selected_area = area;
                }
            }
        }
        return std::isfinite(best_score);
    }
}

void KeypointRefiner::reset()
{
    previous_gray_.release();
    previous_points_.clear();
    previous_valid_.clear();
    marker_areas_.clear();
    rejected_frames_.clear();
    pending_points_.clear();
    previous_size_ = {};
    previous_box_ = {};
    previous_timestamp_ = 0;
    stats_ = {};
}

void KeypointRefiner::refine(const cv::Mat &image, std::vector<keypoint> &points,
                            const cv::Rect2f &box, uint64_t timestamp_ms)
{
    stats_ = {};
    if (image.empty() || image.type() != CV_8UC3 || points.empty() || box.width <= 0 || box.height <= 0)
    {
        reset();
        return;
    }
    const float scale = std::min(1.0f, 960.0f / image.cols);
    cv::Mat gray, reduced;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, reduced, cv::Size(), scale, scale, cv::INTER_AREA);
    const bool gap = previous_timestamp_ &&
                     (timestamp_ms <= previous_timestamp_ || timestamp_ms - previous_timestamp_ > 500);
    const float box_ratio = previous_box_.width > 0 ? box.width / previous_box_.width : 1.0f;
    const cv::Point2f box_center = box.tl() + cv::Point2f(box.width, box.height) * 0.5f;
    const cv::Point2f old_center = previous_box_.tl() + cv::Point2f(previous_box_.width, previous_box_.height) * 0.5f;
    const bool new_target = previous_box_.width > 0 &&
                            (box_ratio < 0.4f || box_ratio > 2.5f ||
                             cv::norm(box_center - old_center) > std::max(100.0f, box.width * 0.65f));
    if (gap || new_target || image.size() != previous_size_ || previous_points_.size() != points.size())
        reset();
    const float dt = previous_timestamp_ && timestamp_ms > previous_timestamp_ ?
                     bounded((timestamp_ms - previous_timestamp_) / 1000.0f, 1.0f / 120, 0.2f) : 1.0f / 30;
    const bool have_history = !previous_gray_.empty();
    if (!have_history)
    {
        previous_points_.resize(points.size());
        previous_valid_.assign(points.size(), 0);
        marker_areas_.assign(points.size(), 0);
        rejected_frames_.assign(points.size(), 0);
        pending_points_.resize(points.size());
    }
    std::vector<cv::Point2f> predictions(points.size());
    std::vector<unsigned char> prediction_valid(points.size(), 0), input_valid(points.size(), 0);
    for (size_t i = 0; i < points.size(); ++i)
        input_valid[i] = std::isfinite(points[i].conf) && points[i].conf >= 0.5f &&
                         inside(cv::Point2f(points[i].x, points[i].y), image.size());

    if (have_history)
    {
        std::vector<cv::Point2f> before, forward, backward;
        std::vector<size_t> ids;
        for (size_t i = 0; i < points.size(); ++i)
            if (previous_valid_[i] && input_valid[i])
            {
                before.push_back(previous_points_[i] * scale);
                ids.push_back(i);
            }
        if (!before.empty())
        {
            std::vector<unsigned char> forward_status, backward_status;
            std::vector<float> forward_error, backward_error;
            const cv::TermCriteria stop(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.01);
            cv::calcOpticalFlowPyrLK(previous_gray_, reduced, before, forward, forward_status,
                                    forward_error, cv::Size(21, 21), 3, stop);
            for (size_t j = 0; j < forward.size(); ++j)
                if (!forward_status[j] || !inside(forward[j], reduced.size()))
                {
                    forward_status[j] = 0;
                    forward[j] = before[j];
                }
            cv::calcOpticalFlowPyrLK(reduced, previous_gray_, forward, backward, backward_status,
                                    backward_error, cv::Size(21, 21), 3, stop);
            for (size_t j = 0; j < ids.size(); ++j)
                if (forward_status[j] && backward_status[j] && forward_error[j] < 30 &&
                    cv::norm(backward[j] - before[j]) / scale < 1.5f && inside(forward[j], reduced.size()))
                {
                    predictions[ids[j]] = forward[j] / scale;
                    prediction_valid[ids[j]] = 1;
                }
        }

        // Coherent whole-target movement must not be mistaken for 17 outliers.
        std::vector<float> dx, dy;
        for (size_t i = 0; i < points.size(); ++i)
            if (prediction_valid[i])
            {
                dx.push_back(points[i].x - predictions[i].x);
                dy.push_back(points[i].y - predictions[i].y);
            }
        if (dx.size() >= 6)
        {
            const cv::Point2f shift(median(dx), median(dy));
            std::vector<float> deviations;
            for (size_t i = 0; i < dx.size(); ++i)
                deviations.push_back(cv::norm(cv::Point2f(dx[i], dy[i]) - shift));
            if (cv::norm(shift) > std::max(10.0f, box.width * 0.012f) &&
                median(deviations) < cv::norm(shift) * 0.5f)
                std::fill(prediction_valid.begin(), prediction_valid.end(), 0);
        }
        std::vector<cv::Point2f> source, destination;
        for (size_t i = 0; i < points.size(); ++i)
            if (prediction_valid[i])
            {
                source.push_back(previous_points_[i]);
                destination.push_back(predictions[i]);
            }
        if (source.size() >= 4)
        {
            cv::Mat inliers;
            const cv::Mat motion = cv::estimateAffinePartial2D(source, destination, inliers,
                                                               cv::RANSAC, 3.0, 200, 0.99);
            if (!motion.empty() && cv::countNonZero(inliers) >= std::max(3, static_cast<int>(source.size() * 0.6)))
                for (size_t i = 0; i < points.size(); ++i)
                    if (!prediction_valid[i] && previous_valid_[i] && input_valid[i])
                    {
                        const auto &p = previous_points_[i];
                        predictions[i] = cv::Point2f(motion.at<double>(0, 0)*p.x + motion.at<double>(0, 1)*p.y + motion.at<double>(0, 2),
                                                      motion.at<double>(1, 0)*p.x + motion.at<double>(1, 1)*p.y + motion.at<double>(1, 2));
                        prediction_valid[i] = inside(predictions[i], image.size()) ? 2 : 0;
                    }
        }
    }
    const float diameter = bounded(box.width * 0.022f, 3.0f, 36.0f);
    const float jump_limit = bounded(diameter * 0.5f, 3.0f, 12.0f);
    for (size_t i = 0; i < points.size(); ++i)
    {
        auto &point = points[i];
        if (!input_valid[i])
        {
            previous_valid_[i] = 0;
            rejected_frames_[i] = 0;
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.conf))
            {
                point.x = point.y = point.conf = 0;
            }
            continue;
        }
        const cv::Point2f raw(point.x, point.y);
        const bool tracked = prediction_valid[i] != 0;
        const cv::Point2f prediction = tracked ? predictions[i] : raw;
        cv::Point2f measurement = raw;
        float area = 0;
        const bool localized = marker_center(image, gray, raw, prediction, tracked, diameter,
                                             marker_areas_[i], measurement, area);
        if (localized)
        {
            ++stats_.color_matches;
        }
        cv::Point2f filtered = measurement;
        bool rejected = false;
        if (tracked)
        {
            ++stats_.flow_matches;
            const float innovation = cv::norm(measurement - prediction);
            if (innovation > jump_limit)
            {
                rejected_frames_[i] = rejected_frames_[i] && cv::norm(measurement - pending_points_[i]) < jump_limit ?
                                      rejected_frames_[i] + 1 : 1;
                pending_points_[i] = measurement;
                if (rejected_frames_[i] < 3)
                {
                    filtered = prediction;
                    point.conf = std::min(point.conf, 0.74f);
                    rejected = true;
                    ++stats_.rejected_jumps;
                }
                else
                    rejected_frames_[i] = 0;
            }
            else
            {
                rejected_frames_[i] = 0;
                const float cutoff = 1.5f + std::min(8.0f, innovation * 0.3f);
                float gain = prediction_valid[i] == 2 ? 0.7f :
                             1.0f - std::exp(-2.0f * static_cast<float>(CV_PI) * cutoff * dt);
                // A visible marker is stronger evidence than an off-center flow
                // prediction. Keep the existing smoothing for network-only input.
                if (localized)
                    gain = std::max(gain, 0.65f);
                filtered = prediction + (measurement - prediction) * gain;
            }
        }
        else
            rejected_frames_[i] = 0;
        if (localized && !rejected)
            marker_areas_[i] = marker_areas_[i] > 0 ? 0.8f * marker_areas_[i] + 0.2f * area : area;
        point.x = filtered.x;
        point.y = filtered.y;
        previous_points_[i] = filtered;
        previous_valid_[i] = 1;
    }
    previous_gray_ = reduced;
    previous_size_ = image.size();
    previous_box_ = box;
    previous_timestamp_ = timestamp_ms;
}
}
}

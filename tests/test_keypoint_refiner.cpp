#include "keypoint_refiner.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using model::pose::keypoint;
using model::pose::KeypointRefiner;
using model::pose::NUM_KEYPOINTS;

namespace
{
    void require(bool condition, const char *message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    std::vector<cv::Point2f> centers(float shift = 0)
    {
        std::vector<cv::Point2f> result;
        for (int i = 0; i < NUM_KEYPOINTS; ++i)
            result.emplace_back(60 + (i % 6) * 70 + shift, 70 + (i / 6) * 80);
        return result;
    }

    cv::Mat image(const std::vector<cv::Point2f> &points,
                  const cv::Scalar &marker_color = cv::Scalar(20, 20, 20))
    {
        cv::Mat frame(360, 640, CV_8UC3, cv::Scalar(220, 220, 220));
        for (const auto &point : points)
            cv::circle(frame, point, 6, marker_color, -1, cv::LINE_AA);
        return frame;
    }

    std::vector<keypoint> observations(const std::vector<cv::Point2f> &truth,
                                      std::mt19937 &random, float sigma = 2.0f)
    {
        std::normal_distribution<float> noise(0, sigma);
        std::vector<keypoint> points;
        for (const auto &point : truth)
            points.emplace_back(point.x + noise(random), point.y + noise(random), 0.95f);
        return points;
    }

    float distance(const keypoint &point, const cv::Point2f &truth)
    {
        return cv::norm(cv::Point2f(point.x, point.y) - truth);
    }
}

int main()
{
    try
    {
        cv::setNumThreads(2);
        std::mt19937 random(42);
        const cv::Rect2f box(20, 20, 500, 260);
        const auto truth = centers();
        const auto frame = image(truth);
        KeypointRefiner refiner;
        for (const auto &marker_color : {cv::Scalar(0, 0, 0), cv::Scalar(65, 65, 65),
                                        cv::Scalar(75, 20, 10), cv::Scalar(150, 150, 150),
                                        cv::Scalar(205, 205, 145)})
        {
            refiner.reset();
            std::vector<keypoint> points;
            for (const auto &point : truth)
                points.emplace_back(point.x + 3, point.y - 2, 0.95f);
            refiner.refine(image(truth, marker_color), points, box, 33);
            require(refiner.stats().color_matches == NUM_KEYPOINTS, "Dark stickers were not all localized");
            for (int i = 0; i < NUM_KEYPOINTS; ++i)
                require(distance(points[i], truth[i]) < 0.5f, "Dark sticker center was not recovered");
        }
        std::cout << "PASS black, gray and reflective cyan stickers on a white body\n";

        for (const auto &marker_color : {cv::Scalar(215, 215, 215), cv::Scalar(245, 245, 245),
                                        cv::Scalar(15, 15, 210)})
        {
            refiner.reset();
            std::vector<keypoint> points;
            for (const auto &point : truth)
                points.emplace_back(point.x + 3, point.y - 2, 0.95f);
            const auto raw = points;
            refiner.refine(image(truth, marker_color), points, box, 33);
            require(refiner.stats().color_matches == 0, "Bright region was treated as a dark sticker");
            for (int i = 0; i < NUM_KEYPOINTS; ++i)
                require(distance(points[i], cv::Point2f(raw[i].x, raw[i].y)) < 0.01f,
                        "Missing dark sticker did not preserve the initial observation");
        }
        std::cout << "PASS low-contrast, bright and saturated-red rejection with raw-point fallback\n";

        // Match the supplied 1920-pixel view: a roughly 1170-pixel aircraft box,
        // flat 28x6 stickers, exposure variation, and network errors beyond 16 px.
        const cv::Rect2f large_box(20, 40, 1170, 600);
        std::vector<cv::Point2f> ellipse_truth;
        for (int i = 0; i < NUM_KEYPOINTS; ++i)
            ellipse_truth.emplace_back(120 + (i % 5) * 210, 180 + (i / 5) * 230);
        cv::Mat reflective_frame(720, 1280, CV_8UC3);
        for (int y = 0; y < reflective_frame.rows; ++y)
        {
            const int brightness = 225 + y / 36;
            reflective_frame.row(y).setTo(cv::Scalar(brightness, brightness, brightness));
        }
        std::vector<keypoint> offset_points;
        for (int i = 0; i < NUM_KEYPOINTS; ++i)
        {
            const cv::Scalar color = i % 2 ? cv::Scalar(205, 205, 145) : cv::Scalar(155, 155, 155);
            cv::ellipse(reflective_frame, ellipse_truth[i], cv::Size(14, 3),
                        (i % 3) * 25.0, 0, 360, color, -1, cv::LINE_AA);
            offset_points.emplace_back(ellipse_truth[i].x + 18, ellipse_truth[i].y - 10, 0.95f);
        }
        refiner.reset();
        refiner.refine(reflective_frame, offset_points, large_box, 33);
        require(refiner.stats().color_matches == NUM_KEYPOINTS, "Reflective ellipses were missed in the expanded search");
        for (int i = 0; i < NUM_KEYPOINTS; ++i)
            require(distance(offset_points[i], ellipse_truth[i]) < 1.5f,
                    "Foreshortened sticker center was not recovered from a 21-pixel error");
        std::cout << "PASS reflective, rotated thin ellipses with illumination gradients and 21-pixel offsets\n";

        refiner.reset();
        cv::Mat clutter(720, 1280, CV_8UC3, cv::Scalar(235, 235, 235));
        const cv::Point2f isolated(300, 300);
        // A cable crosses the expanded window; a real sticker lies inside the
        // search window but outside the allowed correction distance.
        cv::line(clutter, cv::Point(310, 220), cv::Point(310, 380), cv::Scalar(30, 30, 30), 2);
        cv::ellipse(clutter, cv::Point(330, 300), cv::Size(8, 5), 0, 0, 360,
                    cv::Scalar(65, 65, 65), -1, cv::LINE_AA);
        std::vector<keypoint> clutter_points{keypoint(isolated.x, isolated.y, 0.95f)};
        refiner.refine(clutter, clutter_points, large_box, 33);
        require(refiner.stats().color_matches == 0 && distance(clutter_points[0], isolated) < 0.01f,
                "Expanded search snapped to a cable or a marker beyond the correction limit");
        refiner.reset();
        clutter_points[0].x = 309;
        refiner.refine(clutter, clutter_points, large_box, 33);
        require(refiner.stats().color_matches == 1 && distance(clutter_points[0], cv::Point2f(330, 300)) < 1.5f,
                "Cable clutter prevented localization inside the correction limit");
        std::cout << "PASS expanded-window cable rejection and bounded correction\n";

        refiner.reset();
        double raw_squared = 0, refined_squared = 0;
        for (int t = 1; t <= 80; ++t)
        {
            auto points = observations(truth, random);
            for (int i = 0; i < NUM_KEYPOINTS; ++i)
                raw_squared += std::pow(distance(points[i], truth[i]), 2);
            refiner.refine(frame, points, box, t * 33);
            for (int i = 0; i < NUM_KEYPOINTS; ++i)
                refined_squared += std::pow(distance(points[i], truth[i]), 2);
        }
        require(refined_squared < raw_squared * 0.25, "Stationary localization did not improve enough");
        std::cout << "PASS stationary noise: raw RMS=" << std::sqrt(raw_squared / (80*NUM_KEYPOINTS))
                  << ", refined RMS=" << std::sqrt(refined_squared / (80*NUM_KEYPOINTS)) << " px\n";

        refiner.reset();
        double lag = 0;
        for (int t = 1; t <= 35; ++t)
        {
            auto moving_truth = centers(t * 3.0f);
            auto points = observations(moving_truth, random);
            refiner.refine(image(moving_truth), points, cv::Rect2f(20+t*3, 20, 500, 260), t*33);
            for (int i = 0; i < NUM_KEYPOINTS; ++i)
                lag += std::abs(points[i].x - moving_truth[i].x);
        }
        require(lag / (35*NUM_KEYPOINTS) < 1.5, "Motion compensation introduced excessive motion lag");
        std::cout << "PASS real motion: mean horizontal error=" << lag / (35*NUM_KEYPOINTS) << " px\n";

        refiner.reset();
        double fallback_error = 0, signed_lag = 0;
        int measured = 0;
        for (int t = 1; t <= 60; ++t)
        {
            auto moving_truth = centers(t * 2.0f);
            auto points = observations(moving_truth, random);
            // Bright red provides optical-flow texture without passing the dark mask.
            refiner.refine(image(moving_truth, cv::Scalar(15, 15, 210)), points,
                           cv::Rect2f(20+t*2, 20, 500, 260), t*33);
            require(refiner.stats().color_matches == 0, "Fallback test unexpectedly used dark segmentation");
            if (t > 10)
                for (int i = 0; i < NUM_KEYPOINTS; ++i)
                {
                    fallback_error += distance(points[i], moving_truth[i]);
                    signed_lag += points[i].x - moving_truth[i].x;
                    ++measured;
                }
        }
        require(fallback_error / measured < 1.8, "Temporal filtering without color matching is too noisy");
        require(std::abs(signed_lag / measured) < 0.75, "Temporal filtering lags real motion without color matching");
        std::cout << "PASS motion without dark segmentation: mean error=" << fallback_error / measured
                  << ", signed horizontal lag=" << signed_lag / measured << " px\n";

        refiner.reset();
        int rejected = 0;
        for (int t = 1; t <= 25; ++t)
        {
            auto points = observations(truth, random, 0.3f);
            if (t == 15)
                points[0].x += 40;
            refiner.refine(frame, points, box, t*33);
            if (t == 15)
            {
                require(distance(points[0], truth[0]) < 3, "Isolated outlier was not suppressed");
                rejected = refiner.stats().rejected_jumps;
            }
        }
        require(rejected > 0, "Outlier rejection was not exercised");
        std::cout << "PASS isolated 40-pixel neural-network outlier\n";

        auto jumped_truth = centers(70);
        auto jumped = observations(jumped_truth, random, 0.1f);
        refiner.refine(image(jumped_truth), jumped, cv::Rect2f(90, 20, 500, 260), 26*33);
        for (int i = 0; i < NUM_KEYPOINTS; ++i)
            require(distance(jumped[i], jumped_truth[i]) < 3, "Coherent target jump was incorrectly frozen");
        std::cout << "PASS coherent 70-pixel target movement\n";

        auto after_gap_truth = centers(110);
        auto after_gap = observations(after_gap_truth, random, 0.1f);
        refiner.refine(image(after_gap_truth), after_gap, cv::Rect2f(130, 20, 500, 260), 3000);
        for (int i = 0; i < NUM_KEYPOINTS; ++i)
            require(distance(after_gap[i], after_gap_truth[i]) < 2, "Timestamp gap kept stale state");
        std::cout << "PASS timestamp-gap reacquisition\n";

        refiner.reset();
        auto unlocalized = observations(truth, random, 0);
        cv::Mat background(360, 640, CV_8UC3, cv::Scalar(20, 20, 20));
        refiner.refine(background, unlocalized, box, 33);
        for (int i = 0; i < NUM_KEYPOINTS; ++i)
            require(distance(unlocalized[i], truth[i]) < 0.01f, "Dark background was treated as a marker");
        require(refiner.stats().color_matches == 0, "Clipped dark background was accepted");
        std::cout << "PASS dark-background rejection and raw-point fallback\n";

        auto missing = observations(truth, random, 0);
        missing[0].conf = 0;
        missing[1] = keypoint(-5, -5, 0.9f);
        refiner.refine(frame, missing, box, 66);
        require(missing[0].conf == 0, "Lost keypoint confidence was promoted");
        require(std::isfinite(missing[1].x), "Image-boundary handling failed");
        std::cout << "PASS missing-keypoint confidence and image-boundary handling\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

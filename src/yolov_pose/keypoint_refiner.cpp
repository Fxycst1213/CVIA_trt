#include "keypoint_refiner.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace model
{
namespace pose
{
namespace
{
    // Adapted from EllipseMarkerDetector in tools/extract_split_video.py:
    // threshold proposals -> positive normal-gradient edges -> robust ellipse
    // fit -> coverage/contrast/leave-quarter-out stability checks. Work stays
    // inside YOLO-seeded ROIs; no full-frame optical-flow pyramids.
    // Fewer angular samples than the offline annotator, with subpixel normal
    // profiles and the same geometric validation, keep per-frame CPU work bounded.
    constexpr int kSamples = 64;
    constexpr float kMinConfidence = 0.5f;

    float bounded(float value, float low, float high)
    {
        return std::max(low, std::min(value, high));
    }

    bool inside(const cv::Point2f &p, const cv::Size &size)
    {
        return std::isfinite(p.x) && std::isfinite(p.y) &&
               p.x >= 0 && p.y >= 0 && p.x < size.width && p.y < size.height;
    }

    float major(const cv::RotatedRect &e) { return std::max(e.size.width, e.size.height); }
    float minor(const cv::RotatedRect &e) { return std::min(e.size.width, e.size.height); }

    bool valid_ellipse(const cv::RotatedRect &e)
    {
        return std::isfinite(e.center.x) && std::isfinite(e.center.y) &&
               std::isfinite(e.size.width) && std::isfinite(e.size.height) &&
               std::isfinite(e.angle) && minor(e) >= 3 && major(e) <= 400 &&
               major(e) <= minor(e) * 12;
    }

    float median(std::vector<float> values)
    {
        if (values.empty()) return 0;
        const auto middle = values.begin() + values.size() / 2;
        std::nth_element(values.begin(), middle, values.end());
        const float upper = *middle;
        return values.size() % 2 ? upper : (upper + *std::max_element(values.begin(), middle)) * 0.5f;
    }

    struct EllipseGeometry
    {
        cv::Point2f center;
        float a, b, c, s;
        explicit EllipseGeometry(const cv::RotatedRect &e) : center(e.center),
            a(e.size.width * 0.5f), b(e.size.height * 0.5f),
            c(std::cos(e.angle * CV_PI / 180)), s(std::sin(e.angle * CV_PI / 180)) {}

        cv::Point2f rotate(float x, float y) const { return {c*x - s*y, s*x + c*y}; }

        void sample(int i, cv::Point2f &p, cv::Point2f &normal) const
        {
            static const std::array<cv::Point2f, kSamples> unit = [] {
                std::array<cv::Point2f, kSamples> points;
                for (int j = 0; j < kSamples; ++j)
                    points[j] = {float(std::cos(j * 2 * CV_PI / kSamples)),
                                 float(std::sin(j * 2 * CV_PI / kSamples))};
                return points;
            }();
            p = center + rotate(a * unit[i].x, b * unit[i].y);
            normal = rotate(unit[i].x / a, unit[i].y / b);
            normal *= 1.0f / float(cv::norm(normal));
        }

        float residual(const cv::Point2f &p) const
        {
            const cv::Point2f d = p - center;
            const float x = (c*d.x + s*d.y) / a, y = (-s*d.x + c*d.y) / b;
            return (x*x + y*y - 1) / std::max(1e-6f, 2 * std::sqrt(x*x/(a*a) + y*y/(b*b)));
        }
    };

    float pixel(const cv::Mat &image, const cv::Point2f &p)
    {
        const float x = bounded(p.x, 0, image.cols - 1.001f);
        const float y = bounded(p.y, 0, image.rows - 1.001f);
        const int ix = int(x), iy = int(y);
        const float dx = x - ix, dy = y - iy;
        const float *top = image.ptr<float>(iy), *bottom = image.ptr<float>(iy + 1);
        return (1-dy) * ((1-dx)*top[ix] + dx*top[ix+1]) +
               dy * ((1-dx)*bottom[ix] + dx*bottom[ix+1]);
    }

    bool fit(const std::vector<cv::Point2f> &points, cv::RotatedRect &ellipse)
    {
        if (points.size() < 5) return false;
        ellipse = cv::fitEllipse(points);
        return valid_ellipse(ellipse);
    }

    bool refine_edges(const cv::Mat &smooth, const cv::RotatedRect &initial,
                      KeypointRefinementMeasurement &result)
    {
        if (!valid_ellipse(initial)) return false;
        cv::RotatedRect ellipse = initial;
        const float radius = bounded(major(initial) * 0.10f, 5.0f, 11.0f);
        const int half_steps = cvCeil(radius / 0.25f);
        const int columns = half_steps * 2 + 1;
        const float step = radius / half_steps;
        cv::Mat map_x(kSamples, columns, CV_32F), map_y(kSamples, columns, CV_32F), profiles;
        std::vector<cv::Point2f> edges, inliers;
        std::vector<int> indices, inlier_indices;
        int supported = 0;
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            const cv::RotatedRect before = ellipse;
            const EllipseGeometry geometry(ellipse);
            std::array<cv::Point2f, kSamples> boundary, normals;
            for (int i = 0; i < kSamples; ++i)
            {
                geometry.sample(i, boundary[i], normals[i]);
                float *mx = map_x.ptr<float>(i), *my = map_y.ptr<float>(i);
                for (int j = 0; j < columns; ++j)
                {
                    const cv::Point2f p = boundary[i] + normals[i] * ((j - half_steps) * step);
                    mx[j] = p.x; my[j] = p.y;
                }
            }
            cv::remap(smooth, profiles, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            edges.clear(); indices.clear();
            for (int i = 0; i < kSamples; ++i)
            {
                const float *row = profiles.ptr<float>(i);
                int peak = 2;
                float strength = -std::numeric_limits<float>::infinity();
                for (int j = 2; j < columns - 2; ++j)
                {
                    const float gradient = (row[j+1] - row[j-1]) / (2*step);
                    if (gradient > strength) { strength = gradient; peak = j; }
                }
                if (strength < 6 || peak <= 2 || peak >= columns - 3) continue;
                const float left = (row[peak] - row[peak-2]) / (2*step);
                const float right = (row[peak+2] - row[peak]) / (2*step);
                const float denominator = left - 2*strength + right;
                const float fraction = std::abs(denominator) > 1e-6f ?
                    bounded(0.5f*(left-right)/denominator, -0.5f, 0.5f) : 0;
                const cv::Point2f edge = boundary[i] + normals[i] * ((peak-half_steps+fraction)*step);
                // Replicated image/ROI borders must never become measured edges.
                if (edge.x < 2 || edge.y < 2 || edge.x >= smooth.cols-2 || edge.y >= smooth.rows-2) continue;
                edges.push_back(edge); indices.push_back(i);
            }
            supported = int(edges.size());
            if (supported < kSamples * 0.65f || !fit(edges, ellipse)) return false;
            std::vector<int> fitted_indices = indices;
            for (int trim = 0; trim < 2; ++trim)
            {
                const EllipseGeometry fitted(ellipse);
                std::vector<float> errors, deviations;
                for (const auto &p : edges) errors.push_back(fitted.residual(p));
                const float middle = median(errors);
                for (float error : errors) deviations.push_back(std::abs(error-middle));
                const float limit = std::max(0.85f, 3 * 1.4826f * median(deviations));
                inliers.clear(); inlier_indices.clear();
                for (size_t j = 0; j < edges.size(); ++j)
                    if (std::abs(errors[j]-middle) < limit)
                    {
                        inliers.push_back(edges[j]); inlier_indices.push_back(indices[j]);
                    }
                if (inliers.size() < kSamples * 0.55f) return false;
                // Re-fitting an unchanged inlier set gives the same answer.
                // Clean edges usually need just the initial fit in this pass.
                if (inlier_indices == fitted_indices) break;
                if (!fit(inliers, ellipse)) return false;
                fitted_indices = inlier_indices;
            }
            if (cv::norm(ellipse.center-initial.center) > std::max(6.0f, major(initial)*0.32f) ||
                minor(ellipse)/minor(initial) < 0.65f || minor(ellipse)/minor(initial) > 1.55f ||
                major(ellipse)/major(initial) < 0.65f || major(ellipse)/major(initial) > 1.55f) return false;
            float angle_change = std::abs(ellipse.angle-before.angle);
            angle_change = std::min(angle_change, std::abs(180.0f-angle_change));
            if (iteration >= 1 && cv::norm(ellipse.center-before.center) < 0.03f &&
                std::abs(major(ellipse)-major(before)) < 0.08f &&
                std::abs(minor(ellipse)-minor(before)) < 0.08f && angle_change < 0.1f)
                break;
        }
        const EllipseGeometry geometry(ellipse);
        float squared = 0, spread = 0;
        for (const auto &p : inliers) squared += std::pow(geometry.residual(p), 2);
        const float rms = std::sqrt(squared / inliers.size());
        if (rms > std::max(1.2f, major(ellipse)*0.035f)) return false;
        std::vector<float> differences;
        const float contrast_offset = bounded(minor(ellipse)*0.25f, 1.0f, 3.0f);
        for (int i : inlier_indices)
        {
            cv::Point2f p, normal;
            geometry.sample(i, p, normal);
            const auto inner = p - normal*contrast_offset, outer = p + normal*contrast_offset;
            if (!inside(inner, smooth.size()) || !inside(outer, smooth.size())) continue;
            const float outside = pixel(smooth, outer);
            differences.push_back(outside - pixel(smooth, inner));
        }
        // Specular glare may whiten most of the interior. YOLO identifies the
        // marker; the remaining dark-to-light OUTER rim supplies its geometry.
        const float contrast = median(differences);
        if (contrast < 24) return false;
        for (int sector = 0; sector < 4; ++sector)
        {
            std::vector<cv::Point2f> subset;
            for (size_t j = 0; j < inliers.size(); ++j)
                if (inlier_indices[j] / (kSamples/4) != sector) subset.push_back(inliers[j]);
            cv::RotatedRect alternate;
            if (subset.size() < kSamples*0.35f || !fit(subset, alternate)) return false;
            spread = std::max(spread, float(cv::norm(alternate.center-ellipse.center)));
        }
        if (spread > std::max(6.0f, major(ellipse)*0.16f)) return false;
        result.ellipse = ellipse;
        result.coverage = float(supported)/kSamples;
        result.inlier_fraction = float(inliers.size())/kSamples;
        result.edge_rms_px = rms;
        result.contrast = contrast;
        result.center_spread_px = spread;
        return true;
    }

    bool stable(const KeypointRefinementMeasurement &m)
    {
        return m.center_spread_px <= std::max(0.9f, major(m.ellipse)*0.025f);
    }

    bool complete(const KeypointRefinementMeasurement &m)
    {
        return m.coverage >= 0.95f && m.inlier_fraction >= 0.95f && m.contrast >= 30 &&
               m.edge_rms_px <= std::max(0.8f, major(m.ellipse)*0.02f);
    }

    bool consensus(const KeypointRefinementMeasurement &m,
                   const std::vector<KeypointRefinementMeasurement> &observations)
    {
        if (!complete(m)) return false;
        for (const auto &other : observations)
            if (&other != &m && complete(other) && cv::norm(m.ellipse.center-other.ellipse.center) <= 0.65f)
                return true;
        return false;
    }

    bool dark_color(const cv::Mat &image, const cv::RotatedRect &ellipse)
    {
        const EllipseGeometry geometry(ellipse);
        cv::Vec3f mean(0, 0, 0);
        int count = 0;
        for (int i = 0; i < kSamples; i += 8)
        {
            cv::Point2f p, normal; geometry.sample(i, p, normal);
            p = ellipse.center + (p-ellipse.center)*0.55f;
            if (!inside(p, image.size())) continue;
            mean += cv::Vec3f(image.at<cv::Vec3b>(int(p.y), int(p.x))); ++count;
        }
        if (count < kSamples/16) return false;
        mean *= 1.0f/count;
        const float value = std::max({mean[0], mean[1], mean[2]});
        const float low = std::min({mean[0], mean[1], mean[2]});
        return value <= 100 || (value-low)*255 <= value*140;
    }

    struct Candidate { cv::RotatedRect ellipse; float score; };

    float correction_limit(const cv::RotatedRect &ellipse, float expected, float radius)
    {
        return std::min(radius*0.8f, std::max(8.0f, major(ellipse)*0.8f + expected*0.25f));
    }

    bool belongs_to_seed(const cv::Point2f &center, size_t id, const std::vector<cv::Point2f> &seeds,
                         const std::vector<unsigned char> &valid)
    {
        const float distance = cv::norm(center-seeds[id]);
        for (size_t j = 0; j < seeds.size(); ++j)
            if (j != id && valid[j] && cv::norm(center-seeds[j])+2 < distance*0.8f) return false;
        return true;
    }

    std::vector<Candidate> candidates(const cv::Mat &gray, const cv::Point2f &seed,
                                      float expected, float radius, bool quick)
    {
        cv::Mat mask;
        std::vector<int> thresholds;
        if (!quick)
            for (int value = 70; value <= 225; value += 15) thresholds.push_back(value);
        thresholds.push_back(cvRound(cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU)));
        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
        std::vector<Candidate> proposals;
        for (int threshold : thresholds)
        {
            cv::threshold(gray, mask, threshold, 255, cv::THRESH_BINARY_INV);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            for (const auto &contour : contours)
            {
                const float area = cv::contourArea(contour);
                if (contour.size() < 8 || area < 8) continue;
                const cv::Rect bounds = cv::boundingRect(contour);
                if (bounds.x <= 0 || bounds.y <= 0 || bounds.br().x >= gray.cols || bounds.br().y >= gray.rows) continue;
                std::vector<cv::Point2f> boundary;
                const size_t stride = std::max(size_t(1), (contour.size()+95)/96);
                for (size_t j = 0; j < contour.size(); j += stride) boundary.emplace_back(contour[j]);
                const cv::RotatedRect ellipse = cv::fitEllipse(boundary);
                if (!valid_ellipse(ellipse) || major(ellipse) > expected*3.5f) continue;
                const float ellipse_area = float(CV_PI)*ellipse.size.area()*0.25f;
                if (area < ellipse_area*0.65f || area > ellipse_area*1.2f) continue;
                const float distance = cv::norm(ellipse.center-seed);
                const float limit = correction_limit(ellipse, expected, radius);
                if (distance > limit) continue;
                const EllipseGeometry geometry(ellipse);
                float squared = 0;
                for (const auto &p : boundary) squared += std::pow(geometry.residual(p), 2);
                const float rms = std::sqrt(squared/boundary.size());
                if (rms > std::max(1.2f, major(ellipse)*0.04f)) continue;
                const float score = distance + rms*3 + 0.5f*std::abs(std::log(major(ellipse)/expected));
                proposals.push_back({ellipse, score});
            }
        }
        std::sort(proposals.begin(), proposals.end(), [](const Candidate &a, const Candidate &b) { return a.score < b.score; });
        std::vector<Candidate> distinct;
        for (const auto &proposal : proposals)
        {
            bool duplicate = false;
            for (const auto &old : distinct)
            {
                const float distance = cv::norm(proposal.ellipse.center-old.ellipse.center);
                if (distance < 0.1f && std::abs(major(proposal.ellipse)-major(old.ellipse)) < 0.3f &&
                    std::abs(minor(proposal.ellipse)-minor(old.ellipse)) < 0.3f) duplicate = true;
            }
            // A reflective/curved sticker can have several misleading inner
            // contours with similar centers. Preserve different axis estimates
            // so its complete outer boundary remains available for consensus.
            if (!duplicate) distinct.push_back(proposal);
            if (distinct.size() >= 7) break;
        }
        return distinct;
    }

    std::vector<Candidate> arc_candidates(const cv::Mat &gray, const cv::Point2f &seed,
                                          float expected, float radius)
    {
        // A glare patch or image boundary can cut the silhouette with a straight
        // chord. Fit its remaining arcs instead of requiring a closed ellipse.
        // This bounded fallback supplies initial shapes only; every proposal
        // must still pass the normal-gradient checks in the current image.
        cv::Mat mask;
        const int otsu = cvRound(cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU));
        std::vector<Candidate> proposals;
        std::vector<cv::Rect> seen;
        cv::RNG random(0x5eed1234);
        int attempted = 0;
        for (int threshold : {otsu, std::max(20, otsu-25), std::min(235, otsu+25)})
        {
            cv::threshold(gray, mask, threshold, 255, cv::THRESH_BINARY_INV);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            std::sort(contours.begin(), contours.end(), [&](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
                const auto ra = cv::boundingRect(a), rb = cv::boundingRect(b);
                return cv::norm(cv::Point2f(ra.x+ra.width*0.5f, ra.y+ra.height*0.5f)-seed) <
                       cv::norm(cv::Point2f(rb.x+rb.width*0.5f, rb.y+rb.height*0.5f)-seed);
            });
            for (const auto &contour : contours)
            {
                if (contour.size() < 18 || cv::contourArea(contour) < 16) continue;
                const cv::Rect bounds = cv::boundingRect(contour);
                if (bounds.width < 4 || bounds.height < 4 ||
                    std::min(bounds.width, bounds.height) > expected*3.5f) continue;
                bool duplicate = false;
                for (const auto &old : seen)
                    if (cv::norm(old.tl()-bounds.tl()) < 3 && cv::norm(old.br()-bounds.br()) < 3) duplicate = true;
                if (duplicate) continue;
                seen.push_back(bounds);
                std::vector<cv::Point2f> arc;
                const size_t stride = std::max(size_t(1), contour.size()/128);
                for (size_t j = 0; j < contour.size(); j += stride)
                {
                    const auto &p = contour[j];
                    if (p.x > 1 && p.y > 1 && p.x < gray.cols-2 && p.y < gray.rows-2) arc.emplace_back(p);
                }
                if (arc.size() < 18) continue;
                std::vector<cv::Point2f> best_inliers;
                float best_score = -std::numeric_limits<float>::infinity();
                for (int attempt = 0; attempt < 64; ++attempt)
                {
                    std::vector<cv::Point2f> sample;
                    std::array<int, 6> ids;
                    for (int j = 0; j < 6; ++j)
                    {
                        int id;
                        do { id = random.uniform(0, int(arc.size())); }
                        while (std::find(ids.begin(), ids.begin()+j, id) != ids.begin()+j);
                        ids[j] = id; sample.push_back(arc[id]);
                    }
                    cv::RotatedRect ellipse;
                    if (!fit(sample, ellipse) || major(ellipse) > expected*3.5f ||
                        cv::norm(ellipse.center-seed) > correction_limit(ellipse, expected, radius)) continue;
                    const EllipseGeometry geometry(ellipse);
                    std::vector<cv::Point2f> inliers;
                    float error = 0;
                    for (const auto &p : arc)
                    {
                        const float residual = std::abs(geometry.residual(p));
                        if (residual < 1.25f) { inliers.push_back(p); error += residual; }
                    }
                    if (inliers.size() < arc.size()*0.55f) continue;
                    const float score = inliers.size() - error*0.2f;
                    if (score > best_score) { best_score = score; best_inliers = std::move(inliers); }
                }
                cv::RotatedRect ellipse;
                if (fit(best_inliers, ellipse) && major(ellipse) <= expected*3.5f &&
                    cv::norm(ellipse.center-seed) <= correction_limit(ellipse, expected, radius))
                    proposals.push_back({ellipse, float(cv::norm(ellipse.center-seed))/radius});
                if (++attempted >= 3) return proposals;
            }
        }
        return proposals;
    }
}

void KeypointRefiner::reset()
{
    measurements_.clear();
    stats_ = {};
}

void KeypointRefiner::refine(const cv::Mat &image, std::vector<keypoint> &points,
                            const cv::Rect2f &box, uint64_t /*timestamp_ms*/)
{
    stats_ = {};
    if (image.empty() || image.type() != CV_8UC3 || points.empty() ||
        !std::isfinite(box.x) || !std::isfinite(box.y) || !std::isfinite(box.width) ||
        !std::isfinite(box.height) || box.width <= 0 || box.height <= 0)
    {
        reset();
        measurements_.resize(points.size());
        return;
    }
    measurements_.assign(points.size(), {});
    std::vector<cv::Point2f> seeds;
    std::vector<unsigned char> valid;
    for (auto &point : points)
    {
        // Non-finite high-confidence coordinates must not leak into solvePnP.
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.conf))
            point = keypoint();
        seeds.emplace_back(point.x, point.y);
        valid.push_back(std::isfinite(point.conf) && point.conf >= kMinConfidence && inside(seeds.back(), image.size()));
    }
    const float expected = bounded(std::max(box.width, box.height)*0.045f, 8.0f, 180.0f);
    std::vector<unsigned char> ambiguous(points.size(), 0);
    for (size_t i = 0; i < points.size(); ++i)
    {
        if (!valid[i]) continue;
        for (float expansion : {1.6f, 2.8f})
        {
            const float radius = bounded(expected*expansion, 18.0f, 260.0f);
            const cv::Rect roi = cv::Rect(cvFloor(seeds[i].x-radius), cvFloor(seeds[i].y-radius),
                cvCeil(radius*2+1), cvCeil(radius*2+1)) & cv::Rect(0, 0, image.cols, image.rows);
            if (roi.width < 10 || roi.height < 10) continue;
            const cv::Point2f origin(roi.x, roi.y), local_seed = seeds[i]-origin;
            cv::Mat gray, smooth, contour_gray;
            cv::cvtColor(image(roi), gray, cv::COLOR_BGR2GRAY);
            gray.convertTo(smooth, CV_32F);
            cv::GaussianBlur(smooth, smooth, cv::Size(5, 5), 0.8);
            auto acceptable = [&](KeypointRefinementMeasurement &m) {
                const cv::Point2f center = m.ellipse.center+origin;
                return inside(center, image.size()) &&
                    cv::norm(center-seeds[i]) <= correction_limit(m.ellipse, expected, radius) &&
                    belongs_to_seed(center, i, seeds, valid) && dark_color(image(roi), m.ellipse);
            };
            cv::GaussianBlur(gray, contour_gray, cv::Size(3, 3), 0.7);
            std::vector<KeypointRefinementMeasurement> observations;
            auto measure_proposals = [&](const std::vector<Candidate> &proposals)
            {
                for (const auto &proposal : proposals)
                {
                    if (!belongs_to_seed(proposal.ellipse.center+origin, i, seeds, valid)) continue;
                    bool already_measured = false;
                    for (const auto &old : observations)
                        if (stable(old) && cv::norm(old.ellipse.center-proposal.ellipse.center) < 1) already_measured = true;
                    if (already_measured) continue;
                    KeypointRefinementMeasurement m;
                    if (refine_edges(smooth, proposal.ellipse, m) && acceptable(m))
                    {
                        observations.push_back(m);
                        if (consensus(observations.back(), observations))
                        {
                            const bool competitor = std::any_of(proposals.begin(), proposals.end(), [&](const Candidate &other) {
                                return cv::norm(other.ellipse.center-m.ellipse.center) > std::max(3.0f, major(m.ellipse)*0.25f) &&
                                    other.score <= proposal.score+5 && belongs_to_seed(other.ellipse.center+origin, i, seeds, valid);
                            });
                            if (!competitor) break;
                        }
                    }
                }
            };
            // YOLO already localizes a small bimodal region. An Otsu proposal with
            // strong measured support avoids exhaustive thresholds on easy markers.
            measure_proposals(candidates(contour_gray, local_seed, expected, radius, true));
            const bool strong = std::any_of(observations.begin(), observations.end(), [](const KeypointRefinementMeasurement &m) {
                return stable(m) && m.coverage >= 0.95f && m.inlier_fraction >= 0.85f;
            });
            if (!strong) measure_proposals(candidates(contour_gray, local_seed, expected, radius, false));
            const bool full_consensus = std::any_of(observations.begin(), observations.end(),
                [&](const KeypointRefinementMeasurement &m) { return consensus(m, observations); });
            if (!full_consensus && std::none_of(observations.begin(), observations.end(), stable))
                for (const auto &proposal : arc_candidates(contour_gray, local_seed, expected, radius))
                {
                    if (!belongs_to_seed(proposal.ellipse.center+origin, i, seeds, valid)) continue;
                    KeypointRefinementMeasurement m;
                    if (refine_edges(smooth, proposal.ellipse, m) && acceptable(m)) observations.push_back(m);
                }
            float best_score = std::numeric_limits<float>::infinity();
            for (auto &m : observations)
            {
                // A repeatable complete silhouette takes precedence over a low-
                // residual fit to only part of a curved-surface sticker.
                const bool agreed = consensus(m, observations);
                if (full_consensus ? !agreed : !stable(m)) continue;
                m.valid = true;
                const float score = float(cv::norm(m.ellipse.center-local_seed))/correction_limit(m.ellipse, expected, radius) +
                    m.edge_rms_px*0.2f + (1-m.inlier_fraction)*0.3f;
                if (score < best_score) { best_score = score; measurements_[i] = m; }
            }
            auto &selected = measurements_[i];
            if (!selected.valid) continue;
            for (const auto &other : observations)
            {
                if (!other.valid || cv::norm(other.ellipse.center-selected.ellipse.center) < 2) continue;
                const float score = float(cv::norm(other.ellipse.center-local_seed))/correction_limit(other.ellipse, expected, radius) +
                    other.edge_rms_px*0.2f + (1-other.inlier_fraction)*0.3f;
                if (score < best_score+0.12f) { selected.valid = false; ambiguous[i] = 1; break; }
            }
            selected.ellipse.center += origin;
            break; // A measured or ambiguous candidate must not trigger a wider search.
        }
    }
    // One physical ellipse cannot be assigned to two keypoint identities.
    std::vector<unsigned char> duplicate(points.size(), 0);
    for (size_t i = 0; i < points.size(); ++i)
        for (size_t j = i+1; j < points.size(); ++j)
            if (measurements_[i].valid && measurements_[j].valid &&
                cv::norm(measurements_[i].ellipse.center-measurements_[j].ellipse.center) <
                    std::max(1.5f, std::min(minor(measurements_[i].ellipse), minor(measurements_[j].ellipse))*0.25f))
                duplicate[i] = duplicate[j] = 1;
    for (size_t i = 0; i < points.size(); ++i)
    {
        stats_.ambiguous_matches += ambiguous[i];
        if (duplicate[i]) { measurements_[i].valid = false; ++stats_.ambiguous_matches; }
        if (measurements_[i].valid)
        {
            points[i].x = measurements_[i].ellipse.center.x;
            points[i].y = measurements_[i].ellipse.center.y;
            ++stats_.color_matches;
        }
        else if (valid[i]) ++stats_.unrefined;
    }
}
}
}

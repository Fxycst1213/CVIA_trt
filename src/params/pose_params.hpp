#ifndef POSE_PARAMS_hpp
#define POSE_PARAMS_hpp

#include <cstring>
#include <vector>
using namespace std;

// 将 bbox 的定义从 pose.hpp 移到这里
namespace model
{
    namespace pose
    {
        const int NUM_KEYPOINTS = 7;

        struct keypoint
        {
            float x, y, conf;
            keypoint(float x = 0, float y = 0, float conf = 0) : x(x), y(y), conf(conf) {}
        };
        struct bbox
        {
            float x0, x1, y0, y1;
            float confidence;
            bool flg_remove;
            int label;
            vector<keypoint> keypoints;
            bbox() = default;
            bbox(float x0, float y0, float x1, float y1, float conf, int label) : x0(x0), y0(y0), x1(x1), y1(y1),
                                                                                  confidence(conf), flg_remove(false),
                                                                                  label(label)
            {
                keypoints.reserve(NUM_KEYPOINTS);
            };
        };

    } // namespace pose
} // namespace model

#endif // STRUCTURES_HPP
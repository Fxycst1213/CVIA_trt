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
        struct ModelPoint3D
        {
            double x;
            double y;
            double z;
        };

        // 默认关键点：顺序必须与训练标签/模型输出的 P0、P1……严格一致。
        // 网页可以修改坐标和点数；运行时以配置数组长度与模型输出共同校验。
        // 坐标原点为刚体几何中心，坐标单位必须与最终位姿需要的单位一致。
        constexpr ModelPoint3D MODEL_KEYPOINTS_3D[] = {
            {179.617252624, -40.337850737, 17.513833145},
            {178.579489770, -31.449488527, 32.085969252},
            {203.784911818, -26.404937192, 15.758290772},
            {178.104641601, -14.015844432, 45.035654805},
            {190.624677177, -13.346597405, 38.884062510},
            {178.614863843,   2.826463025, 47.693300195},
            {179.749162763,  29.809748516, 39.236784259},
            {197.050679032,  28.559032830, 28.822779855},
            {197.177402103,  35.495516327, 19.641749261},
            {178.457384656,  41.633121462, 26.852492195},
        };

        constexpr int DEFAULT_NUM_KEYPOINTS =
            static_cast<int>(sizeof(MODEL_KEYPOINTS_3D) / sizeof(MODEL_KEYPOINTS_3D[0]));
        constexpr int POSE_CLASS_COUNT = 1;
        static_assert(DEFAULT_NUM_KEYPOINTS >= 4, "solvePnP requires at least four configured keypoints");

        using ModelKeypoints3D = std::vector<double>;

        inline ModelKeypoints3D default_model_keypoints_3d()
        {
            ModelKeypoints3D points;
            points.reserve(DEFAULT_NUM_KEYPOINTS * 3);
            for (int index = 0; index < DEFAULT_NUM_KEYPOINTS; ++index)
            {
                points.push_back(MODEL_KEYPOINTS_3D[index].x);
                points.push_back(MODEL_KEYPOINTS_3D[index].y);
                points.push_back(MODEL_KEYPOINTS_3D[index].z);
            }
            return points;
        }

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
            };
        };

    } // namespace pose
} // namespace model

#endif // STRUCTURES_HPP

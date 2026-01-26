#ifndef TRAJECTORY_KF_H
#define TRAJECTORY_KF_H

#include <opencv2/opencv.hpp>
#include <iostream>

class TrajectoryKF
{
public:
    TrajectoryKF();
    ~TrajectoryKF();

    void init(float x, float y, float z);

    cv::Point3f predict(double dt);
    cv::Point3f update(float x, float y, float z);

    bool isInitialized() const
    {
        return initialized;
    }

private:
    cv::KalmanFilter KF;
    cv::Mat measurement;
    bool initialized = false;

    int consecutive_reject_count = 0;
    const int MAX_REJECT_COUNT = 3;

    // --- 自适应参数配置 ---

    // 1. 基础过程噪声 (对应平稳直线运动)
    // 稍微调大一点点，防止过拟合直线
    const float BASE_Q_POS = 0.5f;
    const float BASE_Q_VEL = 0.8f;

    // 2. 机动判定阈值 (Maneuver Threshold)
    // 预测值和观测值相差超过 1.5cm，认为物体在急转弯/变速
    // const float MANEUVER_THRESHOLD = 18.00f;

    // 3. 离谱阈值 (Outlier Threshold)
    // 预测值和观测值相差超过 50cm，认为绝对是传感器飞了
    const float IMPOSSIBLE_THRESHOLD = 65.0f;
};

#endif // TRAJECTORY_KF_H
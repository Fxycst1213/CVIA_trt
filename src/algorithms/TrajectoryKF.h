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
    void reset();

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

    // --- 自适应参数配置 ---

    // 1. 基础过程噪声 (对应平稳直线运动)
    // 稍微调大一点点，防止过拟合直线
    const float BASE_Q_POS = 0.5f;
    const float BASE_Q_VEL = 0.8f;

    // 离群点由 Pose 的重投影质量与 SE(3) 连续性门控统一处理。KF 只负责平滑
    // 已经通过门控的测量，避免“连续拒绝若干帧后反而接收异常值”的状态跳变。
};

#endif // TRAJECTORY_KF_H

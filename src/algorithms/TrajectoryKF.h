#ifndef TRAJECTORY_KF_H
#define TRAJECTORY_KF_H

#include <opencv2/opencv.hpp>
#include <iostream>

class TrajectoryKF
{
public:
    TrajectoryKF();
    ~TrajectoryKF();

    /**
     * @brief 初始化卡尔曼滤波器
     * @param x, y, z 初始位置
     */
    void init(float x, float y, float z);

    /**
     * @brief 预测下一时刻的位置
     * @param dt 距离上一次更新的时间间隔 (秒)
     * @return 预测的 x, y, z 坐标
     */
    cv::Point3f predict(double dt);

    /**
     * @brief 使用观测值更新滤波器
     * @param x, y, z 观测到的位置 (来自 PnP 或 ZED)
     * @return 修正后的最优估计位置 x, y, z
     */
    cv::Point3f update(float x, float y, float z);
    cv::Point3f Auto_update(float x, float y, float z);

    bool isInitialized() const
    {
        return initialized;
    }

private:
    cv::KalmanFilter KF;
    cv::Mat measurement;
    bool initialized = false;

    // 调试用：打印矩阵
    void printState();
    float base_Q_pos = 1e-5;
};

#endif // TRAJECTORY_KF_H
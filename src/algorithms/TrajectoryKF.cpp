#include "TrajectoryKF.h"

TrajectoryKF::TrajectoryKF()
{
    KF.init(6, 3, 0);
    measurement = cv::Mat::zeros(3, 1, CV_32F);
    initialized = false;
}

TrajectoryKF::~TrajectoryKF() {}

void TrajectoryKF::init(float x, float y, float z)
{
    // 1. 转移矩阵 F
    KF.transitionMatrix = cv::Mat::eye(6, 6, CV_32F);

    // 2. 观测矩阵 H
    KF.measurementMatrix = cv::Mat::eye(3, 6, CV_32F);

    // 3. 过程噪声 Q (初始化为基础值)
    KF.processNoiseCov = cv::Mat::eye(6, 6, CV_32F);
    KF.processNoiseCov.at<float>(0, 0) = BASE_Q_POS;
    KF.processNoiseCov.at<float>(1, 1) = BASE_Q_POS;
    KF.processNoiseCov.at<float>(2, 2) = BASE_Q_POS;
    KF.processNoiseCov.at<float>(3, 3) = BASE_Q_VEL;
    KF.processNoiseCov.at<float>(4, 4) = BASE_Q_VEL;
    KF.processNoiseCov.at<float>(5, 5) = BASE_Q_VEL;

    // 4. 测量噪声 R (保持你之前的设置，相信传感器)
    KF.measurementNoiseCov = cv::Mat::eye(3, 3, CV_32F);
    float R_x = 1.0f;
    float R_y = 1.0f;
    float R_z = 1.0f;
    KF.measurementNoiseCov.at<float>(0, 0) = R_x;
    KF.measurementNoiseCov.at<float>(1, 1) = R_y;
    KF.measurementNoiseCov.at<float>(2, 2) = R_z;

    // 5. P 矩阵
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));

    // 6. 初始状态
    KF.statePost.at<float>(0) = x;
    KF.statePost.at<float>(1) = y;
    KF.statePost.at<float>(2) = z;
    KF.statePost.at<float>(3) = 0;
    KF.statePost.at<float>(4) = 0;
    KF.statePost.at<float>(5) = 0;

    initialized = true;
    consecutive_reject_count = 0;
}

cv::Point3f TrajectoryKF::predict(double dt)
{
    if (!initialized)
        return cv::Point3f(0, 0, 0);

    KF.transitionMatrix.at<float>(0, 3) = (float)dt;
    KF.transitionMatrix.at<float>(1, 4) = (float)dt;
    KF.transitionMatrix.at<float>(2, 5) = (float)dt;

    cv::Mat prediction = KF.predict();
    return cv::Point3f(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2));
}

// 【核心修改】自适应更新函数
cv::Point3f TrajectoryKF::update(float x, float y, float z)
{
    if (!initialized)
    {
        init(x, y, z);
        return cv::Point3f(x, y, z);
    }

    measurement.at<float>(0) = x;
    measurement.at<float>(1) = y;
    measurement.at<float>(2) = z;

    // 1. 计算残差 (Innovation)
    // 获取当前时刻的预测值 (StatePre)
    float pred_x = KF.statePre.at<float>(0);
    float pred_y = KF.statePre.at<float>(1);
    float pred_z = KF.statePre.at<float>(2);

    float dist = std::sqrt(std::pow(x - pred_x, 2) +
                           std::pow(y - pred_y, 2) +
                           std::pow(z - pred_z, 2));

    // 2. 状态机判断

    // --- 情况 A: 离谱噪声 (Impossible) ---
    // 比如一帧跳变 50cm，物理上不可能，直接拒绝
    if (dist > IMPOSSIBLE_THRESHOLD)
    {
        std::cout << "[KF] Reject Outlier___: " << dist << std::endl;
        consecutive_reject_count++;
        if (consecutive_reject_count < MAX_REJECT_COUNT)
        {
            std::cout << "[KF] Reject Outlier: " << dist << std::endl;
            return cv::Point3f(pred_x, pred_y, pred_z);
        }
    }

    // --- 情况 B: 机动模式 (Maneuver) ---
    // 误差在 1.5cm ~ 50cm 之间，说明物体急转弯/反向了
    // 此时预测值还在往前冲，但观测值已经回来了
    // else if (dist > MANEUVER_THRESHOLD)
    // {
    //     std::cout << "[KF] MANEUVER_THRESHOLD: " << dist << std::endl;
    //     float dynamic_Q = 1e-1;
    //     cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(dynamic_Q));
    //     consecutive_reject_count = 0;
    // }

    // --- 情况 C: 稳态模式 (Normal) ---
    // 误差 < 1.5cm，说明运动平滑
    else
    {
        // [策略]: 恢复基础过程噪声 Q
        // 保证直线运动的平滑性
        // 简单恢复对角线即可 (或者用 setIdentity 恢复成 BASE_Q_POS 也行，这里为了严谨分别赋值)
        KF.processNoiseCov.at<float>(0, 0) = BASE_Q_POS;
        KF.processNoiseCov.at<float>(1, 1) = BASE_Q_POS;
        KF.processNoiseCov.at<float>(2, 2) = BASE_Q_POS;
        KF.processNoiseCov.at<float>(3, 3) = BASE_Q_VEL;
        KF.processNoiseCov.at<float>(4, 4) = BASE_Q_VEL;
        KF.processNoiseCov.at<float>(5, 5) = BASE_Q_VEL;

        consecutive_reject_count = 0;
    }

    // 3. 执行更新 (Correct)
    // 无论在 机动模式 还是 稳态模式，都会执行这一步
    cv::Mat corrected = KF.correct(measurement);

    return cv::Point3f(corrected.at<float>(0), corrected.at<float>(1), corrected.at<float>(2));
}
#include "TrajectoryKF.h"

TrajectoryKF::TrajectoryKF()
{
    // 状态向量维度: 6 (x, y, z, vx, vy, vz)
    // 观测向量维度: 3 (x, y, z)
    KF.init(6, 3, 0);
    measurement = cv::Mat::zeros(3, 1, CV_32F);
    initialized = false;
}

TrajectoryKF::~TrajectoryKF() {}

void TrajectoryKF::init(float x, float y, float z)
{
    // 1. 初始化转移矩阵 F (6x6)
    // 状态: [x, y, z, vx, vy, vz]
    // 这里的 dt 在 predict() 函数中会动态更新，这里先设为默认单位矩阵
    KF.transitionMatrix = cv::Mat::eye(6, 6, CV_32F);

    // 2. 初始化观测矩阵 H (3x6)
    // 我们观测的是位置 [x, y, z]，对应状态向量的前三个分量
    KF.measurementMatrix = cv::Mat::eye(3, 6, CV_32F);

    // =========================================================
    // [重点修改 1]: 过程噪声 Q (Process Noise)
    // =========================================================
    // Q 越小 -> 认为物体运动越符合匀速模型（越平滑），对突变反应越慢
    // Q 越大 -> 认为物体运动越剧烈（可随时变速），对突变反应越快

    KF.processNoiseCov = cv::Mat::eye(6, 6, CV_32F);

    // float Q_pos = 1e-4; // 位置的过程噪声
    // float Q_vel = 1e-4; // 速度的过程噪声

    // 如果想要 Y 轴特别平滑（假设Y轴不应该有剧烈加速），可以单独把 Y 轴的 Q 调得更小，例如 1e-5
    // float Q_y_pos = 1e-5;
    float Q_x_pos = 1e-4, Q_y_pos = 1e-4, Q_z_pos = 1e-4;
    // 设置位置噪声 (x, y, z)
    KF.processNoiseCov.at<float>(0, 0) = Q_x_pos;
    KF.processNoiseCov.at<float>(1, 1) = Q_y_pos; // 这里使用通用的 Q_pos，如果需要更粘滞可改为 Q_y_pos
    KF.processNoiseCov.at<float>(2, 2) = Q_z_pos;

    float Q_x_vel = 1e-4, Q_y_vel = 1e-4, Q_z_vel = 1e-4;
    // 设置速度噪声 (vx, vy, vz)
    KF.processNoiseCov.at<float>(3, 3) = Q_x_vel;
    KF.processNoiseCov.at<float>(4, 4) = Q_y_vel;
    KF.processNoiseCov.at<float>(5, 5) = Q_z_vel;

    // =========================================================
    // [重点修改 2]: 测量噪声 R (Measurement Noise)
    // =========================================================
    // R 越大 -> 越不相信传感器观测值（认为噪声大），更依赖预测值
    // R 越小 -> 越相信传感器观测值

    KF.measurementNoiseCov = cv::Mat::eye(3, 3, CV_32F);

    // X 和 Z 轴保持默认（相对信任观测值）
    float R_default = 1e-2;

    float R_x_axis = 0.00005, R_y_axis = 5e-3, R_z_axis = 5e-3;

    KF.measurementNoiseCov.at<float>(0, 0) = R_x_axis; // X
    KF.measurementNoiseCov.at<float>(1, 1) = R_y_axis; // Y (噪声大，权重低)
    KF.measurementNoiseCov.at<float>(2, 2) = R_z_axis; // Z

    // =========================================================

    // 5. 初始化后验错误协方差矩阵 P (初始的不确定性)
    // 设为 1 表示初始状态也有一定的不确定性，随迭代会自动收敛
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));

    // 6. 设置初始状态值
    KF.statePost.at<float>(0) = x;
    KF.statePost.at<float>(1) = y;
    KF.statePost.at<float>(2) = z;
    KF.statePost.at<float>(3) = 0; // 初始速度 x 设为 0
    KF.statePost.at<float>(4) = 0; // 初始速度 y 设为 0
    KF.statePost.at<float>(5) = 0; // 初始速度 z 设为 0

    initialized = true;
}

cv::Point3f TrajectoryKF::predict(double dt)
{
    if (!initialized)
        return cv::Point3f(0, 0, 0);

    // 动态更新转移矩阵中的 dt
    // x' = x + vx * dt
    KF.transitionMatrix.at<float>(0, 3) = (float)dt;
    KF.transitionMatrix.at<float>(1, 4) = (float)dt;
    KF.transitionMatrix.at<float>(2, 5) = (float)dt;

    cv::Mat prediction = KF.predict();

    return cv::Point3f(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2));
}

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

    cv::Mat corrected = KF.correct(measurement);

    return cv::Point3f(corrected.at<float>(0), corrected.at<float>(1), corrected.at<float>(2));
}

// cv::Point3f TrajectoryKF::update(float x, float y, float z)
// {
//     if (!initialized)
//     {
//         init(x, y, z);
//         return cv::Point3f(x, y, z);
//     }

//     // 1. 计算残差 (Residual) = 观测值 - 预测值
//     // 注意：这里需要先拿预测态来比，但 KF.predict() 已经更新了 statePre
//     // 我们手动算一下当前的残差
//     cv::Mat measurement_matrix = cv::Mat::zeros(3, 1, CV_32F);
//     measurement_matrix.at<float>(0) = x;
//     measurement_matrix.at<float>(1) = y;
//     measurement_matrix.at<float>(2) = z;

//     // 获取当前的预测位置 (H * x_minus)
//     // 因为 H 是单位阵，所以直接取 statePre 的前三维
//     float pred_x = KF.statePre.at<float>(0);
//     float pred_y = KF.statePre.at<float>(1);
//     float pred_z = KF.statePre.at<float>(2);

//     float res_x = x - pred_x;
//     float res_y = y - pred_y;
//     float res_z = z - pred_z;

//     // 计算残差的平方模 (或者是马氏距离)
//     float residual_norm_sq = res_x * res_x + res_y * res_y + res_z * res_z;

//     // 2. 自适应逻辑 (AKF 核心)
//     // 设定一个阈值，比如物体最大可能的突变距离是 1cm
//     float threshold_mm = 7.0f;
//     float motion_threshold_sq = threshold_mm * threshold_mm; // 结果是 225.0

//     if (residual_norm_sq > motion_threshold_sq)
//     {
//         // [大动静模式]：残差巨大，说明物体机动了，或者预测偏了
//         // 临时把 Q 撑大，让 KF 赶紧相信观测值，跟过去！
//         float dynamic_Q = base_Q_pos * 1000.0f;
//         cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(dynamic_Q));
//     }
//     else
//     {
//         // [稳态模式]：残差很小，说明物体在匀速运动
//         // 使用非常小的 Q，享受极致丝滑
//         cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(base_Q_pos));
//     }

//     // 3. 标准更新
//     cv::Mat corrected = KF.correct(measurement_matrix);
//     return cv::Point3f(corrected.at<float>(0), corrected.at<float>(1), corrected.at<float>(2));
// }
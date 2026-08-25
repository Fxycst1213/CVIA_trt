#include "TrajectoryKF.h"
#include <algorithm>

TrajectoryKF::TrajectoryKF()
{
    KF.init(6, 3, 0);
    measurement = cv::Mat::zeros(3, 1, CV_32F);
    initialized = false;
}

TrajectoryKF::~TrajectoryKF() {}

void TrajectoryKF::reset()
{
    initialized = false;
    KF.statePre.setTo(0);
    KF.statePost.setTo(0);
    measurement.setTo(0);
}

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
}

cv::Point3f TrajectoryKF::predict(double dt)
{
    if (!initialized)
        return cv::Point3f(0, 0, 0);

    // 防止时钟异常或长时间失配让恒速模型一次外推到不可控距离。长间隔重新
    // 捕获时 Pose 会先 reset，本处只覆盖正常连续帧。
    dt = std::max(0.001, std::min(dt, 2.0));

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

    KF.processNoiseCov.at<float>(0, 0) = BASE_Q_POS;
    KF.processNoiseCov.at<float>(1, 1) = BASE_Q_POS;
    KF.processNoiseCov.at<float>(2, 2) = BASE_Q_POS;
    KF.processNoiseCov.at<float>(3, 3) = BASE_Q_VEL;
    KF.processNoiseCov.at<float>(4, 4) = BASE_Q_VEL;
    KF.processNoiseCov.at<float>(5, 5) = BASE_Q_VEL;

    cv::Mat corrected = KF.correct(measurement);

    return cv::Point3f(corrected.at<float>(0), corrected.at<float>(1), corrected.at<float>(2));
}

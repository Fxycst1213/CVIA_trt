#ifndef __POSE_HPP__
#define __POSE_HPP__

#include <memory>
#include <vector>
#include <string>
#include <array>
#include "NvInfer.h"
#include "logger.hpp"
#include "model.hpp"
#include "../algorithms/TrajectoryKF.h"
// 周期/历史回溯预测已停用；TrajectoryKF 卡尔曼滤波保留。
// #include "../algorithms/PeriodEstimator.h"
#include "../ZEDX/ZEDX.h"
#include "../IRcamera/IRcamera.h"
#include "../params/pose_params.hpp"
// #include "../algorithms/FrameLookbackEstimator.h"

// class LSTMPredictor;

namespace model
{

    namespace pose
    {

        using namespace std;
        enum model
        {
            YOLOV8,
            YOLOV11
        };

        class Pose : public Model
        {

        public:
            Pose(std::string onnx_path, logger::Level level, Params params);
            ~Pose()
            {
                if (m_cudaGraphExec)
                {
                    cudaGraphExecDestroy(m_cudaGraphExec);
                    m_cudaGraphExec = nullptr;
                }
                if (m_cudaGraph)
                {
                    cudaGraphDestroy(m_cudaGraph);
                    m_cudaGraph = nullptr;
                }
            }

        public:
            virtual void setup(void const *data, std::size_t size) override;
            virtual void reset_task() override;
            virtual bool preprocess_cpu(const cv::Mat &img) override;
            virtual bool preprocess_gpu(const cv::Mat &img) override;
            virtual bool postprocess_cpu(const uint64_t &timestamp) override;
            virtual bool postprocess_gpu(const uint64_t &timestamp) override;
            void run_pnp_multi_stage(const uint64_t &timestamp);
            // std::shared_ptr<FrameLookbackEstimator> m_lookback_estimator;
            void run_filter_and_estimation(const uint64_t &timestamp, uint64_t m_frame_counter);
            void run_pnp_single_stage();
            // void run_lstm_predictin();
            void show(string path);
            void refine_keypoints(std::vector<keypoint> &kpt);
            std::vector<float> m_result;
            std::vector<float> uart_result;
            // 刚体坐标系原点 (0,0,0) 在当前红外图像上的重投影。
            cv::Point2d m_reprojected_origin;
            bool m_reprojected_origin_valid = false;
            std::vector<cv::Point2d> m_reprojected_points;
            float linear_map(float val, float in_min, float in_max, float out_min, float out_max);
            std::vector<bbox> m_bboxes;
            bool is_current_frame_good = false;
            bool set_calibration(const std::array<double, 9> &camera_matrix,
                                 const std::array<double, 5> &distortion,
                                 const std::array<double, 16> &extrinsic,
                                 const ModelKeypoints3D &model_keypoints_3d,
                                 bool reset_tracking_state = false);

        private:
            // [修改] 新增标志位，记录当前颜色检测模式 (false=蓝色/近距离, true=红色/远距离)
            bool m_use_red_mode = false; 

            void reset_tracking_after_calibration_change();

            int m_inputSize;
            int m_imgArea;
            int m_outputSize;
            int m_outputBoxes = 0;
            int m_outputFeatures = 0;
            int m_outputClasses = 0;
            int m_numKeypoints = 0;
            // Ultralytics exports may use either [1, boxes, features] or
            // [1, features, boxes]. Keep the engine layout explicit instead
            // of assuming one exporter-specific order in postprocess.
            bool m_outputFeaturesFirst = false;
            cv::Mat _K;
            cv::Mat _diff;
            cv::Mat _p3d;

            cv::Mat _handTrans;
            cv::Mat _wxj2cam;
            // T_M_C: camera coordinates -> NOKOV mocap coordinates.
            cv::Mat mocap_from_camera_inv;
            cv::Mat mocap_from_camera;

            cv::Mat R1, T1;
            cv::Mat _R1_prev;
            cv::Mat _T1_prev;

            cv::Mat R_mat;
            // 最近一次通过几何质量与连续性门控的原始相机系 PnP。
            cv::Mat _accepted_R_mat;
            cv::Mat _accepted_T1;
            uint64_t _accepted_timestamp = 0;
            // 大幅位姿变化必须形成连续候选，避免单帧错点/错误 PnP 分支直接进入输出。
            cv::Mat _pending_R_mat;
            cv::Mat _pending_T1;
            uint64_t _pending_timestamp = 0;
            int _pending_pose_count = 0;
            bool _reset_filter_on_next_measurement = false;
            cv::Vec3d _last_mocap_euler_deg = cv::Vec3d(0.0, 0.0, 0.0);
            bool _has_last_mocap_euler = false;
            TrajectoryKF m_kf;
            uint64_t _last_timestamp = 0;
            uint64_t m_frame_counter = -1;
            // std::shared_ptr<LSTMPredictor> m_lstm;
            // bool m_lstm_ready = false; // 标记 LSTM 是否已经加载成功
        };

        std::shared_ptr<Pose> make_pose(
            std::string onnx_path, logger::Level level, Params params);

    }; // namespace pose
}; // namespace model

#endif //__POSE_HPP__

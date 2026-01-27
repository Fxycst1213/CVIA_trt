#ifndef __POSE_HPP__
#define __POSE_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "logger.hpp"
#include "model.hpp"
#include "../algorithms/TrajectoryKF.h"
#include "../algorithms/PeriodEstimator.h"
#include "../ZEDX/ZEDX.h"
#include "../params/pose_params.hpp"
#include "../algorithms/FrameLookbackEstimator.h"

class LSTMPredictor;

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
            void run_pnp_multi_stage();
            std::shared_ptr<FrameLookbackEstimator> m_lookback_estimator;
            void run_filter_and_estimation(const uint64_t &timestamp, uint64_t m_frame_counter);
            void run_pnp_single_stage();
            void run_lstm_predictin();
            void show(string path);
            void refine_keypoints(std::vector<keypoint> &kpt);
            std::vector<float> m_result;
            std::vector<float> uart_result;
            float linear_map(float val, float in_min, float in_max, float out_min, float out_max);
            std::vector<bbox> m_bboxes;
            bool is_current_frame_good = false;

        private:
            // [修改] 新增标志位，记录当前颜色检测模式 (false=蓝色/近距离, true=红色/远距离)
            bool m_use_red_mode = false; 

            int m_inputSize;
            int m_imgArea;
            int m_outputSize;
            cv::Mat _K;
            cv::Mat _diff;
            cv::Mat _p3d;

            cv::Mat _handTrans;
            cv::Mat _wxj2cam;
            cv::Mat combined_inv;
            cv::Mat combined;

            cv::Mat R1, T1;
            cv::Mat _R1_prev;
            cv::Mat _T1_prev;

            int _stale_frame_count = 0;
            double _candidate_z = 0.0;
            int _candidate_count = 0;
            int _candidate_limit = 1; // 连续多少帧稳定才更新，默认2
            TrajectoryKF m_kf;
            uint64_t _last_timestamp = 0;
            uint64_t m_frame_counter = -1;
            std::shared_ptr<LSTMPredictor> m_lstm;
            bool m_lstm_ready = false; // 标记 LSTM 是否已经加载成功
        };

        std::shared_ptr<Pose> make_pose(
            std::string onnx_path, logger::Level level, Params params);

    }; // namespace pose
}; // namespace model

#endif //__POSE_HPP__
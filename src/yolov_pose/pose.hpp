#ifndef __POSE_HPP__
#define __POSE_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "logger.hpp"
#include "model.hpp"
#include "../algorithms/TrajectoryKF.h"
#include "../ZEDX/ZEDX.h"
#include "../params/pose_params.hpp"

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
            void run_pnp_single_stage();
            void show(string path);
            void refine_keypoints(std::vector<keypoint> &kpt);
            std::vector<double> m_result;

            std::vector<bbox> m_bboxes;
            bool is_current_frame_good = false;

        private:
            int m_inputSize;
            int m_imgArea;
            int m_outputSize;
            cv::Mat _K;
            cv::Mat _diff;
            cv::Mat _p3d;

            cv::Mat R1, T1;
            cv::Mat _R1_prev;
            cv::Mat _T1_prev;

            int _stale_frame_count = 0;
            double _candidate_z = 0.0;
            int _candidate_count = 0;
            int _candidate_limit = 2; // 连续多少帧稳定才更新，默认2
            TrajectoryKF m_kf;
            uint64_t _last_timestamp = 0;
        };

        std::shared_ptr<Pose> make_pose(
            std::string onnx_path, logger::Level level, Params params);

    }; // namespace pose
}; // namespace model

#endif //__POSE_HPP__

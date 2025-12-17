#ifndef __POSE_HPP__
#define __POSE_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "logger.hpp"
#include "model.hpp"

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
        const int NUM_KEYPOINTS = 7;

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
                keypoints.reserve(NUM_KEYPOINTS);
            };
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
            virtual bool preprocess_cpu(cv::Mat &img) override;
            virtual bool preprocess_gpu(cv::Mat &img) override;
            virtual bool postprocess_cpu() override;
            virtual bool postprocess_gpu() override;
            void run_pnp();
            void show();
            void refine_keypoints(std::vector<keypoint> &kpt);
            std::vector<double> m_result;

            std::vector<bbox> m_bboxes;
            bool is_current_frame_good = false;
            cv::Mat final_R, final_T;

        private:
            int m_inputSize;
            int m_imgArea;
            int m_outputSize;
            cv::Mat _K;
            cv::Mat _diff;
            cv::Mat _p3d;

            cv::Mat _R1_prev;
            cv::Mat _T1_prev;

            int _stale_frame_count = 0;
            double _candidate_z = 0.0;
            int _candidate_count = 0;
            int _candidate_limit = 2; // 连续多少帧稳定才更新，默认2
        };

        std::shared_ptr<Pose> make_pose(
            std::string onnx_path, logger::Level level, Params params);

    }; // namespace pose
}; // namespace model

#endif //__POSE_HPP__

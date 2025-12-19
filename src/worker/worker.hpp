#ifndef __WORKER_HPP__
#define __WORKER_HPP__

#include <memory>
#include <vector>
#include "model.hpp"
#include "logger.hpp"
#include "classifier.hpp"
#include "detector.hpp"
#include "pose.hpp"

namespace thread
{

    class Worker
    {
    public:
        Worker(std::string onnxPath, logger::Level level, model::Params params);
        void inference(cv::Mat &img, uint64_t &timestamp);

    public:
        std::shared_ptr<logger::Logger> m_logger;
        // std::shared_ptr<model::Params>           m_params;
        model::Params m_params;
        std::shared_ptr<model::classifier::Classifier> m_classifier;
        std::shared_ptr<model::detector::Detector> m_detector;
        std::shared_ptr<model::pose::Pose> m_pose;
    };

    std::shared_ptr<Worker> create_worker(
        std::string onnxPath, logger::Level level, model::Params params);

}; // namespace thread

#endif //__WORKER_HPP__

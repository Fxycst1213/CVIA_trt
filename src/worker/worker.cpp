#include "worker.hpp"
#include "classifier.hpp"
#include "detector.hpp"
#include "logger.hpp"
#include "memory"

using namespace std;

namespace thread
{

    Worker::Worker(string onnxPath, logger::Level level, model::Params params)
    {
        m_logger = logger::create_logger(level);
        m_params = params;
        // 这里根据task_type选择创建的trt_model的子类，今后会针对detection, segmentation扩充
        if (params.task == model::task_type::CLASSIFICATION)
        {
            m_classifier = model::classifier::make_classifier(onnxPath, level, params);
            m_classifier->init_model();
        }
        else if (params.task == model::task_type::DETECTION)
        {
            m_detector = model::detector::make_detector(onnxPath, level, params);
            m_detector->init_model();
        }
        else if (params.task == model::task_type::POSE)
        {
            m_pose = model::pose::make_pose(onnxPath, level, params); // m_pose 的生命周期由智能指针自动管理
            m_pose->init_model();
        }
    }

    void Worker::inference(cv::Mat &img, const uint64_t &timestamp)
    {
        if (m_params.task == model::task_type::CLASSIFICATION)
        {
            // m_classifier->load_image(imagePath);
            m_classifier->inference(img, timestamp);
        }

        else if (m_params.task == model::task_type::DETECTION)
        {
            // m_detector->load_image(imagePath);
            m_detector->inference(img, timestamp);
        }
        else if (m_params.task == model::task_type::POSE)
        {
            // m_detector->load_image(imagePath);
            m_pose->inference(img, timestamp);
        }
    }

    shared_ptr<Worker> create_worker(
        std::string onnxPath, logger::Level level, model::Params params)
    {
        // 使用智能指针来创建一个实例
        return make_shared<Worker>(onnxPath, level, params);
    }

}; // namespace thread

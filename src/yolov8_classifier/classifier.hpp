#ifndef __CLASSIFIER_HPP__
#define __CLASSIFIER_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "logger.hpp"
#include "model.hpp"
#include "utils.hpp"
namespace model
{

    namespace classifier
    {
        class Classifier : public Model
        {

        public:
            Classifier(std::string onnx_path, logger::Level level, Params params) : Model(onnx_path, level, params) {};

        public:
            virtual void setup(void const *data, std::size_t size) override;
            virtual void reset_task() override;
            virtual bool preprocess_cpu(cv::Mat &img) override;
            virtual bool preprocess_gpu(cv::Mat &img) override;
            virtual bool postprocess_cpu(uint64_t &timestamp) override;
            virtual bool postprocess_gpu(uint64_t &timestamp) override;

        private:
            float m_confidence;
            std::string m_label;
            int m_inputSize;
            int m_imgArea;
            int m_outputSize;
        };

        std::shared_ptr<Classifier> make_classifier(
            std::string onnx_path, logger::Level level, Params params);

    }; // namespace classifier
}; // namespace model

#endif //__CLASSIFIER_HPP__

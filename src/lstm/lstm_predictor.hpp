#ifndef __LSTM_PREDICTOR_HPP__
#define __LSTM_PREDICTOR_HPP__

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime.h>
#include <deque>
#include <vector>
#include <memory>
#include <string>
#include "logger.hpp"

// TRT 10+ 推荐使用 delete，而非 destroy()
struct TrtDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        if (obj)
            delete obj;
    }
};

class LSTMPredictor
{
public:
    struct Config
    {
        std::string onnx_path;
        int input_seq_len = 58;
        int input_dim = 3;
        int output_seq_len = 25;
        int output_dim = 3;
        int target_frame_idx = 22;
        // 节点名称（TRT10 必须匹配 ONNX 里的名字）
        std::string input_node_name = "input";
        std::string output_node_name = "output";
    };

    LSTMPredictor(const Config &config, logger::Level level);
    ~LSTMPredictor();

    bool init();
    bool update(float x, float y, float z);
    std::vector<float> get_prediction() const;

private:
    bool build_engine();
    bool load_engine();

private:
    Config m_config;
    std::shared_ptr<logger::Logger> m_logger;

    // 智能指针使用 TrtDeleter
    std::shared_ptr<nvinfer1::IRuntime> m_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;

    cudaStream_t m_stream = nullptr;

    // TRT 10 不再使用 void* bindings[] 数组
    // 而是直接管理设备指针
    void *m_input_device = nullptr;
    void *m_output_device = nullptr;

    // Host 内存
    float *m_host_input = nullptr;
    float *m_host_output = nullptr;

    int m_input_size_bytes = 0;
    int m_output_size_bytes = 0;

    std::deque<std::vector<float>> m_history_buffer;
    std::vector<float> m_last_prediction;
};

#endif
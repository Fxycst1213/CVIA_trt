#include "lstm_predictor.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

using namespace nvinfer1;
using namespace nvonnxparser;

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                               \
    do                                                                                 \
    {                                                                                  \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess)                                                        \
        {                                                                              \
            LOGE("CUDA error %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        }                                                                              \
    } while (0)
#endif

static std::string getEnginePath(const std::string &onnx_path)
{
    // 1. 提取文件名 (model_multi.onnx)
    size_t last_slash = onnx_path.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? onnx_path : onnx_path.substr(last_slash + 1);

    // 2. 去除后缀 (model_multi)
    size_t last_dot = filename.find_last_of(".");
    std::string stem = (last_dot == std::string::npos) ? filename : filename.substr(0, last_dot);

    // 3. 拼接新路径
    // ⚠️ 确保你的系统里已经创建了 models/engine/ 文件夹，否则保存会失败
    return "models/engine/" + stem + ".engine";
}

static bool fileExists(const std::string &path)
{
    std::ifstream f(path);
    return f.good();
}

LSTMPredictor::LSTMPredictor(const Config &config, logger::Level level)
    : m_config(config)
{
    m_logger = std::make_shared<logger::Logger>(level);

    m_input_size_bytes = 1 * m_config.input_seq_len * m_config.input_dim * sizeof(float);
    m_output_size_bytes = 1 * m_config.output_seq_len * m_config.output_dim * sizeof(float);

    CUDA_CHECK(cudaMallocHost(&m_host_input, m_input_size_bytes));
    CUDA_CHECK(cudaMallocHost(&m_host_output, m_output_size_bytes));

    m_last_prediction.resize(3, 0.0f);
}

LSTMPredictor::~LSTMPredictor()
{
    if (m_stream)
        cudaStreamDestroy(m_stream);
    if (m_input_device)
        cudaFree(m_input_device);
    if (m_output_device)
        cudaFree(m_output_device);
    if (m_host_input)
        cudaFreeHost(m_host_input);
    if (m_host_output)
        cudaFreeHost(m_host_output);
}

bool LSTMPredictor::init()
{
    std::string engine_path = getEnginePath(m_config.onnx_path);
    if (!fileExists(engine_path))
    {
        LOG("Engine not found, building from ONNX...");
        if (!build_engine())
            return false;
    }
    return load_engine();
}

bool LSTMPredictor::build_engine()
{
    // 这里的构建逻辑基本不变，但要注意 delete
    if (!fileExists(m_config.onnx_path))
        return false;

    auto builder = std::shared_ptr<IBuilder>(createInferBuilder(*m_logger), TrtDeleter());
    const uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::shared_ptr<INetworkDefinition>(builder->createNetworkV2(flags), TrtDeleter());
    auto config = std::shared_ptr<IBuilderConfig>(builder->createBuilderConfig(), TrtDeleter());

    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 28);

    // TRT 10 默认就是 FP32，如果没特殊需求先不开 FP16 保持稳定
    // if (builder->platformHasFastFp16()) config->setFlag(BuilderFlag::kFP16);

    auto parser = std::shared_ptr<IParser>(createParser(*network, *m_logger), TrtDeleter());
    if (!parser->parseFromFile(m_config.onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING)))
    {
        LOGE("ONNX parse failed.");
        return false;
    }

    // 动态维度支持
    auto input = network->getInput(0);
    if (input->getDimensions().d[0] == -1)
    {
        auto profile = builder->createOptimizationProfile();
        profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims3(1, m_config.input_seq_len, m_config.input_dim));
        profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims3(1, m_config.input_seq_len, m_config.input_dim));
        profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims3(1, m_config.input_seq_len, m_config.input_dim));
        config->addOptimizationProfile(profile);
    }

    auto plan = std::shared_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config), TrtDeleter());
    if (!plan)
    {
        LOGE("Failed to build serialized engine.");
        return false;
    }

    // 【修改】使用新的路径保存
    std::string engine_path = getEnginePath(m_config.onnx_path);

    // 建议加一行日志，让你知道它想存哪，方便排查文件夹是否存在的问题
    LOG("Saving engine to: %s", engine_path.c_str());

    std::ofstream out(engine_path, std::ios::binary);
    if (!out)
    {
        // 增加文件打开失败的报错，通常是因为 models/engine 文件夹不存在
        LOGE("Failed to open file for writing! Check if 'models/engine/' directory exists.");
        return false;
    }

    out.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    out.close();

    LOG("Engine built successfully.");
    return true;
}

bool LSTMPredictor::load_engine()
{
    std::string engine_path = getEnginePath(m_config.onnx_path);

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good())
    {
        LOGE("Cannot open engine file: %s", engine_path.c_str());
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    m_runtime = std::shared_ptr<IRuntime>(createInferRuntime(*m_logger), TrtDeleter());
    m_engine = std::shared_ptr<ICudaEngine>(m_runtime->deserializeCudaEngine(engine_data.data(), size), TrtDeleter());
    m_context = std::shared_ptr<IExecutionContext>(m_engine->createExecutionContext(), TrtDeleter());

    // 【关键修复】TRT 10 无法使用 getNbBindings
    // 我们直接为已知的节点名分配显存
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    CUDA_CHECK(cudaMalloc(&m_input_device, m_input_size_bytes));
    CUDA_CHECK(cudaMalloc(&m_output_device, m_output_size_bytes));

    return true;
}

bool LSTMPredictor::update(float x, float y, float z)
{
    m_history_buffer.push_back({x, y, z});
    if (m_history_buffer.size() > m_config.input_seq_len)
        m_history_buffer.pop_front();
    if (m_history_buffer.size() < m_config.input_seq_len)
        return false;

    int idx = 0;
    for (auto &v : m_history_buffer)
    {
        m_host_input[idx++] = v[0];
        m_host_input[idx++] = v[1];
        m_host_input[idx++] = v[2];
    }

    // 1. 拷贝数据到 GPU
    CUDA_CHECK(cudaMemcpyAsync(m_input_device, m_host_input, m_input_size_bytes, cudaMemcpyHostToDevice, m_stream));

    // 2. 【关键修复】TRT 10 使用 setTensorAddress 和 enqueueV3
    // 你必须确保 m_config.input_node_name 和 m_config.output_node_name 是正确的
    // 比如 "images" 或 "input"
    m_context->setTensorAddress(m_config.input_node_name.c_str(), m_input_device);
    m_context->setTensorAddress(m_config.output_node_name.c_str(), m_output_device);

    // 如果是动态维度，必须显式设置
    m_context->setInputShape(m_config.input_node_name.c_str(), Dims3(1, m_config.input_seq_len, m_config.input_dim));

    // 3. 执行推理 (enqueueV3)
    bool status = m_context->enqueueV3(m_stream);
    if (!status)
    {
        LOGE("LSTM enqueueV3 failed! Check input/output names.");
        return false;
    }

    // 4. 拷贝回 Host
    CUDA_CHECK(cudaMemcpyAsync(m_host_output, m_output_device, m_output_size_bytes, cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    int off = m_config.target_frame_idx * 3;
    m_last_prediction[0] = m_host_output[off + 0];
    m_last_prediction[1] = m_host_output[off + 1];
    m_last_prediction[2] = m_host_output[off + 2];

    return true;
}

std::vector<float> LSTMPredictor::get_prediction() const
{
    return m_last_prediction;
}
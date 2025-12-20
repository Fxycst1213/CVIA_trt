#include "model.hpp"
#include "../utils/utils.hpp"
#include "../logger/logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "../calibrator/calibrator.hpp"
#include <string>

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

namespace model
{

    Model::Model(string onnx_path, logger::Level level, Params params)
    {
        m_onnxPath = onnx_path;
        m_workspaceSize = WORKSPACESIZE;
        m_logger = make_shared<logger::Logger>(level);
        m_timer = make_shared<timer::Timer>();
        m_params = new Params(params);
        m_enginePath = changePath(onnx_path, "../engine", ".engine", getPrec(params.prec));
        // 改名onnx变engine
    }

    void Model::load_image(string image_path)
    {
        if (!fileExists(image_path))
        {
            LOGE("%s not found", image_path.c_str());
        }
        else
        {
            m_imagePath = image_path;
            LOG("*********************INFERENCE INFORMATION***********************");
            LOG("\tModel:      %s", getFileName(m_onnxPath).c_str());
            LOG("\tImage:      %s", getFileName(m_imagePath).c_str());
            LOG("\tPrecision:  %s", getPrec(m_params->prec).c_str());
        }
    }

    void Model::init_model()
    {
        /* 一个model的engine, context这些一旦创建好了，当多次调用这个模型的时候就没必要每次都初始化了*/
        if (m_context == nullptr)
        {
            if (!fileExists(m_enginePath))
            {
                LOG("%s not found. Building trt engine...", m_enginePath.c_str());
                build_engine();
            }
            else
            {
                LOG("%s has been generated! loading trt engine...", m_enginePath.c_str());
                load_engine();
            }
        }
    }

    bool Model::build_engine()
    {
        auto builder = shared_ptr<IBuilder>(createInferBuilder(*m_logger), destroy_trt_ptr<IBuilder>);
        auto network = shared_ptr<INetworkDefinition>(builder->createNetworkV2(1), destroy_trt_ptr<INetworkDefinition>);
        auto config = shared_ptr<IBuilderConfig>(builder->createBuilderConfig(), destroy_trt_ptr<IBuilderConfig>);
        auto parser = shared_ptr<IParser>(createParser(*network, *m_logger), destroy_trt_ptr<IParser>);
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, m_workspaceSize);
        config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
        if (!parser->parseFromFile(m_onnxPath.c_str(), 1)) // 1 means Verbosity info
        {
            LOGE("Failed to parse ONNX file: %s", m_onnxPath.c_str());
            return false;
        }
        if (builder->platformHasFastFp16() && m_params->prec == model::FP16)
        {
            config->setFlag(BuilderFlag::kFP16);
        }
        else if (builder->platformHasFastInt8() && m_params->prec == model::INT8)
        {
            config->setFlag(BuilderFlag::kINT8);
            config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
            shared_ptr<Int8EntropyCalibrator> calibrator(new Int8EntropyCalibrator(
                64,
                "calibration/calibration_list.txt",
                "calibration/calibration_table.txt",
                3 * 640 * 640, 640, 640));
            config->setInt8Calibrator(calibrator.get());
        }
        auto plan = shared_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config), destroy_trt_ptr<IHostMemory>);

        if (!plan)
        {
            LOGE("Failed to build serialized network!");
            return false;
        }
        save_plan(*plan);
        setup(plan->data(), plan->size());
        LOGV("Before TensorRT optimization");
        print_network(*network, false);
        if (m_engine)
        {
            LOGV("After TensorRT optimization");
            print_network(*network, true);
        }

        return true;
    }

    bool Model::enqueue_bindings()
    {
        m_timer->start_gpu();
        if (m_useCudaGraph && m_isGraphCaptured)
        {
            cudaError_t ret = cudaGraphLaunch(m_cudaGraphExec, m_stream);
            if (ret != cudaSuccess)
            {
                LOGE("CUDA Graph launch failed: %s", cudaGetErrorString(ret));
                return false;
            }
        }
        else
        {
            char const *input_name = m_engine->getIOTensorName(0);
            char const *output_name = m_engine->getIOTensorName(1);
            m_context->setTensorAddress(input_name, m_bindings[0]);
            m_context->setTensorAddress(output_name, m_bindings[1]);

            if (m_useCudaGraph && !m_isGraphCaptured)
            {
                LOG("Starting CUDA Graph Capture...");
                cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal);
                m_context->enqueueV3(m_stream);
                cudaStreamEndCapture(m_stream, &m_cudaGraph);
                cudaGraphInstantiate(&m_cudaGraphExec, m_cudaGraph, nullptr, nullptr, 0);
                m_isGraphCaptured = true;
                LOG("CUDA Graph Captured Successfully!");
                cudaGraphLaunch(m_cudaGraphExec, m_stream);
            }
            else
            {
                m_context->enqueueV3(m_stream);
            }
        }

        m_timer->stop_gpu("trt-inference(GPU)");
        return true;
    }

    bool Model::load_engine()
    {
        if (!fileExists(m_enginePath))
        {
            LOGE("engine does not exits! Program terminated");
            return false;
        }

        vector<unsigned char> modelData;
        modelData = loadFile(m_enginePath);

        setup(modelData.data(), modelData.size());

        return true;
    }

    void Model::save_plan(IHostMemory &plan)
    {
        auto f = fopen(m_enginePath.c_str(), "wb");
        fwrite(plan.data(), 1, plan.size(), f);
        fclose(f);
    }

    void Model::inference(const Resultframe &resultframe)
    {
        m_timer->init();
        reset_task();
        if (m_params->dev == CPU)
        {
            preprocess_cpu(resultframe.rgb);
        }
        else
        {
            preprocess_gpu(resultframe.rgb);
        }

        enqueue_bindings();

        if (m_params->dev == CPU)
        {
            postprocess_cpu(resultframe.timestamp);
        }
        else
        {
            postprocess_gpu(resultframe.timestamp);
        }
    }

    void Model::print_network(INetworkDefinition &network, bool optimized)
    {

        int inputCount = network.getNbInputs();
        int outputCount = network.getNbOutputs();
        string layer_info;

        for (int i = 0; i < inputCount; i++)
        {
            auto input = network.getInput(i);
            LOGV("Input info: %s:%s", input->getName(), printTensorShape(input).c_str());
        }

        for (int i = 0; i < outputCount; i++)
        {
            auto output = network.getOutput(i);
            LOGV("Output info: %s:%s", output->getName(), printTensorShape(output).c_str());
        }

        int layerCount = optimized ? m_engine->getNbLayers() : network.getNbLayers();
        LOGV("network has %d layers", layerCount);

        if (!optimized)
        {
            for (int i = 0; i < layerCount; i++)
            {
                char layer_info[1000];
                auto layer = network.getLayer(i);
                auto input = layer->getInput(0);
                int n = 0;
                if (input == nullptr)
                {
                    continue;
                }
                auto output = layer->getOutput(0);

                LOGV("layer_info: %-40s:%-25s->%-25s[%s]",
                     layer->getName(),
                     printTensorShape(input).c_str(),
                     printTensorShape(output).c_str(),
                     getPrecision(layer->getPrecision()).c_str());
            }
        }
        else
        {
            auto inspector = shared_ptr<IEngineInspector>(m_engine->createEngineInspector());
            for (int i = 0; i < layerCount; i++)
            {
                LOGV("layer_info: %s", inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
            }
        }
    }

    string Model::getPrec(model::precision prec)
    {
        switch (prec)
        {
        case model::precision::FP16:
            return "fp16";
        case model::precision::INT8:
            return "int8";
        default:
            return "fp32";
        }
    }

} // namespace model

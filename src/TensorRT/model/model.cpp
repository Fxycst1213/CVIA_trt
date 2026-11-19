#include "model.hpp"
#include "../utils/utils.hpp"
#include "../logger/logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "../calibrator/calibrator.hpp"
#include <cstdio>
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
        m_timer->start_gpu(m_stream);
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
        m_frame_timings.inference_gpu_ms = m_timer->elapsed_gpu_ms();
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
        const auto frame_start = timer::Clock::now();
        m_frame_timings = FrameTimings{};
        m_timer->init();
        reset_task();
        {
            timer::ScopedCpuTimer stage(m_frame_timings.preprocess_ms);
            if (m_params->dev == CPU)
            {
                preprocess_cpu(resultframe.rgb);
                // Include the CPU path's asynchronous input upload in preprocessing.
                CUDA_CHECK(cudaStreamSynchronize(m_stream));
            }
            else
            {
                preprocess_gpu(resultframe.rgb);
            }
        }

        {
            timer::ScopedCpuTimer stage(m_frame_timings.inference_ms);
            enqueue_bindings(); // Existing GPU timer waits for the inference stream.
        }
        {
            timer::ScopedCpuTimer stage(m_frame_timings.postprocess_ms);
            if (m_params->dev == CPU)
            {
                postprocess_cpu(resultframe.timestamp);
            }
            else
            {
                postprocess_gpu(resultframe.timestamp);
            }
        }
        m_frame_timings.total_ms = timer::elapsed_ms(frame_start);
        // Format/write the log after all measured stages; no per-frame flush or delay.
        log_frame_timings(resultframe.timestamp);
    }

    void Model::log_frame_timings(uint64_t timestamp)
    {
        ++m_timing_frame;
        if (m_timing_frame == 1)
            std::printf("[timing-info] units=ms warmup_frames=20 opencv_threads=%d "
                        "input=%dx%d; postprocess contains output_copy/decode/nms/refine/pnp/filter/transform; "
                        "off=not_called; inference_gpu is part of inference, not additional time\n",
                        cv::getNumThreads(), m_inputImage.cols, m_inputImage.rows);
        const auto value = [](double ms) {
            if (ms < 0)
                return std::string("off");
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%.3f", ms);
            return std::string(buffer);
        };
        const auto &t = m_frame_timings;
        std::printf("[timing] frame=%llu timestamp=%llu warmup=%d "
                    "total_ms=%.3f preprocess_ms=%.3f inference_ms=%.3f postprocess_ms=%.3f "
                    "output_copy_ms=%s decode_ms=%s nms_ms=%s refine_ms=%s "
                    "pnp_ms=%s filter_ms=%s transform_ms=%s inference_gpu_ms=%s\n",
                    static_cast<unsigned long long>(m_timing_frame),
                    static_cast<unsigned long long>(timestamp), m_timing_frame <= 20 ? 1 : 0,
                    t.total_ms, t.preprocess_ms, t.inference_ms, t.postprocess_ms,
                    value(t.output_copy_ms).c_str(), value(t.decode_ms).c_str(), value(t.nms_ms).c_str(),
                    value(t.refine_ms).c_str(), value(t.pnp_ms).c_str(), value(t.filter_ms).c_str(),
                    value(t.transform_ms).c_str(), value(t.inference_gpu_ms).c_str());
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

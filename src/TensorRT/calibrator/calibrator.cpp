#include "NvInfer.h"
#include "calibrator.hpp"
#include "utils.hpp"
#include "logger.hpp"
#include "preprocess.hpp"
#include "cudatools.hpp"
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace nvinfer1;

namespace model
{

    Int8EntropyCalibrator::Int8EntropyCalibrator(
        const int &batchSize,
        const string &calibrationDataPath,
        const string &calibrationTablePath,
        const int &inputSize,
        const int &inputH,
        const int &inputW) :

                             m_batchSize(batchSize),
                             m_inputH(inputH),
                             m_inputW(inputW),
                             m_inputSize(inputSize),
                             m_inputCount(batchSize * inputSize),
                             m_calibrationTablePath(calibrationTablePath)
    {
        m_imageList = loadDataList(calibrationDataPath);
        m_imageList.resize(static_cast<int>(m_imageList.size() / m_batchSize) * m_batchSize);
        std::random_shuffle(m_imageList.begin(), m_imageList.end(),
                            [](int i)
                            { return rand() % i; });
        CUDA_CHECK(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));
    }

    bool Int8EntropyCalibrator::getBatch(
        void *bindings[], const char *names[], int nbBindings) noexcept
    {
        if (m_imageIndex + m_batchSize >= m_imageList.size() + 1)
            return false;

        LOG("%3d/%3d (%3dx%3d): %s",
            m_imageIndex + 1, m_imageList.size(), m_inputH, m_inputW, m_imageList.at(m_imageIndex).c_str());

        cv::Mat input_image;
        for (int i = 0; i < m_batchSize; i++)
        {
            input_image = cv::imread(m_imageList.at(m_imageIndex++));
            preprocess::preprocess_resize_gpu(
                input_image,
                m_deviceInput + i * m_inputSize,
                m_inputH, m_inputW,
                preprocess::tactics::GPU_BILINEAR_CENTER);
        }

        bindings[0] = m_deviceInput;

        return true;
    }

    const void *Int8EntropyCalibrator::readCalibrationCache(size_t &length) noexcept
    {
        void *output;
        m_calibrationCache.clear();

        ifstream input(m_calibrationTablePath, ios::binary);
        input >> noskipws;
        if (m_readCache && input.good())
            copy(istream_iterator<char>(input), istream_iterator<char>(), back_inserter(m_calibrationCache));

        length = m_calibrationCache.size();
        if (length)
        {
            LOG("Using cached calibration table to build INT8 trt engine...");
            output = &m_calibrationCache[0];
        }
        else
        {
            LOG("Creating new calibration table to build INT8 trt engine...");
            output = nullptr;
        }
        return output;
    }

    void Int8EntropyCalibrator::writeCalibrationCache(const void *cache, size_t length) noexcept
    {
        ofstream output(m_calibrationTablePath, ios::binary);
        output.write(reinterpret_cast<const char *>(cache), length);
        output.close();
    }

} // namespace model

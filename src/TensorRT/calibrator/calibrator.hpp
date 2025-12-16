#ifndef __CALIBRATOR_HPP__
#define __CALIBRATOR_HPP__

#include "NvInfer.h"
#include <string>
#include <vector>

namespace model
{

    class Int8EntropyCalibrator : public nvinfer1::IInt8MinMaxCalibrator
    {

    public:
        Int8EntropyCalibrator(
            const int &batchSize,
            const std::string &calibrationSetPath,
            const std::string &calibrationTablePath,
            const int &inputSize,
            const int &inputH,
            const int &inputW);

        ~Int8EntropyCalibrator() {};

        int getBatchSize() const noexcept override { return m_batchSize; };
        bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override;
        const void *readCalibrationCache(std::size_t &length) noexcept override;
        void writeCalibrationCache(const void *ptr, std::size_t legth) noexcept override;

    private:
        const int m_batchSize;
        const int m_inputH;
        const int m_inputW;
        const int m_inputSize;
        const int m_inputCount;
        const std::string m_calibrationTablePath{nullptr};

        std::vector<std::string> m_imageList;
        std::vector<char> m_calibrationCache;

        float *m_deviceInput{nullptr};
        bool m_readCache{true};
        int m_imageIndex;
    };

}; // namespace model

#endif __CALIBRATOR_HPP__

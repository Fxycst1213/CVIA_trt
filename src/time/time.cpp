#include "time.hpp"
namespace timer
{

    Timer::Timer(logger::Level level)
    {
        m_LOG_LEVEL = level;
        _timeElasped = 0;
        _cStart = std::chrono::high_resolution_clock::now();
        _cStop = std::chrono::high_resolution_clock::now();
        cudaEventCreate(&_gStart);
        cudaEventCreate(&_gStop);
    }

    Timer::~Timer()
    {
        cudaEventDestroy(_gStart);
        cudaEventDestroy(_gStop);
    }

    void Timer::start_gpu(cudaStream_t stream)
    {
        _gStream = stream;
        CUDA_CHECK(cudaEventRecord(_gStart, _gStream));
    }

    void Timer::stop_gpu()
    {
        CUDA_CHECK(cudaEventRecord(_gStop, _gStream));
    }

    void Timer::stop_gpu(std::string msg)
    {
        CUDA_CHECK(cudaEventRecord(_gStop, _gStream));
        char buff[100];

        CUDA_CHECK(cudaEventSynchronize(_gStart));
        CUDA_CHECK(cudaEventSynchronize(_gStop));
        CUDA_CHECK(cudaEventElapsedTime(&_timeElasped, _gStart, _gStop));

        sprintf(buff, "\t%-60s uses %.6lf ms", msg.c_str(), _timeElasped);
        this->_timeMsgs.emplace_back(buff);
    }

    void Timer::start_cpu()
    {
        _cStart = std::chrono::high_resolution_clock::now();
    }

    void Timer::stop_cpu()
    {
        _cStop = std::chrono::high_resolution_clock::now();
    }

    void Timer::show()
    {
        for (int i = 0; i < _timeMsgs.size(); i++)
        {
            if (m_LOG_LEVEL == logger::Level::VERB)
            {
                LOGV("%s", _timeMsgs[i].c_str());
            }
            else
            {
                LOG("%s", _timeMsgs[i].c_str());
            }
        }
    }

    void Timer::init()
    {
        _timeMsgs.clear();
    }

} // namespace model

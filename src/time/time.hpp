#ifndef __TIME_HPP__
#define __TIME_HPP__

#include <chrono>
#include <ratio>
#include <string>
#include "cuda_runtime.h"
#include <vector>
#include "logger.hpp"
#include "cudatools.hpp"
namespace timer
{
    using Clock = std::chrono::steady_clock;

    inline double elapsed_ms(Clock::time_point start)
    {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    // Separate clocks allow nested stages without overwriting the parent timer.
    class ScopedCpuTimer
    {
    public:
        explicit ScopedCpuTimer(double &duration_ms)
            : duration_ms_(duration_ms), start_(Clock::now()) {}
        ~ScopedCpuTimer() { duration_ms_ = elapsed_ms(start_); }
        ScopedCpuTimer(const ScopedCpuTimer &) = delete;
        ScopedCpuTimer &operator=(const ScopedCpuTimer &) = delete;

    private:
        double &duration_ms_;
        Clock::time_point start_;
    };

    class Timer
    {
    public:
        using s = std::ratio<1, 1>;
        using ms = std::ratio<1, 1000>;
        using us = std::ratio<1, 1000000>;
        using ns = std::ratio<1, 1000000000>;

    public:
        Timer(logger::Level level = logger::Level::VERB);
        ~Timer();

    public:
        void start_cpu();
        void start_gpu(cudaStream_t stream = nullptr);
        void start_cpu_1();

        void stop_cpu();
        void stop_gpu();
        void stop_cpu_1();

        template <typename span>
        void stop_cpu(std::string msg);
        void stop_gpu(std::string msg);

        void show();
        void init();
        float elapsed_gpu_ms() const { return _timeElasped; }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> _cStart;
        std::chrono::time_point<std::chrono::high_resolution_clock> _cStop;
        cudaEvent_t _gStart;
        cudaEvent_t _gStop;
        cudaStream_t _gStream = nullptr;
        float _timeElasped;
        std::vector<std::string> _timeMsgs;
        logger::Level m_LOG_LEVEL = logger::Level::VERB;
    };

    template <typename span>
    void Timer::stop_cpu(std::string msg)
    {
        _cStop = std::chrono::high_resolution_clock::now();

        char buff[100];
        std::string str;

        if (std::is_same<span, s>::value)
        {
            str = "s";
        }
        else if (std::is_same<span, ms>::value)
        {
            str = "ms";
        }
        else if (std::is_same<span, us>::value)
        {
            str = "us";
        }
        else if (std::is_same<span, ns>::value)
        {
            str = "ns";
        }

        std::chrono::duration<double, span> time = _cStop - _cStart;
        sprintf(buff, "\t%-60s uses %.6lf %s", msg.c_str(), time.count(), str.c_str());
        _timeMsgs.emplace_back(buff);
    }

} // namespace timer

#endif //__TIME_HPP__

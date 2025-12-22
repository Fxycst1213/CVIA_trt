#pragma once
#include <vector>
#include <deque>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <limits>

// 引入 Eigen
#include <Eigen/Dense>
// 引入 FFTW
#include <fftw3.h>

using namespace std;
using Eigen::VectorXd;

class PeriodEstimator {
public:
    // 定义估算模式枚举
    enum EstimationMode {
        MODE_TIME_DOMAIN_ONLY,  // 仅使用时域（半波法）
        MODE_FREQ_DOMAIN_ONLY,  // 仅使用频域（FFT）
        MODE_HYBRID             // 混合模式（根据数据量自动切换）
    };

    // 构造函数
    // max_history: 窗口最大长度
    // min_points:  开始计算的最少点数（用于 单独时域/单独频域 模式的触发门槛）
    // mode:        估算模式，默认为混合模式
    PeriodEstimator(int max_history, int min_points, EstimationMode mode = MODE_HYBRID);

    // 析构函数
    ~PeriodEstimator();

    // 设置模式（运行时动态切换）
    void setMode(EstimationMode mode);

    // 【接口1】输入数据：传入时间戳(秒)和XYZ坐标
    void update(uint64_t timestamp_ns, double x, double y, double z);

    // 【接口2】计算并获取周期
    // output_periods: 传入一个 double[3] 数组，函数会将 x,y,z 的周期填入
    // 返回值: true 表示计算成功，false 表示数据不足或尚未达到计算门槛
    bool getEstimatedPeriods(double* output_periods);

private:
    // --- 内部常量定义 ---
    static constexpr double PI = 3.14159265358979323846;
    
    // 混合模式下的自动切换阈值
    static constexpr int POINTS_PER_PERIOD = 750;
    static constexpr int THRESHOLD_TWO_PERIODS = 1526; // 混合模式下：大于此值切FFT
    static constexpr int HALF_WAVE_WINDOW_SIZE = 916;  // 混合模式下：时域计算的最小窗口/数据量

    // 成员变量
    double last_valid_periods[3]; 
    EstimationMode current_mode; // 当前模式

    // --- 内部数据结构 ---
    enum WindowType {
        RECTANGULAR = 0,
        HANNING,
        HAMMING,
        BLACKMAN
    };

    // --- 成员变量 ---
    int max_history_size;
    int min_points_for_est; // 用户手动设置的计算门槛
    int smooth_radius;

    // 数据缓冲 (X, Y, Z 三轴)
    std::deque<double> time_buffer;
    std::vector<std::deque<double>> axis_buffers; 

    // --- 辅助函数 ---
    void export_axis(int axis_idx, VectorXd &t_out, VectorXd &s_out, bool smooth, bool zscore);
    VectorXd linear_interpolate(const VectorXd &t_vals, const VectorXd &y_vals, const VectorXd &t_grid);
    double calculate_median_dt(const VectorXd &t);
    void apply_window(VectorXd& data, WindowType window_type);
    double calculate_period_fft(const VectorXd &t_vec, const VectorXd &val_vec, double dt_grid);
    double calculate_period_half_wave(const VectorXd &t_vec, const VectorXd &val_vec);
    void detrend_linear(VectorXd &data);
};
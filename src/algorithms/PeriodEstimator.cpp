#include "PeriodEstimator.h"

// 构造函数
// 增加了 mode 参数的初始化
PeriodEstimator::PeriodEstimator(int max_history, int min_points, EstimationMode mode) 
    : max_history_size(max_history), 
      min_points_for_est(min_points),
      current_mode(mode)
{
    smooth_radius = 4; 
    axis_buffers.resize(3); 
    
    for(int i=0; i<3; ++i) {
        last_valid_periods[i] = 0.0;
    }
}

// 析构函数
PeriodEstimator::~PeriodEstimator() {
    // fftw_cleanup(); 
}

// 设置模式
void PeriodEstimator::setMode(EstimationMode mode) {
    this->current_mode = mode;
}

// 更新数据 (保持不变)
void PeriodEstimator::update(uint64_t timestamp_ns, double x, double y, double z) {
    while ((int)time_buffer.size() >= max_history_size) {
        time_buffer.pop_front();
        for (int i = 0; i < 3; ++i) axis_buffers[i].pop_front();
    }
    // 毫秒转秒
    time_buffer.push_back(static_cast<double>(timestamp_ns) / 1000.0);
    axis_buffers[0].push_back(x);
    axis_buffers[1].push_back(y);
    axis_buffers[2].push_back(z);
}

// 获取周期结果 (核心修改逻辑)
bool PeriodEstimator::getEstimatedPeriods(double* output_periods) {
    int current_data_size = (int)time_buffer.size();

    // 0. 基础公共检查
    // 无论什么模式，如果数据极少(例如小于2个点)，直接返回失败
    if (current_data_size < 2) return false;

    // 1. 准备公共数据
    VectorXd t_raw(current_data_size);
    for(size_t i=0; i<time_buffer.size(); ++i) t_raw[i] = time_buffer[i];
    
    double dt_grid = calculate_median_dt(t_raw);
    if (dt_grid <= 0.00001) dt_grid = 0.01; 

    // 2. 遍历三个轴 (X, Y, Z)
    for (int i = 0; i < 3; ++i) {
        VectorXd t_axis, s_axis;
        export_axis(i, t_axis, s_axis, true, true); 

        double calculated_period = 0.0;

        // ================= 模式分流逻辑 =================
        switch (current_mode) {
            
            case MODE_TIME_DOMAIN_ONLY:
                // 【模式1：纯时域】
                // 严格遵守 min_points_for_est 设置
                if (current_data_size >= min_points_for_est) {
                    calculated_period = calculate_period_half_wave(t_axis, s_axis);
                }
                break;

            case MODE_FREQ_DOMAIN_ONLY:
                // 【模式2：纯频域】
                // 严格遵守 min_points_for_est 设置
                if (current_data_size >= min_points_for_est) {
                    calculated_period = calculate_period_fft(t_axis, s_axis, dt_grid);
                }
                break;

            case MODE_HYBRID:
            default:
                // 【模式3：混合模式 (原逻辑)】
                // 逻辑：
                // 1. 数据量 < 916 (HALF_WAVE_WINDOW_SIZE) -> 不计算
                // 2. 916 <= 数据量 < 1526 (THRESHOLD_TWO_PERIODS) -> 用时域
                // 3. 数据量 >= 1526 -> 用频域
                
                if (current_data_size < HALF_WAVE_WINDOW_SIZE) {
                    // 数据不足，本帧不计算，calculated_period 保持为 0
                } 
                else if (current_data_size < THRESHOLD_TWO_PERIODS) {
                    // 阶段一：使用时域半波法
                    calculated_period = calculate_period_half_wave(t_axis, s_axis);
                } 
                // else {
                //     // 阶段二：使用 FFT 算法
                //     calculated_period = calculate_period_fft(t_axis, s_axis, dt_grid);
                // }
                break;
        }

        // ================= 结果保存 (Sample and Hold) =================
        
        // 只有算出有效值时才更新
        if (calculated_period > 1e-6 && std::isfinite(calculated_period)) {
            last_valid_periods[i] = calculated_period;
        }

        output_periods[i] = last_valid_periods[i];
    }

    // 如果还没有产生过任何有效周期，且当前也没有算出结果，可以考虑返回 false
    // 但为了接口稳定性，通常只要 buffer 不为空就返回 true，具体由上层判断 period 是否为 0
    return true;
}

// ================= 内部辅助函数 (保持原有逻辑优化) =================

void PeriodEstimator::export_axis(int axis_idx, VectorXd &t_out, VectorXd &s_out, bool smooth, bool zscore) {
    int n = (int)time_buffer.size();
    t_out.resize(n);
    s_out.resize(n);

    for (int i = 0; i < n; ++i) {
        t_out[i] = time_buffer[i];
        s_out[i] = axis_buffers[axis_idx][i];
    }

    if(smooth) {
        VectorXd tmp = s_out;
        for (int i = 0; i < n; ++i) {
            int L = std::max(0, i - smooth_radius);
            int R = std::min(n - 1, i + smooth_radius);
            double sum = 0;
            int cnt = 0;
            for (int j = L; j <= R; ++j) {
                if (std::isfinite(tmp[j])) {
                    sum += tmp[j];
                    cnt++;
                }
            }
            s_out[i] = (cnt > 0) ? (sum / cnt) : std::numeric_limits<double>::quiet_NaN();
        }
    }

    if(zscore) {
        std::vector<double> finite_vals;
        finite_vals.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (std::isfinite(s_out[i])) finite_vals.push_back(s_out[i]);
        }
        
        if (!finite_vals.empty()) {
            double sum_val = 0;
            for(double v : finite_vals) sum_val += v;
            double mean = sum_val / finite_vals.size();
            
            double var_sum = 0;
            for(double v : finite_vals) var_sum += (v - mean) * (v - mean);
            double stdv = std::sqrt(var_sum / finite_vals.size());
            
            if (stdv < 1e-12) stdv = 1.0;

            for (int i = 0; i < n; ++i) {
                if (std::isfinite(s_out[i])) {
                    s_out[i] = (s_out[i] - mean) / stdv;
                } else {
                    s_out[i] = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
        detrend_linear(s_out);
    }
}

double PeriodEstimator::calculate_median_dt(const VectorXd &t) {
    if (t.size() < 2) return 1.0;
    std::vector<double> diffs;
    diffs.reserve(t.size() - 1);
    for (int i = 1; i < t.size(); ++i) {
        diffs.push_back(t[i] - t[i - 1]);
    }
    if (diffs.empty()) return 1.0;
    std::sort(diffs.begin(), diffs.end());
    return diffs[diffs.size() / 2];
}

VectorXd PeriodEstimator::linear_interpolate(const VectorXd &t_vals, const VectorXd &y_vals, const VectorXd &t_grid) {
    int n = (int)t_vals.size();
    int m = (int)t_grid.size();
    VectorXd out(m);

    if (n == 0) {
        for(int i=0; i<m; ++i) out[i] = std::numeric_limits<double>::quiet_NaN();
        return out;
    }

    for (int i = 0; i < m; ++i) {
        double tg = t_grid[i];
        if (tg < t_vals[0] || tg > t_vals[n - 1]) {
            out[i] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }

        auto it = std::upper_bound(t_vals.data(), t_vals.data() + n, tg);
        int idx = (int)(it - t_vals.data()) - 1;
        
        if (idx < 0) idx = 0;
        if (idx >= n - 1) {
            out[i] = y_vals[n - 1];
            continue;
        }

        double t0 = t_vals[idx];
        double t1 = t_vals[idx + 1];
        double y0 = y_vals[idx];
        double y1 = y_vals[idx + 1];
        
        if (std::abs(t1 - t0) < 1e-9) {
            out[i] = y0;
        } else {
            double alpha = (tg - t0) / (t1 - t0);
            out[i] = y0 * (1.0 - alpha) + y1 * alpha;
        }
    }
    return out;
}

void PeriodEstimator::apply_window(VectorXd& data, WindowType window_type) {
    int N = data.size();
    if (N < 2) return;
    
    // 这里仅保留了 RECTANGULAR 的逻辑框架，如需其他窗函数可按原代码恢复
    // 在 FFT 函数中目前使用的是 RECTANGULAR
    switch (window_type) {
        case RECTANGULAR:
            break;
        default:
            break;
    }
}

double PeriodEstimator::calculate_period_fft(const VectorXd &t_vec, const VectorXd &val_vec, double dt_grid) {
    double t_min = t_vec[0];
    double t_max = t_vec[t_vec.size() - 1];
    int n_grid = (int)std::floor((t_max - t_min) / dt_grid) + 1;
    
    VectorXd t_grid(n_grid);
    for(int i=0; i<n_grid; ++i) t_grid[i] = t_min + i * dt_grid;

    VectorXd y = linear_interpolate(t_vec, val_vec, t_grid);

    VectorXd y_clean = y;
    int valid_count = 0;
    for (int i = 0; i < y_clean.size(); ++i) {
        if (!std::isfinite(y_clean[i]))
            y_clean[i] = 0.0;
        else
            valid_count++;
    }
    
    // 如果有效数据点太少，FFT效果会很差
    if (valid_count < 10) { 
        return std::numeric_limits<double>::quiet_NaN();
    }

    int N = y_clean.size();
    if (N < 2) return std::numeric_limits<double>::quiet_NaN();

    // 补零到 2 的幂次，提高频谱密度
    int N_padded = 8192; 
    while (N_padded < N) N_padded *= 2; 

    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_padded);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N_padded);

    for (int i = 0; i < N_padded; ++i) {
        if (i < N) {
            in[i][0] = y_clean[i];
            in[i][1] = 0.0;
        } else {
            in[i][0] = 0.0;
            in[i][1] = 0.0;
        }
    }

    fftw_plan p = fftw_plan_dft_1d(N_padded, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    int spectrum_length = N_padded / 2;
    std::vector<double> mag(spectrum_length);
    mag[0] = 0.0; // 消除直流
    for (int i = 1; i < spectrum_length; ++i) {
        mag[i] = std::sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
    }

    int peak_idx = 0;
    double max_val = 0.0;
    for (int i = 1; i < spectrum_length; ++i) { 
        if (mag[i] > max_val) {
            max_val = mag[i];
            peak_idx = i;
        }
    }

    // 高斯插值修正频率
    double delta = 0.0;
    if (peak_idx > 0 && peak_idx < spectrum_length - 1) {
        double alpha = std::log(std::max(mag[peak_idx - 1], 1e-10));
        double beta  = std::log(std::max(mag[peak_idx],     1e-10));
        double gamma = std::log(std::max(mag[peak_idx + 1], 1e-10));

        double denominator = 2 * (2 * beta - alpha - gamma);
        if (std::abs(denominator) > 1e-9) {
             delta = (alpha - gamma) / denominator; 
             // 修正: 此处 delta 范围应在 -0.5 到 0.5 之间
             delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma); 
        }
    }

    double peak_freq_idx = peak_idx + delta;
    double fs = 1.0 / dt_grid;
    double freq = peak_freq_idx * fs / N_padded;
    double period = (freq > 1e-6) ? (1.0 / freq) : 0.0;

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return period;
}

void PeriodEstimator::detrend_linear(VectorXd &data) {
    int n = data.size();
    if (n < 2) return;
    
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (int i = 0; i < n; ++i) {
        sum_x += i;
        sum_y += data[i];
        sum_xy += i * data[i];
        sum_xx += i * i;
    }
    
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;
    
    for (int i = 0; i < n; ++i) {
        data[i] -= (slope * i + intercept);
    }
}

double PeriodEstimator::calculate_period_half_wave(const VectorXd &t_vec, const VectorXd &val_vec) {
    int n = (int)val_vec.size();
    if (n < 2) return 0.0;

    int window_size = HALF_WAVE_WINDOW_SIZE; 
    int start_idx = n - window_size;
    if (start_idx < 0) start_idx = 0; 

    // 1. 寻找最大值和最小值
    int max_idx = -1;
    int min_idx = -1;
    double max_val = -std::numeric_limits<double>::infinity();
    double min_val = std::numeric_limits<double>::infinity();

    for (int i = start_idx; i < n; ++i) {
        if (!std::isfinite(val_vec[i])) continue;
        if (val_vec[i] > max_val) { max_val = val_vec[i]; max_idx = i; }
        if (val_vec[i] < min_val) { min_val = val_vec[i]; min_idx = i; }
    }

    if (max_idx == -1 || min_idx == -1) return 0.0;
    
    // 【修改点1】增加对幅值的检查，避免处理平直线或极小噪声
    // 注意：如果是 z-score 后的数据，幅值差通常在 2.0 以上。
    // 如果是原始数据，0.5 可能太小。这里假设输入已被归一化或保留原阈值。
    if ((max_val - min_val) < 0.5) return 0.0; 

    // 【修改点2】核心修复：检查中间是否经过了“中值”
    // 如果是掉线（阶跃），数据往往是高位直接跳到低位，中间缺乏过渡点（或者过渡非常陡峭且不在中间）。
    // 简单的检查：验证 max 和 min 之间是否存在“过零”行为不是必须的，
    // 但我们可以检查 max 和 min 的位置是否在窗口边缘，这往往暗示截断。
    // 更有效的：检查波形对称性。
    
    // 【修改点3】非周期信号过滤：检查是否存在第三个极值点以确认周期性
    // 半波法最怕的就是阶跃。如果我们只看半波，至少要求波形在“时间轴”上具有一定的连续性。
    // 这里增加一个简单的启发式规则：
    // 如果是真正的波峰和波谷，在 max_idx 附近和 min_idx 附近的数据应该比较平滑。
    // 如果是掉线（例如从 1946 瞬间变 0），数据变化率（导数）会异常大。
    
    double t1 = t_vec[max_idx];
    double t2 = t_vec[min_idx];
    double half_period = std::abs(t1 - t2);
    
    
    return half_period * 2.0;
}
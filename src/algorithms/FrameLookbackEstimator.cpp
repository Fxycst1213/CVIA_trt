#include "FrameLookbackEstimator.h"
#include <iostream>

FrameLookbackEstimator::FrameLookbackEstimator(int max_history_frames)
    : max_history_size(max_history_frames)
{
    // 初始化
    for (int i = 0; i < 3; ++i)
    {
        periods[i] = 0.0;
        lookback_offsets[i] = 0.0; // 默认回溯 0 帧
    }
}

FrameLookbackEstimator::~FrameLookbackEstimator()
{
    history_buffer.clear();
}

void FrameLookbackEstimator::setPeriods(double x_period_frames, double y_period_frames, double z_period_frames)
{
    periods[0] = x_period_frames;
    periods[1] = y_period_frames;
    periods[2] = z_period_frames;
}

// 【新增实现】设置具体的回溯量
void FrameLookbackEstimator::setLookbackOffsets(double back_x, double back_y, double back_z)
{
    lookback_offsets[0] = back_x;
    lookback_offsets[1] = back_y;
    lookback_offsets[2] = back_z;
}

void FrameLookbackEstimator::update(uint64_t frame_id, double x, double y, double z)
{
    while ((int)history_buffer.size() >= max_history_size)
    {
        history_buffer.pop_front();
    }

    FrameData data;
    data.frame_id = frame_id;
    data.x = x;
    data.y = y;
    data.z = z;
    history_buffer.push_back(data);
}

// 【核心逻辑修改】
bool FrameLookbackEstimator::getPrediction(double *output_preds)
{
    if (history_buffer.empty())
        return false;

    uint64_t current_id = history_buffer.back().frame_id;
    bool at_least_one_valid = false;

    for (int i = 0; i < 3; ++i)
    {
        // 1. 获取该轴的配置
        double threshold = periods[i];            // 启动门槛（周期）
        double back_frames = lookback_offsets[i]; // 实际回溯量

        // 2. 检查启动条件：
        // 如果数据量还没达到设定的“周期/门槛”，则认为还没准备好
        if (threshold <= 0.1 || history_buffer.size() <= (size_t)threshold)
        {
            output_preds[i] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }

        // 3. 计算目标 ID (Current - Back_Frames)
        // 注意这里用的是 lookback_offsets，不再是 periods
        double target_id = (double)current_id - back_frames;

        // 4. 执行查找
        double val = findValueByFrameId(target_id, i);

        if (!std::isfinite(val))
        {
            output_preds[i] = std::numeric_limits<double>::quiet_NaN();
        }
        else
        {
            output_preds[i] = val;
            at_least_one_valid = true;
        }
    }

    return at_least_one_valid;
}

double FrameLookbackEstimator::findValueByFrameId(double target_id, int axis_index)
{
    // ... (这部分代码保持不变，负责二分查找和插值) ...
    // ... 直接复制你之前提供的 findValueByFrameId 实现即可 ...
    // 为了节省篇幅，这里简写，请保留原有的完整实现

    if (history_buffer.empty())
        return std::numeric_limits<double>::quiet_NaN();
    double min_id = (double)history_buffer.front().frame_id;
    double max_id = (double)history_buffer.back().frame_id;
    if (target_id < min_id || target_id > max_id)
        return std::numeric_limits<double>::quiet_NaN();

    FrameData target_key;
    target_key.frame_id = (uint64_t)std::ceil(target_id);
    auto it = std::lower_bound(history_buffer.begin(), history_buffer.end(), target_key,
                               [](const FrameData &a, const FrameData &b)
                               { return a.frame_id < b.frame_id; });

    if (it == history_buffer.end())
        return std::numeric_limits<double>::quiet_NaN();

    const FrameData &upper_frame = *it;
    if (std::abs((double)upper_frame.frame_id - target_id) < 1e-6)
    {
        if (axis_index == 0)
            return upper_frame.x;
        if (axis_index == 1)
            return upper_frame.y;
        if (axis_index == 2)
            return upper_frame.z;
    }

    if (it == history_buffer.begin())
    {
        if (axis_index == 0)
            return upper_frame.x;
        if (axis_index == 1)
            return upper_frame.y;
        if (axis_index == 2)
            return upper_frame.z;
    }

    auto prev_it = it - 1;
    const FrameData &lower_frame = *prev_it;

    double t0 = (double)lower_frame.frame_id;
    double t1 = (double)upper_frame.frame_id;
    double alpha = (target_id - t0) / (t1 - t0);

    double y0 = 0, y1 = 0;
    if (axis_index == 0)
    {
        y0 = lower_frame.x;
        y1 = upper_frame.x;
    }
    else if (axis_index == 1)
    {
        y0 = lower_frame.y;
        y1 = upper_frame.y;
    }
    else
    {
        y0 = lower_frame.z;
        y1 = upper_frame.z;
    }

    return y0 * (1.0 - alpha) + y1 * alpha;
}

int FrameLookbackEstimator::getBufferSize() const
{
    return (int)history_buffer.size();
}
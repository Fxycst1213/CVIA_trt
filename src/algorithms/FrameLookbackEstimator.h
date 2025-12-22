#pragma once
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>

struct FrameData
{
    uint64_t frame_id;
    double x;
    double y;
    double z;
};

class FrameLookbackEstimator
{
public:
    // 构造函数
    FrameLookbackEstimator(int max_history_frames);
    ~FrameLookbackEstimator();

    // 【配置接口 1】设置“启动门槛”（周期）
    // 只有当缓冲区积累的数据量超过这个值，该轴才开始工作
    void setPeriods(double x_period_frames, double y_period_frames, double z_period_frames);

    // 【配置接口 2】设置“回溯帧数”（具体要往回找多少帧）
    // 计算公式：Target_ID = Current_ID - back_frames
    void setLookbackOffsets(double back_x, double back_y, double back_z);

    // 更新数据
    void update(uint64_t frame_id, double x, double y, double z);

    // 获取预测值
    bool getPrediction(double *output_preds);

    int getBufferSize() const;

private:
    double findValueByFrameId(double target_id, int axis_index);

    int max_history_size;

    // 门槛：只有 buffer size > periods[i] 时才允许查找
    double periods[3];

    // 目标：实际回溯的帧数
    double lookback_offsets[3];

    std::deque<FrameData> history_buffer;
};
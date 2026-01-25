#pragma once
#include "../model.hpp"
#include <deque>
#include <vector>
#include <memory>

namespace model
{
    namespace lstm
    {

        class LSTM : public Model
        {
        public:
            // 构造函数
            LSTM(std::string onnx_path, logger::Level level, Params params);

            std::vector<float> update_and_infer(float frame_id, float x, float y, float z);

        private:
            // 覆写 setup 以适配 LSTM 的输入尺寸（如果基类 setup 是虚函数）
            // 如果基类 setup 不是虚函数，我们在构造函数里通过修改 params 绕过

            // 覆写推理流程
            void inference(const Resultframe &resultframe) override; // 这个基类方法在这里可能用不到，我们自定义推口

            // LSTM 专用处理
            bool preprocess_lstm();
            bool postprocess_lstm(std::vector<float> &result);

        private:
            // 历史数据队列：[frame_id, x, y, z]
            std::deque<std::vector<float>> m_buffer;
            const int m_seq_len = 58;   // 序列长度
            const int m_input_dim = 4;  // 输入维度 (id, x, y, z)
            const int m_output_dim = 3; // 假设输出是 (x, y, z)
        };

        std::shared_ptr<LSTM> make_lstm(std::string onnx_path, logger::Level level, Params params);

    } // namespace lstm
} // namespace model
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <algorithm>
#include <string>
#include "utils.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "pose.hpp"
#include "preprocess.hpp"
#include "cudatools.hpp"
#include "../lstm/lstm_predictor.hpp"

using namespace std;
using namespace nvinfer1;
double deg2rad(double deg)
{
    return deg * CV_PI / 180.0;
};
namespace model
{
    namespace pose
    {

        float iou_calc(bbox bbox1, bbox bbox2)
        {
            auto inter_x0 = std::max(bbox1.x0, bbox2.x0);
            auto inter_y0 = std::max(bbox1.y0, bbox2.y0);
            auto inter_x1 = std::min(bbox1.x1, bbox2.x1);
            auto inter_y1 = std::min(bbox1.y1, bbox2.y1);

            float inter_w = inter_x1 - inter_x0;
            float inter_h = inter_y1 - inter_y0;

            float inter_area = inter_w * inter_h;
            float union_area =
                (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0) +
                (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0) -
                inter_area;

            return inter_area / union_area;
        }

        void Pose::setup(void const *data, size_t size)
        {
            m_runtime = shared_ptr<IRuntime>(createInferRuntime(*m_logger), destroy_trt_ptr<IRuntime>);
            m_engine = shared_ptr<ICudaEngine>(m_runtime->deserializeCudaEngine(data, size), destroy_trt_ptr<ICudaEngine>);
            m_context = shared_ptr<IExecutionContext>(m_engine->createExecutionContext(), destroy_trt_ptr<IExecutionContext>);
            char const *input_name = m_engine->getIOTensorName(0);
            char const *output_name = m_engine->getIOTensorName(1);

            m_inputDims = m_engine->getTensorShape(input_name);
            m_outputDims = m_engine->getTensorShape(output_name);

            CUDA_CHECK(cudaStreamCreate(&m_stream));

            m_inputSize = m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);
            m_imgArea = m_params->img.h * m_params->img.w;
            m_outputSize = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);

            CUDA_CHECK(cudaMallocHost(&m_inputMemory[0], m_inputSize));
            CUDA_CHECK(cudaMallocHost(&m_outputMemory[0], m_outputSize));
            CUDA_CHECK(cudaMalloc(&m_inputMemory[1], m_inputSize));
            CUDA_CHECK(cudaMalloc(&m_outputMemory[1], m_outputSize));

            m_bindings[0] = m_inputMemory[1];
            m_bindings[1] = m_outputMemory[1];
        }

        void Pose::reset_task()
        {
            m_bboxes.clear();
        }

        bool Pose::preprocess_cpu(const cv::Mat &img)
        {
            m_inputImage = img;
            if (m_inputImage.data == nullptr)
            {
                LOGE("ERROR: Image file not founded! Program terminated");
                return false;
            }

            m_timer->start_cpu();

            int input_w = m_inputImage.cols;
            int input_h = m_inputImage.rows;
            int target_w = m_params->img.w;
            int target_h = m_params->img.h;
            float scale = min(float(target_w) / input_w, float(target_h) / input_h);
            int new_w = int(input_w * scale);
            int new_h = int(input_h * scale);

            preprocess::warpaffine_init(input_h, input_w, target_h, target_w);

            cv::Mat tar(target_w, target_h, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::Mat resized_img;
            cv::resize(m_inputImage, resized_img, cv::Size(new_w, new_h));

            int x, y;
            x = (new_w < target_w) ? (target_w - new_w) / 2 : 0;
            y = (new_h < target_h) ? (target_h - new_h) / 2 : 0;

            cv::Rect roi(x, y, new_w, new_h);

            cv::Mat roiOfTar = tar(roi);
            resized_img.copyTo(roiOfTar);

            int index;
            int offset_ch0 = m_imgArea * 0;
            int offset_ch1 = m_imgArea * 1;
            int offset_ch2 = m_imgArea * 2;
            for (int i = 0; i < m_inputDims.d[2]; i++)
            {
                for (int j = 0; j < m_inputDims.d[3]; j++)
                {
                    index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
                    m_inputMemory[0][offset_ch2++] = tar.data[index + 0] / 255.0f;
                    m_inputMemory[0][offset_ch1++] = tar.data[index + 1] / 255.0f;
                    m_inputMemory[0][offset_ch0++] = tar.data[index + 2] / 255.0f;
                }
            }

            CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

            m_timer->stop_cpu<timer::Timer::ms>("preprocess(CPU)");
            return true;
        }

        bool Pose::preprocess_gpu(const cv::Mat &img)
        {
            m_timer->start_gpu();
            m_inputImage = img;
            if (m_inputImage.data == nullptr)
            {
                LOGE("ERROR: file not founded! Program terminated");
                return false;
            }
            m_timer->stop_gpu("preprocess(clone)");

            m_timer->start_gpu();

            preprocess::preprocess_resize_gpu(m_inputImage, m_inputMemory[1],
                                              m_params->img.h, m_params->img.w,
                                              preprocess::tactics::GPU_WARP_AFFINE, m_stream);

            m_timer->stop_gpu("preprocess(GPU)");
            return true;
        }

        void Pose::show(string path)
        {
            cv::Mat vis = m_inputImage.clone();
            float kpt_conf_threshold = 0.5f;
            for (const auto &box : m_bboxes)
            {
                int x0 = static_cast<int>(box.x0);
                int y0 = static_cast<int>(box.y0);
                int x1 = static_cast<int>(box.x1);
                int y1 = static_cast<int>(box.y1);
                cv::rectangle(vis, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 0, 255), 2);
                for (int k = 0; k < box.keypoints.size(); ++k)
                {
                    const auto &kpt = box.keypoints[k];
                    if (kpt.conf < kpt_conf_threshold)
                        continue;

                    int x = static_cast<int>(kpt.x);
                    int y = static_cast<int>(kpt.y);
                    cv::circle(vis, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), -1);
                }
            }
            cv::imwrite(path, vis);
        }

        Pose::Pose(std::string onnx_path, logger::Level level, Params params)
            : Model(onnx_path, level, params)
        {
            //初始化模式为红色(远距离)
            m_use_red_mode = true;

            m_result.resize(6, 0.0f);
            uart_result.resize(3, 0.0f);
            //  一飞院
            _K = (cv::Mat_<double>(3, 3) << 1067.33054757922, 0.0, 949.935792304770,
                  0.0, 1067.37400981335, 525.523361276358,
                  0.0, 0.0, 1.0);

            _diff = (cv::Mat_<float>(1, 5) << -0.0898188725781947, 0.0570779357792198, 0, 0, 0.0421749060858686);

            // new -12
            // _p3d = (cv::Mat_<double>(7, 3) << -253.95, -10.78, -2.17,
            //         -95.4, -6.84, -0.9,
            //         -148.72, 364.33, 58.67,
            //         -118.91, 208.11, 52.76,
            //         -102.54, -226.07, 42.13,
            //         -120.56, -379.84, 46.283,
            //         92.51, -3.9, 4.07);

            //  _p3d = (cv::Mat_<double>(7, 3) << -253.95,	-10.78,	-7.17,
            //             -95.4,	-6.84,	-5.9,
            //             -148.72,	364.33,	53.67,
            //             -118.91,	208.11,	47.76,
            //             -102.54,	-226.07,	37.13,
            //             -120.5,	-379.84,	41.29,
            //             92.51,	-3.9,	-1.07);

            // 0123
            // _p3d = (cv::Mat_<double>(7, 3) << -251.696, -10.28, -10.043,
            //         -93.829, -7.33, -9.62,
            //         -151.899, 363.404, 48.299,
            //         -119.014, 207.414, 42.926,
            //         -100.598, -224.46, 34.07,
            //         -118.778, -379.29, 40.951,
            //         92.84, 2.75, -7.008);

            // 0128
            _p3d = (cv::Mat_<double>(7, 3) << -254.768, -10.362, -12.31,
                    -96.061, -6.026, -11.411,
                    -146.778, 363.68, 45.367,
                    -118.098, 208.233, 40.866,
                    -102.169, -224.97, 34.07,
                    -120.608, -378.954, 40.523,
                    93.057, 1.1, -7.177);

            // old
            // _p3d = (cv::Mat_<double>(7, 3) << -255.41,	-5.3,	-10.9,
            //                                 -97.45,	-4.11,	-11.34,
            //                                 -141.12,	366.99,	49.88,
            //                                 -113.87,	212.02,	44.2,
            //                                 -107.09,	-221.28,	33.62,
            //                                 -128.86,	-374.6, 35.37,
            //                                 93.18,	2.06,	-7.96);

            combined = (cv::Mat_<float>(4, 4) << -4.7331553e-02, -6.4462757e-01, 7.6303029e-01, 1.4811254e+03,
                        9.9347848e-01, 4.8947793e-02, 1.0297883e-01, -8.0326591e+01,
                        -1.0373164e-01, 7.6292819e-01, 6.3810676e-01, 1.3706354e+02,
                        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00);
            combined_inv = combined.inv();

            LSTMPredictor::Config lstm_cfg;
            lstm_cfg.onnx_path = "models/onnx/0409.onnx"; // 【注意】这里填你LSTM模型的路径
            lstm_cfg.input_seq_len = 61;
            lstm_cfg.output_seq_len = 27;
            lstm_cfg.target_frame_idx = 25; // 取第23帧
            m_lstm = std::make_shared<LSTMPredictor>(lstm_cfg, level);
            if (m_lstm->init())
            {
                LOG("LSTM Engine initialized successfully.");
                m_lstm_ready = true;
            }
            else
            {
                LOGE("Failed to initialize LSTM Engine.");
                m_lstm_ready = false;
            }
        }

        bool Pose::postprocess_cpu(const uint64_t &timestamp)
        {

            m_timer->start_cpu();
            int output_size = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0], m_outputMemory[1], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
            CUDA_CHECK(cudaStreamSynchronize(m_stream));

            float conf_threshold = 0.25;
            float nms_threshold = 0.45;
            float kpt_conf_threshold = 0.5f;

            int boxes_count = m_outputDims.d[1];
            int dim_kpts = NUM_KEYPOINTS * 3;
            int class_count = m_outputDims.d[2] - 4 - dim_kpts;
            float *tensor;

            float cx, cy, w, h, obj, prob, conf;
            float x0, y0, x1, y1, u, v, kconf;
            int label;

            for (int i = 0; i < boxes_count; i++)
            {
                tensor = m_outputMemory[0] + i * m_outputDims.d[2];
                label = max_element(tensor + 4, tensor + 4 + class_count) - (tensor + 4);
                conf = tensor[4 + label];
                if (conf < conf_threshold)
                    continue;

                cx = tensor[0];
                cy = tensor[1];
                w = tensor[2];
                h = tensor[3];

                x0 = cx - w / 2;
                y0 = cy - h / 2;
                x1 = x0 + w;
                y1 = y0 + h;
                preprocess::affine_transformation(preprocess::affine_matrix.reverse, x0, y0, &x0, &y0);
                preprocess::affine_transformation(preprocess::affine_matrix.reverse, x1, y1, &x1, &y1);

                vector<keypoint> keypoints;
                keypoints.reserve(NUM_KEYPOINTS);

                int Keypoint_start = 4 + class_count;
                for (int i = 0; i < NUM_KEYPOINTS; ++i)
                {
                    u = tensor[Keypoint_start + i * 3];
                    v = tensor[Keypoint_start + i * 3 + 1];
                    kconf = tensor[Keypoint_start + i * 3 + 2];
                    preprocess::affine_transformation(preprocess::affine_matrix.reverse, u, v, &u, &v);
                    if (kconf >= kpt_conf_threshold)
                    {
                        keypoints.emplace_back(u, v, kconf);
                    }
                    else
                    {
                        keypoints.emplace_back(u, v, 0.0f);
                    }
                }
                bbox pose_box(x0, y0, x1, y1, conf, label);
                pose_box.keypoints = std::move(keypoints);
                m_bboxes.emplace_back(std::move(pose_box));
            }
            LOGD("the count of decoded bbox is %d", m_bboxes.size());

            vector<bbox> final_bboxes;
            final_bboxes.reserve(m_bboxes.size());
            std::sort(m_bboxes.begin(), m_bboxes.end(),
                      [](bbox &box1, bbox &box2)
                      { return box1.confidence > box2.confidence; });

            for (int i = 0; i < m_bboxes.size(); i++)
            {
                if (m_bboxes[i].flg_remove)
                    continue;

                final_bboxes.emplace_back(m_bboxes[i]);
                for (int j = i + 1; j < m_bboxes.size(); j++)
                {
                    if (m_bboxes[j].flg_remove)
                        continue;

                    if (m_bboxes[i].label == m_bboxes[j].label)
                    {
                        if (iou_calc(m_bboxes[i], m_bboxes[j]) > nms_threshold)
                            m_bboxes[j].flg_remove = true;
                    }
                }
            }
            LOGD("the count of bbox after NMS is %d", final_bboxes.size());
            m_bboxes = final_bboxes;

            m_frame_counter++;
            if (!m_bboxes.empty())
            {
                refine_keypoints(m_bboxes[0].keypoints);
            }
            run_pnp_multi_stage();
            run_filter_and_estimation(timestamp, m_frame_counter);
            run_lstm_predictin();

            m_timer->stop_cpu<timer::Timer::ms>("postprocess(CPU)");
            m_timer->show();
            return true;
        }

        bool Pose::postprocess_gpu(const uint64_t &timestamp)
        {
            return postprocess_cpu(timestamp);
        }

        shared_ptr<Pose> make_pose(
            std::string onnx_path, logger::Level level, Params params)
        {
            return make_shared<Pose>(onnx_path, level, params);
        }

        void Pose::refine_keypoints(std::vector<keypoint> &keypoints)
        {
            // [修改] -------------------- 滞回区间逻辑开始 --------------------
            // float dist_enter_red = 2200.0f; // 大于此值切入红色模式
            // float dist_enter_blue = 2160.0f; // 小于此值切回蓝色模式

            float dist_enter_red = 1600.0f; // 大于此值切入红色模式
            float dist_enter_blue = 0.0f;   // 小于此值切回蓝色模式

            if (m_use_red_mode)
            {
                // 当前是红色，检测是否足够近以切换回蓝色
                if (m_result[5] < dist_enter_blue)
                {
                    m_use_red_mode = false;
                }
            }
            else
            {
                // 当前是蓝色，检测是否足够远以切换回红色
                if (m_result[5] > dist_enter_red)
                {
                    m_use_red_mode = true;
                }
            }
            // [修改] -------------------- 滞回区间逻辑结束 --------------------

            // 使用 m_use_red_mode 标志位替代原来的 hard code 判断
            if (!m_use_red_mode)
            {
                // ================== [蓝色通道模式 / 近距离] ==================
                for (auto &kpt : keypoints)
                {
                    if (kpt.conf < 0.75f)
                    {
                        kpt.x = 0.0f;
                        kpt.y = 0.0f;
                        kpt.conf = 0.0f;
                        continue;
                    }
                    int cx = static_cast<int>(kpt.x);
                    int cy = static_cast<int>(kpt.y);

                    int search_side = 28;
                    int half_side = search_side / 2;

                    int x1 = std::max(0, cx - half_side);
                    int y1 = std::max(0, cy - half_side);
                    int x2 = std::min(m_inputImage.cols, x1 + search_side);
                    int y2 = std::min(m_inputImage.rows, y1 + search_side);

                    if (x2 - x1 < 5 || y2 - y1 < 5)
                    {
                        continue;
                    }

                    cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
                    cv::Mat roi = m_inputImage(roi_rect);

                    // 颜色对比提取 (蓝色通道)
                    std::vector<cv::Mat> channels;
                    cv::split(roi, channels);
                    cv::Mat target_img = channels[0]; // B通道

                    cv::Mat mask;
                    cv::threshold(target_img, mask, 130, 255, cv::THRESH_BINARY); // 140 best 一飞院
                    // cv::threshold(target_img, mask, 200, 255, cv::THRESH_BINARY); // 140 best 凯丽

                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    if (contours.empty())
                    {
                        continue; // 直接跳过，kpt 保持原值不变
                    }

                    // 计算相对坐标中心
                    cv::Point2f center_ref((float)(cx - x1), (float)(cy - y1));

                    int best_idx = -1;
                    double max_area = 0;
                    const double DIST_LIMIT = 5.4; // 一飞院
                    // const double DIST_LIMIT = 12;//凯丽

                    for (size_t i = 0; i < contours.size(); ++i)
                    {
                        double area = cv::contourArea(contours[i]);
                        // 一飞院
                        if (area < 2 || area > 110.0)
                            continue;
                        // 凯丽
                        //  if (area < 10 || area > 160.0)
                        //      continue;

                        cv::Moments M = cv::moments(contours[i]);
                        if (M.m00 <= 0)
                            continue;

                        float gx = static_cast<float>(M.m10 / M.m00);
                        float gy = static_cast<float>(M.m01 / M.m00);

                        double dx = gx - center_ref.x;
                        double dy = gy - center_ref.y;
                        double dist = std::sqrt(dx * dx + dy * dy);

                        // 距离过滤
                        if (dist > DIST_LIMIT)
                            continue;

                        // 择优
                        if (area > max_area)
                        {
                            max_area = area;
                            best_idx = i;
                        }
                    }

                    if (best_idx == -1)
                    {
                        // kpt.x = 0.0f;
                        // kpt.y = 0.0f;
                        // kpt.conf = 0.0f; // 认为检测到的点无效（可能是背景噪点），置零
                        continue;
                    }

                    // 计算最终坐标
                    cv::Moments M = cv::moments(contours[best_idx]);
                    float final_gx = static_cast<float>(M.m10 / M.m00);
                    float final_gy = static_cast<float>(M.m01 / M.m00);

                    // 还原到全图坐标
                    float final_x = x1 + final_gx + 0.5f; // 尝试
                    float final_y = y1 + final_gy + 0.5f;

                    double final_dist = std::sqrt(std::pow(final_x - kpt.x, 2) + std::pow(final_y - kpt.y, 2));

                    if (final_dist > DIST_LIMIT)
                    {
                        kpt.x = 0.0f;
                        kpt.y = 0.0f;
                        kpt.conf = 0.0f; // 最终计算结果偏离太大，置零
                        continue;
                    }

                    kpt.x = final_x;
                    kpt.y = final_y;
                }
            }
            else
            {
                // ================== [红色 HSV 模式 / 远距离] ==================
                for (size_t i = 0; i < keypoints.size(); ++i)
                {
                    auto &kpt = keypoints[i];
                    // if (i == 6)
                    // {
                    //     kpt.conf = 0.0f;
                    //     kpt.x = 0.0f; // 建议同时清空坐标，防止误用
                    //     kpt.y = 0.0f;
                    // }

                    if (kpt.conf < 0.75f)
                        continue;

                    int cx = static_cast<int>(kpt.x);
                    int cy = static_cast<int>(kpt.y);

                    // 1. 取 ROI
                    const int search_side = 48;
                    const int half = search_side / 2;

                    int x1 = std::max(0, cx - half);
                    int y1 = std::max(0, cy - half);
                    int x2 = std::min(m_inputImage.cols, x1 + search_side);
                    int y2 = std::min(m_inputImage.rows, y1 + search_side);

                    if (x2 - x1 < 10 || y2 - y1 < 10)
                        continue;

                    cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
                    cv::Mat roi = m_inputImage(roi_rect);

                    // 2. HSV 空间提取红色贴纸
                    cv::Mat hsv;
                    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

                    cv::Mat mask1, mask2, red_mask;
                    cv::inRange(hsv, cv::Scalar(0, 80, 80),
                                cv::Scalar(15, 255, 255), mask1);
                    cv::inRange(hsv, cv::Scalar(160, 80, 80),
                                cv::Scalar(180, 255, 255), mask2);

                    red_mask = mask1 | mask2;

                    // 去一点噪
                    cv::Mat kernel = cv::getStructuringElement(
                        cv::MORPH_ELLIPSE, {3, 3});
                    cv::morphologyEx(red_mask, red_mask,
                                     cv::MORPH_OPEN, kernel);

                    // 3. 找红色轮廓
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(red_mask, contours,
                                     cv::RETR_EXTERNAL,
                                     cv::CHAIN_APPROX_SIMPLE);

                    if (contours.empty())
                        continue; // 回退到 YOLO 点

                    int best_idx = -1;
                    double max_area = 0.0;

                    for (size_t i = 0; i < contours.size(); ++i)
                    {
                        double area = cv::contourArea(contours[i]);
                        if (area < 30.0 || area > 1000.0)
                        {
                            // kpt.x = 0;
                            // kpt.y = 0;
                            // kpt.conf = 0;
                            continue;
                        }

                        if (area > max_area)
                        {
                            max_area = area;
                            best_idx = static_cast<int>(i);
                        }
                    }

                    if (best_idx == -1)
                    {
                        kpt.x = 0.0f;
                        kpt.y = 0.0f;
                        kpt.conf = 0.0f;
                        continue;
                    }

                    // 4. 红色贴纸质心（anchor）
                    cv::Moments M = cv::moments(contours[best_idx]);
                    if (M.m00 <= 0)
                        continue;

                    float rx = static_cast<float>(M.m10 / M.m00);
                    float ry = static_cast<float>(M.m01 / M.m00);

                    float final_x = x1 + rx + 0.5f;
                    float final_y = y1 + ry + 0.5f;

                    // 5. 距离 sanity check
                    const double DIST_LIMIT = 13.0;
                    double dist = std::hypot(final_x - kpt.x,
                                             final_y - kpt.y);

                    if (dist > DIST_LIMIT)
                    {
                        kpt.x = 0.0f;
                        kpt.y = 0.0f;
                        kpt.conf = 0.0f;
                        continue; // 回退到 YOLO 点
                    }

                    // 6. 更新 keypoint
                    kpt.x = final_x;
                    kpt.y = final_y;
                }
            }
        }

        void Pose::run_pnp_multi_stage()
        {
            is_current_frame_good = false; // 重置标记位
            if (m_bboxes.size() >= 1)
            {
                auto &target = m_bboxes[0];
                cv::Mat p3d_Mat = cv::Mat::zeros(7, 3, CV_64FC1);
                cv::Mat p2d_Mat = cv::Mat::zeros(7, 2, CV_64FC1);
                int valid_count = 0;
                for (int i = 0; i < 7; i++)
                {
                    if (target.keypoints[i].conf > 0.75)
                    {
                        p2d_Mat.at<double>(valid_count, 0) = target.keypoints[i].x;
                        p2d_Mat.at<double>(valid_count, 1) = target.keypoints[i].y;
                        _p3d.row(i).copyTo(p3d_Mat.row(valid_count));
                        valid_count++;
                    }
                }
                p3d_Mat.resize(valid_count);
                p2d_Mat.resize(valid_count);

                if (p3d_Mat.rows >= 4)
                {
                    std::vector<int> inliers;
                    bool success = cv::solvePnPRansac(p3d_Mat, p2d_Mat, _K, _diff, R1, T1,
                                                      false, 100, 2.0, 0.99, inliers, cv::SOLVEPNP_SQPNP);

                    // 4. 验证解算质量
                    if (success && inliers.size() >= 4)
                    {
                        is_current_frame_good = true;
                        R1.copyTo(_R1_prev);
                        T1.copyTo(_T1_prev);
                    }
                }
            }
            if (is_current_frame_good && !T1.empty())
            {
                m_result[3] = static_cast<float>(T1.at<double>(0, 0));
                m_result[4] = static_cast<float>(T1.at<double>(1, 0));
                m_result[5] = static_cast<float>(T1.at<double>(2, 0));
            }
        }

        void Pose::run_filter_and_estimation(const uint64_t &timestamp, uint64_t frame_id)
        {
            double dt = 0.0;
            if (_last_timestamp != 0)
            {
                dt = static_cast<double>(timestamp - _last_timestamp) / 1000.0;
            }
            _last_timestamp = timestamp;
            if (dt <= 0.0)
            {
                dt = 0.033;
            }
            std::cout << "时间间隔 :" << dt << std::endl;
            cv::Point3f predicted_pos = m_kf.predict(dt); // 先验估计，预测值
            cv::Point3f kf_result;
            if (!_T1_prev.empty())
            {
                if (is_current_frame_good)
                {
                    kf_result = m_kf.update(m_result[3], m_result[4], m_result[5]);
                }
                else
                {
                    kf_result = predicted_pos;
                }

                // 4. 相机坐标系下滤波后的结果
                // m_result[3] = kf_result.x;
                // m_result[4] = kf_result.y;
                // m_result[5] = kf_result.z;

                // double angle = -0.18;
                double angle = 0;
                cv::Mat vec = (cv::Mat_<float>(3, 1) << kf_result.x, kf_result.y, kf_result.z);
                double rad = deg2rad(angle); // 转换为弧度
                double cos_theta = cos(rad);
                double sin_theta = sin(rad);
                cv::Mat rot_mat = (cv::Mat_<float>(3, 3) << 1, 0, 0,
                                   0, cos_theta, -sin_theta,
                                   0, sin_theta, cos_theta);
                cv::Mat result;
                cv::gemm(rot_mat, vec, 1.0, cv::Mat(), 0.0, result);
                m_result[3] = result.at<float>(0, 0);
                m_result[4] = result.at<float>(1, 0);
                m_result[5] = result.at<float>(2, 0);
                LOG("\tId: %d, [Filter] Ref(Past): x:%.4f, y:%.4f, z:%.4f | Curr(KF): x:%.4f, y:%.4f, z:%.4f",
                    frame_id, m_result[0], m_result[1], m_result[2], m_result[3], m_result[4], m_result[5]);
            }
        }
        void Pose::run_lstm_predictin()
        {
            // 3. === LSTM 推理 ===
            // 逻辑：将 KF 滤波后的平滑数据喂给 LSTM
            if (m_lstm_ready)
            {
                // 推入当前帧 KF 结果，尝试获取未来预测
                // 只有当积累了 58 帧后，update 才会返回 true
                if (m_lstm->update(m_result[3], m_result[4], m_result[5]))
                {
                    std::vector<float> lstm_out = m_lstm->get_prediction();

                    // 【策略选择】
                    m_result[0] = lstm_out[0];
                    m_result[1] = lstm_out[1];
                    m_result[2] = lstm_out[2];

                    // m_result[0] = m_result[3];
                    // m_result[1] = m_result[4];
                    // m_result[2] = m_result[5];

                    // LOGD("LSTM Active: x:%.2f, y:%.2f, z:%.2f", final_x, final_y, final_z);
                }
                else
                {
                    // m_result[0] = 0.0f;
                    // m_result[1] = 0.0f;
                    // m_result[2] = 0.0f;

                    m_result[0] = m_result[3];
                    m_result[1] = m_result[4];
                    m_result[2] = m_result[5];

                    // LOGV("LSTM warming up...");
                }
            }

            cv::Mat point_homogeneous = (cv::Mat_<float>(4, 1) << m_result[0],
                                         m_result[1],
                                         m_result[2],
                                         1.0); // 现在是直接把观测值通过串口发出去，发预测值改0 1 2

            cv::Mat transformed_point = combined * point_homogeneous;

            uart_result[0] = transformed_point.at<float>(0, 0); // 新的 X
            uart_result[1] = transformed_point.at<float>(1, 0); // 新的 Y
            uart_result[2] = transformed_point.at<float>(2, 0); // 新的 Z

            // uart_result[0] = m_result[3]; // 新的 X
            // uart_result[1] = m_result[4]; // 新的 Y
            // uart_result[2] = m_result[5]; // 新的 Z

            LOG("\t wxj: x:%.4f, y:%.4f, z:%.4f",
                uart_result[0], uart_result[1], uart_result[2]);

            // LOG("\t [Filter] Ref(Past): x:%.4f, y:%.4f, z:%.4f | Curr(KF): x:%.4f, y:%.4f, z:%.4f",
            //     m_result[0], m_result[1], m_result[2], m_result[3], m_result[4], m_result[5]);
        }
    };
};
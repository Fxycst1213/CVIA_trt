#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <algorithm>
#include <cmath>
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

            if (m_outputDims.nbDims != 3 || m_outputDims.d[0] != 1 ||
                m_outputDims.d[1] <= 0 || m_outputDims.d[2] != 4 + NUM_CLASSES + NUM_KEYPOINTS * 3)
            {
                LOGE("Pose engine output must be [1, boxes, %d] (%d class, %d keypoints). "
                     "Use x_plane1.onnx and its matching engine cache.",
                     4 + NUM_CLASSES + NUM_KEYPOINTS * 3, NUM_CLASSES, NUM_KEYPOINTS);
            }
            LOG("Pose model: %d keypoints, %d classes.", NUM_KEYPOINTS,
                static_cast<int>(m_outputDims.d[2] - 4 - NUM_KEYPOINTS * 3));

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
            m_result.resize(6, 0.0f);
            uart_result.resize(3,0.0f);

            if (params.keypoints_only)
            {
                uart_result.clear();
                LOGW("2D-only mode: PnP and LSTM are disabled.");
                return;
            }

            //  一飞院
            // _K = (cv::Mat_<double>(3, 3) << 1067.33054757922, 0.0, 949.935792304770,
            //       0.0, 1067.37400981335, 525.523361276358,
            //       0.0, 0.0, 1.0);

            // _diff = (cv::Mat_<float>(1, 5) << -0.0898188725781947, 0.0570779357792198, 0, 0, 0.0421749060858686);
            
            _K = (cv::Mat_<double>(3, 3) << 1075.46954757922, 0.0, 969.05085792304770,
                  0.0, 1081.0180981335, 539.313361276358,
                  0.0, 0.0, 1.0);

            _diff = (cv::Mat_<float>(1, 5) << -0.0909, 0.1061, -0.0014, -0.0011, -0.0332);
            static_assert(NUM_KEYPOINTS == 11, "The x_plane1 CAD model requires nine keypoints");
            _p3d = (cv::Mat_<double>(NUM_KEYPOINTS, 3) <<
                    -1298.178555, 390.302125, -146.34303, // 0: right tail
                    -1320.014862, -130.679327, -115.960107,// 1: middle tail
                    -1327.385123, -341.073521, -106.967758, // 2: left tail
                    -529.742539, 1606.000809, 77.685003, // 3: right wing, outer
                    -388.164682, 1116.515352, 88.559047, // 4: right wing, middle
                    -268.51011, 419.768867, 103.807179, // 5: right wing, inner
                    -337.448481, -413.907372, 150.899049, // 6: left wing, inner
                    -519.734637, -1087.041527, 203.050124, // 7: left wing, outer
                    340.730021, -13.814498, 47.352457,
                    -519.734637, -1087.041527, 203.050124, // 7: left wing, outer
                    340.730021, -13.814498, 47.352457); // 8: nose

            combined = (cv::Mat_<float>(4, 4) <<
                             0.01303241,   0.69184523,  0.7219282,   200.5658,
                             0.99928314,   0.01665291,   -0.03399826,   -72.3532,
                             -0.03554375,  0.72185376,  -0.69113225,  -799.5453,
                             0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0f
                            );
            combined_inv = combined.inv();

            if (has_pnp_model())
            {
                LOG("PnP enabled: %d 3D points in x_plane1 order, wingspan 3445 mm; "
                    "body axes X=nose, Y=right, Z=down; origin at excluded central blue marker.",
                    _p3d.rows);
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
            else
            {
                uart_result.clear();
                LOGW("Model has %d keypoints but the PnP model has %d 3D points. "
                     "Sending 2D detections only; PnP, pose prediction, and serial pose output are disabled.",
                     NUM_KEYPOINTS, _p3d.rows);
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
            m_raw_keypoints.clear();
            if (!m_bboxes.empty())
                m_raw_keypoints = m_bboxes[0].keypoints;

            m_frame_counter++;
            // 2D 点精修：直接修改后续发送的 m_bboxes，不依赖 PnP 是否启用。
            if (!m_bboxes.empty())
                refine_keypoints(m_bboxes[0].keypoints, timestamp);
            else
                m_keypoint_refiner.reset();

            if (has_pnp_model())
            {
                run_pnp_multi_stage();
                // run_filter_and_estimation(timestamp, m_frame_counter);
                cv::Mat point_homogeneous = (cv::Mat_<float>(4, 1) << m_result[3],
                                             m_result[4],
                                             m_result[5],
                                             1.0f);
                cv::Mat transformed_point = combined * point_homogeneous;

                for (int i = 0; i < 3; ++i)
                    m_robot_base_xyz[i] = transformed_point.at<float>(i, 0);

                uart_result[0] = transformed_point.at<float>(0, 0); // 万向节 X
                uart_result[1] = transformed_point.at<float>(1, 0); // 万向节 Y
                uart_result[2] = transformed_point.at<float>(2, 0); // 万向节 Z
            }
            else
            {
                is_current_frame_good = false;
                std::fill(m_result.begin(), m_result.end(), 0.0f);
                uart_result.clear();
            }
            
            if (uart_result.size() >= 3)
                LOG("\t wxj: x:%.4f, y:%.4f, z:%.4f",
                    uart_result[0], uart_result[1], uart_result[2]);

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

        void Pose::refine_keypoints(std::vector<keypoint> &keypoints, uint64_t timestamp_ms)
        {
            if (m_bboxes.empty())
            {
                m_keypoint_refiner.reset();
                return;
            }
            const auto &target = m_bboxes[0];
            const cv::Rect2f box(target.x0, target.y0, target.x1 - target.x0, target.y1 - target.y0);
            m_keypoint_refiner.refine(m_inputImage, keypoints, box, timestamp_ms);
        }

        void Pose::set_camera_calibration(const CameraCalibration &calibration)
        {
            const auto &k = calibration.intrinsics;
            if (k[0] <= 0 || k[1] <= 0 || !calibration.width || !calibration.height)
                throw std::invalid_argument("Invalid camera calibration");
            _K = (cv::Mat_<double>(3, 3) << k[0], 0, k[2], 0, k[1], k[3], 0, 0, 1);
            _diff = cv::Mat(1, 12, CV_64F);
            std::copy(calibration.distortion.begin(), calibration.distortion.end(), _diff.ptr<double>());
        }

        void Pose::copy_pnp_pose(Resultframe &frame) const
        {
            frame.pnp_valid = is_current_frame_good && R1.total() == 3 && T1.total() == 3;
            frame.pnp_rvec.fill(0);
            frame.pnp_tvec.fill(0);
            if (frame.pnp_valid)
                for (int i = 0; i < 3; ++i)
                {
                    frame.pnp_rvec[i] = static_cast<float>(R1.at<double>(i));
                    frame.pnp_tvec[i] = static_cast<float>(T1.at<double>(i));
                }
            // Use the same transformed point as the serial path, before its axis/offset mapping.
            frame.robot_base_valid = has_pnp_model() && !_T1_prev.empty() &&
                std::all_of(m_robot_base_xyz.begin(), m_robot_base_xyz.end(),
                            [](float value) { return std::isfinite(value); });
            frame.robot_base_xyz = frame.robot_base_valid ? m_robot_base_xyz : std::array<float, 3>{};
        }

        void Pose::run_pnp_multi_stage()
        {
            is_current_frame_good = false; // 重置标记位
            if (!has_pnp_model())
                return;
            if (m_bboxes.size() >= 1)
            {
                auto &target = m_bboxes[0];
                if (target.keypoints.size() != NUM_KEYPOINTS)
                    return;
                cv::Mat p3d_Mat = cv::Mat::zeros(NUM_KEYPOINTS, 3, CV_64FC1);
                cv::Mat p2d_Mat = cv::Mat::zeros(NUM_KEYPOINTS, 2, CV_64FC1);
                int valid_count = 0;
                for (int i = 0; i < NUM_KEYPOINTS; i++)
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
                        // OpenCV returns camera_point = Rodrigues(R1) * body_point + T1.
                        // T1 locates the body origin in camera coordinates, in millimeters.
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
            LOGV("Frame interval: %.6f s", dt);
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
                m_result[3] = kf_result.x;
                m_result[4] = kf_result.y;
                m_result[5] = kf_result.z;

                //修手眼矩阵 

                // double angle = -0.18;
                // cv::Mat vec = (cv::Mat_<float>(3, 1) << kf_result.x, kf_result.y, kf_result.z);
                // double rad = deg2rad(angle); // 转换为弧度
                // double cos_theta = cos(rad);
                // double sin_theta = sin(rad);
                // cv::Mat rot_mat = (cv::Mat_<float>(3, 3) << 1, 0, 0,
                //                    0, cos_theta, -sin_theta,
                //                    0, sin_theta, cos_theta);
                // cv::Mat result;
                // cv::gemm(rot_mat, vec, 1.0, cv::Mat(), 0.0, result);
                // m_result[3] = result.at<float>(0, 0);
                // m_result[4] = result.at<float>(1, 0);
                // m_result[5] = result.at<float>(2, 0);

                LOG("\tId: %d, [Filter] Ref(Past): x:%.4f, y:%.4f, z:%.4f | Curr(KF): x:%.4f, y:%.4f, z:%.4f",
                    frame_id, m_result[0], m_result[1], m_result[2], m_result[3], m_result[4], m_result[5]);
            }
        }

    };
};

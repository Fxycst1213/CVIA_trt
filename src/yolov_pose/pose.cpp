#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <algorithm>
#include <chrono>
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
// 未来位置预测已停用；卡尔曼滤波仍由 TrajectoryKF 保留。
// #include "../lstm/lstm_predictor.hpp"

using namespace std;
using namespace nvinfer1;

namespace
{
    cv::Vec3d rotationMatrixToEulerRxRyRzDegrees(const cv::Mat &rotation_matrix)
    {
        CV_Assert(rotation_matrix.rows == 3 && rotation_matrix.cols == 3);

        cv::Mat R;
        rotation_matrix.convertTo(R, CV_64F);

        const double r00 = R.at<double>(0, 0);
        const double r01 = R.at<double>(0, 1);
        const double r10 = R.at<double>(1, 0);
        const double r11 = R.at<double>(1, 1);
        const double r20 = R.at<double>(2, 0);
        const double r21 = R.at<double>(2, 1);
        const double r22 = R.at<double>(2, 2);

        // rx -> ry -> rz 顺序施加旋转，对列向量等价于 R = Rz * Ry * Rx。
        const double sy = std::hypot(r00, r10);
        const bool singular = sy < 1e-6;

        double rx = 0.0;
        double ry = 0.0;
        double rz = 0.0;

        if (!singular)
        {
            rx = std::atan2(r21, r22);
            ry = std::atan2(-r20, sy);
            rz = std::atan2(r10, r00);
        }
        else
        {
            rx = 0.0;
            ry = std::atan2(-r20, sy);
            rz = std::atan2(-r01, r11);
        }

        const double rad_to_deg = 180.0 / CV_PI;
        return cv::Vec3d(rx * rad_to_deg, ry * rad_to_deg, rz * rad_to_deg);
    }

    cv::Mat transformCameraPoseToMocap(const cv::Mat &mocap_from_camera,
                                       const cv::Mat &object_to_camera_rotation,
                                       const cv::Vec3f &object_position_in_camera)
    {
        CV_Assert(mocap_from_camera.rows == 4 && mocap_from_camera.cols == 4);
        CV_Assert(object_to_camera_rotation.rows == 3 && object_to_camera_rotation.cols == 3);

        cv::Mat mocap_from_camera_64;
        cv::Mat object_to_camera_rotation_64;
        mocap_from_camera.convertTo(mocap_from_camera_64, CV_64F);
        object_to_camera_rotation.convertTo(object_to_camera_rotation_64, CV_64F);

        // solvePnP 给出 T_C_O；calibration.extrinsic 是 T_M_C。
        // 对完整位姿左乘得到 T_M_O，位置和姿态一起进入动捕坐标系。
        cv::Mat object_to_camera = cv::Mat::eye(4, 4, CV_64F);
        object_to_camera_rotation_64.copyTo(object_to_camera(cv::Rect(0, 0, 3, 3)));
        object_to_camera.at<double>(0, 3) = object_position_in_camera[0];
        object_to_camera.at<double>(1, 3) = object_position_in_camera[1];
        object_to_camera.at<double>(2, 3) = object_position_in_camera[2];

        return mocap_from_camera_64 * object_to_camera;
    }
}

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

            float inter_w = std::max(0.0f, inter_x1 - inter_x0);
            float inter_h = std::max(0.0f, inter_y1 - inter_y0);

            float inter_area = inter_w * inter_h;
            float union_area =
                (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0) +
                (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0) -
                inter_area;

            return union_area > 0.0f ? inter_area / union_area : 0.0f;
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
                m_outputDims.d[1] <= 0 || m_outputDims.d[2] <= 0)
            {
                LOGE("Unsupported pose output shape: nbDims=%d [%d,%d,%d]",
                     m_outputDims.nbDims,
                     m_outputDims.nbDims > 0 ? m_outputDims.d[0] : -1,
                     m_outputDims.nbDims > 1 ? m_outputDims.d[1] : -1,
                     m_outputDims.nbDims > 2 ? m_outputDims.d[2] : -1);
                return;
            }

            const int expected_features = 4 + POSE_CLASS_COUNT + NUM_KEYPOINTS * 3;
            const int dim1 = m_outputDims.d[1];
            const int dim2 = m_outputDims.d[2];
            if (dim1 == expected_features && dim2 != expected_features)
            {
                m_outputFeaturesFirst = true;
                m_outputFeatures = dim1;
                m_outputBoxes = dim2;
            }
            else if (dim2 == expected_features && dim1 != expected_features)
            {
                m_outputFeaturesFirst = false;
                m_outputBoxes = dim1;
                m_outputFeatures = dim2;
            }
            else
            {
                LOGE("Pose model/config mismatch: output=[1,%d,%d], expected feature dimension=%d for %d classes and %d configured 3D points",
                     dim1, dim2, expected_features, POSE_CLASS_COUNT, NUM_KEYPOINTS);
                return;
            }

            m_outputClasses = POSE_CLASS_COUNT;

            LOG("Pose output: raw=[1,%d,%d], layout=%s, boxes=%d, features=%d, classes=%d, keypoints=%d",
                dim1, dim2,
                m_outputFeaturesFirst ? "features-first" : "boxes-first",
                m_outputBoxes, m_outputFeatures, m_outputClasses, NUM_KEYPOINTS);

            CUDA_CHECK(cudaStreamCreate(&m_stream));

            m_inputSize = m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);
            m_imgArea = m_params->img.h * m_params->img.w;
            m_outputSize = m_outputBoxes * m_outputFeatures * sizeof(float);

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

            return true;
        }

        bool Pose::preprocess_gpu(const cv::Mat &img)
        {
            m_inputImage = img;
            if (m_inputImage.data == nullptr)
            {
                LOGE("ERROR: file not founded! Program terminated");
                return false;
            }
            preprocess::preprocess_resize_gpu(m_inputImage, m_inputMemory[1],
                                              m_params->img.h, m_params->img.w,
                                              preprocess::tactics::GPU_WARP_AFFINE, m_stream);

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
            //  15
            // _K = (cv::Mat_<double>(3, 3) << 1073.357, 0.0, 972.622,
            //       0.0, 1072.1579, 497.2687,
            //       0.0, 0.0, 1.0);

            // _diff = (cv::Mat_<float>(1, 5) << -0.0587, 0.1663, 0.0001541, 0.0026, -0.1357);

            // _K = (cv::Mat_<double>(3, 3) << 1067.695, 0.0, 972.357,
            //       0.0, 1068.264, 504.225,
            //       0.0, 0.0, 1.0);
            _K = (cv::Mat_<double>(3, 3) << 1078.39302318049, 0, 939.772377680377,
                                            0, 1078.53917414318, 595.175265662463,
                                            0.0, 0.0, 1.0);

            _diff = (cv::Mat_<float>(1, 5) << -0.0630336003089920, 0.187345652299030, 0, 0, -0.163375015304349);

            // 点数和坐标都来自 pose_params.hpp 的唯一配置表，避免 NUM_KEYPOINTS
            // 与 _p3d 行数分别修改后不一致。
            _p3d = cv::Mat(NUM_KEYPOINTS, 3, CV_64FC1);
            for (int index = 0; index < NUM_KEYPOINTS; ++index)
            {
                _p3d.at<double>(index, 0) = MODEL_KEYPOINTS_3D[index].x;
                _p3d.at<double>(index, 1) = MODEL_KEYPOINTS_3D[index].y;
                _p3d.at<double>(index, 2) = MODEL_KEYPOINTS_3D[index].z;
            }

            // mocap_from_camera = (cv::Mat_<float>(4, 4) << -4.7331553e-02, -6.4462757e-01, 7.6303029e-01, 1.4811254e+03,
            //             9.9347848e-01, 4.8947793e-02, 1.0297883e-01, -8.0326591e+01,
            //             -1.0373164e-01, 7.6292819e-01, 6.3810676e-01, 1.3706354e+02,
            //             0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00);
            // mocap_from_camera_inv = mocap_from_camera.inv();
            mocap_from_camera = (cv::Mat_<float>(4, 4) <<
                        0.990200045,    -0.102967953,     0.094347611, -1313.326552871,
                        -0.135454307,    -0.543632945,     0.828320802, -1868.458172769,
                        -0.034000028,    -0.832983086,    -0.55225282,  1061.160468608,
                        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00);
            mocap_from_camera_inv = mocap_from_camera.inv();

            // 未来位置预测已停用，不加载/构建 LSTM TensorRT engine。
            // m_lstm_ready = false;
        }

        void Pose::set_calibration(const std::array<double, 9> &camera_matrix,
                                   const std::array<double, 5> &distortion,
                                   const std::array<double, 16> &extrinsic)
        {
            _K = cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone();
            cv::Mat distortion64(1, 5, CV_64F, const_cast<double *>(distortion.data()));
            distortion64.convertTo(_diff, CV_32F);
            cv::Mat transform64(4, 4, CV_64F, const_cast<double *>(extrinsic.data()));
            transform64.convertTo(mocap_from_camera, CV_32F);
            mocap_from_camera_inv = mocap_from_camera.inv();
            LOG("Web calibration loaded: fx=%.3f fy=%.3f cx=%.3f cy=%.3f, extrinsic=T_M_C",
                camera_matrix[0], camera_matrix[4], camera_matrix[2], camera_matrix[5]);
        }

        bool Pose::postprocess_cpu(const uint64_t &timestamp)
        {

            if (m_outputBoxes <= 0 || m_outputFeatures <= 0 || m_outputClasses <= 0)
            {
                LOGE("Pose output layout was not initialized");
                return false;
            }

            int output_size = m_outputBoxes * m_outputFeatures * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0], m_outputMemory[1], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
            CUDA_CHECK(cudaStreamSynchronize(m_stream));

            float conf_threshold = 0.25;
            float nms_threshold = 0.45;
            float kpt_conf_threshold = 0.5f;

            const int boxes_count = m_outputBoxes;
            const int class_count = m_outputClasses;
            auto output_at = [this](int box_index, int feature_index) -> float
            {
                if (m_outputFeaturesFirst)
                    return m_outputMemory[0][feature_index * m_outputBoxes + box_index];
                return m_outputMemory[0][box_index * m_outputFeatures + feature_index];
            };

            float cx, cy, w, h, conf;
            float x0, y0, x1, y1, u, v, kconf;
            int label;

            for (int i = 0; i < boxes_count; i++)
            {
                label = 0;
                conf = output_at(i, 4);
                for (int class_index = 1; class_index < class_count; ++class_index)
                {
                    const float class_confidence = output_at(i, 4 + class_index);
                    if (class_confidence > conf)
                    {
                        conf = class_confidence;
                        label = class_index;
                    }
                }
                if (conf < conf_threshold)
                    continue;

                cx = output_at(i, 0);
                cy = output_at(i, 1);
                w = output_at(i, 2);
                h = output_at(i, 3);

                x0 = cx - w / 2;
                y0 = cy - h / 2;
                x1 = x0 + w;
                y1 = y0 + h;
                preprocess::affine_transformation(preprocess::affine_matrix.reverse, x0, y0, &x0, &y0);
                preprocess::affine_transformation(preprocess::affine_matrix.reverse, x1, y1, &x1, &y1);

                vector<keypoint> keypoints;
                keypoints.reserve(NUM_KEYPOINTS);

                int Keypoint_start = 4 + class_count;
                for (int keypoint_index = 0; keypoint_index < NUM_KEYPOINTS; ++keypoint_index)
                {
                    u = output_at(i, Keypoint_start + keypoint_index * 3);
                    v = output_at(i, Keypoint_start + keypoint_index * 3 + 1);
                    kconf = output_at(i, Keypoint_start + keypoint_index * 3 + 2);
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
            // 未来位置预测已停用；如需恢复，还需恢复头文件中的 LSTM 声明和成员。
            // run_lstm_predictin();

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

        // void Pose::refine_keypoints(std::vector<keypoint> &keypoints)
        // {
        //     // --- 核心调参区 ---
        //     const int search_side = 120;          // ROI 搜索框大小
        //     const double min_area = 60.0;         // 最小面积过滤 (防止微小噪点)
        //     const double max_area = 2500.0;       // 最大面积过滤
        //     const double dist_limit_pixel = 40.0; // 允许偏离 YOLO 初始点的最大像素距离
        //     const int half_side = search_side / 2;

        //     for (auto &kpt : keypoints)
        //     {
        //         // 1. 初步过滤低置信度的点
        //         if (kpt.conf < 0.5f)
        //         {
        //             continue;
        //         }

        //         int cx = static_cast<int>(kpt.x);
        //         int cy = static_cast<int>(kpt.y);

        //         // 2. 定义并截取 ROI
        //         int x1 = std::max(0, cx - half_side);
        //         int y1 = std::max(0, cy - half_side);
        //         int x2 = std::min(m_inputImage.cols, x1 + search_side);
        //         int y2 = std::min(m_inputImage.rows, y1 + search_side);

        //         // ROI 太小则跳过
        //         if (x2 - x1 < 10 || y2 - y1 < 10)
        //         {
        //             continue;
        //         }

        //         cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
        //         cv::Mat roi = m_inputImage(roi_rect);

        //         // 3. 预处理
        //         cv::Mat roi_gray;
        //         if (roi.channels() == 3)
        //         {
        //             cv::cvtColor(roi, roi_gray, cv::COLOR_BGR2GRAY);
        //         }
        //         else
        //         {
        //             roi_gray = roi.clone();
        //         }

        //         // 3.1 [暗场防御] 过滤掉纯黑背景的误检
        //         double min_val, max_val;
        //         cv::minMaxLoc(roi_gray, &min_val, &max_val);
        //         if (max_val < 40.0)
        //         {
        //             continue;
        //         }

        //         // 3.2 局部归一化 (抗曝光波动)
        //         cv::Mat roi_norm;
        //         cv::normalize(roi_gray, roi_norm, 0, 255, cv::NORM_MINMAX);

        //         // 3.3 轻微高斯平滑 (消除毛刺边缘，提升拟合稳定性)
        //         cv::GaussianBlur(roi_norm, roi_norm, cv::Size(3, 3), 0);

        //         // 3.4 大津法 (Otsu) 自动寻找最优分界线
        //         cv::Mat mask;
        //         cv::threshold(roi_norm, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        //         // 4. 查找轮廓
        //         std::vector<std::vector<cv::Point>> contours;
        //         cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        //         if (contours.empty())
        //         {
        //             continue;
        //         }

        //         cv::Point2f center_ref((float)(cx - x1), (float)(cy - y1));
        //         int best_idx = -1;
        //         double min_dist_to_center = std::numeric_limits<double>::max();

        //         // 5. 遍历轮廓进行择优
        //         for (size_t i = 0; i < contours.size(); ++i)
        //         {
        //             double area = cv::contourArea(contours[i]);
        //             if (area < min_area || area > max_area)
        //                 continue;

        //             cv::Moments M = cv::moments(contours[i]);
        //             if (M.m00 <= 0)
        //                 continue;

        //             float gx = static_cast<float>(M.m10 / M.m00);
        //             float gy = static_cast<float>(M.m01 / M.m00);

        //             double dx = gx - center_ref.x;
        //             double dy = gy - center_ref.y;
        //             double dist = std::sqrt(dx * dx + dy * dy);

        //             // 距离过滤
        //             if (dist > dist_limit_pixel)
        //                 continue;

        //             // 选择离 YOLO 初始点最近的高亮区域
        //             if (dist < min_dist_to_center)
        //             {
        //                 min_dist_to_center = dist;
        //                 best_idx = i;
        //             }
        //         }

        //         // 6. 使用椭圆拟合更新 Keypoint 坐标 (亚像素精度)
        //         if (best_idx != -1)
        //         {
        //             // fitEllipse 要求轮廓至少包含 5 个点
        //             if (contours[best_idx].size() >= 5)
        //             {
        //                 cv::RotatedRect fitted_ellipse = cv::fitEllipse(contours[best_idx]);

        //                 // 还原到全图坐标系 (自带亚像素浮点精度)
        //                 float final_x = x1 + fitted_ellipse.center.x;
        //                 float final_y = y1 + fitted_ellipse.center.y;

        //                 double final_dist = std::sqrt(std::pow(final_x - kpt.x, 2) + std::pow(final_y - kpt.y, 2));

        //                 if (final_dist <= dist_limit_pixel)
        //                 {
        //                     kpt.x = final_x;
        //                     kpt.y = final_y;
        //                 }
        //             }
        //             else
        //             {
        //                 // [兜底策略] 如果轮廓太小，退回使用图像矩 (Moments)
        //                 cv::Moments M = cv::moments(contours[best_idx]);
        //                 if (M.m00 > 0)
        //                 {
        //                     float final_x = x1 + static_cast<float>(M.m10 / M.m00) + 0.5f;
        //                     float final_y = y1 + static_cast<float>(M.m01 / M.m00) + 0.5f;

        //                     if (std::sqrt(std::pow(final_x - kpt.x, 2) + std::pow(final_y - kpt.y, 2)) <= dist_limit_pixel)
        //                     {
        //                         kpt.x = final_x;
        //                         kpt.y = final_y;
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        void Pose::refine_keypoints(std::vector<keypoint> &keypoints)
        {
            // --- 核心调参区 ---
            const int search_side = 120;          // ROI 搜索框大小
            const double min_area = 60.0;         // 最小面积过滤 (防止微小噪点)
            const double max_area = 2500.0;       // 最大面积过滤
            const double dist_limit_pixel = 60.0; // 允许偏离 YOLO 初始点的最大像素距离
            const int half_side = search_side / 2;

            for (auto &kpt : keypoints)
            {
                // 1. 初步过滤低置信度的点 (本来就不行，直接跳过，不用改坐标)
                if (kpt.conf < 0.5f)
                {
                    continue;
                }

                int cx = static_cast<int>(kpt.x);
                int cy = static_cast<int>(kpt.y);

                // 2. 定义并截取 ROI
                int x1 = std::max(0, cx - half_side);
                int y1 = std::max(0, cy - half_side);
                int x2 = std::min(m_inputImage.cols, x1 + search_side);
                int y2 = std::min(m_inputImage.rows, y1 + search_side);

                // [缺陷修复 1] ROI 太小（比如点在图像极边缘），无法精修，置为无效
                if (x2 - x1 < 10 || y2 - y1 < 10)
                {
                    kpt.conf = 0.0f;
                    kpt.x = 0.0f;
                    kpt.y = 0.0f;
                    continue;
                }

                cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
                cv::Mat roi = m_inputImage(roi_rect);

                // 3. 预处理
                cv::Mat roi_gray;
                if (roi.channels() == 3)
                {
                    cv::cvtColor(roi, roi_gray, cv::COLOR_BGR2GRAY);
                }
                else
                {
                    roi_gray = roi.clone();
                }

                // 3.1 [暗场防御] 过滤掉纯黑背景的误检
                // [缺陷修复 2] 全是黑底没有反光点，说明是 YOLO 瞎框的，置为无效
                double min_val, max_val;
                cv::minMaxLoc(roi_gray, &min_val, &max_val);
                if (max_val < 40.0)
                {
                    kpt.conf = 0.0f;
                    kpt.x = 0.0f;
                    kpt.y = 0.0f;
                    continue;
                }

                // 3.2 局部归一化 (抗曝光波动)
                cv::Mat roi_norm;
                cv::normalize(roi_gray, roi_norm, 0, 255, cv::NORM_MINMAX);

                // 3.3 轻微高斯平滑 (消除毛刺边缘，提升拟合稳定性)
                cv::GaussianBlur(roi_norm, roi_norm, cv::Size(3, 3), 0);

                // 3.4 大津法 (Otsu) 自动寻找最优分界线
                cv::Mat mask;
                cv::threshold(roi_norm, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

                // 4. 查找轮廓
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

                // [缺陷修复 3] 没有找到任何轮廓，置为无效
                if (contours.empty())
                {
                    kpt.conf = 0.0f;
                    kpt.x = 0.0f;
                    kpt.y = 0.0f;
                    continue;
                }

                cv::Point2f center_ref((float)(cx - x1), (float)(cy - y1));
                int best_idx = -1;
                double min_dist_to_center = std::numeric_limits<double>::max();

                // 5. 遍历轮廓进行择优
                for (size_t i = 0; i < contours.size(); ++i)
                {
                    double area = cv::contourArea(contours[i]);
                    // 注意：这里面的 continue 是跳过当前“不合格”的轮廓，去检查下一个轮廓
                    // 所以这里不能置 0，要等全部遍历完再判断
                    if (area < min_area)
                        continue;

                    cv::Moments M = cv::moments(contours[i]);
                    if (M.m00 <= 0)
                        continue;

                    float gx = static_cast<float>(M.m10 / M.m00);
                    float gy = static_cast<float>(M.m01 / M.m00);

                    double dx = gx - center_ref.x;
                    double dy = gy - center_ref.y;
                    double dist = std::sqrt(dx * dx + dy * dy);

                    // 距离过滤
                    if (dist > dist_limit_pixel)
                        continue;

                    // 选择离 YOLO 初始点最近的高亮区域
                    if (dist < min_dist_to_center)
                    {
                        min_dist_to_center = dist;
                        best_idx = i;
                    }
                }

                // 6. 使用椭圆拟合更新 Keypoint 坐标 (亚像素精度)
                if (best_idx != -1)
                {
                    bool fit_success = false; // 用于标记最终是否成功精修

                    // fitEllipse 要求轮廓至少包含 5 个点
                    if (contours[best_idx].size() >= 5)
                    {
                        cv::RotatedRect fitted_ellipse = cv::fitEllipse(contours[best_idx]);

                        // 还原到全图坐标系 (自带亚像素浮点精度)
                        float final_x = x1 + fitted_ellipse.center.x;
                        float final_y = y1 + fitted_ellipse.center.y;

                        double final_dist = std::sqrt(std::pow(final_x - kpt.x, 2) + std::pow(final_y - kpt.y, 2));

                        if (final_dist <= dist_limit_pixel)
                        {
                            kpt.x = final_x;
                            kpt.y = final_y;
                            fit_success = true; // 成功！
                        }
                    }
                    else
                    {
                        // [兜底策略] 如果轮廓太小，退回使用图像矩 (Moments)
                        cv::Moments M = cv::moments(contours[best_idx]);
                        if (M.m00 > 0)
                        {
                            float final_x = x1 + static_cast<float>(M.m10 / M.m00) + 0.5f;
                            float final_y = y1 + static_cast<float>(M.m01 / M.m00) + 0.5f;

                            if (std::sqrt(std::pow(final_x - kpt.x, 2) + std::pow(final_y - kpt.y, 2)) <= dist_limit_pixel)
                            {
                                kpt.x = final_x;
                                kpt.y = final_y;
                                fit_success = true; // 成功！
                            }
                        }
                    }

                    // [缺陷修复 4] 如果虽然找到了最好的轮廓，但是计算出的最终距离超限了
                    if (!fit_success)
                    {
                        kpt.conf = 0.0f;
                        kpt.x = 0.0f;
                        kpt.y = 0.0f;
                    }
                }
                else
                {
                    // [缺陷修复 5] 遍历了所有轮廓，没有一个满足面积和距离要求
                    kpt.conf = 0.0f;
                    kpt.x = 0.0f;
                    kpt.y = 0.0f;
                }
            }
        }

        void Pose::run_pnp_multi_stage()
        {
            is_current_frame_good = false; // 重置标记位
            m_reprojected_origin_valid = false;
            m_reprojected_points.clear();
            if (m_bboxes.size() >= 1)
            {
                auto &target = m_bboxes[0];
                const int point_count = std::min(
                    std::min(static_cast<int>(target.keypoints.size()), _p3d.rows),
                    NUM_KEYPOINTS);
                cv::Mat p3d_Mat = cv::Mat::zeros(point_count, 3, CV_64FC1);
                cv::Mat p2d_Mat = cv::Mat::zeros(point_count, 2, CV_64FC1);
                int valid_count = 0;
                for (int i = 0; i < point_count; i++)
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

                        // 刚体原点不必是特征点。单独投影 (0,0,0)，网页将它显示为
                        // 几何中心；_p3d 的模型点投影仍用于检查特征点/PnP 对齐质量。
                        const std::vector<cv::Point3d> object_origin(1, cv::Point3d(0.0, 0.0, 0.0));
                        std::vector<cv::Point2d> projected_origin;
                        cv::projectPoints(object_origin, R1, T1, _K, _diff, projected_origin);
                        if (projected_origin.size() == 1 &&
                            std::isfinite(projected_origin[0].x) &&
                            std::isfinite(projected_origin[0].y))
                        {
                            m_reprojected_origin = projected_origin[0];
                            m_reprojected_origin_valid = true;
                        }

                        cv::projectPoints(_p3d, R1, T1, _K, _diff, m_reprojected_points);

                        cv::Rodrigues(R1, R_mat);

                        cv::Mat pnp_transform = cv::Mat::eye(4, 4, CV_64F);
                        cv::Mat R64;
                        cv::Mat T64;
                        R_mat.convertTo(R64, CV_64F);
                        T1.convertTo(T64, CV_64F);
                        T64 = T64.reshape(1, 3);

                        R64.copyTo(pnp_transform(cv::Rect(0, 0, 3, 3)));
                        T64.copyTo(pnp_transform(cv::Rect(3, 0, 1, 3)));

                        cv::Vec3d euler_angles_deg =
                            rotationMatrixToEulerRxRyRzDegrees(pnp_transform(cv::Rect(0, 0, 3, 3)));

                        m_result[0] = static_cast<float>(euler_angles_deg[0]);
                        m_result[1] = static_cast<float>(euler_angles_deg[1]);
                        m_result[2] = static_cast<float>(euler_angles_deg[2]);
                        m_result[3] = static_cast<float>(pnp_transform.at<double>(0, 3));
                        m_result[4] = static_cast<float>(pnp_transform.at<double>(1, 3));
                        m_result[5] = static_cast<float>(pnp_transform.at<double>(2, 3));
                    }

                    static auto last_pnp_log_at = std::chrono::steady_clock::time_point{};
                    const auto now = std::chrono::steady_clock::now();
                    if (last_pnp_log_at.time_since_epoch().count() == 0 ||
                        now - last_pnp_log_at >= std::chrono::seconds(1))
                    {
                        LOG("Pose PnP: boxes=%d, eligible_points=%d/%d, solve=%s, inliers=%d, valid=%s",
                            static_cast<int>(m_bboxes.size()), valid_count, point_count,
                            success ? "true" : "false", static_cast<int>(inliers.size()),
                            is_current_frame_good ? "true" : "false");
                        last_pnp_log_at = now;
                    }
                }
                else
                {
                    static auto last_short_log_at = std::chrono::steady_clock::time_point{};
                    const auto now = std::chrono::steady_clock::now();
                    if (last_short_log_at.time_since_epoch().count() == 0 ||
                        now - last_short_log_at >= std::chrono::seconds(1))
                    {
                        LOGW("Pose PnP skipped: boxes=%d, eligible_points=%d/%d (need at least 4 with confidence > 0.75)",
                             static_cast<int>(m_bboxes.size()), valid_count, point_count);
                        last_short_log_at = now;
                    }
                }
            }
            else
            {
                static auto last_empty_log_at = std::chrono::steady_clock::time_point{};
                const auto now = std::chrono::steady_clock::now();
                if (last_empty_log_at.time_since_epoch().count() == 0 ||
                    now - last_empty_log_at >= std::chrono::seconds(1))
                {
                    LOGW("Pose PnP skipped: no decoded bounding box");
                    last_empty_log_at = now;
                }
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

                // 滤波只更新 T_C_O 的平移，不添加未标定的轴向修正；随后用
                // 同一个齐次变换把位置与姿态一起转换到动捕坐标系。
                const cv::Vec3f filtered_object_position_in_camera(
                    kf_result.x, kf_result.y, kf_result.z);
                const cv::Mat mocap_pose = transformCameraPoseToMocap(
                    mocap_from_camera, R_mat, filtered_object_position_in_camera);
                const cv::Vec3d mocap_euler_angles_deg =
                    rotationMatrixToEulerRxRyRzDegrees(mocap_pose(cv::Rect(0, 0, 3, 3)));

                // 所有发送/记录链路共用 m_result，顺序保持 rx, ry, rz, x, y, z。
                m_result[0] = static_cast<float>(mocap_euler_angles_deg[0]);
                m_result[1] = static_cast<float>(mocap_euler_angles_deg[1]);
                m_result[2] = static_cast<float>(mocap_euler_angles_deg[2]);
                m_result[3] = static_cast<float>(mocap_pose.at<double>(0, 3));
                m_result[4] = static_cast<float>(mocap_pose.at<double>(1, 3));
                m_result[5] = static_cast<float>(mocap_pose.at<double>(2, 3));

                // 高频逐帧日志会阻塞终端并拖慢网页预览；每秒保留一条诊断信息。
                static auto last_filter_log_at = std::chrono::steady_clock::time_point{};
                const auto now = std::chrono::steady_clock::now();
                if (last_filter_log_at.time_since_epoch().count() == 0 ||
                    now - last_filter_log_at >= std::chrono::seconds(1))
                {
                    LOG("\tId: %d, dt:%.4f, [Mocap-frame pose] rx:%.4f, ry:%.4f, rz:%.4f | x:%.4f, y:%.4f, z:%.4f",
                        frame_id, dt, m_result[0], m_result[1], m_result[2],
                        m_result[3], m_result[4], m_result[5]);
                    last_filter_log_at = now;
                }
            }
        }
        /*
         * 未来位置预测已停用。
         * 注意：此处仅注释 LSTM 预测与预测结果输出，TrajectoryKF 卡尔曼滤波代码保持不变。
         *
         * void Pose::run_lstm_predictin()
         * {
         *     // 3. === LSTM 推理 ===
         *     // 逻辑：将 KF 滤波后的平滑数据喂给 LSTM
         *     if (m_lstm_ready)
         *     {
         *         // 推入当前帧 KF 结果，尝试获取未来预测
         *         // 只有当积累了 58 帧后，update 才会返回 true
         *         if (m_lstm->update(m_result[3], m_result[4], m_result[5]))
         *         {
         *             std::vector<float> lstm_out = m_lstm->get_prediction();
         *
         *             // 【策略选择】
         *             m_result[0] = lstm_out[0];
         *             m_result[1] = lstm_out[1];
         *             m_result[2] = lstm_out[2];
         *
         *             // LOGD("LSTM Active: x:%.2f, y:%.2f, z:%.2f", final_x, final_y, final_z);
         *         }
         *         else
         *         {
         *             // m_result[0] = m_result[3];
         *             // m_result[1] = m_result[4];
         *             // m_result[2] = m_result[5];
         *
         *             m_result[0] = 0.0f;
         *             m_result[1] = 0.0f;
         *             m_result[2] = 0.0f;
         *             // LOGV("LSTM warming up...");
         *         }
         *     }
         *
         *     cv::Mat point_homogeneous = (cv::Mat_<float>(4, 1) << m_result[0],
         *                                  m_result[1],
         *                                  m_result[2],
         *                                  1.0);
         *
         *     cv::Mat transformed_point = mocap_from_camera * point_homogeneous;
         *
         *     uart_result[0] = transformed_point.at<float>(0, 0);
         *     uart_result[1] = transformed_point.at<float>(1, 0);
         *     uart_result[2] = transformed_point.at<float>(2, 0);
         *
         *     LOG("\t wxj: x:%.4f, y:%.4f, z:%.4f",
         *         uart_result[0], uart_result[1], uart_result[2]);
         * }
         */
    };
};

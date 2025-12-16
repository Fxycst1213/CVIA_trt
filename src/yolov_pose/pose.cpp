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
using namespace std;
using namespace nvinfer1;

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
            /*
             * pose setup需要做的事情
             *   创建engine, context
             *   设置bindings。这里需要注意，不同版本的yolo的输出binding可能还不一样
             *   分配memory空间。这里需要注意，不同版本的yolo的输出所需要的空间也还不一样
             */

            m_runtime = shared_ptr<IRuntime>(createInferRuntime(*m_logger), destroy_trt_ptr<IRuntime>);
            m_engine = shared_ptr<ICudaEngine>(m_runtime->deserializeCudaEngine(data, size), destroy_trt_ptr<ICudaEngine>);
            m_context = shared_ptr<IExecutionContext>(m_engine->createExecutionContext(), destroy_trt_ptr<IExecutionContext>);
            char const *input_name = m_engine->getIOTensorName(0);
            char const *output_name = m_engine->getIOTensorName(1);

            // 2. 再通过名字获取维度
            m_inputDims = m_engine->getTensorShape(input_name);
            m_outputDims = m_engine->getTensorShape(output_name);

            CUDA_CHECK(cudaStreamCreate(&m_stream));

            m_inputSize = m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);
            m_imgArea = m_params->img.h * m_params->img.w;
            m_outputSize = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);

            // 这里对host和device上的memory一起分配空间
            CUDA_CHECK(cudaMallocHost(&m_inputMemory[0], m_inputSize));
            CUDA_CHECK(cudaMallocHost(&m_outputMemory[0], m_outputSize));
            CUDA_CHECK(cudaMalloc(&m_inputMemory[1], m_inputSize));
            CUDA_CHECK(cudaMalloc(&m_outputMemory[1], m_outputSize));

            // 创建m_bindings，之后再寻址就直接从这里找
            m_bindings[0] = m_inputMemory[1];
            m_bindings[1] = m_outputMemory[1];
        }

        void Pose::reset_task()
        {
            m_bboxes.clear();
        }

        bool Pose::preprocess_cpu(cv::Mat &img)
        {
            /*Preprocess -- yolo的预处理并没有mean和std，所以可以直接skip掉mean和std的计算 */

            /*Preprocess -- 读取数据*/
            // m_inputImage = cv::imread(m_imagePath);
            m_inputImage = img;
            if (m_inputImage.data == nullptr)
            {
                LOGE("ERROR: Image file not founded! Program terminated");
                return false;
            }

            /*Preprocess -- 测速*/
            m_timer->start_cpu();

            /*Preprocess -- resize(手动实现一个CPU版本的letterbox)*/
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

            /* 寻找resize后的图片在背景中的位置*/
            int x, y;
            x = (new_w < target_w) ? (target_w - new_w) / 2 : 0;
            y = (new_h < target_h) ? (target_h - new_h) / 2 : 0;

            cv::Rect roi(x, y, new_w, new_h);

            /* 指定背景图片里居中的图片roi，把resized_img给放入到这个roi中*/
            cv::Mat roiOfTar = tar(roi);
            resized_img.copyTo(roiOfTar);

            /*Preprocess -- host端进行normalization和BGR2RGB, NHWC->NCHW*/
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

            /*Preprocess -- 将host的数据移动到device上*/
            CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

            m_timer->stop_cpu<timer::Timer::ms>("preprocess(CPU)");
            return true;
        }

        bool Pose::preprocess_gpu(cv::Mat &img)
        {
            /*Preprocess -- yolo的预处理并没有mean和std，所以可以直接skip掉mean和std的计算 */

            /*Preprocess -- 读取数据*/
            // m_inputImage = cv::imread(m_imagePath);
            m_inputImage = img.clone();
            if (m_inputImage.data == nullptr)
            {
                LOGE("ERROR: file not founded! Program terminated");
                return false;
            }

            /*Preprocess -- 测速*/
            m_timer->start_gpu();

            /*Preprocess -- 使用GPU进行warpAffine, 并将结果返回到m_inputMemory中*/
            preprocess::preprocess_resize_gpu(m_inputImage, m_inputMemory[1],
                                              m_params->img.h, m_params->img.w,
                                              preprocess::tactics::GPU_WARP_AFFINE, m_stream);

            m_timer->stop_gpu("preprocess(GPU)");
            return true;
        }

        void Pose::show()
        {
            float kpt_conf_threshold = 0.5f;
            for (const auto &box : m_bboxes)
            {
                int x0 = static_cast<int>(box.x0);
                int y0 = static_cast<int>(box.y0);
                int x1 = static_cast<int>(box.x1);
                int y1 = static_cast<int>(box.y1);
                cv::rectangle(m_inputImage, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 0, 255), 2); // 线宽2
                for (int k = 0; k < box.keypoints.size(); ++k)
                {
                    const auto &kpt = box.keypoints[k];
                    if (kpt.conf < kpt_conf_threshold)
                        continue;

                    int x = static_cast<int>(kpt.x);
                    int y = static_cast<int>(kpt.y);
                    cv::circle(m_inputImage, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), -1);
                }
            }
            cv::imwrite("keypoints_result.jpg", m_inputImage);
        }

        bool Pose::postprocess_cpu()
        {
            m_timer->start_cpu();

            /*Postprocess -- 将device上的数据移动到host上*/
            int output_size = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0], m_outputMemory[1], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
            CUDA_CHECK(cudaStreamSynchronize(m_stream));
            /*Postprocess -- yolov8的postprocess需要做的事情*/
            /*
             * 1. 把bbox从输出tensor拿出来，并进行decode，把获取的bbox放入到m_bboxes中
             * 2. 把decode得到的m_bboxes根据nms threshold进行NMS处理
             * 3. 把最终得到的bbox绘制到原图中
             */

            float conf_threshold = 0.25; // 用来过滤decode时的bboxes
            float nms_threshold = 0.45;  // 用来过滤nms时的bboxes
            float kpt_conf_threshold = 0.5f;
            /*Postprocess -- 1. decode*/
            /*
             * 我们需要做的就是将[batch, bboxes, ch]转换为vector<bbox>
             * 几个步骤:
             * 1. 从每一个bbox中对应的ch中获取cx, cy, width, height
             * 2. 对每一个bbox中对应的ch中，找到最大的class label, 可以使用std::max_element
             * 3. 将cx, cy, width, height转换为x0, y0, x1, y1
             * 4. 因为图像是经过resize了的，所以需要根据resize的scale和shift进行坐标的转换(这里面可以根据preprocess中的到的affine matrix来进行逆变换)
             * 5. 将转换好的x0, y0, x1, y1，以及confidence和classness给存入到box中，并push到m_bboxes中，准备接下来的NMS处理
             */
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
                // Keypoint

                vector<keypoint> keypoints;
                keypoints.reserve(NUM_KEYPOINTS); // 预分配17个关键点空间

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

            /*Postprocess -- 2. NMS*/
            /*
             * 几个步骤:
             * 1. 做一个IoU计算的lambda函数
             * 2. 将m_bboxes中的所有数据，按照confidence从高到低进行排序
             * 3. 最终希望是对于每一个class，我们都只有一个bbox，所以对同一个class的所有bboxes进行IoU比较，
             *    选取confidence最大。并与其他的同类bboxes的IoU的重叠率最大的同时IoU > IoU threshold
             */

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

            /*Postprocess -- 2. 精修关键点*/
            /*
             */
            refine_point();
            m_timer->stop_cpu<timer::Timer::ms>("postprocess(CPU)");
            m_timer->show();
            return true;
        }

        bool Pose::postprocess_gpu()
        {
            return postprocess_cpu();
        }

        shared_ptr<Pose> make_pose(
            std::string onnx_path, logger::Level level, Params params)
        {
            return make_shared<Pose>(onnx_path, level, params);
        }
        void Pose::refine_point()
        {
            if (m_bboxes.size() == 0)
            {
                return;
            }
            bbox box = m_bboxes[0];

            return;
        }
    }; // namespace pose
}; // namespace model

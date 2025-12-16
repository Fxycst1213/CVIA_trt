#include "opencv2/opencv.hpp"
#include "preprocess.hpp"
#include "utils.hpp"
#include "time.hpp"
#include "cudatools.hpp"
namespace preprocess
{
    uint8_t *d_src = nullptr;
    float *d_mean = nullptr;
    float *d_std = nullptr;
    void affine_transformation_cpu(
        float trans_matrix[6],
        int src_x, int src_y,
        float *tar_x, float *tar_y)
    {
        *tar_x = trans_matrix[0] * src_x + trans_matrix[1] * src_y + trans_matrix[2];
        *tar_y = trans_matrix[3] * src_x + trans_matrix[4] * src_y + trans_matrix[5];
    }

    // 根据比例进行缩放 (CPU版本)
    cv::Mat preprocess_resize_cpu(
        cv::Mat &src,
        const int &tar_h, const int &tar_w,
        float *d_mean, float *d_std,
        tactics tac)
    {
        cv::Mat tar;

        int height = src.rows;
        int width = src.cols;
        float dim = std::max(height, width);
        int resizeH = ((height / dim) * tar_h);
        int resizeW = ((width / dim) * tar_w);

        int xOffSet = (tar_w - resizeW) / 2;
        int yOffSet = (tar_h - resizeH) / 2;

        resizeW = tar_w;
        resizeH = tar_h;

        /*BGR2RGB*/
        cv::cvtColor(src, src, cv::COLOR_BGR2RGB);

        /*Resize*/
        switch (tac)
        {
        case tactics::CPU_NEAREST:
            cv::resize(src, tar, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_LINEAR);
            break;
        case tactics::CPU_BILINEAR:
            cv::resize(src, tar, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_NEAREST);
            break;
        default:
            LOGE("ERROR: Wrong CPU resize tactics selected. Program terminated");
            exit(1);
        }
        return tar;
    }

    // 根据比例进行缩放 (GPU版本)
    void preprocess_resize_gpu(
        cv::Mat &h_src, float *d_tar,
        const int &tar_h, const int &tar_w,
        float *h_mean, float *h_std,
        tactics tac, cudaStream_t stream)
    {
        int height = h_src.rows;
        int width = h_src.cols;
        int chan = 3;

        int src_size = height * width * chan * sizeof(uint8_t);
        CUDA_CHECK(cudaMemcpyAsync(d_src, h_src.data, src_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_mean, h_mean, 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_std, h_std, 3 * sizeof(float), cudaMemcpyHostToDevice, stream));

        // device上处理resize, BGR2RGB的核函数
        resize_bilinear_gpu(d_tar, d_src, tar_w, tar_h, width, height, d_mean, d_std, tac, stream);
    }
    void preprocess_resize_gpu(
        cv::Mat &h_src, float *d_tar,
        const int &tar_h, const int &tar_w,
        tactics tac, cudaStream_t stream)
    {
        int height = h_src.rows;
        int width = h_src.cols;
        int chan = 3;

        int src_size = height * width * chan * sizeof(uint8_t);
        CUDA_CHECK(cudaMemcpyAsync(d_src, h_src.data, src_size, cudaMemcpyHostToDevice, stream));

        // device上处理resize, BGR2RGB的核函数
        resize_bilinear_gpu(d_tar, d_src, tar_w, tar_h, width, height, tac, stream);
    }

    void init_process(int h, int w)
    {
        int size = h * w * 3 * sizeof(uint8_t);
        CUDA_CHECK(cudaMalloc(&d_src, size));
        CUDA_CHECK(cudaMalloc(&d_mean, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_std, 3 * sizeof(float)));
    }
    void destroy_process()
    {
        if (d_src)
        {
            cudaFree(d_src);
            d_src = nullptr;
        }
        if (d_mean)
        {
            cudaFree(d_mean);
            d_mean = nullptr;
        }
        if (d_std)
        {
            cudaFree(d_std);
            d_std = nullptr;
        }
    }
} // namespace process

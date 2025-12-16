#include "cuda_runtime_api.h"
#include "stdio.h"
#include <iostream>
#include "preprocess.hpp"
#define NORM_SCALE 0.003921569f // 1.0f / 255.0f
namespace preprocess
{

    TransInfo trans;
    AffineMatrix affine_matrix;

    void warpaffine_init(int srcH, int srcW, int tarH, int tarW)
    {
        trans.src_h = srcH;
        trans.src_w = srcW;
        trans.tar_h = tarH;
        trans.tar_w = tarW;
        affine_matrix.init(trans);
    }

    __host__ __device__ void affine_transformation(
        float trans_matrix[6],
        int src_x, int src_y,
        float *tar_x, float *tar_y)
    {
        *tar_x = trans_matrix[0] * src_x + trans_matrix[1] * src_y + trans_matrix[2];
        *tar_y = trans_matrix[3] * src_x + trans_matrix[4] * src_y + trans_matrix[5];
    }

    __global__ void nearest_BGR2RGB_nhwc2nchw_norm_kernel(
        float *tar, uint8_t *src,
        int tarW, int tarH,
        int srcW, int srcH,
        float scaled_w, float scaled_h,
        float *d_mean, float *d_std)
    {
        // nearest neighbour -- resized之后的图tar上的坐标
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // nearest neighbour -- 计算最近坐标
        int src_y = floor((float)y * scaled_h);
        int src_x = floor((float)x * scaled_w);

        if (src_x < 0 || src_y < 0 || src_x > srcW || src_y > srcH)
        {
            // nearest neighbour -- 对于越界的部分，不进行计算
        }
        else
        {
            // nearest neighbour -- 计算tar中对应坐标的索引
            int tarIdx = y * tarW + x;
            int tarArea = tarW * tarH;

            // nearest neighbour -- 计算src中最近邻坐标的索引
            int srcIdx = (src_y * srcW + src_x) * 3;

            // nearest neighbour -- 实现nearest beighbour的resize + BGR2RGB + nhwc2nchw + norm
            tar[tarIdx + tarArea * 0] = (src[srcIdx + 2] / 255.0f - d_mean[2]) / d_std[2];
            tar[tarIdx + tarArea * 1] = (src[srcIdx + 1] / 255.0f - d_mean[1]) / d_std[1];
            tar[tarIdx + tarArea * 2] = (src[srcIdx + 0] / 255.0f - d_mean[0]) / d_std[0];
        }
    }

    __global__ void bilinear_BGR2RGB_nhwc2nchw_norm_kernel(
        float *tar, uint8_t *src,
        int tarW, int tarH,
        int srcW, int srcH,
        float scaled_w, float scaled_h,
        float *d_mean, float *d_std)
    {

        // bilinear interpolation -- resized之后的图tar上的坐标
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
        int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
        int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
        int src_y2 = src_y1 + 1;
        int src_x2 = src_x1 + 1;

        if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW)
        {
            // bilinear interpolation -- 对于越界的坐标不进行计算
        }
        else
        {
            // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
            float th = ((y + 0.5) * scaled_h - 0.5) - src_y1;
            float tw = ((x + 0.5) * scaled_w - 0.5) - src_x1;

            // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
            float a1_1 = (1.0 - tw) * (1.0 - th); // 右下
            float a1_2 = tw * (1.0 - th);         // 左下
            float a2_1 = (1.0 - tw) * th;         // 右上
            float a2_2 = tw * th;                 // 左上

            // bilinear interpolation -- 计算4个坐标所对应的索引
            int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3; // 左上
            int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3; // 右上
            int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3; // 左下
            int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3; // 右下

            // bilinear interpolation -- 计算resized之后的图的索引
            int tarIdx = y * tarW + x;
            int tarArea = tarW * tarH;

            // bilinear interpolation -- 实现bilinear interpolation的resize + BGR2RGB + NHWC2NCHW normalization
            // 注意，这里tar和src进行遍历的方式是不一样的
            tar[tarIdx + tarArea * 0] =
                (round((a1_1 * src[srcIdx1_1 + 2] +
                        a1_2 * src[srcIdx1_2 + 2] +
                        a2_1 * src[srcIdx2_1 + 2] +
                        a2_2 * src[srcIdx2_2 + 2])) /
                     255.0f -
                 d_mean[2]) /
                d_std[2];

            tar[tarIdx + tarArea * 1] =
                (round((a1_1 * src[srcIdx1_1 + 1] +
                        a1_2 * src[srcIdx1_2 + 1] +
                        a2_1 * src[srcIdx2_1 + 1] +
                        a2_2 * src[srcIdx2_2 + 1])) /
                     255.0f -
                 d_mean[1]) /
                d_std[1];

            tar[tarIdx + tarArea * 2] =
                (round((a1_1 * src[srcIdx1_1 + 0] +
                        a1_2 * src[srcIdx1_2 + 0] +
                        a2_1 * src[srcIdx2_1 + 0] +
                        a2_2 * src[srcIdx2_2 + 0])) /
                     255.0f -
                 d_mean[0]) /
                d_std[0];
        }
    }

    __global__ void bilinear_BGR2RGB_nhwc2nchw_shift_norm_kernel(
        float *tar, uint8_t *src,
        int tarW, int tarH,
        int srcW, int srcH,
        float scaled_w, float scaled_h,
        float *d_mean, float *d_std)
    {
        // resized之后的图tar上的坐标
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
        int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
        int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
        int src_y2 = src_y1 + 1;
        int src_x2 = src_x1 + 1;

        if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW)
        {
            // bilinear interpolation -- 对于越界的坐标不进行计算
        }
        else
        {
            // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
            float th = (float)y * scaled_h - src_y1;
            float tw = (float)x * scaled_w - src_x1;

            // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
            float a1_1 = (1.0 - tw) * (1.0 - th); // 右下
            float a1_2 = tw * (1.0 - th);         // 左下
            float a2_1 = (1.0 - tw) * th;         // 右上
            float a2_2 = tw * th;                 // 左上

            // bilinear interpolation -- 计算4个坐标所对应的索引
            int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3; // 左上
            int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3; // 右上
            int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3; // 左下
            int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3; // 右下

            // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
            y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
            x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);

            // bilinear interpolation -- 计算resized之后的图的索引
            int tarIdx = (y * tarW + x) * 3;
            int tarArea = tarW * tarH;

            // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB + shift + nhwc2nchw
            tar[tarIdx + tarArea * 0] =
                (round((a1_1 * src[srcIdx1_1 + 2] +
                        a1_2 * src[srcIdx1_2 + 2] +
                        a2_1 * src[srcIdx2_1 + 2] +
                        a2_2 * src[srcIdx2_2 + 2])) /
                     255.0f -
                 d_mean[2]) /
                d_std[2];

            tar[tarIdx + tarArea * 1] =
                (round((a1_1 * src[srcIdx1_1 + 1] +
                        a1_2 * src[srcIdx1_2 + 1] +
                        a2_1 * src[srcIdx2_1 + 1] +
                        a2_2 * src[srcIdx2_2 + 1])) /
                     255.0f -
                 d_mean[1]) /
                d_std[1];

            tar[tarIdx + tarArea * 2] =
                (round((a1_1 * src[srcIdx1_1 + 0] +
                        a1_2 * src[srcIdx1_2 + 0] +
                        a2_1 * src[srcIdx2_1 + 0] +
                        a2_2 * src[srcIdx2_2 + 0])) /
                     255.0f -
                 d_mean[0]) /
                d_std[0];
        }
    }

    __global__ void nearest_BGR2RGB_nhwc2nchw_kernel(
        float *tar, uint8_t *src,
        int tarW, int tarH,
        int srcW, int srcH,
        float scaled_w, float scaled_h)
    {
        // nearest neighbour -- resized之后的图tar上的坐标
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // nearest neighbour -- 计算最近坐标
        int src_y = floor((float)y * scaled_h);
        int src_x = floor((float)x * scaled_w);

        if (src_x < 0 || src_y < 0 || src_x > srcW || src_y > srcH)
        {
            // nearest neighbour -- 对于越界的部分，不进行计算
        }
        else
        {
            // nearest neighbour -- 计算tar中对应坐标的索引
            int tarIdx = y * tarW + x;
            int tarArea = tarW * tarH;

            // nearest neighbour -- 计算src中最近邻坐标的索引
            int srcIdx = (src_y * srcW + src_x) * 3;

            // nearest neighbour -- 实现nearest beighbour的resize + BGR2RGB + nhwc2nchw + norm
            tar[tarIdx + tarArea * 0] = src[srcIdx + 2] / 255.0f;
            tar[tarIdx + tarArea * 1] = src[srcIdx + 1] / 255.0f;
            tar[tarIdx + tarArea * 2] = src[srcIdx + 0] / 255.0f;
        }
    }

    __global__ void bilinear_BGR2RGB_nhwc2nchw_kernel(
        float *tar, uint8_t *src,
        int tarW, int tarH,
        int srcW, int srcH,
        float scaled_w, float scaled_h)
    {

        // bilinear interpolation -- resized之后的图tar上的坐标
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
        int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
        int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
        int src_y2 = src_y1 + 1;
        int src_x2 = src_x1 + 1;

        if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW)
        {
            // bilinear interpolation -- 对于越界的坐标不进行计算
        }
        else
        {
            // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
            float th = ((y + 0.5) * scaled_h - 0.5) - src_y1;
            float tw = ((x + 0.5) * scaled_w - 0.5) - src_x1;

            // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
            float a1_1 = (1.0 - tw) * (1.0 - th); // 右下
            float a1_2 = tw * (1.0 - th);         // 左下
            float a2_1 = (1.0 - tw) * th;         // 右上
            float a2_2 = tw * th;                 // 左上

            // bilinear interpolation -- 计算4个坐标所对应的索引
            int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3; // 左上
            int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3; // 右上
            int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3; // 左下
            int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3; // 右下

            // bilinear interpolation -- 计算resized之后的图的索引
            int tarIdx = y * tarW + x;
            int tarArea = tarW * tarH;

            // bilinear interpolation -- 实现bilinear interpolation的resize + BGR2RGB + NHWC2NCHW normalization
            // 注意，这里tar和src进行遍历的方式是不一样的
            tar[tarIdx + tarArea * 0] =
                round((a1_1 * src[srcIdx1_1 + 2] +
                       a1_2 * src[srcIdx1_2 + 2] +
                       a2_1 * src[srcIdx2_1 + 2] +
                       a2_2 * src[srcIdx2_2 + 2])) /
                255.0f;

            tar[tarIdx + tarArea * 1] =
                round((a1_1 * src[srcIdx1_1 + 1] +
                       a1_2 * src[srcIdx1_2 + 1] +
                       a2_1 * src[srcIdx2_1 + 1] +
                       a2_2 * src[srcIdx2_2 + 1])) /
                255.0f;

            tar[tarIdx + tarArea * 2] =
                round((a1_1 * src[srcIdx1_1 + 0] +
                       a1_2 * src[srcIdx1_2 + 0] +
                       a2_1 * src[srcIdx2_1 + 0] +
                       a2_2 * src[srcIdx2_2 + 0])) /
                255.0f;
        }
    }

    __global__ void bilinear_BGR2RGB_nhwc2nchw_shift_kernel(
        float *tar, uint8_t *src,
        int tarW, int tarH,
        int srcW, int srcH,
        float scaled_w, float scaled_h)
    {
        // resized之后的图tar上的坐标
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
        int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
        int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
        int src_y2 = src_y1 + 1;
        int src_x2 = src_x1 + 1;

        if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW)
        {
            // bilinear interpolation -- 对于越界的坐标不进行计算
        }
        else
        {
            // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
            float th = (float)y * scaled_h - src_y1;
            float tw = (float)x * scaled_w - src_x1;

            // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
            float a1_1 = (1.0 - tw) * (1.0 - th); // 右下
            float a1_2 = tw * (1.0 - th);         // 左下
            float a2_1 = (1.0 - tw) * th;         // 右上
            float a2_2 = tw * th;                 // 左上

            // bilinear interpolation -- 计算4个坐标所对应的索引
            int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3; // 左上
            int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3; // 右上
            int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3; // 左下
            int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3; // 右下

            // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
            y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
            x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);

            // bilinear interpolation -- 计算resized之后的图的索引
            int tarIdx = y * tarW + x;
            int tarArea = tarW * tarH;

            // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB + shift + nhwc2nchw
            tar[tarIdx + tarArea * 0] =
                round((a1_1 * src[srcIdx1_1 + 2] +
                       a1_2 * src[srcIdx1_2 + 2] +
                       a2_1 * src[srcIdx2_1 + 2] +
                       a2_2 * src[srcIdx2_2 + 2])) /
                255.0f;

            tar[tarIdx + tarArea * 1] =
                round((a1_1 * src[srcIdx1_1 + 1] +
                       a1_2 * src[srcIdx1_2 + 1] +
                       a2_1 * src[srcIdx2_1 + 1] +
                       a2_2 * src[srcIdx2_2 + 1])) /
                255.0f;

            tar[tarIdx + tarArea * 2] =
                round((a1_1 * src[srcIdx1_1 + 0] +
                       a1_2 * src[srcIdx1_2 + 0] +
                       a2_1 * src[srcIdx2_1 + 0] +
                       a2_2 * src[srcIdx2_2 + 0])) /
                255.0f;
        }
    }

    __global__ void warpaffine_BGR2RGB_kernel(
        float *__restrict__ tar,         // 输出指针 (加入 __restrict__ 帮助编译器优化)
        const uint8_t *__restrict__ src, // 输入指针 (加入 const 和 __restrict__ 启用 L1 Cache)
        TransInfo trans,
        AffineMatrix affine_matrix)
    {
        // 计算全局坐标
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // 边界检查：直接退出多余线程，避免无效计算
        if (x >= trans.tar_w || y >= trans.tar_h)
            return;

        // 1. 坐标变换 (FMA优化)
        // 假设 affine_matrix.reverse 是 float[6]
        // src_x = m0 * x + m1 * y + m2
        // src_y = m3 * x + m4 * y + m5
        // 这里使用 center offset (+0.5f) 是标准的 OpenCV 行为
        float src_x = affine_matrix.reverse[0] * (x + 0.5f) + affine_matrix.reverse[1] * (y + 0.5f) + affine_matrix.reverse[2];
        float src_y = affine_matrix.reverse[3] * (x + 0.5f) + affine_matrix.reverse[4] * (y + 0.5f) + affine_matrix.reverse[5];

        // 调整回 0-index
        src_x -= 0.5f;
        src_y -= 0.5f;

        // 2. 双线性插值坐标计算
        int src_x_low = floorf(src_x);
        int src_y_low = floorf(src_y);
        int src_x_high = src_x_low + 1;
        int src_y_high = src_y_low + 1;

        // 计算权重 (利用 float 精度)
        float ly = src_y - src_y_low;
        float lx = src_x - src_x_low;
        float hy = 1.0f - ly;
        float hx = 1.0f - lx;

        // 预计算权重乘积
        float w1 = hx * hy;
        float w2 = lx * hy;
        float w3 = hx * ly;
        float w4 = lx * ly;

        // 3. 计算输出索引 (Coalesced Access)
        // 输出是 planar (CHW)，每个线程写 float，这是合并访问，效率很高
        int tarIdx = y * trans.tar_w + x;
        int tarArea = trans.tar_w * trans.tar_h;

        // 初始化颜色值 (默认填充值，通常为 114/255 或 0)
        // 原始代码没有处理这里，导致Padding区域是乱码
        float c0 = 0.5f; // B (or R depending on logic)
        float c1 = 0.5f; // G
        float c2 = 0.5f; // R (or B)

        // 4. 边界检查与数据读取
        // 检查是否在源图像范围内 (使用 fast integer logic)
        // 注意：只要 TopLeft 点在范围内，部分插值也是可能的，但为了高性能通常全匹配
        if (src_x_low >= 0 && src_x_high < trans.src_w &&
            src_y_low >= 0 && src_y_high < trans.src_h)
        {
            // 计算源图像基地址
            // 优化：BGR是紧凑排列，stride = width * 3
            int stride = trans.src_w * 3;
            int ptr1 = src_y_low * stride + src_x_low * 3;
            int ptr2 = src_y_high * stride + src_x_low * 3; // 下一行

            // 利用 L1 Cache 读取 4 个点
            // Top-Left
            uint8_t v1_0 = src[ptr1 + 0];
            uint8_t v1_1 = src[ptr1 + 1];
            uint8_t v1_2 = src[ptr1 + 2];

            // Top-Right
            uint8_t v2_0 = src[ptr1 + 3];
            uint8_t v2_1 = src[ptr1 + 4];
            uint8_t v2_2 = src[ptr1 + 5];

            // Bottom-Left
            uint8_t v3_0 = src[ptr2 + 0];
            uint8_t v3_1 = src[ptr2 + 1];
            uint8_t v3_2 = src[ptr2 + 2];

            // Bottom-Right
            uint8_t v4_0 = src[ptr2 + 3];
            uint8_t v4_1 = src[ptr2 + 4];
            uint8_t v4_2 = src[ptr2 + 5];

            // 5. 插值计算 (FMA) + BGR2RGB Swapping
            // 原始代码：
            // Plane 0 = src[+2] (R)
            // Plane 1 = src[+1] (G)
            // Plane 2 = src[+0] (B)

            // 计算 R (Source index +2) -> Target Plane 0
            c0 = (w1 * v1_2 + w2 * v2_2 + w3 * v3_2 + w4 * v4_2) * NORM_SCALE;

            // 计算 G (Source index +1) -> Target Plane 1
            c1 = (w1 * v1_1 + w2 * v2_1 + w3 * v3_1 + w4 * v4_1) * NORM_SCALE;

            // 计算 B (Source index +0) -> Target Plane 2
            c2 = (w1 * v1_0 + w2 * v2_0 + w3 * v3_0 + w4 * v4_0) * NORM_SCALE;
        }
        // 如果出界，保留默认值 (Padding)，比如黑色 0.0f 或 灰色 0.5f

        // 写入全局内存
        tar[tarIdx + tarArea * 0] = c0;
        tar[tarIdx + tarArea * 1] = c1;
        tar[tarIdx + tarArea * 2] = c2;
    }

    void resize_bilinear_gpu(
        float *d_tar, uint8_t *d_src,
        int tarW, int tarH,
        int srcW, int srcH,
        float *d_mean, float *d_std,
        tactics tac, cudaStream_t stream)
    {
        dim3 dimBlock(32, 32, 1);
        dim3 dimGrid(tarW / 32, tarH / 32, 1);

        // scaled resize
        float scaled_h = (float)srcH / tarH;
        float scaled_w = (float)srcW / tarW;
        float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

        switch (tac)
        {
        case tactics::GPU_NEAREST:
            nearest_BGR2RGB_nhwc2nchw_norm_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h, d_mean, d_std);
            break;
        case tactics::GPU_NEAREST_CENTER:
            nearest_BGR2RGB_nhwc2nchw_norm_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale, d_mean, d_std);
            break;
        case tactics::GPU_BILINEAR:
            bilinear_BGR2RGB_nhwc2nchw_norm_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h, d_mean, d_std);
            break;
        case tactics::GPU_BILINEAR_CENTER:
            bilinear_BGR2RGB_nhwc2nchw_shift_norm_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale, d_mean, d_std);
            break;
        default:
            LOGE("ERROR: Wrong GPU resize tactics selected. Program terminated");
            exit(1);
        }
    }

    void resize_bilinear_gpu(
        float *d_tar, uint8_t *d_src,
        // uint8_t *cpu_mat,
        int tarW, int tarH,
        int srcW, int srcH,
        tactics tac, cudaStream_t stream)
    {
        dim3 dimBlock(16, 16, 1);
        dim3 dimGrid(tarW / 16 + 1, tarH / 16 + 1, 1);

        // scaled resize
        float scaled_h = (float)srcH / tarH;
        float scaled_w = (float)srcW / tarW;
        float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

        switch (tac)
        {
        case tactics::GPU_NEAREST:
            nearest_BGR2RGB_nhwc2nchw_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
            break;
        case tactics::GPU_NEAREST_CENTER:
            nearest_BGR2RGB_nhwc2nchw_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale);
            break;
        case tactics::GPU_BILINEAR:
            bilinear_BGR2RGB_nhwc2nchw_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
            break;
        case tactics::GPU_BILINEAR_CENTER:
            bilinear_BGR2RGB_nhwc2nchw_shift_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale);
            break;
        case tactics::GPU_WARP_AFFINE:
            warpaffine_init(srcH, srcW, tarH, tarW);
            warpaffine_BGR2RGB_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_tar, d_src, trans, affine_matrix);
            break;
        default:
            LOGE("ERROR: Wrong GPU resize tactics selected. Program terminated");
            exit(1);
        }
    }

} // namespace preprocess

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>

#include "ops/attention.h"
#include "cuda/cuda_check.h"

constexpr int THREADPERBLOCKDIM = 16;

template <typename T>
__global__ void scale_mask_kernel(T *qk, const T *mask, float scalar, std::int64_t num_heads, std::int64_t seq)
{
    std::int64_t row = blockDim.y * blockIdx.y + threadIdx.y;
    std::int64_t col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_heads * seq && col < seq)
    {
        // qk[h,i,j]
        std::int64_t i = row % seq;
        if (mask)
        {
            qk[row * seq + col] = (T)(static_cast<float>(qk[row * seq + col]) * scalar + static_cast<float>(mask[i * seq + col]));
        }
        else
        {
            if (i < col)
            {
                qk[row * seq + col] = (T)(-FLT_MAX);
            }
            else
            {
                qk[row * seq + col] = (T)(static_cast<float>(qk[row * seq + col]) * scalar);
            }
        }
    }
}

void scale_mask_launch(void *qk, const void *mask, float scalar, int64_t num_heads, int64_t seq, DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCKDIM, THREADPERBLOCKDIM);
    dim3 blockPerGrid((seq + threadPerBlock.x - 1) / threadPerBlock.x, (num_heads * seq + threadPerBlock.y - 1) / threadPerBlock.y);
    if (dtype == DType::Float32)
    {
        scale_mask_kernel<float><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(qk), static_cast<const float *>(mask), scalar, num_heads, seq);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (dtype == DType::BFloat16)
    {
        scale_mask_kernel<__nv_bfloat16><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(qk), static_cast<const __nv_bfloat16 *>(mask), scalar, num_heads, seq);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else
        throw std::runtime_error("unsupported dtype");
}
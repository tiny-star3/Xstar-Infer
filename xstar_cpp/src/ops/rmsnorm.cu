#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "ops/rmsnorm.h"
#include "cuda/cuda_check.h"
#include "cuda/dtype_cast.h"

constexpr int THREADPERBLOCK = 256;

template <typename T>
__global__ void rmsnorm_kernel(T *out,
                               const T *x,
                               const T *weight,
                               std::int64_t hidden,
                               std::int64_t num_rows,
                               float eps)
{
    std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    __shared__ float partial[THREADPERBLOCK];
    partial[threadIdx.x] = 0;
    __syncthreads();
    for (std::int64_t i = threadIdx.x; i < hidden; i += blockDim.x)
    {
        float val = toFloat(x[row * hidden + i]);
        partial[threadIdx.x] += val * val;
    }
    __syncthreads();
    // 树归约
    for (std::int64_t i = blockDim.x / 2; i >= 32; i /= 2)
    {
        if (threadIdx.x < i)
        {
            partial[threadIdx.x] += partial[threadIdx.x + i];
        }
        __syncthreads();
    }
    // warp shuffle
    if (threadIdx.x < warpSize)
    {
        float sum_val = partial[threadIdx.x];
        for (std::int64_t offset = 16; offset > 0; offset /= 2)
        {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        }
        if (threadIdx.x == 0)
        {
            partial[threadIdx.x] = rsqrtf(sum_val / hidden + eps);
        }
    }
    __syncthreads();
    float inv_rms = partial[0];
    for (std::int64_t i = threadIdx.x; i < hidden; i += blockDim.x)
    {
        out[row * hidden + i] = T(toFloat(x[row * hidden + i]) * inv_rms * toFloat(weight[i]));
    }
}

void rmsnorm_launch(void *out,
                    const void *x,
                    const void *weight,
                    std::int64_t hidden,
                    std::int64_t num_rows,
                    float eps,
                    DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCK);
    dim3 blockPerGrid(num_rows);
    if (dtype == DType::Float32)
    {
        rmsnorm_kernel<float><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(out), static_cast<const float *>(x), static_cast<const float *>(weight), hidden, num_rows, eps);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (dtype == DType::BFloat16)
    {
        rmsnorm_kernel<__nv_bfloat16><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(weight), hidden, num_rows, eps);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else
        throw std::runtime_error("unsupported dtype");
}
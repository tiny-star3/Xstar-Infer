#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>

#include "ops/add.h"
#include "cuda/cuda_check.h"

constexpr int THREADPERBLOCK = 256;

template <typename T>
__global__ void add_kernel(T *out, const T *a, const T *b, std::int64_t numel)
{
    for (std::int64_t i = blockDim.x * blockIdx.x + threadIdx.x; i < numel; i += gridDim.x * blockDim.x)
    {
        out[i] = a[i] + b[i];
    }
}

void add_launch(void *out, const void *a, const void *b, std::int64_t numel, DType dtype)
{
    static int cached_sm = -1;
    if (cached_sm < 0)
    {
        // 获取设备属性
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        cached_sm = prop.multiProcessorCount;
    }
    dim3 threadPerBlock(THREADPERBLOCK);
    // grid-stride 固定 grid:min(SM数 × 4, ceil(numel/blockDim)), 运行时查 SM 数
    int BLOCKPERGRID = (numel + threadPerBlock.x - 1) / threadPerBlock.x;
    if (BLOCKPERGRID > cached_sm * 4)
    {
        BLOCKPERGRID = cached_sm * 4;
    }
    dim3 blockPerGrid(BLOCKPERGRID);
    if (dtype == DType::Float32)
    {
        add_kernel<float><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(out), static_cast<const float *>(a), static_cast<const float *>(b), numel);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (dtype == DType::BFloat16)
    {
        add_kernel<__nv_bfloat16><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(a), static_cast<const __nv_bfloat16 *>(b), numel);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else
        throw std::runtime_error("unsupported dtype");
}
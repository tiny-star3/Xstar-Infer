#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstring>

#include "ops/embedding.h"
#include "cuda/cuda_check.h"

constexpr int THREADPERBLOCK = 256;

template <typename T>
__global__ void embedding_kernel(T *out,
                                 const T *weight,
                                 const std::int64_t *d_ids,
                                 std::int64_t numel,
                                 std::int64_t hidden)
{
    std::int64_t row = blockIdx.x;
    for (std::int64_t i = threadIdx.x; i < hidden; i += blockDim.x)
    {
        out[row * hidden + i] = weight[d_ids[row] * hidden + i];
    }
}

void embedding_launch(void *out,
                      const void *weight,
                      const std::int64_t *d_ids,
                      std::int64_t numel,
                      std::int64_t hidden,
                      DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCK);
    dim3 blockPerGrid(numel);
    if (dtype == DType::Float32)
    {
        embedding_kernel<float><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(out), static_cast<const float *>(weight), d_ids, numel, hidden);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (dtype == DType::BFloat16)
    {
        embedding_kernel<__nv_bfloat16><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(weight), d_ids, numel, hidden);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else
        throw std::runtime_error("unsupported dtype");
}
#include "cuda_runtime.h"
#include "cuda_bf16.h"

#include "ops/head_split.h"
#include "cuda/cuda_check.h"

constexpr int THREADPERBLOCK = 256;

template <typename T>
__global__ void head_split_kernel(T *out,
                                  const T *t,
                                  std::int64_t heads,
                                  std::int64_t seq,
                                  std::int64_t head_dim)
{
    std::int64_t h = blockIdx.x / seq;
    std::int64_t s = blockIdx.x % seq;

    for (std::int64_t i = threadIdx.x; i < head_dim; i += blockDim.x)
    {
        out[h * seq * head_dim + s * head_dim + i] = t[s * heads * head_dim + h * head_dim + i];
    }
}

void head_split_launch(void *out,
                       const void *t,
                       std::int64_t heads,
                       std::int64_t seq,
                       std::int64_t head_dim,
                       DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCK);
    dim3 blockPerGrid(heads * seq);
    if (dtype == DType::Float32)
    {
        head_split_kernel<float><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(out), static_cast<const float *>(t), heads, seq, head_dim);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (dtype == DType::BFloat16)
    {
        head_split_kernel<__nv_bfloat16><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(t), heads, seq, head_dim);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else
        throw std::runtime_error("unsupported dtype");
}

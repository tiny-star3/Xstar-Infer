#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "ops/rope.h"
#include "cuda/cuda_check.h"
#include "cuda/dtype_cast.h"

// 32 个连续 ch, 1 个 coalesced transaction
constexpr int THREADPERBLOCKDIMX = 32;
constexpr int THREADPERBLOCKDIMY = 8;
constexpr int THREADPERBLOCKDIMZ = 4;

template <typename T>
__global__ void rope_kernel(T *out,
                            const T *x,
                            const float *cache,
                            const std::int64_t *d_positions,
                            std::int64_t num_outer,
                            std::int64_t dim,
                            std::int64_t seq_len,
                            std::int64_t half)
{
    std::int64_t h = blockDim.z * blockIdx.z + threadIdx.z;
    std::int64_t s = blockDim.y * blockIdx.y + threadIdx.y;
    std::int64_t ch = blockDim.x * blockIdx.x + threadIdx.x;

    if (h < num_outer && s < seq_len && ch < dim / 2)
    {
        std::int64_t p = d_positions[s];
        float cos = cache[p * (dim / 2) + ch];
        float x1 = toFloat(x[h * seq_len * dim + s * dim + ch]);
        float sin = cache[half + p * (dim / 2) + ch];
        float x2 = toFloat(x[h * seq_len * dim + s * dim + dim / 2 + ch]);
        out[h * seq_len * dim + s * dim + ch] = (T)(cos * x1 - sin * x2);
        out[h * seq_len * dim + s * dim + dim / 2 + ch] = (T)(sin * x1 + cos * x2);
    }
}

void rope_launch(void *out,
                 const void *x,
                 const float *cache,
                 const std::int64_t *d_positions,
                 std::int64_t num_outer,
                 std::int64_t dim,
                 std::int64_t seq_len,
                 std::int64_t half,
                 DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCKDIMX, THREADPERBLOCKDIMY, THREADPERBLOCKDIMZ);
    dim3 blockPerGrid((dim / 2 + threadPerBlock.x - 1) / threadPerBlock.x, (seq_len + threadPerBlock.y - 1) / threadPerBlock.y, (num_outer + threadPerBlock.z - 1) / threadPerBlock.z);
    if (dtype == DType::Float32)
    {
        rope_kernel<float><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(out), static_cast<const float *>(x), cache, d_positions, num_outer, dim, seq_len, half);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (dtype == DType::BFloat16)
    {
        rope_kernel<__nv_bfloat16><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(x), cache, d_positions, num_outer, dim, seq_len, half);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else
        throw std::runtime_error("unsupported dtype");
}
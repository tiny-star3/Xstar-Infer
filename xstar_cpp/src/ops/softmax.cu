#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>

#include "ops/softmax.h"
#include "cuda/cuda_check.h"
#include "cuda/dtype_cast.h"

constexpr int THREADPERBLOCK = 256;

template <typename T>
__global__ void softmax_kernel(T *out,
                               const T *x,
                               std::int64_t outer_size,
                               std::int64_t dim_size,
                               std::int64_t inner_size)
{
    // 最大值
    __shared__ float m_partial[THREADPERBLOCK];
    // 和
    __shared__ float l_partial[THREADPERBLOCK];
    std::int64_t outer_idx = blockIdx.x / inner_size;
    std::int64_t inner_idx = blockIdx.x % inner_size;
    float m_final = -FLT_MAX;
    float l_final = 0;
    if (threadIdx.x < dim_size)
    {
        m_final = toFloat(x[outer_idx * dim_size * inner_size + threadIdx.x * inner_size + inner_idx]);
        l_final = 1;
    }
    for (std::int64_t i = threadIdx.x + blockDim.x; i < dim_size; i += blockDim.x)
    {
        float old_final = m_final;
        float val = toFloat(x[outer_idx * dim_size * inner_size + i * inner_size + inner_idx]);
        m_final = fmaxf(m_final, val);
        // __expf(CUDA 快速 intrinsic, ~22 位)
        l_final = l_final * __expf(old_final - m_final) + __expf(val - m_final);
    }
    m_partial[threadIdx.x] = m_final;
    l_partial[threadIdx.x] = l_final;
    __syncthreads();

    // 树归约
    for (std::int64_t i = blockDim.x / 2; i >= 32; i /= 2)
    {
        if (threadIdx.x < i)
        {
            float m1 = m_partial[threadIdx.x];
            float m2 = m_partial[threadIdx.x + i];
            if (m1 >= m2)
            {
                l_partial[threadIdx.x] = l_partial[threadIdx.x] + l_partial[threadIdx.x + i] * __expf(m2 - m1);
            }
            else
            {
                l_partial[threadIdx.x] = l_partial[threadIdx.x] * __expf(m1 - m2) + l_partial[threadIdx.x + i];
                m_partial[threadIdx.x] = m2;
            }
        }
        __syncthreads();
    }

    // warp shuffle
    if (threadIdx.x < warpSize)
    {
        float m1 = m_partial[threadIdx.x];
        float l1 = l_partial[threadIdx.x];
        for (std::int64_t i = warpSize / 2; i > 0; i /= 2)
        {
            float m2 = __shfl_down_sync(0xffffffff, m1, i);
            float l2 = __shfl_down_sync(0xffffffff, l1, i);
            if (m1 >= m2)
            {
                l1 = l1 + l2 * __expf(m2 - m1);
            }
            else
            {
                l1 = l1 * __expf(m1 - m2) + l2;
                m1 = m2;
            }
        }
        if (threadIdx.x == 0)
        {
            m_partial[threadIdx.x] = m1;
            l_partial[threadIdx.x] = l1;
        }
    }
    __syncthreads();
    m_final = m_partial[0];
    l_final = l_partial[0];
    for (std::int64_t i = threadIdx.x; i < dim_size; i += blockDim.x)
    {
        out[outer_idx * dim_size * inner_size + i * inner_size + inner_idx] = (T)(__expf(toFloat(x[outer_idx * dim_size * inner_size + i * inner_size + inner_idx]) - m_final) / l_final);
    }
}

void softmax_launch(void *out,
                    const void *x,
                    std::int64_t outer_size,
                    std::int64_t dim_size,
                    std::int64_t inner_size,
                    DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCK);
    dim3 blockPerThread(outer_size * inner_size);
    if (dtype == DType::Float32)
    {
        softmax_kernel<float><<<blockPerThread, threadPerBlock>>>(static_cast<float *>(out), static_cast<const float *>(x), outer_size, dim_size, inner_size);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (dtype == DType::BFloat16)
    {
        softmax_kernel<__nv_bfloat16><<<blockPerThread, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(x), outer_size, dim_size, inner_size);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else
        throw std::runtime_error("unsupported dtype");
}
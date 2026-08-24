#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "ops/mlp.h"
#include "cuda/cuda_check.h"
#include "cuda/dtype_cast.h"

constexpr int THREADPERBLOCKDIM = 8;
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;
// compute 侧(TM 整数)
static_assert(BM % THREADPERBLOCKDIM == 0, "BM must be divisible by thread rows");
static_assert(BN % THREADPERBLOCKDIM == 0, "BN must be divisible by thread cols");
// load 侧(load 粒度整数)
static_assert(BK % THREADPERBLOCKDIM == 0, "BK must be divisible by thread rows (load granularity)");
constexpr int TM = BM / THREADPERBLOCKDIM;
constexpr int TN = BN / THREADPERBLOCKDIM;
constexpr int VEC = BK / THREADPERBLOCKDIM;

// trait
// f32 用 float4(16B=4 float)、bf16 用 uint2(8B=4 bf16)
template <typename T>
struct vec_type;
template <>
struct vec_type<float>
{
    using type = float4;
};
template <>
struct vec_type<__nv_bfloat16>
{
    using type = uint2;
};
template <typename T>
using vec_t = typename vec_type<T>::type;

// 内部链接
static __device__ float silu(float x)
{
    // 永远不会算 exp 的正参数
    if (x >= 0)
    {
        return x * (1.0f / (1.0f + expf(-x)));
    }
    else
    {
        return x * (expf(x) / (1.0f + expf(x)));
    }
}

template <typename T>
__global__ void gemm_silu_and_mul_kernel(T *act,
                                         const T *x,
                                         const T *W,
                                         std::int64_t m,
                                         std::int64_t k,
                                         std::int64_t intermediate,
                                         std::int64_t ldc)
{
    std::int64_t row = blockIdx.y * BM;
    std::int64_t col = blockIdx.x * BN;
    __shared__ T smem_gate[BN][BK], smem_up[BN][BK], smem_x[BM][BK];
    float acc_gate[TM][TN] = {0.0f};
    float acc_up[TM][TN] = {0.0f};
    for (std::int64_t kk = 0; kk < k; kk += BK)
    {
        for (std::int64_t i = 0; i < TM; i++)
        {
            if (row + i * blockDim.y + threadIdx.y < m && kk + threadIdx.x * VEC + VEC < k)
            {
                vec_t<T> val = *reinterpret_cast<const vec_t<T> *>(&x[(row + i * blockDim.y + threadIdx.y) * k + kk + threadIdx.x * VEC]);
                T *p = reinterpret_cast<T *>(&val);
                for (int j = 0; j < VEC; j++)
                {
                    smem_x[i * blockDim.y + threadIdx.y][threadIdx.x * VEC + j] = p[j];
                }
            }
            else
            {
                for (int j = 0; j < VEC; j++)
                {
                    if (row + i * blockDim.y + threadIdx.y < m && kk + threadIdx.x * VEC + j < k)
                    {
                        smem_x[i * blockDim.y + threadIdx.y][threadIdx.x * VEC + j] = x[(row + i * blockDim.y + threadIdx.y) * k + kk + threadIdx.x * VEC + j];
                    }
                    else
                    {
                        smem_x[i * blockDim.y + threadIdx.y][threadIdx.x * VEC + j] = 0;
                    }
                }
            }
        }
        for (std::int64_t i = 0; i < TN; i++)
        {
            if (col + i * blockDim.x + threadIdx.x < intermediate && kk + threadIdx.y * VEC + VEC <= k)
            {
                vec_t<T> val = *reinterpret_cast<const vec_t<T> *>(&W[(col + i * blockDim.x + threadIdx.x) * k + kk + threadIdx.y * VEC]);
                T *p = reinterpret_cast<T *>(&val);
                for (int j = 0; j < VEC; j++)
                {
                    smem_gate[i * blockDim.x + threadIdx.x][threadIdx.y * VEC + j] = p[j];
                }
            }
            else
            {
                for (int j = 0; j < VEC; j++)
                {
                    if (col + i * blockDim.x + threadIdx.x < intermediate && kk + threadIdx.y * VEC + j < k)
                    {
                        smem_gate[i * blockDim.x + threadIdx.x][threadIdx.y * VEC + j] = W[(col + i * blockDim.x + threadIdx.x) * k + kk + threadIdx.y * VEC + j];
                    }
                    else
                    {
                        smem_gate[i * blockDim.x + threadIdx.x][threadIdx.y * VEC + j] = 0;
                    }
                }
            }
            if (col + i * blockDim.x + threadIdx.x < intermediate && kk + threadIdx.y * VEC + VEC <= k)
            {
                vec_t<T> val = *reinterpret_cast<const vec_t<T> *>(&W[(col + intermediate + i * blockDim.x + threadIdx.x) * k + kk + threadIdx.y * VEC]);
                T *p = reinterpret_cast<T *>(&val);
                for (int j = 0; j < VEC; j++)
                {
                    smem_up[i * blockDim.x + threadIdx.x][threadIdx.y * VEC + j] = p[j];
                }
            }
            else
            {
                for (int j = 0; j < VEC; j++)
                {
                    if (col + i * blockDim.x + threadIdx.x < intermediate && kk + threadIdx.y * VEC + j < k)
                    {
                        smem_up[i * blockDim.x + threadIdx.x][threadIdx.y * VEC + j] = W[(col + intermediate + i * blockDim.x + threadIdx.x) * k + kk + threadIdx.y * VEC + j];
                    }
                    else
                    {
                        smem_up[i * blockDim.x + threadIdx.x][threadIdx.y * VEC + j] = 0;
                    }
                }
            }
        }
        __syncthreads();
        for (std::int64_t i = 0; i < TM; i++)
        {
            for (std::int64_t kkk = 0; kkk < BK; kkk++)
            {
                float regx = toFloat(smem_x[i * blockDim.y + threadIdx.y][kkk]);
                for (std::int64_t j = 0; j < TN; j++)
                {
                    acc_gate[i][j] += regx * toFloat(smem_gate[j * blockDim.x + threadIdx.x][kkk]);
                    acc_up[i][j] += regx * toFloat(smem_up[j * blockDim.x + threadIdx.x][kkk]);
                }
            }
        }
        __syncthreads();
    }

    for (std::int64_t i = 0; i < TM; i++)
    {
        for (std::int64_t j = 0; j < TN; j++)
        {
            if (row + i * blockDim.y + threadIdx.y < m && col + j * blockDim.x + threadIdx.x < intermediate)
            {
                act[(row + i * blockDim.y + threadIdx.y) * ldc + col + j * blockDim.x + threadIdx.x] = static_cast<T>(silu(acc_gate[i][j]) * acc_up[i][j]);
            }
        }
    }
}

void gemm_silu_and_mul_launch(void *act,
                              const void *x,
                              const void *W,
                              std::int64_t m,
                              std::int64_t k,
                              std::int64_t intermediate,
                              std::int64_t ldc,
                              DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCKDIM, THREADPERBLOCKDIM);
    dim3 blockPerGrid((intermediate + BN - 1) / BN, (m + BM - 1) / BM);
    if (k % 4 != 0)
        throw std::runtime_error("gemm_launch: lda/ldb must be multiple of 4 for vectorized load");
    if (dtype == DType::Float32)
    {
        gemm_silu_and_mul_kernel<float><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(act), static_cast<const float *>(x), static_cast<const float *>(W), m, k, intermediate, ldc);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (dtype == DType::BFloat16)
    {
        gemm_silu_and_mul_kernel<__nv_bfloat16><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(act), static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(W), m, k, intermediate, ldc);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else
        throw std::runtime_error("unsupported dtype");
}
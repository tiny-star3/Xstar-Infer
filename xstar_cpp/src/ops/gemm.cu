#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "ops/gemm.h"
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

// 模板编译期分流
// 二进制两份(两个特化), 零运行时开销, no-trans 路径二进制隔离
template <typename T, bool TRANSB, bool HAS_BIAS>
__global__ void gemm_kernel(T *C,
                            const T *A,
                            const T *B,
                            const T *bias,
                            std::int64_t m,
                            std::int64_t k,
                            std::int64_t n,
                            std::int64_t lda,
                            std::int64_t ldb,
                            std::int64_t ldc)
{
    // tile 全局行/列起点
    std::int64_t row = blockIdx.y * BM;
    std::int64_t col = blockIdx.x * BN;
    // transB 为 false 时 smemB[BK][BN], 为 true 时 smemB[BN][BK]
    __shared__ T smemA[BM][BK], smemB[BK * BN];
    __shared__ T smemBias[BN];
    // 一个 thread 负责 TM*TN 个数据
    float acc[TM][TN] = {0.0f};
    for (std::int64_t kk = 0; kk < k; kk += BK)
    {
        for (std::int64_t i = 0; i < TM; i++)
        {
            std::int64_t A_idx = (row + i * blockDim.y + threadIdx.y) * lda + kk + threadIdx.x * VEC;
            if ((reinterpret_cast<std::uintptr_t>(&A[A_idx]) % sizeof(vec_t<T>)) == 0 && row + i * blockDim.y + threadIdx.y < m && kk + threadIdx.x * VEC + VEC <= k)
            {
                // reinterpret
                // 被加载的精确地址必须 16 对齐,否则非法访存
                vec_t<T> val = *reinterpret_cast<const vec_t<T> *>(&A[A_idx]);
                // 拆包
                T *p = reinterpret_cast<T *>(&val);
                for (int vv = 0; vv < VEC; vv++)
                {
                    smemA[i * blockDim.y + threadIdx.y][threadIdx.x * VEC + vv] = p[vv];
                }
            }
            else
            {
                // "部分越界"(末组起点在界、末尾越界)
                for (int vv = 0; vv < VEC; vv++)
                {
                    if (row + i * blockDim.y + threadIdx.y < m && kk + threadIdx.x * VEC + vv < k)
                    {
                        smemA[i * blockDim.y + threadIdx.y][threadIdx.x * VEC + vv] = A[(row + i * blockDim.y + threadIdx.y) * lda + kk + threadIdx.x * VEC + vv];
                    }
                    else
                    {
                        smemA[i * blockDim.y + threadIdx.y][threadIdx.x * VEC + vv] = 0;
                    }
                }
            }
        }
        // 编译期进行条件判断
        if constexpr (TRANSB)
        {
            for (std::int64_t i = 0; i < TN; i++)
            {
                std::int64_t B_idx = (col + i * blockDim.x + threadIdx.x) * ldb + kk + threadIdx.y * VEC;
                if ((reinterpret_cast<std::uintptr_t>(&B[B_idx]) % sizeof(vec_t<T>)) == 0 && col + i * blockDim.x + threadIdx.x < n && kk + threadIdx.y * VEC + VEC <= k)
                {
                    // reinterpret
                    // 被加载的精确地址必须 16 对齐,否则非法访存
                    vec_t<T> val = *reinterpret_cast<const vec_t<T> *>(&B[B_idx]);
                    // 拆包
                    T *p = reinterpret_cast<T *>(&val);
                    for (int vv = 0; vv < VEC; vv++)
                    {
                        smemB[(i * blockDim.x + threadIdx.x) * BK + threadIdx.y * VEC + vv] = p[vv];
                    }
                }
                else
                {
                    for (int vv = 0; vv < VEC; vv++)
                    {
                        if (col + i * blockDim.x + threadIdx.x < n && kk + threadIdx.y * VEC + vv < k)
                        {
                            smemB[(i * blockDim.x + threadIdx.x) * BK + threadIdx.y * VEC + vv] = B[(col + i * blockDim.x + threadIdx.x) * ldb + kk + threadIdx.y * VEC + vv];
                        }
                        else
                        {
                            smemB[(i * blockDim.x + threadIdx.x) * BK + threadIdx.y * VEC + vv] = 0;
                        }
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < VEC; i++)
            {
                std::int64_t B_idx = (kk + i * blockDim.y + threadIdx.y) * ldb + col + threadIdx.x * VEC;
                if ((reinterpret_cast<std::uintptr_t>(&B[B_idx]) % sizeof(vec_t<T>)) == 0 && col + threadIdx.x * VEC + VEC <= n && kk + i * blockDim.y + threadIdx.y < k)
                {
                    // reinterpret
                    // 被加载的精确地址必须 16 对齐,否则非法访存
                    // 原本的列访问变为行访问, 可以使用 float4 一次读取 4 个连续元素
                    vec_t<T> val = *reinterpret_cast<const vec_t<T> *>(&B[B_idx]);
                    // 拆包
                    T *p = reinterpret_cast<T *>(&val);
                    for (int vv = 0; vv < VEC; vv++)
                    {
                        smemB[(i * blockDim.y + threadIdx.y) * BN + threadIdx.x * VEC + vv] = p[vv];
                    }
                }
                else
                {
                    for (int vv = 0; vv < VEC; vv++)
                    {
                        if (col + threadIdx.x * VEC + vv < n && kk + i * blockDim.y + threadIdx.y < k)
                        {
                            smemB[(i * blockDim.y + threadIdx.y) * BN + threadIdx.x * VEC + vv] = B[(kk + i * blockDim.y + threadIdx.y) * ldb + col + threadIdx.x * VEC + vv];
                        }
                        else
                        {
                            smemB[(i * blockDim.y + threadIdx.y) * BN + threadIdx.x * VEC + vv] = 0;
                        }
                    }
                }
            }
        }
        __syncthreads();
        for (std::int64_t i = 0; i < TM; i++)
        {
            for (std::int64_t kkk = 0; kkk < BK; kkk++)
            {
                float regA = toFloat(smemA[i * blockDim.y + threadIdx.y][kkk]);
                for (std::int64_t j = 0; j < TN; j++)
                {
                    if constexpr (TRANSB)
                    {
                        acc[i][j] += regA * toFloat(smemB[(j * blockDim.x + threadIdx.x) * BK + kkk]);
                    }
                    else
                    {
                        acc[i][j] += regA * toFloat(smemB[kkk * BN + j * blockDim.x + threadIdx.x]);
                    }
                }
            }
        }
        __syncthreads();
    }
    // bias load
    if constexpr (HAS_BIAS)
    {
        // bias 与 K 无关, 整个 tile 只需 BN 个元素, 一次 load, 不进 K-loop
        // 一个 warp 协同填满 smemBias: 8 个 thread 各 load 4 列(或按需分配覆盖 BN=32)
        for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < BN; t += blockDim.x * blockDim.y)
        {
            // tile 内第 t 列对应全局列 col+t; 越界(boundary tile 的 n 不整除)填 0
            smemBias[t] = (col + t < n) ? bias[col + t] : T(0);
        }
        __syncthreads();
    }

    for (std::int64_t i = 0; i < TM; i++)
    {
        for (std::int64_t j = 0; j < TN; j++)
        {
            if (row + i * blockDim.y + threadIdx.y < m && col + j * blockDim.x + threadIdx.x < n)
            {
                float a = acc[i][j];
                if constexpr (HAS_BIAS)
                {
                    a += toFloat(smemBias[j * blockDim.x + threadIdx.x]);
                }
                C[(row + i * blockDim.y + threadIdx.y) * ldc + col + j * blockDim.x + threadIdx.x] = static_cast<T>(a);
            }
        }
    }
}

void gemm_launch(void *C,
                 const void *A,
                 const void *B,
                 const void *bias,
                 std::int64_t m,
                 std::int64_t k,
                 std::int64_t n,
                 bool transB,
                 std::int64_t lda,
                 std::int64_t ldb,
                 std::int64_t ldc,
                 DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCKDIM, THREADPERBLOCKDIM);
    dim3 blockPerGrid((n + BN - 1) / BN, (m + BM - 1) / BM);
    if (dtype == DType::Float32)
    {
        // transB 是运行时值, 模板实参要编译期, 运行时分派
        if (transB && bias != nullptr)
            gemm_kernel<float, true, true><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(C), static_cast<const float *>(A), static_cast<const float *>(B), static_cast<const float *>(bias), m, k, n, lda, ldb, ldc);
        else if (transB && bias == nullptr)
            gemm_kernel<float, true, false><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(C), static_cast<const float *>(A), static_cast<const float *>(B), nullptr, m, k, n, lda, ldb, ldc);
        else if (!transB && bias != nullptr)
            gemm_kernel<float, false, true><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(C), static_cast<const float *>(A), static_cast<const float *>(B), static_cast<const float *>(bias), m, k, n, lda, ldb, ldc);
        else
            gemm_kernel<float, false, false><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(C), static_cast<const float *>(A), static_cast<const float *>(B), nullptr, m, k, n, lda, ldb, ldc);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (dtype == DType::BFloat16)
    {
        // transB 是运行时值, 模板实参要编译期, 运行时分派
        if (transB && bias != nullptr)
            gemm_kernel<__nv_bfloat16, true, true><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(C), static_cast<const __nv_bfloat16 *>(A), static_cast<const __nv_bfloat16 *>(B), static_cast<const __nv_bfloat16 *>(bias), m, k, n, lda, ldb, ldc);
        else if (transB && bias == nullptr)
            gemm_kernel<__nv_bfloat16, true, false><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(C), static_cast<const __nv_bfloat16 *>(A), static_cast<const __nv_bfloat16 *>(B), nullptr, m, k, n, lda, ldb, ldc);
        else if (!transB && bias != nullptr)
            gemm_kernel<__nv_bfloat16, false, true><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(C), static_cast<const __nv_bfloat16 *>(A), static_cast<const __nv_bfloat16 *>(B), static_cast<const __nv_bfloat16 *>(bias), m, k, n, lda, ldb, ldc);
        else
            gemm_kernel<__nv_bfloat16, false, false><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(C), static_cast<const __nv_bfloat16 *>(A), static_cast<const __nv_bfloat16 *>(B), nullptr, m, k, n, lda, ldb, ldc);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else
        throw std::runtime_error("unsupported dtype");
}
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <cmath>

#include "ops/paged_attention.h"
#include "cuda/cuda_check.h"
#include "cuda/dtype_cast.h"

constexpr int THREADPERBLOCKDIM = 256;
// query 块大小
constexpr int Br = 64;
// key/value 块大小
constexpr int Bc = 64;
// 目前只支持 head_dim=64, 一个 lane 持有 1 query 行, 16 key/head_dim 列
// constexpr int ROW_PER_THREAD = 1;
constexpr int COL_PER_THREAD = 16;
// 一行 query 被 QUAD 共享
// constexpr int QUAD = 4;

template <typename T, bool IS_DECODE, int HEAD_DIM>
__global__ void paged_attention_kernel(T *out, const T *Q,
                                       const T *pool, const int *d_block_table,
                                       std::int64_t layer_stride_elems, int block_elems, int layer,
                                       int num_heads, int num_kv_heads,
                                       int seq_q, int seq_k, int block_size, float scalar)
{
    int qb = blockIdx.x;
    int batch = blockIdx.y;
    int head = blockIdx.z;
    int kv_head = head / (num_heads / num_kv_heads);
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    // 本 lane 的 query 行
    int row = warp * 8 + lane / 4;
    // 本 lane 的 head_dim/key 起始列
    int col_offset = (lane % 4) * 16;

    __shared__ T Qs[Br][HEAD_DIM];
    __shared__ T Ks[Bc][HEAD_DIM];
    __shared__ T Vs[Bc][HEAD_DIM];
    float O[COL_PER_THREAD] = {0};
    // lane 行最大值
    float m = -FLT_MAX;
    // lane 行累加
    float l = 0;

    // 合作加载 Qs
    for (int i = threadIdx.x; i < Br * HEAD_DIM; i += blockDim.x)
    {
        int local_row = i / HEAD_DIM;
        int dim = i % HEAD_DIM;
        int global_row = qb * Br + local_row;
        if (global_row < seq_q)
        {
            Qs[local_row][dim] = Q[batch * num_heads * seq_q * HEAD_DIM + head * seq_q * HEAD_DIM + global_row * HEAD_DIM + dim];
        }
        else
        {
            Qs[local_row][dim] = (T)0;
        }
    }
    __syncthreads();

    for (int i = 0; i < seq_k; i += Bc)
    {
        if constexpr (!IS_DECODE)
        {
            // 无 mask(纯 causal)且整块未来: 全掩贡献 0, 跳过; additive-mask 未来块有真值不跳
            if (i >= qb * Br + Br)
                continue;
        }
        // causal 三态: 整块未来->skip(见上), 整块过去->full_past 快路径(免逐元素), 跨对角线->逐元素; full_past 仅无 mask 路径用
        bool full_past = (i + Bc - 1 <= qb * Br);
        float m_old = m;
        float l_old = l;
        l = 0;
        // 合作加载 Ks, Vs
        for (int j = threadIdx.x; j < Bc * HEAD_DIM; j += blockDim.x)
        {
            int local_row = j / HEAD_DIM;
            int dim = j % HEAD_DIM;
            // 逻辑 token 号
            int global_key = i + local_row;
            if (global_key < seq_k)
            {
                // 物理块
                int block = d_block_table[global_key / block_size];
                int slot = global_key % block_size;
                Ks[local_row][dim] = pool[layer * layer_stride_elems + block * block_elems + kv_head * block_size * HEAD_DIM + slot * HEAD_DIM + dim];
                Vs[local_row][dim] = pool[layer * layer_stride_elems + block * block_elems + num_kv_heads * block_size * HEAD_DIM + kv_head * block_size * HEAD_DIM + slot * HEAD_DIM + dim];
            }
            else
            {
                Ks[local_row][dim] = (T)0;
                Vs[local_row][dim] = (T)0;
            }
        }
        __syncthreads();

        // Q @ K
        // lane 负责的 16 行 key
        float acc_s[COL_PER_THREAD] = {0};
        int global_query = qb * Br + row;
        int global_key = i + col_offset;
        for (int k = 0; k < HEAD_DIM; k++)
        {
            float Q_num = toFloat(Qs[row][k]);
            for (int j = 0; j < COL_PER_THREAD; j++)
            {
                acc_s[j] += Q_num * toFloat(Ks[col_offset + j][k]);
            }
        }
        if constexpr (IS_DECODE)
        {
            for (int t = 0; t < COL_PER_THREAD; t++)
            {
                if (global_query < seq_q && global_key + t < seq_k)
                {
                    acc_s[t] = acc_s[t] * scalar;
                }
                else
                {
                    acc_s[t] = -FLT_MAX;
                }
            }
        }
        else
        {
            if (full_past)
            {
                // 整块过去(免逐元素 causal)
                for (int t = 0; t < COL_PER_THREAD; t++)
                {
                    if (global_query < seq_q && global_key + t < seq_k)
                    {
                        acc_s[t] = acc_s[t] * scalar;
                    }
                    else
                    {
                        acc_s[t] = -FLT_MAX;
                    }
                }
            }
            else
            {
                // 跨对角线
                for (int t = 0; t < COL_PER_THREAD; t++)
                {
                    if (global_query < seq_q && global_key + t < seq_k && global_query >= global_key + t)
                    {
                        acc_s[t] = acc_s[t] * scalar;
                    }
                    else
                    {
                        acc_s[t] = -FLT_MAX;
                    }
                }
            }
        }

        // online softmax
        for (int j = 0; j < COL_PER_THREAD; j++)
        {
            float local_m_old = m;
            m = fmaxf(m, acc_s[j]);
            l = l * __expf(local_m_old - m) + __expf(acc_s[j] - m);
        }
        // quad allreduce
        float m2 = __shfl_xor_sync(0xffffffff, m, 2);
        float l2 = __shfl_xor_sync(0xffffffff, l, 2);
        if (m < m2)
        {
            l = l * __expf(m - m2) + l2;
            m = m2;
        }
        else
        {
            l = l + l2 * __expf(m2 - m);
        }
        m2 = __shfl_xor_sync(0xffffffff, m, 1);
        l2 = __shfl_xor_sync(0xffffffff, l, 1);
        if (m < m2)
        {
            l = l * __expf(m - m2) + l2;
            m = m2;
        }
        else
        {
            l = l + l2 * __expf(m2 - m);
        }

        // P @ V
        float P = 0;
        float N[HEAD_DIM] = {0};
        float ratio = __expf(m_old - m);
        l += ratio * l_old;
        for (int j = 0; j < COL_PER_THREAD; j++)
        {
            P = __expf(acc_s[j] - m);
            for (int k = 0; k < HEAD_DIM; k++)
            {
                N[k] += P * toFloat(Vs[col_offset + j][k]);
            }
        }
        // quad allreduce
        for (int j = 0; j < HEAD_DIM; j++)
        {
            N[j] += __shfl_xor_sync(0xffffffff, N[j], 2);
            N[j] += __shfl_xor_sync(0xffffffff, N[j], 1);
        }
        // 累加进 O
        for (int j = 0; j < COL_PER_THREAD; j++)
        {
            O[j] = O[j] * ratio + N[col_offset + j];
        }
        __syncthreads();
    }

    // 存回 out
    for (int i = 0; i < COL_PER_THREAD; i++)
    {
        int gq = qb * Br + row;
        if (gq < seq_q)
        {
            out[batch * seq_q * (num_heads * HEAD_DIM) + gq * (num_heads * HEAD_DIM) + head * HEAD_DIM + col_offset + i] = (T)(O[i] / l);
        }
    }
}

void paged_attention_launch(void *out, const void *Q,
                            const void *pool, const int *d_block_table,
                            std::int64_t layer_stride_elems, int block_elems, int layer,
                            int batch, int num_heads, int num_kv_heads,
                            int seq_q, int seq_k, int head_dim, int block_size,
                            bool is_decode, DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCKDIM);
    // Q 行块 batch head
    dim3 blockPerGrid((seq_q + Br - 1) / Br, batch, num_heads);
    float scalar = 1.0 / sqrt(head_dim);
    if (dtype == DType::Float32)
    {
        if (head_dim == 64)
        {
            if (is_decode)
            {
                paged_attention_kernel<float, true, 64><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(out), static_cast<const float *>(Q), static_cast<const float *>(pool), d_block_table, layer_stride_elems, block_elems, layer, num_heads, num_kv_heads, seq_q, seq_k, block_size, scalar);
            }
            else
            {
                paged_attention_kernel<float, false, 64><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(out), static_cast<const float *>(Q), static_cast<const float *>(pool), d_block_table, layer_stride_elems, block_elems, layer, num_heads, num_kv_heads, seq_q, seq_k, block_size, scalar);
            }
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        else if (head_dim == 128)
        {
            throw std::runtime_error("unsupported head_dim");
            // CHECK_CUDA(cudaGetLastError());
            // CHECK_CUDA(cudaDeviceSynchronize());
        }
        else if (head_dim == 256)
        {
            throw std::runtime_error("unsupported head_dim");
            // CHECK_CUDA(cudaGetLastError());
            // CHECK_CUDA(cudaDeviceSynchronize());
        }
        else
            throw std::runtime_error("unsupported head_dim");
    }
    else if (dtype == DType::BFloat16)
    {
        if (head_dim == 64)
        {
            if (is_decode)
            {
                paged_attention_kernel<__nv_bfloat16, true, 64><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(Q), static_cast<const __nv_bfloat16 *>(pool), d_block_table, layer_stride_elems, block_elems, layer, num_heads, num_kv_heads, seq_q, seq_k, block_size, scalar);
            }
            else
            {
                paged_attention_kernel<__nv_bfloat16, false, 64><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(Q), static_cast<const __nv_bfloat16 *>(pool), d_block_table, layer_stride_elems, block_elems, layer, num_heads, num_kv_heads, seq_q, seq_k, block_size, scalar);
            }
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        else if (head_dim == 128)
        {
            throw std::runtime_error("unsupported head_dim");
            // CHECK_CUDA(cudaGetLastError());
            // CHECK_CUDA(cudaDeviceSynchronize());
        }
        else if (head_dim == 256)
        {
            throw std::runtime_error("unsupported head_dim");
            // CHECK_CUDA(cudaGetLastError());
            // CHECK_CUDA(cudaDeviceSynchronize());
        }
        else
            throw std::runtime_error("unsupported head_dim");
    }
    else
        throw std::runtime_error("unsupported dtype");
}
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <cmath>

#include "ops/paged_attention.h"
#include "cuda/cuda_check.h"
#include "cuda/dtype_cast.h"

constexpr int THREADPERBLOCKDIM = 256;
// key/value 块大小
constexpr int Bc = 64;
// 目前只支持 head_dim=64, 一个 warp 负责 1 query 行, 一个 lane 持有 1 query 行, 2 key/head_dim 列
constexpr int NUM_WRAPS = THREADPERBLOCKDIM / 32;
constexpr int COL_PER_THREAD = 2;

template <typename T, int HEAD_DIM>
__global__ void paged_attention_decode_kernel(T *out, const T *Q, const T *pool,
                                              const int *d_block_table_2d, const int *cu_seqlens_k,
                                              int num_seqs, int max_blocks_per_seq,
                                              std::int64_t layer_stride_elems, int block_elems, int layer,
                                              int num_heads, int num_kv_heads, int head_dim, int block_size,
                                              float scalar)
{
    int seq_block = blockIdx.x;
    int head = blockIdx.y;
    int kv_head = head / (num_heads / num_kv_heads);
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    // 本 lane 的 head_dim/key 起始列
    int col_offset = lane * (COL_PER_THREAD);

    float O[COL_PER_THREAD] = {0};
    // lane 行最大值
    float m = -FLT_MAX;
    // lane 行累加
    float l = 0;
    T Qr[COL_PER_THREAD];
    T Kr[Bc][COL_PER_THREAD];
    T Vr[Bc][COL_PER_THREAD];

    // 当前 lane 负责的 query 行
    int lane_global_query = seq_block * NUM_WRAPS + warp;

    // 加载 Qr
    for (int i = 0; i < COL_PER_THREAD; i++)
    {
        if (lane_global_query < num_seqs)
        {
            Qr[i] = Q[head * num_seqs * HEAD_DIM + lane_global_query * HEAD_DIM + col_offset + i];
        }
        else
        {
            Qr[i] = (T)0;
        }
    }

    std::int64_t seq_k = (lane_global_query < num_seqs) ? cu_seqlens_k[lane_global_query + 1] - cu_seqlens_k[lane_global_query] : 0;
    for (int i = 0; i < seq_k; i += Bc)
    {
        float m_old = m;
        float l_old = l;
        l = 0;
        // 加载 Kr, Vr
        for (int local_row = 0; local_row < Bc; local_row++)
        {
            for (int local_col = 0; local_col < COL_PER_THREAD; local_col++)
            {
                // 逻辑 token 号
                int global_key = i + local_row;
                if (global_key < seq_k)
                {
                    // 物理块
                    int block = d_block_table_2d[(lane_global_query)*max_blocks_per_seq + global_key / block_size];
                    int slot = global_key % block_size;
                    Kr[local_row][local_col] = pool[layer * layer_stride_elems + block * block_elems + kv_head * block_size * HEAD_DIM + slot * HEAD_DIM + col_offset + local_col];
                    Vr[local_row][local_col] = pool[layer * layer_stride_elems + block * block_elems + num_kv_heads * block_size * HEAD_DIM + kv_head * block_size * HEAD_DIM + slot * HEAD_DIM + col_offset + local_col];
                }
                else
                {
                    Kr[local_row][local_col] = (T)0;
                    Vr[local_row][local_col] = (T)0;
                }
            }
        }
        // Q @ K
        // lane 负责的 2 行 key
        float acc_s[Bc] = {0};
        for (int k = 0; k < COL_PER_THREAD; k++)
        {
            float Q_num = toFloat(Qr[k]);
            for (int j = 0; j < Bc; j++)
            {
                acc_s[j] += Q_num * toFloat(Kr[j][k]);
            }
        }
        for (int t = 0; t < Bc; t++)
        {
            if (lane_global_query < num_seqs && i + t < seq_k)
            {
                acc_s[t] = acc_s[t] * scalar;
            }
            else
            {
                acc_s[t] = -FLT_MAX;
            }
        }

        // 归约部分和
        for (int j = 0; j < Bc; j++)
        {
            acc_s[j] += __shfl_xor_sync(0xffffffff, acc_s[j], 16);
            acc_s[j] += __shfl_xor_sync(0xffffffff, acc_s[j], 8);
            acc_s[j] += __shfl_xor_sync(0xffffffff, acc_s[j], 4);
            acc_s[j] += __shfl_xor_sync(0xffffffff, acc_s[j], 2);
            acc_s[j] += __shfl_xor_sync(0xffffffff, acc_s[j], 1);
        }

        // online softmax
        for (int j = 0; j < Bc; j++)
        {
            float local_m_old = m;
            m = fmaxf(m, acc_s[j]);
            l = l * __expf(local_m_old - m) + __expf(acc_s[j] - m);
        }

        // P @ V
        float P = 0;
        float N[COL_PER_THREAD] = {0};
        float ratio = __expf(m_old - m);
        l += ratio * l_old;
        for (int j = 0; j < Bc; j++)
        {
            P = __expf(acc_s[j] - m);
            for (int k = 0; k < COL_PER_THREAD; k++)
            {
                N[k] += P * toFloat(Vr[j][k]);
            }
        }
        // 累加进 O
        for (int j = 0; j < COL_PER_THREAD; j++)
        {
            O[j] = O[j] * ratio + N[j];
        }
    }

    // 存回 out
    for (int i = 0; i < COL_PER_THREAD; i++)
    {
        if (lane_global_query < num_seqs)
        {
            out[lane_global_query * (num_heads * HEAD_DIM) + head * HEAD_DIM + col_offset + i] = (T)(O[i] / l);
        }
    }
}

void paged_attention_decode_launch(void *out, const void *Q, const void *pool,
                                   const int *d_block_table_2d, const int *cu_seqlens_k,
                                   int num_seqs, int max_blocks_per_seq,
                                   std::int64_t layer_stride_elems, int block_elems, int layer,
                                   int num_heads, int num_kv_heads, int head_dim, int block_size,
                                   DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCKDIM);
    dim3 blockPerGrid((num_seqs + NUM_WRAPS - 1) / NUM_WRAPS, num_heads);
    float scalar = 1.0 / sqrt(head_dim);
    if (dtype == DType::Float32)
    {
        if (head_dim == 64)
        {
            paged_attention_decode_kernel<float, 64><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(out), static_cast<const float *>(Q), static_cast<const float *>(pool), d_block_table_2d, cu_seqlens_k, num_seqs, max_blocks_per_seq, layer_stride_elems, block_elems, layer, num_heads, num_kv_heads, head_dim, block_size, scalar);
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
            paged_attention_decode_kernel<__nv_bfloat16, 64><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(Q), static_cast<const __nv_bfloat16 *>(pool), d_block_table_2d, cu_seqlens_k, num_seqs, max_blocks_per_seq, layer_stride_elems, block_elems, layer, num_heads, num_kv_heads, head_dim, block_size, scalar);
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
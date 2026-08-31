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
// 将 seq_k 划分为 num_splits 个并行计算, 然后合并最终结果
// mid_lse[s][seq][head]
// mid_o[s][seq][head][c]

template <typename T, int HEAD_DIM>
__global__ void paged_attention_decode_split_kernel(float *mid_o, float *mid_lse,
                                                    const T *Q, const T *pool,
                                                    const int *d_block_table_2d, const int *cu_seqlens_k,
                                                    int num_seqs, int max_blocks_per_seq, int num_splits,
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
    std::int64_t seq_k = (lane_global_query < num_seqs) ? cu_seqlens_k[lane_global_query + 1] - cu_seqlens_k[lane_global_query] : 0;
    // 对齐 split 边界到 Bc 倍数
    int kv_len_per_split = ((seq_k + num_splits - 1) / num_splits + Bc - 1) / Bc * Bc;
    int split_start = blockIdx.z * kv_len_per_split;
    int split_end = min(split_start + kv_len_per_split, (int)seq_k);

    // 当 seq_k < num_splits
    if (split_start >= seq_k)
    {
        if (lane_global_query < num_seqs)
        {
            if (lane == 0)
            {
                // lse = m + log(l),也就是 log(Σ e^{s_j})
                mid_lse[blockIdx.z * num_seqs * num_heads + lane_global_query * num_heads + head] = -FLT_MAX;
            }
        }
        return;
    }

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

    for (int i = split_start; i < split_end; i += Bc)
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
                if (global_key < split_end)
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
            if (lane_global_query < num_seqs && i + t < split_end)
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

    for (int i = 0; i < COL_PER_THREAD; i++)
    {
        if (lane_global_query < num_seqs)
        {
            // f32 归一化 O'
            mid_o[blockIdx.z * num_seqs * num_heads * HEAD_DIM + lane_global_query * num_heads * HEAD_DIM + head * HEAD_DIM + col_offset + i] = O[i] / l;
        }
    }
    if (lane_global_query < num_seqs)
    {
        if (lane == 0)
        {
            // lse = m + log(l),也就是 log(Σ e^{s_j})
            mid_lse[blockIdx.z * num_seqs * num_heads + lane_global_query * num_heads + head] = m + logf(l);
        }
    }
}

template <typename T, int HEAD_DIM>
__global__ void paged_attention_decode_split_reduce_kernel(T *out,
                                                           const float *mid_o, const float *mid_lse,
                                                           int num_seqs, int num_splits,
                                                           int num_heads)
{
    int seq_block = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    // 本 lane 的 key 起始列
    int col_offset = lane * (COL_PER_THREAD);
    // 当前 lane 负责的 query 行
    int lane_global_query = seq_block * NUM_WRAPS + warp;
    if (lane_global_query >= num_seqs)
        return;

    // 数值上不能直接 e^{lse_s}, lse 可能很大导致 exp 溢出, 所以跑 running max, 一直除以当前最大值 e^{n_max}, 包括 e_sum, 最后除法相互抵消
    float e_max = -FLT_MAX, e_sum = 0.0f;
    float acc[COL_PER_THREAD] = {0};
    for (int s = 0; s < num_splits; s++)
    {
        float lse = mid_lse[s * num_seqs * num_heads + lane_global_query * num_heads + head];
        // 空 split
        if (lse == -FLT_MAX)
            continue;
        float n_max = fmaxf(lse, e_max);
        float old_scale = __expf(e_max - n_max);
        float w = __expf(lse - n_max);
        for (int k = 0; k < COL_PER_THREAD; k++)
        {
            acc[k] = acc[k] * old_scale + w * mid_o[s * num_seqs * num_heads * HEAD_DIM + lane_global_query * num_heads * HEAD_DIM + head * HEAD_DIM + col_offset + k];
        }
        e_sum = e_sum * old_scale + w;
        e_max = n_max;
    }
    for (int k = 0; k < COL_PER_THREAD; k++)
    {
        out[lane_global_query * (num_heads * HEAD_DIM) + head * HEAD_DIM + col_offset + k] = (T)(acc[k] / e_sum);
    }
}

void paged_attention_decode_split_launch(float *mid_o, float *mid_lse,
                                         const void *Q, const void *pool,
                                         const int *d_block_table_2d, const int *cu_seqlens_k,
                                         int num_seqs, int max_blocks_per_seq, int num_splits,
                                         std::int64_t layer_stride_elems, int block_elems, int layer,
                                         int num_heads, int num_kv_heads, int head_dim, int block_size,
                                         DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCKDIM);
    dim3 blockPerGrid((num_seqs + NUM_WRAPS - 1) / NUM_WRAPS, num_heads, num_splits);
    float scalar = 1.0 / sqrt(head_dim);
    if (dtype == DType::Float32)
    {
        if (head_dim == 64)
        {
            paged_attention_decode_split_kernel<float, 64><<<blockPerGrid, threadPerBlock>>>(mid_o, mid_lse, static_cast<const float *>(Q), static_cast<const float *>(pool), d_block_table_2d, cu_seqlens_k, num_seqs, max_blocks_per_seq, num_splits, layer_stride_elems, block_elems, layer, num_heads, num_kv_heads, head_dim, block_size, scalar);
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
            paged_attention_decode_split_kernel<__nv_bfloat16, 64><<<blockPerGrid, threadPerBlock>>>(mid_o, mid_lse, static_cast<const __nv_bfloat16 *>(Q), static_cast<const __nv_bfloat16 *>(pool), d_block_table_2d, cu_seqlens_k, num_seqs, max_blocks_per_seq, num_splits, layer_stride_elems, block_elems, layer, num_heads, num_kv_heads, head_dim, block_size, scalar);
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

void paged_attention_decode_split_reduce_launch(void *out,
                                                const float *mid_o, const float *mid_lse,
                                                int num_seqs, int num_splits,
                                                int num_heads, int head_dim,
                                                DType dtype)
{
    dim3 threadPerBlock(THREADPERBLOCKDIM);
    dim3 blockPerGrid((num_seqs + NUM_WRAPS - 1) / NUM_WRAPS, num_heads);
    if (dtype == DType::Float32)
    {
        if (head_dim == 64)
        {
            paged_attention_decode_split_reduce_kernel<float, 64><<<blockPerGrid, threadPerBlock>>>(static_cast<float *>(out), mid_o, mid_lse, num_seqs, num_splits, num_heads);
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
            paged_attention_decode_split_reduce_kernel<__nv_bfloat16, 64><<<blockPerGrid, threadPerBlock>>>(static_cast<__nv_bfloat16 *>(out), mid_o, mid_lse, num_seqs, num_splits, num_heads);
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

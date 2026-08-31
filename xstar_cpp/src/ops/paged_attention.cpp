#include <stdexcept>

#include "ops/paged_attention.h"
#include "cuda/cuda_allocator.h"

constexpr int SPLIT_KV_THRESHOLD = 512;
constexpr int SPLIT_KV_NUM_SPLITS = 8;

Tensor paged_attention(const BlockManager &bm, int layer, const Tensor &Q, const int *d_block_table, std::int64_t seq_k, int num_heads, int num_kv_heads, bool is_decode)
{
    std::int64_t seq_q = Q.shape()[Q.shape().size() - 2];
    std::int64_t head_dim = Q.shape()[Q.shape().size() - 1];

    if (num_heads % num_kv_heads != 0)
        throw std::runtime_error("rep not integral");

    std::int64_t batch = 1;
    Tensor result(std::vector<std::int64_t>{seq_q, num_heads * head_dim}, Q.dtype(), Q.device());

    if (Q.device() == Device::CUDA)
    {
        std::int64_t dz = static_cast<std::int64_t>(dtype_size(Q.dtype()));

        paged_attention_launch(result.data(), Q.data(), bm.pool_ptr(), d_block_table, bm.layer_stride() / dz, bm.block_bytes() / dz, layer, batch, num_heads, num_kv_heads, seq_q, seq_k, head_dim, bm.block_size(), is_decode, Q.dtype());

        return result;
    }
    else
        throw std::runtime_error("unsupported device");
}

Tensor paged_attention(const BlockManager &bm, int layer, const Tensor &Q, const int *d_block_table_2d, const int *cu_seqlens_q, const int *cu_seqlens_k, int num_seqs, int max_blocks_per_seq, int max_seqlen_q, int max_seqlen_k, int num_heads, int num_kv_heads, bool is_decode)
{
    std::int64_t head_dim = Q.shape().back();

    if (num_heads % num_kv_heads != 0)
        throw std::runtime_error("rep not integral");

    Tensor result(std::vector<std::int64_t>{Q.shape()[1], num_heads * head_dim}, Q.dtype(), Q.device());

    if (Q.device() == Device::CUDA)
    {
        std::int64_t dz = static_cast<std::int64_t>(dtype_size(Q.dtype()));

        if (is_decode)
        {
            if (max_seqlen_k > SPLIT_KV_THRESHOLD)
            {
                int num_splits = SPLIT_KV_NUM_SPLITS;
                float *mid_o = (float *)cuda_alloc(num_splits * num_seqs * num_heads * head_dim * sizeof(float));
                float *mid_lse = (float *)cuda_alloc(num_splits * num_seqs * num_heads * sizeof(float));
                paged_attention_decode_split_launch(mid_o, mid_lse, Q.data(), bm.pool_ptr(),
                                                    d_block_table_2d, cu_seqlens_k, num_seqs, max_blocks_per_seq, num_splits,
                                                    bm.layer_stride() / dz, bm.block_bytes() / dz, layer,
                                                    num_heads, num_kv_heads, head_dim, bm.block_size(), Q.dtype());
                paged_attention_decode_split_reduce_launch(result.data(), mid_o, mid_lse,
                                                           num_seqs, num_splits, num_heads, head_dim, Q.dtype());
                cuda_free(mid_o);
                cuda_free(mid_lse);
            }
            else
            {
                paged_attention_decode_launch(result.data(), Q.data(), bm.pool_ptr(), d_block_table_2d, cu_seqlens_k, num_seqs, max_blocks_per_seq, bm.layer_stride() / dz, bm.block_bytes() / dz, layer, num_heads, num_kv_heads, head_dim, bm.block_size(), Q.dtype());
            }
        }
        else
        {
            paged_attention_prefill_launch(result.data(), Q.data(), bm.pool_ptr(), d_block_table_2d, cu_seqlens_q, cu_seqlens_k, num_seqs, max_blocks_per_seq, max_seqlen_q, bm.layer_stride() / dz, bm.block_bytes() / dz, layer, num_heads, num_kv_heads, head_dim, bm.block_size(), Q.dtype());
        }

        return result;
    }
    else
        throw std::runtime_error("unsupported device");
}

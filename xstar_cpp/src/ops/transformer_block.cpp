#include <stdexcept>
#include <vector>
#include <algorithm>

#include "ops/transformer_block.h"
#include "bfloat16.h"
#include "ops/rmsnorm.h"
#include "ops/linear.h"
#include "ops/rope.h"
#include "ops/attention.h"
#include "ops/mlp.h"
#include "ops/head_split.h"
#include "ops/add.h"
#include "ops/attention_fa2.h"
#include "ops/paged_attention.h"
#include "ops/paged_write.h"
#include "cuda/cuda_allocator.h"

Tensor transformer_block(
    const Tensor &x,
    std::int64_t num_heads,
    const Tensor &ln1_w, const Tensor &ln2_w, float eps,
    const Tensor &q_w, const Tensor *q_b,
    const Tensor &k_w, const Tensor *k_b,
    const Tensor &v_w, const Tensor *v_b,
    const Tensor &o_w,
    const Tensor &gate_up_w, const Tensor &down_w,
    const Tensor &cache, const std::int64_t *positions,
    const Tensor *mask)
{
    if (x.shape().size() != 2)
        throw std::runtime_error("block rank mismatch");

    std::int64_t seq = x.shape()[x.shape().size() - 2];
    std::int64_t hidden = x.shape().back();
    if (num_heads <= 0)
        throw std::runtime_error("num_heads must be positive");
    std::int64_t head_dim = q_w.shape()[0] / num_heads;
    if (head_dim <= 0)
        throw std::runtime_error("head_dim must be positive");
    std::int64_t num_key_value_heads = k_w.shape()[0] / head_dim;

    if (q_w.shape()[0] % num_heads != 0)
        throw std::runtime_error("head_dim not integral");
    if (k_w.shape()[0] % head_dim != 0)
        throw std::runtime_error("num_kv not integral");
    if (num_heads % num_key_value_heads != 0)
        throw std::runtime_error("rep not integral");
    if (k_w.shape()[0] != v_w.shape()[0])
        throw std::runtime_error("k/v proj out mismatch");
    if (q_w.shape()[1] != hidden)
        throw std::runtime_error("hidden mismatch");
    if (o_w.shape()[1] != num_heads * head_dim)
        throw std::runtime_error("o_proj in mismatch");
    if (o_w.shape()[0] != hidden)
        throw std::runtime_error("o_proj out mismatch");
    if (ln1_w.shape()[0] != hidden || ln2_w.shape()[0] != hidden)
        throw std::runtime_error("ln weight mismatch");

    Tensor ln1_x = rmsnorm(x, ln1_w, eps);
    Tensor Q = linear(ln1_x, q_w, q_b);
    Tensor K = linear(ln1_x, k_w, k_b);
    Tensor V = linear(ln1_x, v_w, v_b);

    Tensor Q_head_split = head_split(Q, num_heads);
    Tensor K_head_split = head_split(K, num_key_value_heads);
    Tensor V_head_split = head_split(V, num_key_value_heads);

    Q_head_split = rope(Q_head_split, cache, positions);
    K_head_split = rope(K_head_split, cache, positions);
    // Tensor attn_out = attention(Q_head_split, K_head_split, V_head_split, mask);
    // FA2 仅 CUDA(CPU 回退到四步复合 attention)
    Tensor attn_out = (Q_head_split.device() == Device::CUDA) ? attention_fa2(Q_head_split, K_head_split, V_head_split, mask) : attention(Q_head_split, K_head_split, V_head_split, mask);
    Tensor attention_x(add(x, linear(attn_out, o_w, nullptr)));
    return add(attention_x, mlp(rmsnorm(attention_x, ln2_w, eps), gate_up_w, down_w));
}

Tensor transformer_block(
    const Tensor &x,
    std::int64_t num_heads,
    const Tensor &ln1_w, const Tensor &ln2_w, float eps,
    const Tensor &q_w, const Tensor *q_b,
    const Tensor &k_w, const Tensor *k_b,
    const Tensor &v_w, const Tensor *v_b,
    const Tensor &o_w,
    const Tensor &gate_up_w, const Tensor &down_w,
    const Tensor &rope_cache, const std::int64_t *positions,
    const Tensor *mask,
    KVCache &kv_cache, bool is_decode, std::int64_t layer_idx)
{
    if (x.shape().size() != 2)
        throw std::runtime_error("block rank mismatch");

    std::int64_t seq = x.shape()[x.shape().size() - 2];
    std::int64_t hidden = x.shape().back();
    if (num_heads <= 0)
        throw std::runtime_error("num_heads must be positive");
    std::int64_t head_dim = q_w.shape()[0] / num_heads;
    if (head_dim <= 0)
        throw std::runtime_error("head_dim must be positive");
    std::int64_t num_key_value_heads = k_w.shape()[0] / head_dim;

    if (q_w.shape()[0] % num_heads != 0)
        throw std::runtime_error("head_dim not integral");
    if (k_w.shape()[0] % head_dim != 0)
        throw std::runtime_error("num_kv not integral");
    if (num_heads % num_key_value_heads != 0)
        throw std::runtime_error("rep not integral");
    if (k_w.shape()[0] != v_w.shape()[0])
        throw std::runtime_error("k/v proj out mismatch");
    if (q_w.shape()[1] != hidden)
        throw std::runtime_error("hidden mismatch");
    if (o_w.shape()[1] != num_heads * head_dim)
        throw std::runtime_error("o_proj in mismatch");
    if (o_w.shape()[0] != hidden)
        throw std::runtime_error("o_proj out mismatch");
    if (ln1_w.shape()[0] != hidden || ln2_w.shape()[0] != hidden)
        throw std::runtime_error("ln weight mismatch");

    Tensor ln1_x = rmsnorm(x, ln1_w, eps);
    Tensor Q = linear(ln1_x, q_w, q_b);
    Tensor K = linear(ln1_x, k_w, k_b);
    Tensor V = linear(ln1_x, v_w, v_b);

    Tensor Q_head_split = head_split(Q, num_heads);
    Tensor K_head_split = head_split(K, num_key_value_heads);
    Tensor V_head_split = head_split(V, num_key_value_heads);

    Q_head_split = rope(Q_head_split, rope_cache, positions);
    K_head_split = rope(K_head_split, rope_cache, positions);

    // 写 cache
    kv_cache.write(layer_idx, K_head_split, V_head_split, is_decode);
    // 读 cache
    Tensor K_for_attn = is_decode ? kv_cache.k_view(layer_idx) : std::move(K_head_split);
    Tensor V_for_attn = is_decode ? kv_cache.v_view(layer_idx) : std::move(V_head_split);

    // Tensor attn_out = attention(Q_head_split, K_head_split, V_head_split, mask);
    // FA2 仅 CUDA(CPU 回退到四步复合 attention)
    Tensor attn_out = (Q_head_split.device() == Device::CUDA) ? attention_fa2(Q_head_split, K_for_attn, V_for_attn, mask) : attention(Q_head_split, K_for_attn, V_for_attn, mask);
    Tensor attention_x(add(x, linear(attn_out, o_w, nullptr)));
    return add(attention_x, mlp(rmsnorm(attention_x, ln2_w, eps), gate_up_w, down_w));
}

Tensor transformer_block(const Tensor &x,
                         std::int64_t num_heads,
                         const Tensor &ln1_w, const Tensor &ln2_w, float eps,
                         const Tensor &q_w, const Tensor *q_b,
                         const Tensor &k_w, const Tensor *k_b,
                         const Tensor &v_w, const Tensor *v_b,
                         const Tensor &o_w,
                         const Tensor &gate_up_w, const Tensor &down_w,
                         const Tensor &rope_cache, const std::int64_t *positions,
                         const Tensor *mask,
                         BlockManager &bm, PagedKVCache &kv_cache, bool is_decode, std::int64_t layer_idx)
{
    if (x.shape().size() != 2)
        throw std::runtime_error("block rank mismatch");

    std::int64_t seq = x.shape()[x.shape().size() - 2];
    std::int64_t hidden = x.shape().back();
    if (num_heads <= 0)
        throw std::runtime_error("num_heads must be positive");
    std::int64_t head_dim = q_w.shape()[0] / num_heads;
    if (head_dim <= 0)
        throw std::runtime_error("head_dim must be positive");
    std::int64_t num_key_value_heads = k_w.shape()[0] / head_dim;

    if (q_w.shape()[0] % num_heads != 0)
        throw std::runtime_error("head_dim not integral");
    if (k_w.shape()[0] % head_dim != 0)
        throw std::runtime_error("num_kv not integral");
    if (num_heads % num_key_value_heads != 0)
        throw std::runtime_error("rep not integral");
    if (k_w.shape()[0] != v_w.shape()[0])
        throw std::runtime_error("k/v proj out mismatch");
    if (q_w.shape()[1] != hidden)
        throw std::runtime_error("hidden mismatch");
    if (o_w.shape()[1] != num_heads * head_dim)
        throw std::runtime_error("o_proj in mismatch");
    if (o_w.shape()[0] != hidden)
        throw std::runtime_error("o_proj out mismatch");
    if (ln1_w.shape()[0] != hidden || ln2_w.shape()[0] != hidden)
        throw std::runtime_error("ln weight mismatch");

    Tensor ln1_x = rmsnorm(x, ln1_w, eps);
    Tensor Q = linear(ln1_x, q_w, q_b);
    Tensor K = linear(ln1_x, k_w, k_b);
    Tensor V = linear(ln1_x, v_w, v_b);

    Tensor Q_head_split = head_split(Q, num_heads);
    Tensor K_head_split = head_split(K, num_key_value_heads);
    Tensor V_head_split = head_split(V, num_key_value_heads);

    Q_head_split = rope(Q_head_split, rope_cache, positions);
    K_head_split = rope(K_head_split, rope_cache, positions);

    // 写 cache
    kv_cache.write(layer_idx, bm, K_head_split, V_head_split, is_decode);

    // Paged_Attention 仅 CUDA(CPU 回退到四步复合 attention)
    Tensor attn_out = (Q_head_split.device() == Device::CUDA) ? paged_attention(bm, layer_idx, Q_head_split, kv_cache.d_block_table(), kv_cache.cursor(), num_heads, num_key_value_heads, is_decode) : attention(Q_head_split, K_head_split, V_head_split, mask);
    Tensor attention_x(add(x, linear(attn_out, o_w, nullptr)));
    return add(attention_x, mlp(rmsnorm(attention_x, ln2_w, eps), gate_up_w, down_w));
}

Tensor transformer_block(const Tensor &x,
                         std::int64_t num_heads,
                         const Tensor &ln1_w, const Tensor &ln2_w, float eps,
                         const Tensor &q_w, const Tensor *q_b,
                         const Tensor &k_w, const Tensor *k_b,
                         const Tensor &v_w, const Tensor *v_b,
                         const Tensor &o_w,
                         const Tensor &gate_up_w, const Tensor &down_w,
                         const Tensor &rope_cache,
                         BlockManager &bm,
                         std::vector<PagedKVCache *> &kv_caches,
                         bool is_decode, std::int64_t layer_idx,
                         const std::int64_t *positions,
                         const std::vector<std::int64_t> &cu_seqlens_q_host)
{
    if (x.shape().size() != 2)
        throw std::runtime_error("block rank mismatch");
    if (x.device() == Device::CPU)
        throw std::runtime_error("multi-request block requires CUDA");

    std::int64_t seq = x.shape()[x.shape().size() - 2];
    std::int64_t hidden = x.shape().back();
    if (num_heads <= 0)
        throw std::runtime_error("num_heads must be positive");
    std::int64_t head_dim = q_w.shape()[0] / num_heads;
    if (head_dim <= 0)
        throw std::runtime_error("head_dim must be positive");
    std::int64_t num_key_value_heads = k_w.shape()[0] / head_dim;

    if (q_w.shape()[0] % num_heads != 0)
        throw std::runtime_error("head_dim not integral");
    if (k_w.shape()[0] % head_dim != 0)
        throw std::runtime_error("num_kv not integral");
    if (num_heads % num_key_value_heads != 0)
        throw std::runtime_error("rep not integral");
    if (k_w.shape()[0] != v_w.shape()[0])
        throw std::runtime_error("k/v proj out mismatch");
    if (q_w.shape()[1] != hidden)
        throw std::runtime_error("hidden mismatch");
    if (o_w.shape()[1] != num_heads * head_dim)
        throw std::runtime_error("o_proj in mismatch");
    if (o_w.shape()[0] != hidden)
        throw std::runtime_error("o_proj out mismatch");
    if (ln1_w.shape()[0] != hidden || ln2_w.shape()[0] != hidden)
        throw std::runtime_error("ln weight mismatch");

    Tensor ln1_x = rmsnorm(x, ln1_w, eps);
    Tensor Q = linear(ln1_x, q_w, q_b);
    Tensor K = linear(ln1_x, k_w, k_b);
    Tensor V = linear(ln1_x, v_w, v_b);

    Tensor Q_head_split = head_split(Q, num_heads);
    Tensor K_head_split = head_split(K, num_key_value_heads);
    Tensor V_head_split = head_split(V, num_key_value_heads);

    Q_head_split = rope(Q_head_split, rope_cache, positions);
    K_head_split = rope(K_head_split, rope_cache, positions);

    // 逐 seq write cache (每层写自己 region)
    int num_seqs = kv_caches.size();
    std::int64_t sum_q = Q_head_split.shape()[1];
    int block_size = kv_caches[0]->block_size();
    std::vector<int> slot_mapping(sum_q);

    // N 个 cache 各自 prepare_meta (只 layer 0 真 做)
    // 造整批 slot_mapping[sum_q] (host)
    for (int s = 0; s < num_seqs; s++)
    {
        std::int64_t start = cu_seqlens_q_host[s];
        std::int64_t len = cu_seqlens_q_host[s + 1] - start;
        kv_caches[s]->prepare_meta(layer_idx, bm, len, is_decode);

        const auto &bt = kv_caches[s]->block_table();
        if (is_decode)
        {
            std::int64_t pos = kv_caches[s]->cursor() - 1;
            slot_mapping[start] = bt[pos / block_size] * block_size + pos % block_size;
        }
        else
        {
            for (std::int64_t i = 0; i < len; i++)
            {
                slot_mapping[start + i] = bt[i / block_size] * block_size + i % block_size;
            }
        }
    }
    // 整批一次 paged_write
    paged_write(bm, layer_idx, K_head_split, V_head_split, slot_mapping.data());

    // 拼 2D block_table (host 收集 -> h2d)
    int max_blocks = 0;
    for (auto *kv : kv_caches)
    {
        max_blocks = std::max(max_blocks, (int)kv->block_table().size());
    }
    std::vector<int> block_table_2d(num_seqs * max_blocks, 0);
    for (int s = 0; s < num_seqs; s++)
    {
        const auto &bt = kv_caches[s]->block_table();
        for (int b = 0; b < (int)bt.size(); b++)
        {
            block_table_2d[s * max_blocks + b] = bt[b];
        }
    }
    int *d_block_table_2d = static_cast<int *>(cuda_alloc(num_seqs * max_blocks * sizeof(int)));
    cuda_memcpy_h2d(d_block_table_2d, block_table_2d.data(), num_seqs * max_blocks * sizeof(int));

    // cu_seqlens_k (device) = [0] + cumsum(post-write cursors)
    std::vector<int> cu_seqlens_k(num_seqs + 1, 0);
    for (int s = 0; s < num_seqs; s++)
    {
        cu_seqlens_k[s + 1] = cu_seqlens_k[s] + (int)kv_caches[s]->cursor();
    }
    int *d_cu_seqlens_k = static_cast<int *>(cuda_alloc((num_seqs + 1) * sizeof(int)));
    cuda_memcpy_h2d(d_cu_seqlens_k, cu_seqlens_k.data(), (num_seqs + 1) * sizeof(int));

    // cu_seqlens_q (device) + max_seqlen_q
    int *d_cu_seqlens_q = nullptr;
    int max_seqlen_q = 0;
    if (!is_decode)
    {
        d_cu_seqlens_q = static_cast<int *>(cuda_alloc((num_seqs + 1) * sizeof(int)));
        std::vector<int> cu_seqlens_q(num_seqs + 1);
        for (int s = 0; s <= num_seqs; s++)
        {
            cu_seqlens_q[s] = (int)cu_seqlens_q_host[s];
        }
        cuda_memcpy_h2d(d_cu_seqlens_q, cu_seqlens_q.data(), (num_seqs + 1) * sizeof(int));
        for (int s = 0; s < num_seqs; s++)
        {
            max_seqlen_q = std::max(max_seqlen_q, (int)(cu_seqlens_q_host[s + 1] - cu_seqlens_q_host[s]));
        }
    }

    // 多请求 Paged_Attention 仅 CUDA(CPU 回退到四步复合 attention)
    Tensor attn_out = paged_attention(bm, layer_idx, Q_head_split, d_block_table_2d, d_cu_seqlens_q, d_cu_seqlens_k, num_seqs, max_blocks, max_seqlen_q, num_heads, num_key_value_heads, is_decode);

    cuda_free(d_block_table_2d);
    cuda_free(d_cu_seqlens_k);
    if (d_cu_seqlens_q)
        cuda_free(d_cu_seqlens_q);

    Tensor attention_x(add(x, linear(attn_out, o_w, nullptr)));
    return add(attention_x, mlp(rmsnorm(attention_x, ln2_w, eps), gate_up_w, down_w));
}
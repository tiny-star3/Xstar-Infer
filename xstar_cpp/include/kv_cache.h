#pragma once
#include <cstdint>
#include "tensor.h"

/**
 * Continuous (non-paged) KV cache for single-request incremental decoding.
 * Preallocated to max_seq_len per layer, written by cursor (NOT torch.cat-append -- avoids HF DynamicCache's per-step full-history reallocation and transient ~2x peak).
 *
 * State:
 *   k_cache[layer] / v_cache[layer]: (num_kv_heads, max_seq_len, head_dim), dtype = weight dtype.
 *   cursor (int64): live token count; also the new token's absolute position.
 *
 * Stores POST-RoPE K and raw V (V is never RoPE'd in transformer_block -- only RoPE Q,K).
 *
 * NOT a model attribute: passed into qwen2_forward like rope_cache.
 * Weights are static / load-once / move-only; KVCache is per-sequence runtime state -- different lifecycle.
 * (Industrial: HF passes cache as a forward arg; vLLM/SGLang hold it at the runner because they run a scheduler -- single request needs neither, so pass-in suffices.)
 *
 * Compiled as g++ (CUDA-header-free, like block_manager.cpp):
 *   GPU copies go through the cuda_allocator d2d wrapper, NOT raw cudaMemcpy; CPU copies via std::memcpy.
 */
class KVCache
{
public:
    /**
     * Preallocate per-layer K/V buffers to max_seq_len (zero-filled; unwritten tail is garbage and must never be read -- k_view/v_view cut to cursor).
     * device follows the forward (CPU for the cuda_vs_cpu parity path, CUDA for the real model).
     */
    KVCache(std::int64_t num_layers, std::int64_t num_kv_heads,
            std::int64_t max_seq_len, std::int64_t head_dim,
            DType dtype, Device device);

    /**
     * Live token count; the new token's absolute position in decode.
     */
    std::int64_t cursor() const;

    /**
     * Copy this layer's rope-output K / raw V into the cache.
     *   is_decode=false (prefill): K,V are (num_kv_heads, seq, head_dim); write [0, seq); set cursor=seq (idempotent across layers -- all layers set the same value, no accumulation).
     *   is_decode=true  (decode):  K,V are (num_kv_heads, 1, head_dim).
     *     cursor advance happens ONCE per forward at layer 0 (cursor++ first, L->L+1), then row=cursor-1 is written; layers 1..N-1 see the stable cursor and also write row=cursor-1.
     *     All layers thus write the same row (cursor-1) and k_view returns [0, cursor) including the just-written token at row cursor-1.
     * Caller passes K/V AFTER rope (K) / after head_split only (V); the block does not compute offsets.
     *
     * Load-bearing invariants (the cache does NOT defend these -- caller / forward must):
     *   - forward MUST call write in order 0..N-1 with NO skipped layers, else cursor (advanced at layer 0) desynchronizes from the rows actually written.
     *   - prefill exactly ONCE on a fresh cache (cursor=0), then decode N times; a second prefill overwrites (cursor reset to seq, prior history lost).
     *
     * Copy granularity is NOT one memcpy: the buffer is (num_kv_heads, max_seq_len, head_dim) row-major, so one token's K/V is num_kv_heads disjoint rows (each head_dim wide, max_seq_len*head_dim apart) -- NOT contiguous.
     * Prefill writes a contiguous seq*head_dim run per head; decode writes one row per head.
     * Implement the per-head offset; do not assume one flat copy works for decode.
     *
     * Out-of-range guard: write throws "kv cache full: cursor >= max_seq_len" when a decode would write past the preallocated buffer (cursor == max_seq_len). Prefill is guarded by the seq >= max_seq_len_ shape check.
     */
    void write(std::int64_t layer, const Tensor &K, const Tensor &V, bool is_decode);

    /**
     * OWNED, CONTIGUOUS copy of this layer's LIVE region [0, cursor): shape (num_kv_heads, cursor, head_dim).
     * The buffer is (num_kv_heads, max_seq_len, head_dim) row-major with head-stride = max_seq_len*head_dim, but attention ops index K/V as contiguous (head-stride = cursor*head_dim) -- a mismatch (GQA: num_kv_heads>1) that reads garbage.
     * k_view/v_view COMPACT to a freshly-allocated contiguous tensor (per-head gather: source head-stride = max_seq_len*head_dim, dest head-stride = cursor*head_dim) so the consumer's contiguous indexing is correct.
     * This is PATH A (stopgap): decode KV-read bandwidth ~2x (read-strided + write-contiguous + re-read-contiguous). PATH B (paged kernel, sub-block B) replaces this with a stride-aware kernel (head-stride passed in, no compact) -- continuous cache is the block_size=max_seq_len, 1-block special case of paged. Remove the compact when B lands.
     * The tail [cursor, max_seq_len) is garbage and is excluded (only [0, cursor) is copied).
     * Decode attention consumes this as K/V (seq_k = cursor, includes the just-written token at row cursor-1).
     * Prefill does NOT call this (reads the temp rope output directly via std::move -- numerically identical to the cache, minimizes M6 regression risk); sub-block B will switch prefill to read the cache too.
     */
    Tensor k_view(std::int64_t layer) const;
    Tensor v_view(std::int64_t layer) const;

private:
    std::int64_t num_layers_, num_kv_heads_, max_seq_len_, head_dim_;
    DType dtype_;
    Device device_;
    std::int64_t cursor_;
    std::vector<Tensor> k_cache_; // [num_layers], each (num_kv_heads, max_seq_len, head_dim)
    std::vector<Tensor> v_cache_;
};

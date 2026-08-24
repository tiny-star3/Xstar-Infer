#pragma once
#include <cstdint>
#include <vector>

#include "tensor.h"
#include "block_manager.h"

/**
 * Paged per-sequence KV cache (M9 sub-block B): holds a logical->physical block_table + cursor, NOT any KV buffer -- K/V live in the shared BlockManager pool (passed to write/attn, not owned).
 * Paged analog of KVCache (continuous); KVCache is KEPT as the perf baseline.
 * Industrial form:
 *   vLLM/SGLang keep the pool global and the per-seq object holds only the block_table (+ an id to look it up); single-request here, so the block_table lives on this object directly.
 *
 * State:
 *   block_table (host std::vector<int>): logical block i -> physical block_id (from bm.alloc).
 *     Grows by one block when cursor crosses a block boundary (prefill: ceil(seq/BS) blocks at once; decode: +1 block every BS steps).
 *     The paged_attention kernel reads a DEVICE copy of this.
 *   d_block_table (device int*): device-resident copy of block_table (industrial form -- kernel reads device, scheduler updates each step).
 *     Re-copied (or appended) when block_table changes.
 *     Lifetime: owned, cuda_alloc'd/freed with the object.
 *   cursor (int64): live token count = new token's absolute position (same semantics as KVCache).
 *
 * Stores POST-RoPE K and raw V (same as KVCache), via paged_write into bm's pool.
 * NOT a model attr (passed into qwen2_forward like rope_cache/KVCache).
 * Compiled g++ (CUDA-header-free); device copies go through the cuda_allocator h2d wrapper.
 *
 * block_size: physical block token capacity (== bm.block_size(); must match). 0.5B GQA: BS=16.
 */
class PagedKVCache
{
public:
    PagedKVCache(std::int64_t num_kv_heads, std::int64_t head_dim,
                 std::int64_t max_seq_len, std::int64_t block_size,
                 DType dtype, Device device);
    ~PagedKVCache();

    std::int64_t cursor() const;
    /**
     * host view (host_id -> physical block_id)
     */
    const std::vector<int> &block_table() const;
    /**
     * device-resident copy for the attn kernel
     */
    const int *d_block_table() const;
    int block_size() const;

    /**
     * paged_write this layer's rope-output K / raw V into bm's pool at the cursor's slot(s).
     *   is_decode=false (prefill): K,V are (nkv, seq, hd); alloc ceil(seq/BS) blocks (if cursor==0), build slot_mapping = block_table[logical]*BS + slot_in_block for each token, paged_write.
     *     cursor=seq.
     *   is_decode=true  (decode):  K,V are (nkv, 1, hd); cursor++ (ONCE at layer 0, same as KVCache), alloc +1 block if cursor crosses a boundary, slot_mapping = [last_block_id*BS + (cursor-1)%BS], paged_write.
     *     All layers write the same slot; d_block_table is current.
     * Caller passes K/V AFTER rope (K) / head_split only (V); bm is the shared pool.
     *
     * CONTRACT (caller/forward must defend; cache does NOT):
     *   - forward calls write in order 0..N-1, no skipped layers (cursor advanced at layer 0).
     *   - prefill ONCE on a fresh cache (cursor=0); a second prefill frees old blocks + re-allocs.
     *   - bm must have enough free blocks for the prompt (prefill) / +1 (decode boundary) -- throws "paged kv cache: insufficient free blocks" (propagated from bm.alloc) if not.
     *   - bm.num_layers() spans all layers; write targets layer `layer` in the pool.
     *   - block_size == bm.block_size() (checked at construction).
     * Throws "paged kv cache full: cursor >= max_seq_len" when a decode would exceed max_seq_len.
     */
    void write(std::int64_t layer, BlockManager &bm, const Tensor &K, const Tensor &V, bool is_decode);

    /**
     * Meta-only half of write (Phase 3 multi-request): alloc blocks + advance cursor + grow block_table + d_block_table h2d, NO K/V scatter.
     * Lets the caller gather a whole-batch slot_mapping once and paged_write all seqs in one launch (vLLM reshape_and_cache form).
     * Gated layer==0; MUST precede slot_mapping build (decode slot uses post-increment cursor, prefill slot uses the grown block_table).
     *   len: prefill = prompt length (drives ceil(len/BS) alloc); decode = 1 (unused, alloc is cursor%BS-gated).
     */
    void prepare_meta(std::int64_t layer, BlockManager &bm, std::int64_t len, bool is_decode);

    /**
     * Restore to fresh-cache state for Recompute preemption.
     * Pre-condition: scheduler has already bm.free(block_table()).
     * Does NOT touch bm (no reference by design).
     */
    void reset();

private:
    std::int64_t num_kv_heads_;
    std::int64_t head_dim_;
    std::int64_t max_seq_len_;
    std::int64_t block_size_; // BS, == bm.block_size() (checked first write)
    DType dtype_;
    Device device_;
    std::int64_t cursor_;            // live token count, same semantics as KVCache
    std::vector<int> block_table_;   // host: logical block i -> physical block_id
    int *d_block_table_;             // device copy, cuda_alloc'd; resized+recopied when block_table_ grows
    std::int64_t d_block_table_cap_; // capacity of d_block_table_ (in ints), to know when to realloc
    bool block_size_checked_;        // one-shot: first write asserts bm.block_size()==block_size_
};

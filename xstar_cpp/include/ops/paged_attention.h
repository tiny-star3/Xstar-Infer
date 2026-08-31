#pragma once

#include "tensor.h"
#include "block_manager.h"

/**
 * Paged attention (M9 sub-block B, read half).
 * Reads K/V from BlockManager pool via block_table (logical block -> physical block_id), not from a contiguous buffer.
 *   decode:  Q=[nh,1,hd],  seq_k=cursor
 *   prefill: Q=[nh,seq,hd], seq_k>=seq_q (self-attn when seq_k==seq_q; extend when seq_k>seq_q, e.g. radix adopt_prefix: Q = residual, K/V = full [0,cursor) via block_table)
 *   GQA: Q=nh heads, K/V=nkv heads, rep=nh/nkv
 * K/V element offset for logical token t:
 *   block=block_table[t/BS], slot=t%BS
 *   K=pool + layer*layer_stride_elems + block*block_elems + head*BS*hd + slot*hd + d
 *   V=... + nkv*BS*hd + head*BS*hd + slot*hd + d
 * Same [nkv,BS,hd] K-then-V layout as paged_write (offsets in elements).
 * block_table: device const int* len ceil(seq_k/BS).
 * batch=1; batch axis in grid dim -> future N only touches launcher.
 * Same online-softmax order as FA2 -> bit-exact vs contiguous path.
 * Causal alignment: query rows are LOCAL to Q (residual), keys are ABSOLUTE ([0,seq_k)).
 * offset = seq_k - seq_q; query local p attends keys <= offset + p.
 * Skip/full_past/compare all use the absolute diagonal. Degenerates to old form when seq_k==seq_q.
 */
Tensor paged_attention(const BlockManager &bm, int layer,
                       const Tensor &Q,
                       const int *d_block_table, std::int64_t seq_k,
                       int num_heads, int num_kv_heads,
                       bool is_decode);

/**
 * CUDA launch (CUDA-free decl so .cpp links without CUDA header).
 */
void paged_attention_launch(void *out, const void *Q,
                            const void *pool, const int *d_block_table,
                            std::int64_t layer_stride_elems, int block_elems, int layer,
                            int batch, int num_heads, int num_kv_heads,
                            int seq_q, int seq_k, int head_dim, int block_size,
                            bool is_decode, DType dtype);

/**
 * Paged attention, multi-request (Phase 3): prefill varlen + decode multi-seq.
 *   prefill: Q=[sum_q, num_heads, hd] (varlen, concatenated);
 *            3D grid(ceil(max_seqlen_q/Br), num_seqs, num_heads);
 *            blockIdx.x = intra-seg q block, y = seq, z = head; seq located by direct index cu_seqlens_q[seq] (no binary search);
 *            short-seg early-exit (qb*Br >= seq_q); causal INTRA-segment (seg-local), extend form:
 *            q_global = (seq_k-seq_q) + seg_q_pos vs k_global = seg_k_pos;
 *            degenerates to self-attn causal when seq_k==seq_q (Phase3 full-prefill path unchanged).
 *   decode:  Q=[num_seqs, num_heads, hd] (1 Q token/seq);
 *            grid(ceil(num_seqs/NUM_WARPS), num_heads); NUM_WARPS seqs packed per block;
 *            per-seq seq_k from cu_seqlens_k; cu_seqlens_q UNUSED in decode.
 *            max_seqlen_k (host) = max per-seq seq_k: when it exceeds SPLIT_KV_THRESHOLD, decode routes to the split-KV two-stage kernel (per-split partial + LSE merge), else the single decode kernel. Unused in prefill.
 *   K/V read from pool via 2D block_table [num_seqs, max_blocks_per_seq] (logical block -> physical);
 *            K/V spans the full seq_k (including radix-forked prefix blocks), Q covers only the extend segment.
 *   caller contract: prefill takes TWO cu_seqlens (q = extend-segment cumsum, k = full-length cumsum);
 *            k segment length >= q segment length (extend). decode cu_seqlens_k semantics unchanged (cumsum of cursors).
 *   N independent PagedKVCache (per-seq block_table + cursor); caller (forward) gathers them into the 2D block_table + cu_seqlens arrays (vLLM form: per-seq List[int] -> [num_seqs, max_blocks]).
 *   max_seqlen_q: max seg q-length (prefill grid.x basis; ignored in decode).
 *   GQA: Q=num_heads, K/V=num_kv_heads, rep=num_heads/num_kv_heads.
 *   No padding mask: decode per-seq independent; prefill varlen segments isolate requests (intra-seg causal only).
 *   result rows = Q.shape()[0] (decode: num_seqs; prefill: sum_q). Same online-softmax order as FA2.
 */
Tensor paged_attention(const BlockManager &bm, int layer,
                       const Tensor &Q,
                       const int *d_block_table_2d, // [num_seqs, max_blocks_per_seq], device
                       const int *cu_seqlens_q,     // [num_seqs+1], device; decode: [0,1,2,..]
                       const int *cu_seqlens_k,     // [num_seqs+1], device; decode: cumsum(cursor)
                       int num_seqs, int max_blocks_per_seq, int max_seqlen_q, int max_seqlen_k,
                       int num_heads, int num_kv_heads,
                       bool is_decode);

/**
 * CUDA launch (CUDA-free decl so .cpp links without CUDA header).
 */
void paged_attention_decode_launch(void *out, const void *Q, const void *pool,
                                   const int *d_block_table_2d, const int *cu_seqlens_k,
                                   int num_seqs, int max_blocks_per_seq,
                                   std::int64_t layer_stride_elems, int block_elems, int layer,
                                   int num_heads, int num_kv_heads, int head_dim, int block_size,
                                   DType dtype);

void paged_attention_prefill_launch(void *out, const void *Q, const void *pool,
                                    const int *d_block_table_2d,
                                    const int *cu_seqlens_q, const int *cu_seqlens_k,
                                    int num_seqs, int max_blocks_per_seq, int max_seqlen_q,
                                    std::int64_t layer_stride_elems, int block_elems, int layer,
                                    int num_heads, int num_kv_heads, int head_dim, int block_size,
                                    DType dtype);

void paged_attention_decode_split_launch(float *mid_o, float *mid_lse,
                                         const void *Q, const void *pool,
                                         const int *d_block_table_2d, const int *cu_seqlens_k,
                                         int num_seqs, int max_blocks_per_seq, int num_splits,
                                         std::int64_t layer_stride_elems, int block_elems, int layer,
                                         int num_heads, int num_kv_heads, int head_dim, int block_size,
                                         DType dtype);

void paged_attention_decode_split_reduce_launch(void *out,
                                                const float *mid_o, const float *mid_lse,
                                                int num_seqs, int num_splits,
                                                int num_heads, int head_dim,
                                                DType dtype);

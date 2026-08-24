#pragma once
#include "tensor.h"

/**
 * FlashAttention-2: fused Q@K^T -> softmax -> scores@V in one pass, scores never materialized.
 * SAME I/O contract as attention() (thin GQA, already-projected Q/K/V, merged output, o_proj is caller's job) -> drop-in replacement for the four-step composite, parity-testable against attention() on identical inputs.
 *
 * Contract (SAME as attention()):
 *   Q: (num_heads, seq_q, head_dim); K,V: (num_kv_heads, seq_k, head_dim); rep=num_heads/num_kv_heads, h shares KV (h/rep), no repetition.
 *   mask: optional additive (seq_q, seq_k) broadcast across heads; nullptr -> causal on the fly.
 *   out: (seq_q, num_heads*head_dim), dtype == Q.dtype.
 *
 * FlashAttention-2 core:
 *   K/V tiled into Bc-blocks; a Q-tile block SERIALLY loops over all K/V blocks (the seq axis is NOT parallelized -- block j+1 depends on block j's running (m,l,O)).
 *   Online softmax maintains running m=rowmax, l=rowsum, O=output,
 *   rescaling at each block to a unified max:
 *     m_new=max(m_old,m_j); O*=exp(m_old-m_new); l*=exp(m_old-m_new);  -- OLD accumulators, (m_old-m_new) NEGATIVE -> shrink
 *     P=exp(S_j-m_new); O+=P@V_j; l+=sum(P);                            -- current block uses UNIFIED m_new, NOT local m_j
 *   Decode (seq_q=1, Q at position seq_k-1): Br=1, all keys visible -> NO mask branch (compile-pruned).
 *   Online softmax degenerates to cross-K/V-block combine (still needed -- K/V is long, must be tiled).
 *
 * Numerics: Q@K^T and P@V accumulate in f32; scale 1/sqrt(head_dim) folded into Q@K^T in-f32; online softmax in f32; output RNE-cast to T once.
 *   NOT bit-exact vs attention() (softmax combine order differs) -> allclose, tolerance PROBED per prefill/decode (do not copy attention()'s tolerance).
 *   BFloat16: one downcast at output.
 *
 * Parity: oracle is attention() -- same Q/K/V/mask, allclose(fa2_out, ref_out). No new oracle.
 * Preconditions: Q,K,V 3-D contiguous same dtype (Float32|BFloat16), same seq_k, same head_dim, rep integral.
 * Throws std::runtime_error on rank/shape/dtype/device mismatch, rep not integral, unsupported dtype.
 */
Tensor attention_fa2(const Tensor &Q, const Tensor &K, const Tensor &V, const Tensor *mask);

/**
 * Launch the FlashAttention-2 kernel (.cu side; declared CUDA-free so attention_fa2.cpp links it without a CUDA header).
 * One templated kernel <bool IS_DECODE>, dispatched at runtime; two thin launch shells (prefill carries mask, decode omits it).
 *
 * Grid prefill: (ceil(seq_q/Br), batch, num_heads); grid decode: (1, batch, num_heads). K/V axis serial (online-softmax dependency).
 * Qs[Br][head_dim] loaded once; Ks/Vs[Bc][head_dim] reloaded each iter; O per lane: O[COL_PER_THREAD], scalar m, scalar l in REGISTERS (row-parallel, one query row per 4-lane quad; O never materialized).
 * GQA via h/rep. Br/Bc=64.
 */
void flash_attention2_prefill_launch(void *out, const void *Q, const void *K, const void *V, const void *mask,
                                     int batch, int num_heads, int num_kv_heads, int seq_q, int seq_k, int head_dim, DType dtype);
void flash_attention2_decode_launch(void *out, const void *Q, const void *K, const void *V,
                                    int batch, int num_heads, int num_kv_heads, int seq_k, int head_dim, DType dtype);

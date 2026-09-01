#pragma once
#include "tensor.h"
#include "kv_cache.h"
#include "paged_kv_cache.h"

/**
 * Pre-norm Transformer decoder block (Qwen2), two residual sub-layers:
 *     x = x + attn( input_layernorm(x) )
 *     x = x + mlp ( post_attention_layernorm(x) )
 * This is the FIRST integration op: it assembles the already-tested single ops (rmsnorm, linear, rope, attention, mlp) into a full block, adding only the two residual adds (inlined -- three-call rule not met; extract an `add` op only when a third same-shape residual consumer appears).
 *
 * Attention stays THIN: this block owns the q/k/v/o projections + RoPE, then calls the thin attention op (which consumes already-projected, already-rotated Q/K/V and does NOT repeat KV -- GQA is realized by attention's internal h/rep indexing).
 * So the block's attention path is:
 *   ln1_x = rmsnorm(x, ln1_w, eps)
 *   Q = linear(ln1_x, q_w, q_b)   -> rearrange to (num_heads, seq, head_dim)
 *   K = linear(ln1_x, k_w, k_b)   -> rearrange to (num_key_value_heads, seq, head_dim)   // NOT repeated
 *   V = linear(ln1_x, v_w, v_b)   -> rearrange to (num_key_value_heads, seq, head_dim)   // NOT repeated
 *   Q = rope(Q, cache, positions);  K = rope(K, cache, positions)
 *   attn_out = attention(Q, K, V, mask)   -> (seq, num_heads * head_dim) merged
 *   x = x + linear(attn_out, o_w, nullptr)          // residual 1 (o_proj bias=False)
 *   x = x + mlp( rmsnorm(x, ln2_w, eps), gate_up_w, down_w )   // residual 2
 *
 * Head-split order CONTRACT (must match attention's h/rep assumption):
 *   Q/K/V come out of linear as (..., seq, heads*head_dim).
 *   Splitting the last axis into (heads, head_dim) is row-major: head h occupies columns [h*head_dim, (h+1)*head_dim).
 *   attention's GQA index K[h/rep] assumes query head h maps to KV head h/rep, i.e. the merged head axis is ordered (kv0, kv0, ..., kv1, kv1, ...) -- each KV head contiguous for `rep` slots.
 *   This matches xstar/layers/attention.py's rearrange("... (k_heads rep) ...").
 *   The block's head-split MUST produce this order; a wrong split (e.g. interleaved) silently breaks GQA.
 *
 * RoPE cache + positions are PASSED IN (stateless, like rope op itself): cache is the shared (2, max_seq_len, head_dim/2) table, positions is the int64 buffer of length seq.
 * The block does NOT own or build the cache -- Qwen2Model builds one cache and passes it to every block.
 *
 * Numerics: this op adds NO new floating-point logic -- every compute step delegates to an already-tested op.
 * The two residual adds are plain elementwise add; bf16 residuals follow each operand's dtype (upcast to add, downcast once) -- see the inline add note below.
 * f32 block vs Python reference: allclose (each sub-op is allclose vs its own reference, errors compose).
 * bf16 block: allclose with the project's bf16 discipline (the residual downcast path is the one new rounding site -- measure it, do not assume 1e-2).
 *
 * Parameter groups:
 *   norm:     ln1_w, ln2_w (RMSNorm weights, shape (hidden,)), eps
 *   attn:     q_w, q_b, k_w, k_b, v_w, v_b (projections, q/k/v bias=True -> non-null), o_w (bias=False)
 *   mlp:      gate_up_w (2*intermediate, hidden), down_w (hidden, intermediate)
 *   runtime:  x (seq, hidden), cache (2, max_seq, hd/2), positions (seq,), mask (seq,seq) or nullptr
 *
 * Shapes:
 *   x:        (seq_len, hidden)            -- rank == 2, contiguous, last axis = hidden
 *   ln1_w,ln2_w: (hidden,)                       -- 1-D
 *   q_w: (num_heads*head_dim, hidden);  q_b: (num_heads*head_dim,)
 *   k_w,v_w: (num_key_value_heads*head_dim, hidden);  k_b,v_b: (num_key_value_heads*head_dim,)
 *   o_w: (hidden, num_heads*head_dim)
 *   gate_up_w: (2*intermediate, hidden);  down_w: (hidden, intermediate)
 *   cache: (2, max_seq_len, head_dim/2), f32;  positions: int64[seq_len] in [0, max_seq_len)
 *   mask: (seq, seq) additive or nullptr (nullptr -> attention builds causal)
 *   out: (seq_len, hidden) -- same shape as x, dtype == x.dtype
 *
 * Precondition: hidden = num_heads*head_dim = num_key_value_heads*head_dim + (GQA via rep); num_heads % num_key_value_heads == 0; all weights match x.dtype; ln/q/k/v/o/mlp shapes consistent.
 * Throws std::runtime_error on: any sub-op's validation failure (surfaced), plus block-local checks (head_dim divides hidden, rep integral).
 */
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
    const Tensor *mask);

/**
 * Incremental variant: reads/writes a continuous KV cache.
 *
 * K/V flow change vs the non-incremental block:
 *   K = linear -> head_split -> rope;  V = linear -> head_split (no rope).
 *   kv_cache->write(layer_idx, K_head, V_head, is_decode)  -- copies rope-output K / raw V into cache attention reads K/V FROM the cache when is_decode (history + new); prefill still reads the temp rope output directly (numerically identical to cache -- minimizes M6 regression risk; paged sub-block B will switch prefill to read the cache too).
 *
 * is_decode dispatch (explicit, not inferred from seq_q==1):
 *   false (prefill): x is the full prompt (seq rows); write fills [0, seq); attention over full seq.
 *   true  (decode):  x is 1 new token (1 row); write appends at cursor; attention reads [0, cursor].
 *
 * layer_idx: which cache layer to write/read (forward passes the loop index i).
 *   The non-incremental block is stateless and reused across layers; this variant needs the index because the cache is per-layer.
 *
 * positions: caller supplies the new token's absolute position(s); for decode that is [cursor].
 *   The block does NOT track position -- it trusts positions matches x's seq dim (as in M6/M8).
 *
 * Parity: oracle is the non-incremental transformer_block (same weights, same x).
 *   The cache copy is a contiguous bf16 memcpy (bit-exact, no arithmetic), so the only new numeric site is that decode attention runs 1 Q vs the prefill last-row Q -- mathematically equal but 24-layer bf16 accumulation order may differ from full-sequence prefill.
 *   Tolerance PROBED end-to-end, not assumed from M6.
 */
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
                         KVCache &kv_cache, bool is_decode, std::int64_t layer_idx);

/**
 * Paged transformer_block: K/V paged_written into bm's pool, paged_attention reads pool via block_table.
 * K=linear->head_split->rope, V=linear->head_split.
 * kv_cache.write -> paged_write; attention reads pool in BOTH modes (no is_decode?k_view:std::move(K) fork -- prefill/decode unify on write pool + read pool).
 *
 * is_decode: false=prefill (fills [0,seq), full-seq causal, seq_q=seq_k=seq); true=decode (fills cursor slot, 1-Q over [0,cursor), seq_q=1, no mask).
 *   layer_idx: pool layer (base = layer_idx*layer_stride).
 *
 * Parity: oracle = continuous transformer_block(KVCache&); pool write bit-exact bf16 scatter, new site is paged attn vs continuous FA2 on same K/V (tile=64==Bc, bit-exact goal).
 *   Tolerance PROBED.
 */
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
                         BlockManager &bm, PagedKVCache &kv_cache, bool is_decode, std::int64_t layer_idx);

/**
 * Multi-request transformer block (Phase 3): N PagedKVCache -> one batched paged_attention call.
 *   prefill: x=[sum_q, hidden] (varlen concat); per-seq write + 2D block_table gather + cu_seqlens.
 *   decode:  x=[num_seqs, hidden] (1 token/seq); per-seq 1-token slice.
 * Caller (forward) builds `positions` (varlen: prefill=seg-local arange concat, decode=per-seq cursor) before this.
 * `cu_seqlens_q_host` (len num_seqs+1, q cumsum) drives BOTH K/V slice offsets AND the prefill cu_seqlens_q device array: prefill = [0, len0, len0+len1, ...]; decode = [0,1,...,num_seqs] (each seq 1 token).
 * This layer gathers block_tables into a 2D device buffer (padded with 0, unread past seq_k -- matches vLLM) + builds cu_seqlens_k (device, [0]+cumsum of post-write cursors).
 * Per-seq K/V slice = non-owning view (byte offset into the varlen-concat K/V tensor), passed to write.
 * Device buffers (d_bt2d, d_cu_k, d_cu_q) are temporary per-call (freed after kernel sync).
 * num_splits: passed through UNMODIFIED to paged_attention (decode only; prefill ignores it). The block makes no split-policy decision -- threshold/chunk/cap live in paged_attention (single source of truth).
 */
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
                         const std::vector<std::int64_t> &cu_seqlens_q_host, int num_splits);

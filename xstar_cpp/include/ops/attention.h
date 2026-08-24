#pragma once
#include "tensor.h"

/**
 * Grouped-Query Attention, THIN variant: the score + weighted-sum core only.
 * Projections (q/k/v/o_proj) and RoPE are the CALLER's job -- this op consumes already-projected, already-rotated Q/K/V and emits the merged multi-head output, ready for the caller's o_proj. (Decision A: thin, not end-to-end.)
 *
 * Contract:
 *   Q: (num_heads,           seq, head_dim)  -- already RoPE-applied
 *   K: (num_key_value_heads, seq, head_dim)  -- already RoPE-applied, NOT repeated
 *   V: (num_key_value_heads, seq, head_dim)  -- NOT repeated (V receives no RoPE)
 *   mask: optional ADDITIVE mask, (seq, seq), broadcast across heads.
 *         nullptr  -> build a causal mask on the fly (qi >= kj visible, else -inf).
 *         non-null -> add to the scaled scores; caller puts -inf / large negative at hidden positions.
 *                     EITHER/OR: an external mask REPLACES the causal mask, it does not stack on top.
 *   out: (seq, num_heads * head_dim) -- heads merged along the last axis, dtype == Q.dtype.
 *        o_proj is NOT applied here.
 *
 * GQA via head indexing (NO materialization of repeated K/V):
 *   rep = num_heads / num_key_value_heads. Query head h shares KV head (h / rep).
 *   K/V stay at num_key_value_heads copies; the per-head loop indexes the shared KV -- no rep-fold memory blowup (the production choice; vLLM does not repeat).
 *
 * Numerics (the "reduce/accumulate stays f32" family):
 *   - Q@K^T reduces over head_dim; scores@V reduces over keys.
 *     Both accumulate in f32 (same long-reduction discipline as matmul).
 *   - The 1/sqrt(head_dim) scale runs in f32.
 *   - softmax is DELEGATED to the softmax op (f32 internal max+sum, output dtype = input dtype). attention does NOT re-downcast after softmax -- softmax already returns the working dtype. (Cross-op contract: do NOT .to(bf16) the softmax output a second time.)
 *   - Output dtype == input dtype.
 *     BFloat16 path: bf16 in, f32-accumulate the two reductions, softmax f32-internal, RNE-cast out.
 *     NOT bit-exact vs the Python reference (different reduction accumulation) -> allclose.
 *   - Cross-branch numerics DEVIATION (bf16 + additive-mask ONLY; mirrors linear.h's analogous note):
 *       The CPU bf16 branch (attention.cpp) scales and adds mask in TWO separate bf16 stores:
 *         qk = qk*scalar           (upcast f32 -> scale -> RNE downcast to bf16, store #1)
 *         qk = qk + mask           (bf16+bf16 -> ... -> RNE downcast to bf16, store #2)   = TWO casts.
 *       The CUDA scale_mask kernel fuses both into one f32 expression and stores ONCE:
 *         qk = (T)(qk*scalar + mask)   = ONE cast.
 *       So in bf16-with-additive-mask, CUDA-attention differs from CPU-attention by the downcast-order gap (~1 ULP, on the order of 1e-2 relative).
 *       This is the SAME fused-vs-staged gap accepted in linear (M5a), and CUDA is CLOSER to torch's fused-addmm semantics, NOT a bug.
 *       The f32 path has no downcast -> both branches agree (1e-4). The causal branch (mask==nullptr) has no mask-add -> the only store is the scale, one cast on both sides -> also agrees.
 *       => Only "bf16 + explicit additive mask" sits in the deviation window; the CUDA test must use CPU as oracle with bf16 rtol/atol=1e-2 (self-consistency style, same as test_cpp_cuda_linear.py).
 *
 * Shapes are DERIVED, not passed: num_heads = Q.shape[0], num_key_value_heads = K.shape[0], head_dim = Q.shape[2], seq = Q.shape[1]. rep must divide evenly.
 *
 * Precondition:
 *   - Q, K, V are 3-D, contiguous, same dtype, same seq, same head_dim.
 *   - num_heads % num_key_value_heads == 0.
 *   - mask (when present) is (seq, seq), same dtype as Q.
 *
 * Throws std::runtime_error on: rank/shape/dtype mismatch, rep not integral.
 *
 * Edge case: a fully-masked query row (all keys hidden) -> softmax 0/0 = NaN.
 *   The causal branch never hits this (the diagonal is always visible).
 *   An external additive mask that hides the diagonal WOULD -- that is the caller's contract, not attention's.
 */
Tensor attention(const Tensor &Q, const Tensor &K, const Tensor &V, const Tensor *mask);

/**
 * In-place scale + (causal | additive-mask) on the raw Q@K^T scores, before softmax.
 * Folds the per-element work that the CPU attention branch does in its i,j loop into one elementwise CUDA kernel -- called between the Q@K^T gemm and softmax (attention.cpp device branch step 2).
 *
 * Operates on qk viewed as (num_heads * seq, seq) row-major: a thread owns ONE element qk[row, col].
 *   row = global query position across all heads (head h, query i -> row = h*seq + i);
 *   col = key index j. The causal key i is recovered as i = row % seq (per-head query index, mask broadcasts across heads).
 *
 * Three-branch semantics, MUST match attention.cpp's CPU branch:
 *   - mask != nullptr (additive):  qk[row,col] = qk[row,col]*scalar + mask[i*seq + col]   -- applied to EVERY (i,j), bypasses causal.
 *       An external mask REPLACES the causal mask, never stacks (caller's contract).
 *   - mask == nullptr AND i < col: qk[row,col] = -inf  (future key hidden).  Written as (T)(-FLT_MAX).
 *   - mask == nullptr AND i >= col: qk[row,col] = qk[row,col]*scalar  (visible key, scaled).
 *   mask==null vs mask!=null is a RUNTIME branch on the pointer (kernel `if (mask)`), NOT a template -- one kernel serves both.
 *
 * Scale runs in f32: qk upcast to float, * scalar (1/sqrt(head_dim)), downcast back to T on store.
 *   This is where the bf16+mask cross-branch numerics deviation lives -- see the note at the bottom of attention.h.
 *
 * The "-inf" sentinel: written as (T)(-FLT_MAX), NOT -std::numeric_limits<float>::infinity().
 *   Probe-verified: for T in {float, __nv_bfloat16}, static_cast<T>(-FLT_MAX) and static_cast<T>(-inf) produce IDENTICAL 16-bit patterns (f32 -FLT_MAX RNE-overflows to -inf in bf16; both -> 0xFF80).
 *   So the CPU branch's -inf and this kernel's -FLT_MAX are bit-identical post-cast -- the discrepancy is cosmetic, NOT a numerics bug.
 *   (Side note, out of attention's path: NaN cast differs CPU-vs-CUDA, 0x7FC0 vs 0x7FFF -- a known Phase-1 boundary, never hit in normal inference.)
 *
 * Grid: (ceil(seq/16), ceil(num_heads*seq/16)), 16x16 threads/block. row dim spans num_heads*seq (a few thousand blocks at most for 0.5B: num_heads*seq ~ 16*2048 = 32k -> ~2k blocks).
 *   Elementwise, so THREADPERBLOCKDIM=16 (256 threads/block) is independent of gemm's 8.
 * Launch is synchronous (cudaDeviceSynchronize) -- caller does NOT sync.
 *
 * Args:
 *   qk        -- device buffer, shape (num_heads, seq, seq) row-major, dtype == dtype. MODIFIED IN PLACE.
 *   mask      -- device buffer shape (seq, seq) row-major, dtype == dtype; OR nullptr for causal.
 *   scalar    -- 1/sqrt(head_dim), passed in by the caller (attention.cpp computes it).
 *   num_heads -- number of QUERY heads (K/V heads folded in via the per-head gemm loop, not here).
 *   seq       -- sequence length (both qk dims 2,3 and the mask dims).
 *   dtype     -- Float32 | BFloat16 (dispatches the template T).
 * Throws std::runtime_error on unsupported dtype. No alignment guard (elementwise, no vectorized load).
 */
void scale_mask_launch(void *qk, const void *mask, float scalar, int64_t num_heads, int64_t seq, DType dtype);
#pragma once
#include "tensor.h"

/**
 * 2-D matrix multiply: out = A @ B.
 * This is the first op with a LONG accumulation chain: each output element is a dot product of length K (the shared inner dimension).
 * For Qwen2 hidden=896, that is 896 products summed -- a long chain repeated for every output element, so accumulation error does NOT average out; it accumulates systematically.
 *
 * Numerics (the core decision): the accumulation runs in float32 regardless of input dtype.
 * Each product is upcast to f32, the K-length sum stays in f32, and the result is cast to the output dtype only at the end.
 * This is the "reduce in f32" principle from rmsnorm, scaled up: a long bf16 chain (8-bit mantissa, ~3 decimal digits) would suffer catastrophic big-eats-small long before K=896, so bf16 accumulation is not viable.
 * Output dtype follows A (== B).
 *   - Float32  path: f32 accumulate; NOT bit-exact vs PyTorch (different accumulation order / tiling), use allclose.
 *   - BFloat16 path: bf16 inputs upcast to f32 to accumulate, result RNE-cast to bf16.
 *                    Matches "bf16 in, f32 accumulate" (verify HF Qwen2's matmul path does this; if HF keeps bf16 accumulate the tolerance must loosen).
 *
 * Shapes (2-D only):
 *   A:   (M, K)  -- 2-D, row-major contiguous, float dtype
 *   B:   (K, N)  -- 2-D, row-major contiguous, float dtype
 *   out: (M, N)  -- dtype == A.dtype (== B.dtype)
 * Precondition: A.dtype == B.dtype (Float32 or BFloat16); A.shape[1] == B.shape[0].
 * Throws std::runtime_error on: A or B not 2-D, dtype mismatch, inner-dim mismatch.
 * Note: batched matmul (leading batch dims, e.g. attention's per-head Q@K^T) is NOT supported here -- it is deferred to the attention stage.
 * Callers needing a batch must collapse leading dims + share B (the Linear case) or wait for batched.
 */
Tensor matmul(const Tensor &A, const Tensor &B);

#pragma once
#include "tensor.h"

/**
 * Linear (fully-connected) layer: out = x @ W^T + bias.
 * Own loop that bakes in the PyTorch nn.Linear convention:
 *   the stored weight has shape (out_features, in_features) and the forward applies its TRANSPOSE, so x (..., in) @ W^T (in, out) -> (..., out).
 * Transpose by index in the linear loop.
 *
 * Leading dims of x are collapsed to num_rows = x.numel / in (same flattening as rmsnorm: any (..., in) treated as (num_rows, in)), and the SAME weight is applied to every row.
 * This is the shared-B 2-D matmul case (not batched):
 *   (num_rows, in) @ (in, out) -> (num_rows, out), then reshape back to (..., out).
 *
 * bias is optional (nullptr means no bias).
 * When present it is 1-D of length out_features and is added (in f32, then cast to output dtype) to every row.
 *
 * Numerics: identical to matmul -- f32 accumulation, output dtype follows x.
 *   - Float32  path: allclose vs PyTorch, NOT bit-exact (f32 accumulation order differs from PyTorch's kernel).
 *     (Note: this is a different baseline from gemm.h's "bit-exact vs the old per-op loops" -- that is an internal refactor gate on f32 self-consistency, not a parity claim vs PyTorch.)
 *   - BFloat16 path: bf16 in, f32 accumulate, RNE-cast out.
 *
 * Shapes:
 *   x:      (..., in_features)    -- rank >= 1, row-major contiguous, last axis = in
 *   weight: (out_features, in_features) -- 2-D, stored TRANSPOSED (nn.Linear convention)
 *   bias:   (out_features,)       -- 1-D, optional (nullptr = no bias)
 *   out:    (..., out_features)   -- x's leading dims + out_features; dtype == x.dtype
 * Precondition: x.dtype == weight.dtype (== bias.dtype when present);
 *               weight.shape[1] == x.shape[-1] (in_features match);
 *               bias.shape[0] == weight.shape[0] (out_features match) when present.
 * Throws std::runtime_error on: rank/shape/dtype mismatch.
 * Device branch (M5, CUDA):
     - When x.device() == CUDA, linear dispatches to gemm_launch with transB=true, has_bias=(bias!=nullptr), ldc=n (contiguous output).
         bias is fused into the GEMM store epilogue (BiasAdd), NOT a separate post-loop -- see gemm.h's epilogue note.
         This is the FIRST real consumer of gemm's BiasAdd epilogue (M5a) and of gemm's transB=true path (M4b); until M5 both were test-only.
     - Alignment precondition (CUDA only): in_features (= k = lda = ldb) must be multiple of 4, else throws "lda/ldb must be multiple of 4 for vectorized load".
         The CPU branch has no such constraint (no vectorized load).
         This guard mirrors gemm.cpp's; it is the caller's responsibility to pass 4-aligned in_features (Qwen2.5 hidden=896 / intermediate=4864 both satisfy it; tests use k=8).
     - Numerics DEVIATION between the two branches (bf16 with bias only):
         CPU branch adds bias to an ALREADY-DOWNCAST bf16 (downcast acc -> +bias -> downcast again = TWO casts).
         CUDA branch adds bias in f32 then downcasts ONCE (acc+bias -> downcast = ONE cast), matching torch F.linear.
         So in bf16-with-bias, CUDA-linear is bit-exact vs torch F.linear (0.0 measured) but differs from CPU-linear by the downcast-order deviation (~0.0156).
         The cuda test (test_cpp_cuda_linear.py) uses CPU as oracle (self-consistency style), so its bf16-with-bias case sits near rtol=1e-2 -- this is the known cross-branch numerics gap, NOT a bug.
         f32 has no downcast, both branches agree (1e-5).
  - result inherits x.device() (allocation follows x), so the device branch writes to a GPU buffer in place.
 * Note: bias passed as a raw Tensor* (nullptr-optional) rather than a Tensor to keep the no-bias case zero-cost and unambiguous; the binding layer decides how to expose this to Python (e.g. None -> nullptr).
 */
Tensor linear(const Tensor &x, const Tensor &weight, const Tensor *bias);

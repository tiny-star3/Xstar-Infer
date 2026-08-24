#pragma once
#include "tensor.h"

/**
 * Root-mean-square layer norm (Qwen2/Llama style): no mean subtraction.
 * Normalizes over the LAST axis (hidden); all leading dims are treated as batch (flattened to num_rows = numel/hidden).
 * Matches PyTorch RMSNorm which accepts arbitrary leading dims "(..., hidden_size)".
 *   out[..., j] = (x[..., j] / sqrt(mean(x[...]^2 over hidden) + eps)) * weight[j]
 *
 * Numerics: accumulation (sum of squares, mean, eps, sqrt, scale) runs in float32 regardless of input dtype; input/output dtype follows x (== weight).
 *   - Float32  path: exact, used for bit-exact parity testing.
 *   - BFloat16 path: production; bf16 upcast to f32 to accumulate, normalized result cast back to bf16, then multiplied by bf16 weight (matches PyTorch CPU bf16: upcast, mul, round-to-nearest-even).
 * eps is added INSIDE the sqrt to avoid division by zero.
 *
 * Shapes:
 *   x:      (..., hidden)  — rank >= 1, row-major contiguous, last axis = hidden
 *   weight: (hidden,)      — 1D, per-channel scale, no bias
 *   out:    same shape & dtype as x
 * Precondition: x.dtype == weight.dtype (Float32 or BFloat16); weight.shape().size()==1, x.shape().back() == weight.shape()[0].
 * Throws std::runtime_error on dtype mismatch / rank mismatch / shape mismatch.
 */
Tensor rmsnorm(const Tensor &x, const Tensor &weight, float eps);

/**
 * Launch the rmsnorm kernel -- the .cu side, called by the cpp orchestration.
 *
 * Layering: declared here (CUDA-free, pure C++) so rmsnorm.cpp (g++) can call it; defined in rmsnorm.cu (nvcc) where the <<<>>> launch lives.
 * Same invariant as concat_launch: host orchestration links the symbol without touching CUDA headers.
 *
 * Ownership / where each pointer lives (caller = rmsnorm.cpp device branch):
 *   - out     : GPU buffer for the OUTPUT (result.data()), sized numel*dtype_size. Kernel WRITES.
 *   - x       : GPU input (x.data()), row-major contiguous, LAST axis = hidden (the reduce axis).
 *   - weight  : GPU 1D weight (weight.data()), length hidden, per-channel scale.
 *   - hidden  : reduction axis length (last dim of x).
 *   - num_rows: numel / hidden = number of independent rows. Launches one BLOCK per row.
 *   - eps     : added inside sqrt (same as CPU).
 *   - dtype   : Float32 or BFloat16 -- the launch wrapper dispatches to the matching kernel (compute is f32 either way; dtype only changes how x/weight are read and out written).
 *
 * Kernel behavior (one block per row, threads cooperate -- M3's first reduction):
 *   each thread accumulates a partial sum-of-squares in f32 (upcast if bf16) over its slice of the row;
 *   block reduces the partials to one total via shared-memory tree + warp shuffle;
 *   thread 0 computes inv_rms = 1/sqrtf(sum/hidden + eps), broadcasts via shared memory;
 *   all threads write out[j] = x[j] * inv_rms * weight[j].
 *
 * Sync: cudaGetLastError (launch config) + cudaDeviceSynchronize (execution) + CHECK_CUDA, same two-stage as concat_launch.
 */
void rmsnorm_launch(void *out,
                    const void *x,
                    const void *weight,
                    std::int64_t hidden,
                    std::int64_t num_rows,
                    float eps,
                    DType dtype);

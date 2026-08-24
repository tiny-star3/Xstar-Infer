#pragma once
#include "tensor.h"

/**
 * SwiGLU MLP (Qwen2):  out = down( silu(gate) * up(x) ),  silu(g) = g * sigmoid(g).
 * gate and up are FUSED into one Linear of width 2*intermediate and split along the last axis at runtime -- one GEMM, not two (matches xstar/layers/mlp.py, so the 1a parity oracle stays structurally identical to the Python reference).
 * Both projections are bias-free.
 *
 * Fusion is SCHEDULING-level (one gate_up GEMM), not a kernel-internal fused MLP:
 *   gate_up = linear(x, gate_up_weight)  -> (num_rows, 2*intermediate) contiguous
 *   gate, up = the two equal column halves of gate_up (gate first, up second), taken by POINTER OFFSET -- strided sub-matrices (row stride 2*intermediate, NOT intermediate), zero copy.
 *              They are NOT contiguous and cannot be fed back to linear (which assumes lda = in).
 *   act = silu(gate) * up, materialized into a NEW contiguous (num_rows, intermediate) buffer so down's linear gets lda = intermediate.
 *   out  = linear(act, down_weight)       -> (num_rows, hidden) contiguous
 * Three buffer allocations (gate_up, act, out).
 * An industrial fused-MLP CUDA kernel folds the gate/up split + silu + the down GEMM's elementwise into one pass and drops the act re-materialization; that is a Phase 5 kernel concern, not this Phase 1 CPU op.
 * The act re-materialization is the honest cost of not having that kernel -- a talking point, not a defect to hide.
 *
 * Leading dims of x are collapsed: num_rows = x.numel / hidden (same flattening as linear/rmsnorm -- any (..., hidden) is (num_rows, hidden), shared weights per row).
 *
 * Numerics (the "reduce/accumulate stays f32" family -- mlp owns the NONLINEAR precision; linear/gemm_cpu own the GEMM precision):
 *   - The two GEMMs (gate_up, down) inherit linear's contract (bf16 in, f32 accumulate, RNE-cast out).
 *     mlp does NOT touch GEMM precision.
 *   - sigmoid(g) and the silu(g)*up product run in f32 INTERMEDIATE: gate/up are upcast from the working dtype, the whole silu*up expression is evaluated in f32, and ONLY the act write is RNE-downcast to the working dtype.
 *     This is the ONE place mlp touches precision-- bf16-domain exp would amplify rounding through the nonlinearity.
 *   - sigmoid is overflow-safe: the branched form (x>=0 -> 1/(1+exp(-x)); x<0 -> exp(x)/(1+exp(x))) never evaluates exp of a positive argument, so expf cannot overflow for any |gate|.
 *     (The naive 1/(1+exp(-x)) also self-corrects -- exp(-x)->+inf, 1/(1+inf)=0 -- but relies on inf arithmetic and is less accurate in the subnormal window; branched is the textbook stable choice.)
 *   - Output dtype == input dtype.
 *     Float32  path: allclose vs the Python reference (not bit-exact -- f32-intermediate sigmoid vs torch's own path may differ in the last ULP).
 *     BFloat16 path: bf16 in, f32 sigmoid+product, RNE-cast act, f32-accumulate GEMMs, RNE-cast out -> allclose(rtol=1e-2, atol=1e-2).
 *
 * Shapes (derived, not passed):
 *   x:              (..., hidden)            -- rank >= 1, contiguous, last axis = hidden
 *   gate_up_weight: (2*intermediate, hidden) -- 2-D, stored TRANSPOSED (nn.Linear convention)
 *   down_weight:    (hidden, intermediate)   -- 2-D, stored TRANSPOSED
 *   out:            (..., hidden)            -- x's leading dims + hidden; dtype == x.dtype
 *
 * Precondition:
 *   - x.dtype == gate_up_weight.dtype == down_weight.dtype.
 *   - x rank >= 1; both weights 2-D.
 *   - gate_up_weight.shape[1] == x.shape[-1]            (hidden; surfaced from gate_up linear)
 *   - gate_up_weight.shape[0] is even                   (gate/up split is two equal halves; mlp-local)
 *   - down_weight.shape[1] == gate_up_weight.shape[0]/2 (intermediate consistency; mlp-local)
 *   - down_weight.shape[0] == x.shape[-1]               (down out == hidden; mlp-local)
 *
 * Throws std::runtime_error on: rank/shape/dtype mismatch (surfaced from the linear calls and the mlp-local even-half / intermediate / down-out checks).
 *
 * Note: bias-free by contract (Qwen2 SwiGLU has no bias on either projection); there is no bias parameter.
 * Adding one would re-introduce the "bias after downcast" rounding trap (cf. linear) and is deliberately omitted.
 */
Tensor mlp(const Tensor &x, const Tensor &gate_up_weight, const Tensor &down_weight);

/**
 * Fused GEMM + SwiGLU for the MLP gate_up projection (M5 mlp, paired-tile).
 * Computes the gate_up GEMM AND silu(gate)*up in ONE kernel, so the (m, 2*intermediate) gate_up tensor never lands in global memory:
 *   gate = x @ W^T over W rows [0, intermediate);  up = x @ W^T over W rows [intermediate, 2*intermediate)
 *   act[i,j] = silu(gate[i,j]) * up[i,j]
 *
 * Paired-tile: one block computes BOTH halves -- two accumulators (accG, accU) over two B smems (smemBg, smemBu) whose W row ranges differ by exactly `intermediate`, sharing one A tile (smemA).
 * Store fuses silu*mul, writes only `act`. (M4 proved a plain store-time fuse can't do this: gate col j and up col j+intermediate land in different blockIdx.x when intermediate >> BN.)
 *
 * transB is FIXED true (nn.Linear convention: W is (2*intermediate, hidden), forward is x @ W^T).
 * NOT templated -- only mlp uses this kernel; no dead no-trans branch.
 *
 * Tiling: BM=BN=BK=32, 8x8 threads, TM=TN=VEC=4 (same as gemm_kernel, not tuned).
 *   grid = (ceil(intermediate/BN), ceil(m/BM))  -- x is intermediate (output cols), NOT 2*intermediate.
 *   blockIdx.x = act column-block; same col maps to W gate rows [col,col+BN) AND up rows [col+intermediate, ...).
 *
 * Args:
 *   act            -- GPU out, (m, intermediate) row-major, ldc = intermediate. WRITES only.
 *   x              -- GPU in, (m, hidden) contiguous (lda == k). READS.
 *   W              -- GPU in, (2*intermediate, hidden) row-major, stored TRANSPOSED. ldb = k. Must be %4 (guard inside).
 *   m              -- num_rows.
 *   k              -- hidden (K dim, lda == ldb == k).
 *   intermediate   -- act col count AND gate/up split.
 *   dtype          -- Float32 | BFloat16.
 *
 * Numerics (matches mlp.cpp helper):
 *   - k-dots accumulate f32 (accG, accU); toFloat from cuda/dtype_cast.h.
 *   - silu(g)=g*sigmoid(g), sigmoid OVERFLOW-SAFE branched (g>=0: 1/(1+expf(-g)); g<0: expf(g)/(1+expf(g)))
 *     -- CPU contract carried verbatim, do NOT simplify to 1/(1+expf(-g)).
 *   - silu(g)*u all f32, ONE RNE downcast at act write == mlp.cpp single downcast.
 *     So bf16 fused path ALIGNS with CPU oracle by construction -- no downcast-order deviation (unlike linear bf16+bias / attention bf16+mask).
 *     Expect bf16 diff SMALLER than those.
 *
 * Boundary: OOB x/W loaded as 0; OOB act not written.
 *   Gate half and up half bounded INDEPENDENTLY at the W row index (gate clips at intermediate, up clips at 2*intermediate) -- separate guards, do not share.
 *
 * Resource: double acc (32 f32) + double B smem (3072 elem, f32 12KB / bf16 6KB).
 *   Registers are the risk on sm_75 (gemm_kernel already 87.5%-occupied); verify with --ptxas-options=-v.
 *   Correctness first, occupancy Phase 5; if spill, fall back to BM=BN=16.
 *
 * Sync: cudaGetLastError + cudaDeviceSynchronize + CHECK_CUDA. Throws if ldb%4 != 0.
 */
void gemm_silu_and_mul_launch(void *act,
                              const void *x,
                              const void *W,
                              std::int64_t m,
                              std::int64_t k,
                              std::int64_t intermediate,
                              std::int64_t ldc,
                              DType dtype);

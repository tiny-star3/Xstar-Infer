#pragma once
#include "tensor.h"

/**
 * Numerically stable softmax over a single axis.
 *
 * For each slice along `dim`, computes exp(x - max(x)) / sum(exp(x - max(x))) in float32 and writes back in the input dtype.
 * The max-shift keeps the largest exponent at 0 so exp never overflows; the shift is a per-slice constant so it cancels in the normalized sum (result unchanged).
 *
 * Numerics (the "reduce stays f32" family -- rmsnorm reduce, rope cache, matmul/linear accumulation, now softmax's max+sum):
 *   - Both reduces (max and sum) run in f32 REGARDLESS of input dtype.
 *   - Output dtype == input dtype: the f32 probabilities are RNE-cast back at the very last step.
 *     No intermediate downcast.
 *   - Float32  path: NOT bit-exact vs PyTorch (different reduction accumulation / no FMA contraction guarantee) -> allclose.
 *   - BFloat16 path: bf16 in, f32 max+sum+exp, RNE-cast out
 *     This MATCHES PyTorch CPU bf16 softmax's DOWNCAST PATTERN: PyTorch also runs the max/sum/exp reduction in f32 and downcasts ONCE at the end (NOT per-step -- there is no intermediate bf16 downcast on either side; verified that the downcast count matches).
 *     The two paths are NOT always bit-exact: PyTorch's vectorized f32 reduction order can differ from a naive sequential sum by a ULP on some inputs (probed: bit-exact on many inputs but not all across 200 seeds).
 *     The residual ULP gap is reduction-order / expf, NOT downcast count -> allclose.
 *
 * This op owns the f32 precision decision -- there is NO dtype parameter and NO caller-side downcast.
 * Attention feeds logits straight in and consumes the (already-downcast) probabilities; it must NOT downcast a second time.
 *
 * Shapes:
 *   x:   arbitrary rank (>= 1), row-major contiguous, any dtype in {f32, bf16}.
 *   dim: the axis to normalize over; supports negative indexing (-1 == last axis). The other axes are preserved.
 *   out: same shape and dtype as x; slices along `dim` sum to 1 (up to the output dtype's representable precision).
 *
 * Precondition:
 *   - x is contiguous (this op does not handle strided inputs).
 *   - dim in [-rank, rank).
 *
 * Throws std::runtime_error on: unsupported dtype, dim out of range.
 *
 * Edge case: a fully-masked slice (all elements == -inf) yields 0/0 = NaN.
 * Softmax does NOT mask or sanitize -- callers must keep at least one element finite.
 * Attention's causal mask always leaves the diagonal visible, so the attention path never hits this; an external additive mask that hides the diagonal WOULD, and that is the caller's contract, not softmax's.
 */
Tensor softmax(const Tensor &x, int64_t dim);

/**
 * Launch the softmax kernel -- the .cu side, called by the cpp orchestration.
 *
 * Layering: declared here (CUDA-free, pure C++) so softmax.cpp (g++) can call it; defined in softmax.cu (nvcc) where the <<<>>> launch lives.
 * Same invariant as rmsnorm_launch / concat_launch: host orchestration links the symbol without touching CUDA headers.
 *
 * Arbitrary-axis via 3D collapse (the industrial leaf-softmax pattern; cf. PyTorch aten/src/ATen/native/cuda/SoftMax.cu, comment:
 *   "assume that our input has been flattened to have only three dimension: outer x dim x inner"):
 *   the caller (softmax.cpp device branch) flattens the contiguous row-major tensor to THREE scalars and passes them by value -- NO strides array, NO row_start, NO dim reach the kernel:
 *     outer_size = product of all axes BEFORE the (normalized) dim,
 *     dim_size   = length of the reduce axis == shape[dim],
 *     inner_size = product of all axes AFTER  the (normalized) dim.
 *   Every element then lives at linear offset outer_idx*(dim_size*inner_size) + d*inner_size + inner_idx,
 *   so the contiguous contract (softmax.h: "x is contiguous") makes this exact and the arbitrary-stride row_start path (used by the CPU oracle) is NOT needed on GPU.
 *   The `dim` argument is consumed by this collapse and is not forwarded.
 *
 * Ownership / where each pointer lives (caller = softmax.cpp device branch):
 *   - out       : GPU buffer for the OUTPUT (result.data()), numel*dtype_size. Kernel WRITES.
 *   - x         : GPU input (x.data()), row-major contiguous; rank already collapsed to 3D by caller.
 *   - outer_size: # independent slices along the leading axes.
 *   - dim_size  : the reduce-axis length -- the axis softmax normalizes over.
 *   - inner_size: # independent slices along the trailing axes; parallelizes WITH outer as one block per (outer_idx, inner_idx) pair.
 *   - dtype     : Float32 or BFloat16 -- dispatches to the matching kernel (compute is f32 either way; dtype only changes how x is read and out written).
 *
 * Grid/block (one block per slice; M3's second reduction -- the online-softmax step toward M6's multi-block merge; deliberately NOT the 3-pass industrial leaf softmax, which keeps max and sum as separate passes -- online merges them into one scan so the (m,l) rescaling-merge primitive is exercised here):
 *   grid  = outer_size * inner_size   (blockIdx.x = slice index; reverse as outer_idx = idx / inner_size, inner_idx = idx % inner_size).
 *   block = 256 (THREADPERBLOCK), grid-stride over dim_size.
 *   Within a slice, each thread keeps a running (m_local, l_local) over its grid-stride slice of dim_size; block reduces the 256 (m,l) pairs by shared-memory tree + warp shuffle with the RESCALING MERGE as the combine (NOT plain add -- l must be rescaled by exp(old_m - new_m) when a larger m wins; skipping this leaves sum systematically small and is INVISIBLE when the slice max is the FIRST element, so tests must place the max mid/late). thread 0 broadcasts (m_final, l_final) via shared memory with a block-wide __syncthreads() BEFORE the read (same race fix as rmsnorm); all threads then write  out[data_offset + d*inner_size] = exp(x - m_final) / l_final.
 *
 * Numerics (matches softmax.h contract): max, sum-of-exp, exp all run in f32 regardless of dtype; bf16 is RNE-cast ONCE at the write (no intermediate downcast).
 *   - f32 path: NOT bit-exact vs CPU (reduction order + expf) -> allclose, ~1e-6 (measure before asserting).
 *   - bf16 path: output in [0,1]; 1 ULP at 1.0 ~= 0.0078, so tolerance is SMALLER than rmsnorm's 0.03125 -- measure before asserting, do not copy that number.
 *
 * Sync: cudaGetLastError (launch config) + cudaDeviceSynchronize (execution) + CHECK_CUDA, two-stage as concat_launch / rmsnorm_launch.
 */
void softmax_launch(void *out,
                    const void *x,
                    std::int64_t outer_size,
                    std::int64_t dim_size,
                    std::int64_t inner_size,
                    DType dtype);

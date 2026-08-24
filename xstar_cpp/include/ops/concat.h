#pragma once
#include <vector>
#include "tensor.h"

#define MAX_RANK 10

/**
 * Concatenate N tensors along an axis -- the device-side join, general form.
 *
 * Why this exists: Phase 1's loader concatenates gate_proj + up_proj into the fused gate_up weight on the CPU (qwen2_model.cpp, two std::memcpy).
 * M2 writes the device version so M6 can fuse gate_up ON the GPU after H2D, avoiding a CPU concat + extra H2D.
 * The general (N inputs, arbitrary axis) form is chosen over a 2-input/axis-0 special case as a kernel-writing exercise; the ONLY real consumer is gate_up (2 inputs, axis 0), so the general path is exercised by synthetic tests, not by the model.
 *
 * Layering (same invariant as cuda_allocator):
 *   - concat.h / concat.cpp: pure C++ orchestration (g++-compilable, no CUDA headers).
 *     Validates dtype/rank/axis consistency, computes the output shape (axis dim = sum of input axis dims, others must match), allocates a contiguous output Tensor on the inputs' device, and calls the launch wrapper.
 *   - concat.cu: the __global__ kernel + launch wrapper (nvcc).
 *     The kernel maps each  output element to (which input it belongs to, offset within that input) by scanning input axis sizes, then copies one element.
 *     This per-element dispatch is correct-first; tiling/overlap is a Phase 5 concern.
 *
 * Contiguity contract: all inputs AND the output are row-major contiguous (Tensor's only layout).
 * No arbitrary strides to handle -- the coordinate math uses the standard row-major offset formula.
 *
 * Device: GPU-only. All inputs must be on CUDA (CPU inputs are rejected in concat.cpp); the output lives on CUDA too.
 * There is NO CPU code path: the kernel is __global__, it cannot run on the host.
 * M2 tests verify device concat against torch.cat (an INDEPENDENT CPU reference, not this op's CPU path): to_cuda the inputs, run device concat, to_cpu the result, compare bit-exact to torch.cat of the original CPU tensors.
 *
 * Throws std::runtime_error on: empty inputs, rank mismatch, dtype mismatch, non-axis dim mismatch, axis out of [-rank, rank).
 */
Tensor concat(const std::vector<const Tensor *> &inputs, int axis);

/**
 * Launch the concat kernel -- the .cu side of concat, called by the cpp orchestration.
 *
 * Layering: declared here (CUDA-free, pure C++) so concat.cpp (g++) can call it; defined in concat.cu (nvcc) where the <<<>>> launch lives.
 * Same invariant as cuda_alloc: the host orchestration links the symbol without touching any CUDA header.
 *
 * Ownership / where each pointer lives (caller = concat.cpp prepares all of these):
 *   - out          : GPU buffer for the OUTPUT tensor (result.data()), already allocated, sized out_numel * dtype_size bytes.
 *                    The kernel WRITES here.
 *   - d_ptrs       : GPU array of N void* -- each inputs[i].data() (a GPU pointer for CUDA inputs).
 *                    concat.cpp builds this array on the host, h2d's it to a GPU buffer, passes that GPU buffer's address here.
 *                    The kernel reads d_ptrs[k] to get input k's data pointer. (Indirection: N pointers must themselves live on the GPU before the kernel can read them.)
 *   - d_axis_sizes : GPU array of N int64 -- each inputs[i].shape()[axis].
 *                    Kernel scans it to map out_coord[axis] -> (which input k, in-axis offset).
 *                    h2d'd by caller.
 *   - d_out_shape  : GPU array of rank int64 -- the FULL output shape (NOT non-axis only).
 *                    Kernel uses it to invert idx -> out_coord (row-major), and (with axis_sizes + k) to derive input k's shape for the in_idx computation.
 *                    h2d'd by caller.
 *   - n, rank, axis, dtype_size, out_numel : scalars (passed by value, trivially copyable).
 *
 * Kernel behavior (per-element, one thread per output element, correctness-first):
 *   idx = global thread index; if idx >= out_numel return.
 *   1. idx -> out_coord[0..rank-1] by row-major inversion against d_out_shape.
 *   2. scan d_axis_sizes: subtract from out_coord[axis] until it fits in input k; the remainder is in_coord[axis]; other in_coord[d] = out_coord[d].
 *   3. in_coord -> in_idx (row-major offset within input k, using input k's shape = d_out_shape with the axis dim replaced by d_axis_sizes[k]).
 *   4. copy dtype_size bytes from d_ptrs[k][in_idx] to out[idx].
 *
 * No validation here -- all shape/dtype/rank/axis checks live in concat.cpp.
 * The launch trusts its caller (mirrors cuda_alloc_kernel: pointer-level, trusts the Tensor op).
 *
 * Sync: M2 correctness-first calls cudaDeviceSynchronize after launch and CHECK_CUDA's it, so a kernel error surfaces as a thrown runtime_error at the call site rather than silently corrupting a later op.
 * Async (no sync, overlap with next op) is a Phase 5 concern gated on a stream abstraction.
 */
void concat_launch(void *out,
                   void **d_ptrs,
                   const std::int64_t *d_axis_sizes,
                   const std::int64_t *d_out_shape,
                   int n,
                   int rank,
                   int axis,
                   int dtype_size,
                   std::int64_t out_numel);

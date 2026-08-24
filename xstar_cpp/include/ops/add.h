#include <stdint.h>

#include "tensor.h"

/**
 * Elementwise add (residual connection): out = a + b, same shape & dtype.
 * Used for the two residual adds in transformer_block (x + attn_out, x + mlp_out).
 *
 * Numerics: a single IEEE add per element -- NO mul, so NO FMA contraction (unlike rope).
 *   - Float32 path: one f32 add (RNE) on both backends -- bit-exact (torch.equal).
 *   - BFloat16 path: the CPU bfloat16 has only operator float() (no operator+), so both operands upcast to f32 (lossless: bf16 is the high 16 bits of f32), add in f32 (the sum of two bf16 values is f32-exact -- 7 mantissa bits added stays within f32's 23, so the add rounds once), then RNE-downcast to bf16 ONCE at the store. GPU __nv_bfloat16 operator+ follows the SAME path on sm_75 (no native bf16 ALU): SASS shows PRMT (bf16->f32 zero-extend) -> FFMA x1 (= a*1+b, the f32 add) -> a bit-manipulation RNE sequence (LOP3/SHF/SEL) for f32->bf16. Same bits -> bit-exact (torch.equal).
 * Pure elementwise, no reduction, no cross-element dependency.
 *
 * Shapes: a and b must have identical shape (any rank) and identical dtype. out has the same shape & dtype, on the same device as a (and b, enforced by the device-mismatch check).
 *
 * Precondition:
 *   - a.shape == b.shape (every dim)
 *   - a.dtype == b.dtype, and is Float32 or BFloat16
 *   - a.device == b.device
 * Throws std::runtime_error on: shape mismatch, dtype mismatch, or device mismatch.
 */
Tensor add(const Tensor &a, const Tensor &b);

/**
 * Launch the CUDA elementwise-add kernel. Internal helper -- called only by add() on the Device::CUDA branch.
 *
 * No temp device buffer (both a and b are caller tensors): launch neither allocs nor frees -- NO leak surface, no no_leak test.
 * Sync: ends with CHECK_CUDA(cudaGetLastError()) + CHECK_CUDA(cudaDeviceSynchronize()), matching the other _launch ops.
 *
 * Kernel: elementwise grid-stride loop, templated on T for BOTH reasons -- the load/store width (LDG.32/16) AND the + operator (float vs __nv_bfloat16 differ). blockDim=256, grid covers numel; consecutive threads hit consecutive i -> a[i], b[i], out[i] all coalesced (1 transaction each for read-a, read-b, write-out).
 *
 * Params:
 *   out   -- device ptr, numel elements, dtype == a.dtype
 *   a, b  -- device ptrs, numel elements each, same dtype
 *   numel -- element count
 *   dtype -- Float32 or BFloat16
 */
void add_launch(void *out, const void *a, const void *b, std::int64_t numel, DType dtype);

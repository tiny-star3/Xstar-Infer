#pragma once
#include <vector>
#include <cstdint>

#include "tensor.h"

/**
 * Rotary position embedding (RoPE), Qwen2/Llama split-half style.
 * Rotates each of the dim/2 two-D subspaces of x by an angle set by the token's position.
 * Pairs are split-half: the first half x1 = x[..., :d/2] and the second half x2 = x[..., d/2:] are paired as (x_i, x_{i+d/2}).
 * This is mathematically equivalent to HF's rotate_half (x*cos + rotate_half(x)*sin), written in expanded form:
 *   out[..., :d/2]   = cos * x1 - sin * x2
 *   out[..., d/2:]   = sin * x1 + cos * x2
 *
 * Stateless: the cos/sin cache is passed in as a Tensor (precomputed once by the caller, e.g. via the Python reference's _init_cache).
 * The op only consumes it -- it does not build or own the cache.
 * This mirrors rmsnorm/embedding taking precomputed data (weight) as input.
 *
 * Cache layout (matches xstar/layers/rope.py _init_cache):
 *   cache: (2, max_seq_len, dim/2)  -- 3-D, row-major, float dtype.
 *           leading axis stacks [cos, sin]: cache[0] = cos table, cache[1] = sin table.
 *           row p holds the cos/sin for absolute position p; column f is frequency channel f.
 *
 * Numerics: the rotation runs in f32 internally; the cache stays Float32 (precondition: cache.dtype == Float32, enforced by a throw), so the caller passes the raw f32 cache and does NOT pre-downcast it.
 * There is no reduction here (only elementwise mul/add), so unlike rmsnorm there is no f32 accumulation step.
 * The bf16 downcast policy DIFFERS between the two backends, so bf16 is NOT bit-exact across CPU/GPU (nor with PyTorch eager):
 *   - Float32 path: pure f32 on both backends and op order aligned, but NOT bit-exact: nvcc contracts `cos*x1 - sin*x2` / `sin*x1 + cos*x2` into FFMA (default fmad ON, no -fmad=false in the build; FFMA confirmed in SASS), so each expression rounds twice (the a*b folds into the fma, only the addend mul and the fma each round once), while the CPU loop does separate mul/mul/sub (three roundings) -- ~1 ULP. Tested with allclose(atol=probed), NOT torch.equal.
 *   - CPU BFloat16: downcasts cos/sin to bfloat16 BEFORE the mul (static_cast<bfloat16>), runs mul/add in f32, downcasts the result ONCE at the store -- not bit-exact with eager.
 *   - GPU BFloat16: keeps cos/sin as f32 (NO downcast), upcasts x via toFloat, downcasts the result ONCE at the store -- more precise than CPU bf16, so not bit-exact with CPU bf16 either. See rope_launch.
 *
 * Shapes:
 *   x:          (..., seq_len, dim)      -- rank >= 2, row-major contiguous; last axis = dim (= head_dim), second-to-last = seq_len; leading dims = batch/heads.
 *   cache:      (2, max_seq_len, dim/2)  -- 3-D, float dtype; dim/2 = dim / 2.
 *   positions:  int64 buffer of length seq_len; positions[s] is the absolute position of token s.
 *               The same positions apply to ALL leading dims (every head/batch at slot s shares the position).
 *               Pass 0..seq_len-1 for a contiguous prefill.
 *   out:        same shape & dtype as x.
 *
 * Precondition:
 *   - x.rank >= 2; cache.rank == 3 and cache.shape[0] == 2
 *   - cache.dtype == Float32
 *   - x.shape[-1] == 2 * cache.shape[2]      (dim == 2 * (dim/2))
 *   - positions length == x.shape[-2]        (== seq_len)
 *   - every position in [0, cache.shape[1])  (max_seq_len)
 * Throws std::runtime_error on:
 *   - rank mismatch, dtype mismatch, or shape mismatch
 *   - any position < 0 or position >= max_seq_len (out-of-range position)
 * Note: positions are INDICES (like embedding ids), not values -- passed as a raw int64 buffer so the core library stays free of pybind11; the binding layer unpacks a numpy int64 array.
 * A per-batch positions shape ((..., seq_len) matching x's leading dims) is NOT supported here;
 * if M6 attention needs it, extend the signature then.
 */
Tensor rope(const Tensor &x, const Tensor &cache, const std::int64_t *positions);

/**
 * Launch the CUDA RoPE kernel (split-half rotary embedding). Internal helper -- called only by rope() on the Device::CUDA branch.
 *
 * d_positions is caller-managed (same pattern as embedding's ids):
 *   rope() cuda_alloc's a temp device buffer, cudaMemcpy's positions H2D, calls this launch, then cuda_free's the buffer.
 *   This launch does NOT own/free d_positions -- it only hands the device pointer to the kernel.
 *
 * cache is ALWAYS Float32 (enforced by rope() before reaching here), hence the const float* type.
 * The kernel uses cos/sin DIRECTLY as f32 (NO downcast to x.dtype) and upcasts x via toFloat; the rotation runs in f32, then the result is downcast to x.dtype ONCE at the store ((T)(...)).
 * This is MORE precise than the CPU bf16 loop (which downcasts cos/sin to bfloat16 before the mul). Neither path is bit-exact with the CPU: the f32 path differs by ~1 ULP from GPU FMA contraction (see rope() docstring), and the bf16 path additionally differs from the cos/sin downcast policy.
 * The caller passes the raw f32 cache and does NOT pre-downcast it.
 *
 * The kernel is templated on x's dtype (Float32 / __nv_bfloat16) and dispatched inside this launch on `dtype`.
 * Unlike embedding (pure byte-gather, shared kernel, no template), rope has float arithmetic so the dtype must specialize the math -- same reason rmsnorm/mlp are templated.
 *
 * Ends with cudaDeviceSynchronize: the caller's cuda_free(d_positions) is only safe once the kernel has finished reading d_positions.
 *
 * Op order matches the CPU loop (cos*x1 - sin*x2 / sin*x1 + cos*x2, NO cross-element reduction), but NEITHER path is bit-exact: f32 differs by ~1 ULP from GPU FMA contraction (FFMA vs CPU mul/mul/sub), bf16 additionally differs from the cos/sin downcast policy (GPU keeps f32, more precise). The parity test pins both tolerances by probe.
 */
void rope_launch(void *out,
                 const void *x,
                 const float *cache,
                 const std::int64_t *d_positions,
                 std::int64_t num_outer,
                 std::int64_t dim,
                 std::int64_t seq_len,
                 std::int64_t half,
                 DType dtype);

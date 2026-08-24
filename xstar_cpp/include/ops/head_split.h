#include <stdint.h>

#include <tensor.h>
#include <dtype.h>

/**
 * Head split: reshape + transpose the Q/K/V projection output into per-head layout for RoPE/attention.
 * Maps the projection's (seq, heads*head_dim) output to (heads, seq, head_dim) by swapping the heads and seq axes:
 *   out[i, j, k] = in[j, i*head_dim + k]    (i=head, j=seq, k=channel)
 * Equivalently, view in as (seq, heads, head_dim) row-major and transpose(0,1) -> (heads, seq, head_dim).
 *
 * Pure data movement -- NO floating-point arithmetic (a strided byte copy), so the output is bit-exact for both Float32 and BFloat16 (tested with torch.equal, NOT allclose).
 * This is the embedding family (copy by index), NOT the rope family (float math): no dtype-dependent ARITHMETIC. The kernel is still templated internally on dtype (head_split_kernel<T>), but ONLY to set the load/store width (LDG.32 for Float32 / LDG.16 for BFloat16, verified in SASS) -- there is no per-dtype math, no FMA, no tolerance probe, and the result is bit-exact (torch.equal).
 *
 * Shapes:
 *   t:   (seq, heads*head_dim)   -- 2-D, row-major contiguous; the linear() projection output. Last axis packs heads*head_dim (heads outer, head_dim inner).
 *   out: (heads, seq, head_dim)  -- 3-D, row-major contiguous; heads is the NEW leading axis.
 *
 * Precondition:
 *   - t.rank == 2
 *   - t.dtype is Float32 or BFloat16
 *   - heads > 0 and t.shape[1] % heads == 0    (head_dim = t.shape[1] / heads is integral)
 * Throws std::runtime_error on:
 *   - rank mismatch, dtype unsupported, heads not positive, or head_dim not integral
 * Note: head_split MATERIALIZES the transpose (a gmem read+write). Production inference (vLLM/SGLang/flash-attention) does NOT materialize it -- the attention kernel reads Q/K/V with strides directly, making this a non-materializing view. Materializing here is an M6-parity simplification; fusing it into attention's strided GEMM is a Phase 2 optimization.
 */
Tensor head_split(const Tensor &t, std::int64_t heads);

/**
 * Launch the CUDA head-split kernel (a strided transpose copy). Internal helper -- called only by head_split() on the Device::CUDA branch.
 *
 * No temp device buffer (unlike embedding/rope's d_ids/d_positions): both t and out are caller tensors, so this launch neither allocs nor frees anything -- there is NO leak surface here and no cuda_free to pair. (Hence no no_leak test for head_split.)
 *
 * Sync: launch ends with CHECK_CUDA(cudaGetLastError()) + CHECK_CUDA(cudaDeviceSynchronize()), matching embedding_launch.
 *
 * Kernel: pure typed copy, NO floating-point arithmetic. Templated internally on T (dispatched in the launch on dtype) so each load/store uses the dtype's natural width (LDG.32 for float, LDG.16 for __nv_bfloat16) instead of a byte loop (LDG.U8) -- this is the ONLY reason for the template; the math is dtype-independent. (A byte memcpy with a runtime length compiles to LDG.U8, 4x/2x the instructions; SASS-verified.)
 * One block per (head, seq) segment: grid = (heads*seq). blockIdx.x -> (h = blockIdx.x/seq, s = blockIdx.x%seq). Each block copies ONE contiguous head_dim-element segment:
 *   out[h*seq*head_dim + s*head_dim + k] = in[s*heads*head_dim + h*head_dim + k]
 * threadIdx.x strides over k in [0, head_dim) with stride blockDim.x (head_dim may exceed blockDim.x); the k < head_dim guard handles the tail, so no separate col/real_bytes math.
 * k is the contiguous axis in BOTH in and out (for fixed head,seq), so reads and writes are coalesced.
 *
 * Params:
 *   out      -- device ptr, (heads, seq, head_dim), dtype == t.dtype()
 *   t        -- device ptr, (seq, heads*head_dim), row-major contiguous
 *   heads    -- number of heads blockPerGrid(heads*seq)
 *   seq      -- sequence length blockPerGrid(heads*seq)
 *   head_dim -- elements per head (segment length)
 *   dtype    -- Float32 or BFloat16 (sets byte stride only)
 */
void head_split_launch(void *out,
                       const void *t,
                       std::int64_t heads,
                       std::int64_t seq,
                       std::int64_t head_dim,
                       DType dtype);

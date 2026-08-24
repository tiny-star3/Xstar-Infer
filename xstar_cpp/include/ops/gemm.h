#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>

#include "tensor.h"

/**
 * BLAS-style GEMM with leading dimensions -- the shared kernel behind matmul / linear / attention.
 * Two layers:
 *   * gemm_cpu<T>  -- pointer-level template, ONE body for both dtypes.
 *     The cast pair static_cast<float>(T) on the way in and static_cast<T>(float) on the way out is identity for float and a real upcast / RNE-downcast for bfloat16, so a single loop body serves both.
 *     This is the only place the accumulation loop exists; factoring it out is the whole point of gemm -- the loop was duplicated, f32/bf16-pairwise, across matmul, linear, and the two attention reductions (4 sites).
 *   * gemm(A, B, transB) -- the Tensor-level op: validates shapes/dtype, allocates a contiguous (m, n) result, computes contiguous leading dimensions, and dispatches to gemm_cpu<float> / gemm_cpu<bfloat16>.
 *
 * Transpose contract (transB ONLY, no transA):
 *   A is ALWAYS (m, k), no-transpose. Across all four call sites the first operand is (rows, reduce) with the reduce index contiguous, so transA has no consumer and is omitted -- add it only when one appears.
 *   transB = false -> B is (k, n):      B[kk, j] at B_ptr[kk * ldb + j]
 *   transB = true  -> B is (n, k) (i.e. B^T is consumed):
 *                                       B[kk, j] at B_ptr[j  * ldb + kk]
 *   With transB the transpose is realized by index, not by a memory copy -- the linear / attention-Q@K^T choice (weight stored (out, in), K stored (seq, hd)).
 *
 * Leading dimensions (lda, ldb, ldc) -- BLAS convention:
 *   A[i, kk] at A_ptr[i * lda + kk],  C[i, j] at C_ptr[i * ldc + j],  B as above.
 *   ldc exists because attention's scores@V writes each head's (seq, head_dim) output into a STRIDED slice of the merged (seq, num_heads * head_dim) result -- row stride num_heads * head_dim, not head_dim.
 *   A contiguous-only output gemm cannot express that without a temporary buffer + scatter; a leading dimension lets the kernel write the strided slice in place.
 *   For the contiguous cases (matmul, linear, attention Q@K^T) the caller passes ldc == n (and lda == k, ldb == the stored B row stride).
 *
 * Usage map (which site is which transB):
 *   transB = false: matmul (A@B),             attention scores@V (attn @ V)
 *   transB = true : linear (x @ W^T),         attention Q@K^T    (Q @ K^T)
 *
 * Numerics (the "reduce / accumulate stays f32" family):
 *   - The k-length dot product accumulates in float regardless of T.
 *   - T = float: the casts are identities, f32 in / f32 out.
 *     The kernel MUST preserve the existing accumulation order so f32 stays BIT-EXACT vs the current per-op loops; that bit-exactness is the refactor's correctness gate (the existing matmul/linear/attention tests must stay green, not merely allclose).
 *   - T = bfloat16: upcast each operand via static_cast<float>, accumulate f32, RNE-cast out via static_cast<bfloat16>.
 *     Non-bit-exactness (vs PyTorch) stems from the accumulation order; there is NO INTERMEDIATE downcast -- the f32 accumulator is RNE-cast to bf16 exactly once, at output (the static_cast<T> on the assignment line).
 *
 * Caller responsibilities (what stays OUT of gemm):
 *   - bias stays in linear: linear adds bias AFTER gemm's output downcast.
 *     gemm_cpu writes the RNE-downcast bf16 result; linear then reads it back, upcasts to f32, adds bias, and RNE-downcasts again -- so bias hits an already-downcast value, NOT the f32 accumulator.
 *     This is a DEVIATION from PyTorch CPU bf16 F.linear, which adds bias to the f32 accumulator BEFORE the single output downcast (verified: ref==bias-in-f32 is bit-exact; ref==bias-after-downcast differs by ~0.03).
 *     The post-downcast add is forced by this refactor's split: gemm owns only the GEMM + output cast, so bias is a separate per-op step that cannot reach inside the accumulator.
 *     The extra round-trip is absorbed by linear's allclose budget.
 *     gemm itself has no bias.
 *   - per-head loop + GQA head indexing (h / rep) + additive/causal mask + softmax stay in attention.
 *     attention calls gemm_cpu directly, per head, passing a strided ldc for scores@V (merged output) and a contiguous ldc for Q@K^T.
 *
 * Preconditions:
 *   - A is 2-D (m, k); B is 2-D and matches transB ((k, n) or (n, k)).
 *   - A.dtype() == B.dtype(); C carries the same element type; Float32 or BFloat16.
 *   - inner dimension agrees: A.shape()[1] == (transB ? B.shape()[1] : B.shape()[0]).
 *
 * Throws std::runtime_error on: rank mismatch, dtype mismatch, inner-dim mismatch, unsupported dtype (Tensor gemm only -- gemm_cpu is pointer-level and trusts its caller).
 */
template <typename T>
void gemm_cpu(const T *A, const T *B, T *C, std::int64_t m, std::int64_t k, std::int64_t n, bool transB, std::int64_t lda, std::int64_t ldb, std::int64_t ldc)
{
    // lda: A 的行跨度
    // ldb: B 的行跨度
    // ldc: C 的行跨度
    // 用于描述"跨步的子矩阵"
    std::vector<float> temp(n, 0);
    for (size_t i = 0; i < m; i++)
    {
        std::fill(temp.begin(), temp.end(), 0);
        for (size_t kk = 0; kk < k; kk++)
        {
            // s 在 j 循环中是常量
            float s = static_cast<float>(A[i * lda + kk]);
            for (size_t j = 0; j < n; j++)
            {
                // B 访问 strided(转置索引的代价), C 连续
                temp[j] += s * static_cast<float>(transB ? B[kk + j * ldb] : B[kk * ldb + j]);
            }
        }
        for (size_t j = 0; j < n; j++)
        {
            C[i * ldc + j] = static_cast<T>(temp[j]);
        }
    }
}

/**
 * Tensor-level GEMM op.
 * m = A.shape()[0], k = A.shape()[1], n = (transB ? B.shape()[0] : B.shape()[1]).
 * Allocates a contiguous (m, n) result of A.dtype() and dispatches by dtype.
 */
Tensor gemm(const Tensor &A, const Tensor &B, bool transB);

/**
 * Launch the tiled GEMM kernel -- the .cu side, called by the cpp orchestration.
 *
 * Layering: declared here (CUDA-free, pure C++) so gemm.cpp (g++) can call it; defined in gemm.cu (nvcc) where the <<<>>> launch lives.
 * Same invariant as rmsnorm_launch / softmax_launch / concat_launch: host orchestration links the symbol without touching CUDA headers.
 *
 * This is the GPU counterpart of the CPU gemm_cpu<T> (above) -- SAME pointer-level contract, SAME numerics, DIFFERENT implementation (tiled).
 * The CPU gemm_cpu<T> stays as the bit-trusted correctness oracle; the GPU kernel is validated against it via allclose (NOT bit-exact -- GPU k-reduction order differs from CPU sequential).
 *
 * Scope (M4a/b/M5a -- CORRECTNESS, not tuning):
 *   - transB == false AND true are BOTH implemented (M4b): the kernel is templated <bool TRANSB>, dispatched at runtime in gemm_launch; the two specializations are binary-isolated (no-trans path unaffected by transB edits).
 *     transB=true reads B as (n,k) along the k dimension (contiguous, vectorizable) into a transposed smemB layout [BN][BK]; the compute-side read index is flipped via if constexpr. Same per-load alignment guard (actual load address % sizeof(vec_t<T>) == 0), routing misaligned loads to the scalar slow path.
 *   - Epilogue (M5a): kernel templated further <bool HAS_BIAS>. Two epilogues:
 *       HAS_BIAS=false -> Identity (plain store, C = acc). Consumed by matmul, attention, gemm(Tensor).
 *       HAS_BIAS=true  -> BiasAdd (C = downcast(acc + bias[col])). Consumed by linear (device branch).
 *     HAS_BIAS is selected by `bias != nullptr` in gemm_launch's runtime dispatch (no separate has_bias param); `if constexpr (HAS_BIAS)` compile-prunes the bias load + bias-add, so Identity specialization carries NO bias code (binary == M4 Identity).
 *     This mirrors cuBLASLt CUBLASLT_EPILOGUE_BIAS / CUTLASS epilogue-template fusion (design, not pasted code): bias is fused into the GEMM store to avoid a separate bias kernel's C read+write round-trip.
 *     SiluAndMul is NOT this template's epilogue: gate/up split along the N output dim, d=intermediate_size (4864) >> BN (32), so gate col j and up col j+d always land in different blockIdx.x -- a per-block store-time epilogue cannot reach across blocks.
 *     SiluAndMul needs a paired-tile restructure (M5 mlp).
 *   - Tile params are constexpr, chosen SMALL for debuggability (dense boundary cases), NOT tuned for sm_75.
 *     Tuning (large tile, double buffer, bank-conflict swizzle, splitK) is Phase 5 -- RE-DONE per architecture when renting A100/H100 (Phase 5) -- logic is card-independent, only the constexpr tile params change.
 *     This is why tile params are constexpr, not hardcoded in logic.
 *
 * Tiling (card-independent logic; one block computes one (BM,BN) sub-block of C):
 *   grid  = (ceil(N/BN), ceil(M/BM))   -- 2D: blockIdx.x = N column-block, blockIdx.y = M row-block
 *   block = 64 threads (8x8 = THREADPERBLOCKDIM; one thread owns a TM*TN=4*4 register tile).
 *   K is tiled in steps of BK; each step loads A:(BM,BK) and B:(BK,BN) into shared memory, all threads compute acc += smemA * smemB (accumulate in f32 registers), two __syncthreads() per K-step (after load, after compute).
 *   The f32 accumulator never leaves registers across the K loop.
 *
 * Boundary (card-independent HARD requirement, not tuning):
 *   M/N/K need not be multiples of BM/BN/BK.
 *   Boundary tiles CLIP: out-of-bounds A/B elements are loaded as 0 (contribute 0 to the dot product, matching CPU which just skips them); out-of-bounds C elements are NOT written.
 *   Skipping this reads/writes garbage -- same OOB family as the softmax dim_size<blockDim bug.
 *
 * Ownership / where each pointer lives (caller = gemm.cpp device branch):
 *   - C: GPU output buffer (result.data()), m*n*dtype_size. Kernel WRITES.
 *   - A: GPU input (A.data()), (m, k) no-trans, row-major. Assumes contiguous (lda == k).
 *   - B: GPU input (B.data()), (k, n) if transB==false / (n, k) if transB==true, row-major.
 *     ldb == n (no-trans) or ldb == k (transB). lda/ldb are NOT required to be multiples of 4 -- any value works.
 *   - m, k, n: GEMM dimensions (A is (m,k), B is (k,n), C is (m,n)).
 *   - transB: false (B is (k,n)) and true (B is (n,k), B^T consumed) are both implemented.
 *   - lda, ldb: leading dimensions, ANY value (no multiple-of-4 requirement).
 *               The kernel's per-load alignment guard checks the ACTUAL load address (reinterpret_cast<std::uintptr_t>(&ptr[idx]) % sizeof(vec_t<T>) == 0):
 *                 VEC-aligned loads take the float4/uint2 fast path, misaligned ones fall to the per-element scalar slow path (scalar reads need only element-size alignment, always satisfied).
 *                 gemm_launch no longer throws on misalignment.
 *                 This guards TWO independent misalignment sources the old "lda%4" check missed:
 *                   (1) row-stride misalignment (lda not multiple of 4, e.g. attention scores@V with seq%4!=0 -> some rows' start addresses misalign), AND
 *                   (2) BASE-pointer misalignment (caller passes a sliced pointer like attention's attn_weights + h*seq*seq*elem_size, whose byte offset is not a multiple of vec_t size).
 *                       The old "idx % VEC" offset-only guard caught (1) but NOT (2); the actual-address guard catches both.
 *                 Aligned launches (lda/ldb%4==0 AND base 256-aligned: linear/mlp/Q@K^T) take the identical fast path as before (guard is trivially true) -> M4/M5 numerics unchanged.
 *                 Misaligned launches (attention scores@V A-side, seq%4!=0) degrade to scalar -- correct, slower, only on that small op; Phase 5 replaces scores@V with FA2 fusion.
 *     ldc: arbitrary (only the epilogue store uses it, the main loop is ldc-free).
 *          ldc==n for contiguous output (matmul/linear/gemm Tensor); ldc==num_heads*head_dim for attention scores@V's strided slice into the merged output -- this is the strided-C path, exercised by attention (M5) which is its first real consumer.
 *   - dtype: Float32 or BFloat16 -- dispatches to gemm_kernel<float> / <__nv_bfloat16>.
 *            Compute is f32 either way (toFloat upcast, accumulate f32, (T) RNE downcast), reusing cuda/dtype_cast.h from M3.
 *
 * Numerics (matches the CPU gemm_cpu<T> contract above):
 *   - k-dot accumulates in f32 regardless of T (toFloat on read, (T) RNE on write).
 *   - f32 path: NOT bit-exact vs CPU (GPU k-tile reduction order != CPU sequential) -> allclose.
 *     Tolerance MEASURED then set (M9): grows with K (reduction length) -- do not copy a number.
 *   - bf16 path: one downcast at write; larger tolerance than f32 (downcast ULP dominates).
 *
 * Sync: cudaGetLastError (launch config) + cudaDeviceSynchronize (execution) + CHECK_CUDA, two-stage as the other _launch functions.
 *
 * NOTE on the epilogue (M5a, done): the store loop is the fuse point for output transforms.
 *   - Identity (HAS_BIAS=false): C[i][j] = downcast(acc). The bias-load and bias-add blocks are compile-pruned (if constexpr), so this specialization is binary-identical to the M5 store.
 *   - BiasAdd (HAS_BIAS=true): bias is loaded ONCE into smemBias[BN] after the K-loop (bias is K-independent, so it does not enter the K-loop), one __syncthreads, then store does C = downcast(acc + toFloat(smemBias[j*bdx+tx])).
 *       Numerics (bf16): ONE downcast (f32 acc+bias -> bf16), matching torch F.linear semantics -- NOT matching CPU linear, which does two downcasts (downcast acc -> add bias -> downcast).
 *       So GPU-linear vs CPU-linear in bf16 differ by the downcast-order deviation (~0.0156 measured), absorbed by rtol=1e-2. vs torch F.linear the GPU path is bit-exact (0.0, both do one downcast).
 *       This is why the bf16-with-bias cuda test uses CPU as oracle (self-consistency with the gemm test style) and sits near the tolerance edge, while vs torch it would be exact.
 *   - smemBias is declared OUTSIDE if constexpr (shared by both specializations): Identity specialization allocates 32 elements (bf16 64B, 1.5% of tile smem) it never reads -- a trade accepted because putting smemBias inside if constexpr would break scope (the store reads it from outside the block).
 *       The 64B does not affect occupancy (registers, not smem, are the limiter at 87.5%).
 */
void gemm_launch(void *C,
                 const void *A,
                 const void *B,
                 const void *bias,
                 std::int64_t m,
                 std::int64_t k,
                 std::int64_t n,
                 bool transB,
                 std::int64_t lda,
                 std::int64_t ldb,
                 std::int64_t ldc,
                 DType dtype);

#pragma once
#include <vector>
#include <cstdint>

#include "tensor.h"

/**
 * Embedding lookup (token-id gather): maps each integer token id to the corresponding row of a 2-D weight table.
 * No floating-point computation -- this is a pure gather (copy rows by index), so the output is bit-exact and matches torch.nn.functional.embedding with padding_idx=None.
 *   out[..., :] = weight[ids[...], :]
 *
 * Token ids are INDICES, not values: they do NOT enter the Tensor/DType system(DType stays float-only). Only the weight table is a Tensor (it reuses the mmap zero-copy view).
 * ids are passed as a raw int64 buffer + shape so the core library stays free of pybind11; the binding layer unpacks a numpy int64 array into (pointer, shape).
 *
 * Dtype: output dtype == weight.dtype() (a gather moves bytes; the integer id does not participate in floating-point arithmetic).
 * Input id precision is irrelevant to output precision.
 *
 * Shapes:
 *   weight: (vocab_size, hidden)  -- 2-D, row-major contiguous, float dtype
 *   ids:    (...,)                -- rank >= 1, int64, any leading dims (batch/seq)
 *   out:    (..., hidden)         -- ids_shape with `hidden` appended; dtype == weight.dtype()
 *
 * Precondition:
 *   - weight.shape().size() == 2 and weight.dtype() is Float32 or BFloat16
 *   - ids_shape.size() >= 1
 *   - every id in [0, vocab_size), where vocab_size = weight.shape()[0]
 * Throws std::runtime_error on:
 *   - weight not 2-D, or weight not a float dtype
 *   - empty ids_shape
 *   - any id < 0 or id >= vocab_size (out-of-range index)
 * Note: out-of-range ids are treated as ERRORS (single-device case).
 * A mask strategy (clamp to 0 + zero output) is deferred to the tensor-parallel stage.
 */
Tensor embedding(const Tensor &weight, const std::int64_t *ids, const std::vector<std::int64_t> &ids_shape);

/**
 * Launch the CUDA embedding gather kernel. Internal helper -- called only by embedding()'s CUDA branch, not part of the public API.
 *
 * Pointers: out, weight, d_ids are ALL device pointers.
 *   d_ids is the device copy of the host ids buffer, owned by the caller: caller does cuda_alloc + cuda_memcpy_h2d BEFORE this call and cuda_free AFTER.
 *   launch neither allocs nor frees d_ids (mirrors rmsnorm_launch: launch = pure kernel start).
 *
 * Sync: launch ends with cudaDeviceSynchronize, so on return the kernel is done and the caller's subsequent cuda_free(d_ids) is safe (no use-after-free of the index buffer).
 *
 * Kernel: pure byte gather, NO floating-point arithmetic. Templated internally on T (dispatched in the launch on dtype) so each load/store uses the dtype's natural width (LDG.32 for float, LDG.16 for __nv_bfloat16, SASS-verified) instead of a byte loop (LDG.U8) -- this is the ONLY reason for the template; the math is dtype-independent. (A byte memcpy with a runtime length compiles to LDG.U8, 4x/2x the instructions.)
 * One block per token (grid = numel), 256 threads per block stride over the row:
 *    for (j = threadIdx.x; j < hidden; j += blockDim.x) out[i*hidden + j] = weight[d_ids[i]*hidden + j]
 * The j < hidden guard handles the tail (no separate col/real_bytes math). j is the row's contiguous fast axis, so reads and writes are coalesced. Bit-exact by construction (a gather moves bytes, no arithmetic).
 *
 * Params:
 *   out     -- device ptr, (numel, hidden), dtype == weight.dtype()
 *   weight  -- device ptr, (vocab, hidden), row-major contiguous
 *   d_ids   -- device ptr, numel int64 indices, each already range-checked in [0, vocab) by caller
 *   numel   -- number of token ids
 *   hidden  -- row length in elements
 *   dtype   -- Float32 or BFloat16 (selects the LDG.32 / LDG.16 load width via the template, not a byte stride)
 */
void embedding_launch(void *out,
                      const void *weight,
                      const std::int64_t *d_ids,
                      std::int64_t numel,
                      std::int64_t hidden,
                      DType dtype);
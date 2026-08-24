#pragma once
#include <cstddef>

/**
 * Minimal GPU memory allocator -- the bottom layer of Xstar's CUDA memory stack.
 *
 * Two layers, kept deliberately separate by abstraction level:
 *   - THIS layer (cuda_allocator): raw byte-level alloc/free on the GPU.
 *     Backend today is cudaMalloc / cudaFree.
 *     Swappable later (e.g. one big pre-allocated buffer carved by offset) WITHOUT changing this interface.
 *   - BlockManager: paging -- integer block indices into a pool, free-list, logical<->physical block table.
 *     It lives ABOVE this allocator and speaks in block indices, NOT void* + size.
 *     So free-list / paging logic is NOT pushed down here.
 *     The split is by abstraction level: this layer owns a physical GPU pointer; BlockManager owns a logical block number.
 *     Mixing them would entangle "how much VRAM" with "which block of which sequence".
 *
 * Why no Device parameter: CUDA-only.
 * CPU tensors never route through here -- they keep std::malloc / std::free in tensor.cpp (Phase 1 bit-trusted oracle, left untouched).
 * tensor.cpp branches on its device_ field and calls cuda_alloc/free only on the CUDA branch.
 *
 * Why cuda_free takes no size: it mirrors cudaFree's signature.
 * A size argument would only matter for an in-allocator size-class free-list, but paging's free-list is keyed by block index in BlockManager, not by byte size here.
 * Keep the surface minimal; add size later ONLY if a size-class pool actually appears.
 *
 * Return / errors:
 *   - cuda_alloc throws std::runtime_error on CUDA failure (via CHECK_CUDA macro), the CPU path throws bad_alloc, the CUDA path throws runtime_error; both signal failure by exception, so callers do NOT branch on nullptr.
 *   - cuda_free(nullptr) is a no-op (matches std::free / cudaFree semantics).
 *
 * Preconditions:
 *   - bytes > 0 is the normal case; bytes == 0 follows cudaMalloc(0)'s implementation-defined behavior and is best avoided by callers.
 */
void *cuda_alloc(std::size_t bytes);
void cuda_free(void *ptr);

/**
 * Raw byte copies -- the bottom-layer transport, mirroring cudaMemcpy (HostToDevice / DeviceToHost / DeviceToDevice).
 *
 * Three wrappers, one per cudaMemcpy kind:
 *   - cuda_memcpy_h2d : host -> GPU
 *   - cuda_memcpy_d2h : GPU  -> host
 *   - cuda_memcpy_d2d : GPU  -> GPU   (new: consumed by BlockManager CoW -- copies one block's contiguous KV bytes to a freshly allocated block in the same pool)
 *
 * Why raw (void*, no Tensor): same layering as cuda_alloc/free.
 * The Tensor-level to_cuda/to_cpu (in tensor.cpp) own shape/dtype/device bookkeeping and call h2d/d2h for the actual byte move; these just move bytes between two pointers the caller already sized.
 * d2d is NOT a Tensor-level path: BlockManager speaks in block indices into a single pre-allocated pool, so it computes (kv_pool + id*block_bytes) pointers itself and calls d2d directly -- no Tensor round-trip.
 * Keeping Tensor logic OUT of here means tensor.cpp / block_manager.cpp link the symbol without including any CUDA header (g++-compilable), same rule as cuda_alloc.
 *
 * Sync, not async: cudaMemcpy (blocking) is correct-first.
 * vLLM's C++ side uses cudaMemcpyAsync + an explicit stream to overlap H2D with compute; that is a Phase 5 optimization, gated on a stream abstraction that does not exist yet.
 * Do NOT reach for async here -- without a stream it would run on the default stream and gain nothing while adding sync pitfalls.
 *
 * Errors: throw std::runtime_error on CUDA failure (the CUDACHECK pattern).
 * These failures are recoverable-ish (caller can catch), unlike cuda_free in a destructor, so throwing (not aborting) is correct here.
 *
 * Preconditions:
 *   - dst and src are non-null, both on the side their kind requires:
 *       h2d: dst=GPU, src=host;  d2h: dst=host, src=GPU;  d2d: dst=GPU, src=GPU.
 *   - bytes > 0.
 */
void cuda_memcpy_h2d(void *dst, const void *src, std::size_t bytes);
void cuda_memcpy_d2h(void *dst, const void *src, std::size_t bytes);
void cuda_memcpy_d2d(void *dst, const void *src, std::size_t bytes);

/**
 * Current free GPU memory in bytes (cudaMemGetInfo).
 * Used by the no-leak probe: free bytes must stay flat across N alloc/free cycles (a drop = free didn't release).
 * This is the same API vLLM's memory profiler uses to size the KV-cache pool after weights+activations are accounted for -- a real industrial primitive, not a test hack.
 */
std::size_t cuda_free_bytes();

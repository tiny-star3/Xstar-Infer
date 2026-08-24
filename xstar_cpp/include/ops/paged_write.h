#pragma once
#include "tensor.h"
#include "block_manager.h"

/**
 * Paged KV-cache write: scatter rope'd K and raw V for num_tokens tokens into the BlockManager pool, one token per physical slot, indirect-addressed by slot_mapping.
 * Write half of M9 sub-block B paging (read half = Block 3 paged decode kernel).
 *
 * Intra-block layout [nkv, BS, hd] (head outer, same shape as M8 FA2), K-then-V two regions:
 *   element offsets (kernel indexes pool as T*, so all terms below are in ELEMENTS, not bytes):
 *     K[L][B][h][slot][d] = pool + L*layer_stride_elems + B*block_elems + h*BS*hd + slot*hd + d
 *     V[L][B][h][slot][d] = pool + L*layer_stride_elems + B*block_elems + nkv*BS*hd + h*BS*hd + slot*hd + d
 * K region starts at offset 0; V region element offset = nkv*BS*hd.
 *
 * The layer_stride_elems/block_elems passed to the kernel are BlockManager's byte counts (layer_stride / block_bytes) divided by sizeof(T) in paged_write() (the conversion happens on the host before launch).
 * Bytes per block: K 4096 + V 4096 = nkv*BS*hd*2 + nkv*BS*hd*2 = 8192 = block_bytes.
 *
 * Contract:
 *   1. K is rope'd, V is raw (V is never rope'd). Both [nkv, num_tokens, hd] contiguous CUDA.
 *   2. slot_mapping[num_tokens]: physical slot_id (vLLM form); kernel recovers block_id = slot_id / BS, slot = slot_id % BS.
 *   3. GQA: written per num_kv_heads (nkv), NOT repeated; query-head repetition is the read kernel's job (Block 3), not the write's.
 *   4. Partial: writes exactly num_tokens tokens; untouched slots in a partially-written block keep their previous value (no zero-padding).
 *
 * Addressing params are taken from the BlockManager (pool_ptr, layer_stride/block_bytes as byte counts), then divided by sizeof(T) on the host to become the element-count params (layer_stride_elems / block_elems) the kernel receives.
 * Called per layer (layer is a runtime parameter, not a grid dimension). GPU-only (no CPU path).
 *
 * Numerics: pure bf16 bit-move (no arithmetic) -> bit-exact; no tolerance probe needed.
 * slot_mapping is a host int array; paged_write() H2D-copies it to device (rope positions pattern).
 */
void paged_write(const BlockManager &bm, int layer,
                 const Tensor &K, const Tensor &V,
                 const int *slot_mapping);

/**
 * CUDA launch (declared CUDA-free so paged_write.cpp links it without a CUDA header).
 */
void paged_write_launch(void *pool,
                        const void *K, const void *V,
                        const int *d_slot_mapping,
                        std::int64_t layer_stride_elems, int block_elems, int layer,
                        int num_tokens, int nkv, int hd, int block_size,
                        DType dtype);

#pragma once
#include <vector>

#include "tensor.h" // Device
#include "device.h"

/**
 * Physical block handle with an inlined free-list doubly-linked list node.
 *
 * block_id is the KV-tensor first-axis index (physical block number = row index), NOT a pointer.
 * ref_cnt: reference count. alloc sets 1; fork increments the whole chain; free decrements, and only when it reaches 0 is the block returned to the free list.
 *   M7 does no prefix caching; the "pin against eviction" use of ref_cnt is reserved for the Phase 4 radix layer, but the counting logic is implemented correctly now.
 * prev_free/next_free: inlined free-list links for O(1) detach/attach with no find() and no iterator invalidation (vLLM V1 inlines the same way; a std::list container would need an O(n) find, defeating the point of a doubly-linked list).
 *   Sentinel head/tail nodes (block_id = -1) eliminate null-check branches.
 *
 * Phase 4 radix reuse: this struct is unchanged; a last_access_time field is added then, when a "block was read" event exists to refresh it (M7 has no such event, so the field would be dead).
 */
struct KVCacheBlock
{
    int block_id;
    int ref_cnt;
    KVCacheBlock *prev_free;
    KVCacheBlock *next_free;
};

/**
 * GPU VRAM page pool -- the physical layer (PagedAttention paradigm).
 *
 * One large KV tensor is pre-allocated (num_blocks * block_bytes, reusing cuda_alloc) and sliced
 * into block_ids; blocks are NOT cuda_alloc'd individually (that reintroduces fragmentation, the very problem PagedAttention solves).
 * block_id is the KV-tensor row index; a paged kernel indexes via block_table[seq][logical_idx] -> block_id -> kv_ptr + block_id*stride (the paged kernel is FA2's job; M7 does not touch kernels).
 *
 * Free list: doubly-linked list with sentinels. M7 uses LIFO -- free prepends to head, alloc pops from head.
 * This matches vLLM V1's non-cached path (source comment: "LIFO reuse of non-cached blocks for better GPU locality") and SGLang's default path (need_sort=False, torch.cat to prefix == push to head == a stack).
 * M7 has no cached blocks (no prefix cache), so LRU semantics do not exist yet -- LRU is a property of cached blocks going to the tail (Phase 4), not of the free list's default end.
 * M7's head end IS the head end of the Phase 4 two-ended structure: when Phase 4 adds cached blocks, a "cached blocks append to tail" branch is added, the free list becomes two-ended (head = LIFO non-cached, tail = LRU cached), matching vLLM V1 free_blocks; M7's head end does not need rework.
 *
 * Layering (for Phase 4 radix reuse): the pool (this class) owns only physical alloc/free/ref/fork/CoW; the logical->physical mapping (BlockTable, a separate layer) holds the block_id sequence, and the Phase 4 radix layer operates on BlockTable, not on this pool.
 *
 * CoW (copy-on-write): after a whole-chain fork, writing a shared block triggers CoW. Two stages:
 *   logical (this class): on write with ref_cnt>1 -> alloc a new block -> redirect the caller's BlockTable to it -> decrement the old block's ref_cnt.
 *   physical (synchronous, in this class): cudaMemcpy D2D copies the old block's KV bytes to the new block.
 *     vLLM/SGLang do the physical copy asynchronously on the worker/kernel side; M7 has no scheduler and no paged kernel, so it is done synchronously here (blocking; sufficient for parity testing).
 *     Trigger condition: ref_cnt>1 AND write; ref_cnt==1 writes directly, no CoW.
 *   Scenarios with no CoW caller (greedy/single sampling) never trigger it; CoW is for beam search / parallel sampling.
 */
class BlockManager
{
public:
    /**
     * Pre-allocate num_layers * num_blocks blocks.
     * The pool is laid out [num_layers, num_blocks, block_bytes] row-major (layer outer, block inner), so layer L's blocks are CONTIGUOUS (== vLLM gpu_cache[L] per-layer tensor) but a single block_id across layers is NOT contiguous (strided by layer_stride).
     * num_layers defaults to 1 == exact M7 behavior (M7 tests unchanged).
     */
    BlockManager(int num_blocks, int block_size, int kv_slot_bytes, Device dev = Device::CUDA, int num_layers = 1);

    ~BlockManager();

    BlockManager(const BlockManager &) = delete;
    BlockManager &operator=(const BlockManager &) = delete;

    /**
     * Pop num blocks from the free-list head, set ref_cnt=1, return their physical block_ids.
     * Throws std::runtime_error if the free list is empty.
     */
    std::vector<int> alloc(int num);

    /**
     * Decrement each block's ref_cnt; return blocks that reach 0 to the free-list head (LIFO end).
     * Blocks with ref_cnt>0 are not reclaimed (pinned).
     * M7 routes everything through the non-cached path (uniform head); when Phase 4 adds the cached branch, cached blocks go to the tail instead, gated by a block_hash check added to this method.
     */
    void free(const std::vector<int> &block_ids);

    /**
     * Increment each block's ref_cnt (Phase 4 prefix pin; no caller in M7, but the API is kept).
     */
    void ref(const std::vector<int> &block_ids);

    /**
     * Whole-chain fork: increment ref_cnt for every block_id in src_table and return a new block table pointing at the SAME physical blocks (no data copy -- OS-style fork).
     * CoW is deferred until a shared block is written via write_block.
     */
    std::vector<int> fork(const std::vector<int> &src_table);

    /**
     * Write a block.
     * If ref_cnt>1:
     *   trigger CoW (alloc a new block + D2D copy + decrement the old block's ref_cnt + set the new block's ref_cnt=1), return the new block_id.
     * If ref_cnt==1:
     *   no CoW, return the original block_id. The caller updates its BlockTable to point at the returned id.
     */
    int write_block(int block_id);

    /**
     * Cross-layer whole-logical-block copy (matches vLLM copy_blocks kernel which copies per layer, not one big contiguous memcpy -- the [layers,blocks,bytes] layout makes a single block's layers non-contiguous).
     * D2D copy primitive (used by CoW; public so tests can verify the copy is bit-exact directly):
     *   copy block_bytes bytes from block src_id to block dst_id.
     */
    void cow_copy(int src_id, int dst_id);

    /**
     * Free-list length. Invariant: num_free() + num_allocated() == num_blocks.
     */
    int num_free() const;

    /**
     * Number of blocks currently allocated (ref_cnt>0).
     */
    int num_allocated() const;

    /**
     * test-only
     */
    int block_ref_cnt(int block_id) const;

    /**
     * kv_pool_ is a pointer to the device for the entire pool.
     */
    void *pool_ptr() const;

    /**
     * layer_stride_, kernel calculates layer L base.
     */
    int64_t layer_stride() const;

    /**
     * Bytes per block (block_size * kv_slot_bytes); intra-block K-then-V layout is 2 regions of nkv*BS*hd*2 bytes each.
     * Exposed so paged_write can compute V-region offset.
     */
    int block_bytes() const;

    /**
     * Tokens per physical block (block_size).
     */
    int block_size() const;

    /**
     * kv_pool_ + layer*layer_stride_(convenience).
     */
    void *layer_base(int layer) const;

    /**
     * Pool layer count (constructor arg); layer L base = L * layer_stride.
     * Paged forward asserts bm.num_layers() == cfg.num_hidden_layers.
     */
    int num_layers() const;

private:
    // never resized after construction; free-list pointers into it are stable
    std::vector<KVCacheBlock> blocks_;
    KVCacheBlock head_sentinel_, tail_sentinel_;
    void *kv_pool_; // cuda_alloc pre-allocated large buffer
    // block_size_ = 一个 block 容纳的 KV slot 数(token 数)
    // block_bytes_ = 一个 block 占的字节数
    int num_blocks_, block_size_, block_bytes_;
    int num_layers_;
    // 层内连续,跨层步长
    std::int64_t layer_stride_;
    // 目前只支持 CUDA, 后面扩展 CPU
    Device dev_;
    int num_allocated_;

    // 防止 block_id 越界
    void check_id(int block_id) const;
    // 防 double-free 把 ref_cnt 打负 / 把块二次塞进 free list
    void check_allocated(int block_id) const;
    // 从 head 弹出一个 free block, 返回其 block_id; 空链表报错
    // node 脱钩(prev/next 置空)
    int pop_free_head();
    // 把 block(已在 blocks_ 里, ref_cnt 刚到 0)插回 head 端(LIFO)
    void push_free_head(KVCacheBlock *block);
};

/**
 * Logical->physical mapping, a separate layer (for Phase 4 radix reuse).
 *
 * The physical block_id sequence occupied by one sequence.
 * Logical block i = token_pos // block_size, which is directly the vector index (implicit indexing, no extra map).
 * The Phase 4 radix layer operates here (prefix -> which physical blocks), not on the BlockManager pool.
 * M7 keeps it minimal: it holds the id sequence; append/query APIs are added as needed (append the ids returned by alloc).
 */
struct BlockTable
{
    std::vector<int> physical_ids;
};

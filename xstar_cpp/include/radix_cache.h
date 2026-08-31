#pragma once
#include <map>
#include <vector>

#include "block_manager.h"

/**
 * Radix prefix-cache tree node.
 * Maps a token-prefix segment to physical blocks for cross-request reuse.
 * LRU links + lock_ref are NODE-level; BlockManager keeps its single LIFO free-list unchanged.
 *   lock_ref (node): drives eviction. block ref_cnt (bm): drives physical release. Don't conflate.
 */
struct RadixNode
{
    // fork key = first block's tokens (block_size), or full residual (< block_size) for residual leaves
    std::map<std::vector<int>, RadixNode *> children;
    RadixNode *parent = nullptr;
    // token segment, len = block_count * block_size
    std::vector<int> key;
    // physical block_ids, size = ceil(len(key)/block_size). Split at block boundary -> whole blocks only.
    std::vector<int> block_table;
    // >0 = pinned, not evictable. inc/dec walk UP parent chain.
    int lock_ref = 0;
    RadixNode *lru_prev = nullptr;
    RadixNode *lru_next = nullptr;
    // explicit flag, don't infer from lock_ref==0
    bool in_lru = false;
};

/**
 * Node-level LRU doubly-linked list, sentinel head/tail.
 * head = oldest (evict here), tail = newest (insert/move-to-back here).
 * LRU order IS list order, no heap.
 * Only LEAF nodes belong here.
 * INVARIANT: in_lru == true <=> on the list.
 */
class LRUList
{
public:
    LRUList();
    ~LRUList() = default;

    /**
     * Append at tail (MRU end).
     * Pre: !node->in_lru.
     */
    void push_back(RadixNode *node);

    /**
     * Detach + return head (LRU).
     * Pre: non-empty; throws if empty.
     * Clears in_lru.
     */
    RadixNode *pop_front();

    /**
     * Move an on-list node to tail (just accessed).
     * Pre: node->in_lru.
     */
    void move_to_back(RadixNode *node);

    /**
     * Detach (pin/split/delete).
     * Pre: node->in_lru. Clears in_lru.
     */
    void remove(RadixNode *node);

    bool empty() const;
    int size() const;

private:
    RadixNode head_sentinel_;
    RadixNode tail_sentinel_;
    int size_ = 0;
};

/**
 * Radix prefix-cache tree. Cached prefix -> fork blocks (bm.fork, no prefill recompute); residual re-prefilled.
 * Drives BlockManager via bm.fork (reuse), bm.ref (insert), and bm.free (evict). block_size must == bm.block_size().
 * CONTRACT: insert() records an ALREADY-allocated block_table (caller alloc'd + prefilled) and takes ONE ref on the blocks it newly records; tree never allocs.
 *           Caller passes BLOCK-ALIGNED tokens + block_table: tokens truncated to floor(len/block_size)*block_size, block_table covers only those whole blocks.
 *           The residual < block_size (tokens + its blocks) is kept by the caller (request-level), NEVER enters the tree.
 *           If tokens fully match an existing node, returns it WITHOUT overwriting block_table; caller must not re-insert an already-cached prefix (insert is for new prefixes only).
 *           match_prefix returns a BLOCK-ALIGNED length; caller forks returned blocks directly, re-prefills the rest; matched length = len(blocks)*BS.
 *           evict() frees via bm.free + cascades empty parents; returns < need_blocks if LRU empties -> caller falls back to Recompute.
 */
class RadixTree
{
public:
    explicit RadixTree(int block_size);
    ~RadixTree();

    /**
     * Match tokens against the tree; input truncated to a block-aligned length (residual < block_size does NOT participate).
     * Returns (blocks of the matched prefix, node where match ended).
     *   blocks: physical block ids in position order (root segment first), the WHOLE matched chain -- caller forks this list directly.
     *           Matched length is derivable: len(blocks) * block_size().
     *           Terminal node may be partially matched (floored to block boundary).
     *   node:   pin anchor only (inc/dec walk up from here) -- NOT a block source. Root when no match.
     * Refreshes LRU order for visited nodes. Caller: fork(blocks) -> adopt_prefix(forked); skip all when blocks empty.
     */
    std::pair<std::vector<int>, RadixNode *> match_prefix(const std::vector<int> &tokens);

    /**
     * Insert a block-aligned token segment + its already-allocated block_table (residual kept by caller; NOT passed).
     * Takes a ref (bm.ref) on every block it newly records -- tree owns 1 ref per held block; so each recorded block's ref_cnt == 1 (alloc) + N (forked) + 1 (tree).
     * Caller's finish-side bm.free drops its own share.
     * Full-match re-cache returns the existing node WITHOUT adding refs or overwriting block_table.
     * New leaf enters LRUList at tail (lock_ref=0); caller inc_lock_ref to pin. Returns the leaf (or existing node on full match).
     */
    RadixNode *insert(const std::vector<int> &tokens, const std::vector<int> &block_table, BlockManager &bm);

    /**
     * Pin node + all ancestors (walk UP parent chain).
     * 0->1 removes from LRUList.
     */
    void inc_lock_ref(RadixNode *node);

    /**
     * Unpin node + ancestors (reverse of inc).
     * 1->0 leaf re-enters LRUList at tail via push_back (NOT move_to_back: a pinned node is never on the list, so dec always pushes a fresh entry, never moves an existing one).
     * Interior nodes don't enter.
     */
    void dec_lock_ref(RadixNode *node);

    /**
     * Evict LRU leaves until >= need_blocks freed (bm.free + cascade empty parents).
     * Returns blocks freed.
     */
    int evict(int need_blocks, BlockManager &bm);

    /**
     * test-only: evictable leaf count.
     */
    int lru_size() const;

    /**
     * evictable block count.
     */
    int evictable_blocks() const;

private:
    /**
     * Split child at block-aligned split_len: new_node takes key[:split_len]+block_table[:split_len/BS], child keeps the rest.
     * Both halves whole blocks. Only invoked on whole-block nodes (key >= block_size); residual leaves (< block_size) are never split.
     */
    RadixNode *_split_node(RadixNode *child, int split_len);

    /**
     * Remove a leaf: detach from parent, remove from LRUList if present, bm.free(block_table), delete.
     * No cascade.
     */
    void _delete_leaf(RadixNode *node, BlockManager &bm);

    RadixNode *root_;
    LRUList lru_;
    int block_size_;
    int evictable_blocks_;
};

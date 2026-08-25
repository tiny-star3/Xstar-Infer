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
    // fork key = block-aligned token segment
    // key 为子节点第一个块的 token
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
 * Drives BlockManager only via bm.fork (reuse) and bm.free (evict). block_size must == bm.block_size().
 * CONTRACT: insert() records an ALREADY-allocated block_table (caller alloc'd + prefilled); tree does not alloc.
 *           match_prefix returns a BLOCK-ALIGNED length; caller forks block_table[:matched/BS], re-prefills the rest.
 *           evict() frees via bm.free + cascades empty parents; returns < need_blocks if LRU empties -> caller falls back to Recompute.
 */
class RadixTree
{
public:
    explicit RadixTree(int block_size);
    ~RadixTree();

    /**
     * Walk tree matching tokens.
     * Returns (block-aligned matched_length, node where match ended).
     * Residual not reused.
     */
    std::pair<int, RadixNode *> match_prefix(const std::vector<int> &tokens);

    /**
     * Insert a token segment + its already-allocated block_table.
     * Splits existing node on partial overlap.
     * Returns the leaf.
     */
    RadixNode *insert(const std::vector<int> &tokens, const std::vector<int> &block_table);

    /** Pin node + all ancestors (walk UP parent chain). 0->1 removes from LRUList. */
    void inc_lock_ref(RadixNode *node);

    /** Unpin node + ancestors (reverse of inc). 1->0 leaf re-enters LRUList at tail; interior nodes don't. */
    void dec_lock_ref(RadixNode *node);

    /** Evict LRU leaves until >= need_blocks freed (bm.free + cascade empty parents). Returns blocks freed. */
    int evict(int need_blocks, BlockManager &bm);

    /** test-only: evictable leaf count. */
    int lru_size() const;

private:
    /** Split child at block-aligned split_len: new_node takes key[:split_len]+block_table[:split_len/BS], child keeps the rest. Both halves whole blocks. */
    RadixNode *_split_node(RadixNode *child, int split_len);

    /** Remove a leaf: detach from parent, remove from LRUList if present, bm.free(block_table), delete. No cascade. */
    void _delete_leaf(RadixNode *node, BlockManager &bm);

    RadixNode *root_;
    LRUList lru_;
    int block_size_;
};

import pytest
import sys

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 空树匹配
def test_match_empty():
    tree = xstar_cpp.RadixTree(4)
    blocks, _ = tree.match_prefix([1, 2, 3, 4])

    assert len(blocks) == 0


# 单节点 + 回读
def test_insert_then_match():
    block_size = 4
    tree = xstar_cpp.RadixTree(block_size)
    bm = xstar_cpp.BlockManager(10, block_size, 512, xstar_cpp.Device.CUDA)
    block_table = bm.alloc(1)
    tree.insert([1, 2, 3, 4], block_table, bm)
    # 前4命中, 后4残留不匹配
    blocks, node = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])

    # block_size=4, 余下 [5..8] 没这叉 → 停在 leaf
    assert blocks == block_table
    assert node.key == [1, 2, 3, 4]
    assert node.block_table == block_table


# 全命中不覆盖
def test_insert_full_match_no_overwrite():
    block_size = 4
    tree = xstar_cpp.RadixTree(block_size)
    bm = xstar_cpp.BlockManager(10, block_size, 512, xstar_cpp.Device.CUDA)
    block_table = bm.alloc(1)
    block_table2 = bm.alloc(1)
    tree.insert([1, 2, 3, 4], block_table, bm)
    node = tree.insert([1, 2, 3, 4], block_table2, bm)

    assert node.block_table == block_table
    assert tree.lru_size() == 1


# 残留不参与
def test_insert_truncation():
    block_size = 4
    tree = xstar_cpp.RadixTree(block_size)
    bm = xstar_cpp.BlockManager(10, block_size, 512, xstar_cpp.Device.CUDA)
    block_table = bm.alloc(2)
    node = tree.insert([1, 2, 3, 4, 5, 6], block_table, bm)

    assert node.key == [1, 2, 3, 4]
    assert tree.lru_size() == 1


# 共享前 4 分叉触发 split, 两条分支都能完整穿到 leaf
def test_split_aligned():
    block_size = 4
    tree = xstar_cpp.RadixTree(block_size)
    bm = xstar_cpp.BlockManager(10, block_size, 512, xstar_cpp.Device.CUDA)
    block_table = bm.alloc(2)
    block_table2 = bm.alloc(1)
    tree.insert([1, 2, 3, 4, 5, 6, 7, 8], block_table, bm)
    tree.insert([1, 2, 3, 4, 9, 9, 9, 9], [block_table[0], block_table2[0]], bm)
    blocks1, _ = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
    blocks2, _ = tree.match_prefix([1, 2, 3, 4, 9, 9, 9, 9])

    assert blocks1 == block_table and blocks2 == [block_table[0], block_table2[0]]
    assert tree.lru_size() == 2


# split 后旧 child 的 in_lru 状态
def test_split_leave_child_lru():
    block_size = 4
    tree = xstar_cpp.RadixTree(block_size)
    bm = xstar_cpp.BlockManager(10, block_size, 512, xstar_cpp.Device.CUDA)
    block_table = bm.alloc(2)
    block_table2 = bm.alloc(1)
    tree.insert([1, 2, 3, 4, 5, 6, 7, 8], block_table, bm)
    tree.insert([1, 2, 3, 4, 9, 9, 9, 9], [block_table[0], block_table2[0]], bm)
    _, node = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])

    assert tree.lru_size() == 2
    assert node.in_lru == True


# 0→1 从 LRU 移除
def test_inc_lock_ref_removes_from_lru():
    block_size = 4
    tree = xstar_cpp.RadixTree(block_size)
    bm = xstar_cpp.BlockManager(10, block_size, 512, xstar_cpp.Device.CUDA)
    block_table = bm.alloc(1)
    leaf = tree.insert([1, 2, 3, 4], block_table, bm)
    tree.inc_lock_ref(leaf)

    assert tree.lru_size() == 0 and leaf.lock_ref == 1 and leaf.in_lru == False


# 1→0 回 LRU
def test_dec_lock_ref_returns_to_lru():
    block_size = 4
    tree = xstar_cpp.RadixTree(block_size)
    bm = xstar_cpp.BlockManager(10, block_size, 512, xstar_cpp.Device.CUDA)
    block_table = bm.alloc(1)
    leaf = tree.insert([1, 2, 3, 4], block_table, bm)
    tree.inc_lock_ref(leaf)
    tree.dec_lock_ref(leaf)

    assert tree.lru_size() == 1 and leaf.lock_ref == 0 and leaf.in_lru == True


# inc child 会把 parent upper 也 pin
def test_inc_lock_ref_walks_up():
    block_size = 4
    tree = xstar_cpp.RadixTree(block_size)
    bm = xstar_cpp.BlockManager(10, block_size, 512, xstar_cpp.Device.CUDA)
    block_table = bm.alloc(2)
    block_table2 = bm.alloc(1)
    tree.insert([1, 2, 3, 4, 5, 6, 7, 8], block_table, bm)
    tree.insert([1, 2, 3, 4, 9, 9, 9, 9], [block_table[0], block_table2[0]], bm)
    _, upper = tree.match_prefix([1, 2, 3, 4])
    _, child = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
    tree.inc_lock_ref(child)

    assert upper.lock_ref == 1 and child.lock_ref == 1


# need_blocks=10 > 树里 1 块, 返回能踢的实际数
def test_evict_insufficient():
    block_size = 4

    bm = xstar_cpp.BlockManager(10, block_size, 512, xstar_cpp.Device.CUDA)
    tree = xstar_cpp.RadixTree(block_size)
    block_table = bm.alloc(1)
    tree.insert([1, 2, 3, 4], block_table, bm)
    freed = tree.evict(10, bm)

    assert freed == 1 and tree.lru_size() == 0


# evict 一次后空 parent 进 LRU 能被二次 evict
def test_evict_cascade_empty_parent():
    block_size = 4

    bm = xstar_cpp.BlockManager(10, block_size, 512, xstar_cpp.Device.CUDA)
    tree = xstar_cpp.RadixTree(block_size)
    block_table = bm.alloc(2)
    block_table2 = bm.alloc(1)
    tree.insert([1, 2, 3, 4, 5, 6, 7, 8], block_table, bm)
    tree.insert([1, 2, 3, 4, 9, 9, 9, 9], [block_table[0], block_table2[0]], bm)
    _, upper = tree.match_prefix([1, 2, 3, 4])
    _, child = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])

    assert upper.in_lru == False and tree.lru_size() == 2

    freed = tree.evict(2, bm)
    assert freed == 2

    assert upper.in_lru == True and tree.lru_size() == 1

    freed = tree.evict(1, bm)
    assert freed == 1

    assert tree.lru_size() == 0


# insert 时, 加入 leaf, parent 如果在 lru_list 移除
def test_insert_lru_invariant():
    bs = 16
    bm = xstar_cpp.BlockManager(8, bs, 256, xstar_cpp.Device.CUDA, 1)
    tree = xstar_cpp.RadixTree(bs)

    # 2 块的 seqA (32 tokens)
    seqA = list(range(32))
    blocksA = bm.alloc(2)
    tree.insert(seqA, blocksA, bm)
    # print(f"after insert A: lru={tree.lru_size()} evictable={tree.evictable_blocks()}")
    assert tree.lru_size() == 1

    # 延长序列 seqA+seqB (64 tokens, 4 块) -- 修复前: insert 停在 A 叶子下挂孩子, A 带孩子留在 LRU
    blocksAB = blocksA + bm.alloc(2)
    tree.insert(seqA + list(range(100, 132)), blocksAB, bm)
    # print(f"after insert AB: lru={tree.lru_size()} evictable={tree.evictable_blocks()}")
    assert tree.lru_size() == 1, f"LRU invariant broken: lru={tree.lru_size()}"

    # 逐块驱逐 -- 修复前这里 SIGSEGV (孤儿 parent 悬空)
    freed_total = 0
    while tree.evictable_blocks() > 0:
        freed_total += tree.evict(6, bm)
    # print(
    #     f"evicted all: freed={freed_total} lru={tree.lru_size()} evictable={tree.evictable_blocks()}"
    # )
    assert tree.lru_size() == 0 and tree.evictable_blocks() == 0

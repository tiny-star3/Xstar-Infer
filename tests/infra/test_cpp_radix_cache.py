import pytest
import sys

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 空树匹配
def test_match_empty():
    tree = xstar_cpp.RadixTree(4)
    matched, _ = tree.match_prefix([1, 2, 3, 4])

    assert matched == 0


# 单节点 + 回读
def test_insert_then_match():
    tree = xstar_cpp.RadixTree(4)
    tree.insert([1, 2, 3, 4], [10])
    # 前4命中, 后4残留不匹配
    matched, node = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])

    # block_size=4, 余下 [5..8] 没这叉 → 停在 leaf
    assert matched == 4
    assert node.key == [1, 2, 3, 4]
    assert node.block_table == [10]


# 全命中不覆盖
def test_insert_full_match_no_overwrite():
    tree = xstar_cpp.RadixTree(4)
    tree.insert([1, 2, 3, 4], [10])
    node = tree.insert([1, 2, 3, 4], [99])

    assert node.block_table == [10]
    assert tree.lru_size() == 1


# 残留不参与
def test_insert_truncation():
    tree = xstar_cpp.RadixTree(4)
    node = tree.insert([1, 2, 3, 4, 5, 6], [10, 20])

    assert node.key == [1, 2, 3, 4]
    assert tree.lru_size() == 1


# 共享前 4 分叉触发 split, 两条分支都能完整穿到 leaf
def test_split_aligned():
    tree = xstar_cpp.RadixTree(4)
    tree.insert([1, 2, 3, 4, 5, 6, 7, 8], [10, 20])
    tree.insert([1, 2, 3, 4, 9, 9, 9, 9], [10, 40])
    m1, _ = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
    m2, _ = tree.match_prefix([1, 2, 3, 4, 9, 9, 9, 9])

    assert m1 == 8 and m2 == 8
    assert tree.lru_size() == 2


# split 后旧 child 的 in_lru 状态
def test_split_leave_child_lru():
    tree = xstar_cpp.RadixTree(4)
    tree.insert([1, 2, 3, 4, 5, 6, 7, 8], [10, 20])
    tree.insert([1, 2, 3, 4, 9, 9, 9, 9], [10, 40])
    _, node = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])

    assert tree.lru_size() == 2
    assert node.in_lru == True


# 0→1 从 LRU 移除
def test_inc_lock_ref_removes_from_lru():
    tree = xstar_cpp.RadixTree(4)
    leaf = tree.insert([1, 2, 3, 4], [10])
    tree.inc_lock_ref(leaf)

    assert tree.lru_size() == 0 and leaf.lock_ref == 1 and leaf.in_lru == False


# 1→0 回 LRU
def test_dec_lock_ref_returns_to_lru():
    tree = xstar_cpp.RadixTree(4)
    leaf = tree.insert([1, 2, 3, 4], [10])
    tree.inc_lock_ref(leaf)
    tree.dec_lock_ref(leaf)

    assert tree.lru_size() == 1 and leaf.lock_ref == 0 and leaf.in_lru == True


# inc child 会把 parent upper 也 pin
def test_inc_lock_ref_walks_up():
    tree = xstar_cpp.RadixTree(4)
    tree.insert([1, 2, 3, 4, 5, 6, 7, 8], [10, 20])
    tree.insert([1, 2, 3, 4, 9, 9, 9, 9], [10, 40])
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
    tree.insert([1, 2, 3, 4], block_table)
    freed = tree.evict(10, bm)

    assert freed == 1 and tree.lru_size() == 0


# evict 一次后空 parent 进 LRU 能被二次 evict
def test_evict_cascade_empty_parent():
    block_size = 4

    bm = xstar_cpp.BlockManager(10, block_size, 512, xstar_cpp.Device.CUDA)
    tree = xstar_cpp.RadixTree(block_size)
    block_table = bm.alloc(2)
    block_table2 = bm.alloc(1)
    tree.insert([1, 2, 3, 4, 5, 6, 7, 8], block_table)
    tree.insert([1, 2, 3, 4, 9, 9, 9, 9], [block_table[0], block_table2[0]])
    _, upper = tree.match_prefix([1, 2, 3, 4])
    _, child = tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])

    assert upper.in_lru == False and tree.lru_size() == 2

    freed = tree.evict(2, bm)
    assert freed == 2

    assert upper.in_lru == True and tree.lru_size() == 1

    freed = tree.evict(1, bm)
    assert freed == 1

    assert tree.lru_size() == 0

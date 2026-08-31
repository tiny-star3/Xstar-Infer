import pytest
import sys

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 插树 → fork 复用 → free 请求份 → 驱逐，整圈 num_allocated 归零
def test_ownership_insert_fork_free_evict_returns_to_zero():
    block_size = 4
    bm = xstar_cpp.BlockManager(16, block_size, 512, xstar_cpp.Device.CUDA)
    tree = xstar_cpp.RadixTree(block_size)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    blocks = bm.alloc(2)  # ref_cnt 0→1 (请求 R1 持有)
    tree.insert(tokens, blocks, bm)  # 树 ref → 2 (R1 1 + 树 1)
    assert bm.block_ref_cnt(0) == 2
    assert bm.block_ref_cnt(1) == 2

    bm.free(blocks)  # R1 finish, free 自己那份 → 1 (树)
    assert bm.block_ref_cnt(0) == 1
    assert bm.block_ref_cnt(1) == 1
    assert bm.num_allocated() == 2  # 树还持有,物理块没回 free list

    forked = bm.fork(blocks)  # R2 命中 → 2 (树 1 + R2 1)
    assert bm.block_ref_cnt(0) == 2

    bm.free(forked)  # R2 finish → 1 (树)
    assert bm.block_ref_cnt(0) == 1
    assert bm.num_allocated() == 2

    freed = tree.evict(2, bm)  # 树还最后一份 → 0
    assert freed == 2
    assert bm.num_allocated() == 0  # 归零


# insert 的 full-match 分支不能再 ref 一次（树已持有），否则 ref_cnt 变 3 泄漏
def test_ownership_full_match_no_duplicate_ref():
    block_size = 4
    bm = xstar_cpp.BlockManager(16, block_size, 512, xstar_cpp.Device.CUDA)
    tree = xstar_cpp.RadixTree(block_size)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    t1 = bm.alloc(2)
    tree.insert(tokens, t1, bm)
    assert bm.block_ref_cnt(t1[0]) == 2  # 请求份 + 树份

    t2 = bm.alloc(2)
    node = tree.insert(tokens, t2, bm)  # full match,不 ref
    assert node.block_table == t1
    assert bm.block_ref_cnt(t1[0]) == 2  # 没多 ref,还是 2

    # R1(t1) finish free → 树 1; R2(t2) 是孤儿,free 归零
    bm.free(t1)
    assert bm.block_ref_cnt(t1[0]) == 1  # 树仍持有
    bm.free(t2)
    assert bm.num_allocated() == 2  # 只剩树那份 t1


# 树里有 2 个 leaf 时，evict 要全部驱掉才归零
# 卡 evict 的 cascade / 多 leaf 循环有没有漏 free 某一支
def test_ownership_two_leaves_evict_all_returns_to_zero():
    block_size = 4
    bm = xstar_cpp.BlockManager(16, block_size, 512, xstar_cpp.Device.CUDA)
    tree = xstar_cpp.RadixTree(block_size)
    tokens_a = [1, 2, 3, 4]
    tokens_b = [5, 6, 7, 8]  # 无共享前缀,两个独立 leaf

    a = bm.alloc(1)
    b = bm.alloc(1)
    tree.insert(tokens_a, a, bm)
    tree.insert(tokens_b, b, bm)
    bm.free(a)
    bm.free(b)
    assert bm.num_allocated() == 2

    tree.evict(1, bm)  # 只驱 1 块 → 只赶走 1 个 leaf
    assert bm.num_allocated() == 1  # 还剩 1 个 leaf 持有 1 块

    tree.evict(1, bm)  # 再驱 1 块
    assert bm.num_allocated() == 0  # 归零


# 命中匹配后 inc_lock_ref pin 住节点，此时 evict 驱不动它——这是 preempt 释放 pin 的反向保证（dec 后能驱，inc 后不能驱）
def test_ownership_pinned_node_not_evictable():
    block_size = 4
    bm = xstar_cpp.BlockManager(16, block_size, 512, xstar_cpp.Device.CUDA)
    tree = xstar_cpp.RadixTree(block_size)

    blocks = bm.alloc(1)
    node = tree.insert([1, 2, 3, 4], blocks, bm)
    bm.free(blocks)
    matched_blocks, matched_node = tree.match_prefix([1, 2, 3, 4])  # 命中
    tree.inc_lock_ref(matched_node)  # 模拟命中后 pin

    assert tree.evictable_blocks() == 0  # pinned 不计入可驱逐
    assert tree.evict(1, bm) == 0  # 驱不出
    assert bm.num_allocated() == 1  # 块还在

    tree.dec_lock_ref(matched_node)  # 释放 pin(Step5 语义)
    assert tree.evict(1, bm) == 1  # 现在能驱
    assert bm.num_allocated() == 0

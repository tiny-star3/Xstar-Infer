import pytest
import sys

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# LIFO 顺序 + 回收 + 计数不变量
def test_block_manager_alloc_free_lifo():
    num_blocks = 4
    block_size = 16
    kv_slot_bytes = 512
    dev = xstar_cpp.Device.CUDA

    manager = xstar_cpp.BlockManager(num_blocks, block_size, kv_slot_bytes, dev)
    blocks = manager.alloc(2)
    assert blocks == [0, 1]
    assert manager.num_free() == 2 and manager.num_allocated() == 2
    manager.free([0])
    blocks = manager.alloc(1)
    assert blocks == [0]
    assert manager.num_free() == 2 and manager.num_allocated() == 2

    # 边界, 空池
    num_blocks = 0
    block_size = 16
    kv_slot_bytes = 512
    dev = xstar_cpp.Device.CUDA

    manager = xstar_cpp.BlockManager(num_blocks, block_size, kv_slot_bytes, dev)
    blocks = manager.alloc(0)
    assert blocks == []
    with pytest.raises(RuntimeError, match="insufficient free blocks"):
        manager.alloc(1)


# pinned 语义(ref_cnt>0 不回收)
def test_block_manager_refcnt_pin():
    num_blocks = 4
    block_size = 16
    kv_slot_bytes = 512
    dev = xstar_cpp.Device.CUDA

    manager = xstar_cpp.BlockManager(num_blocks, block_size, kv_slot_bytes, dev)
    A = manager.alloc(1)[0]
    assert manager.block_ref_cnt(A) == 1
    manager.fork([A])
    assert manager.block_ref_cnt(A) == 2
    manager.free([A])
    assert manager.num_allocated() == 1 and manager.num_free() == 3
    manager.free([A])
    assert manager.num_allocated() == 0 and manager.num_free() == 4
    A = manager.alloc(1)[0]
    assert manager.num_allocated() == 1
    manager.free([A])
    with pytest.raises(RuntimeError, match="block not allocated"):
        manager.free([A])


# CoW 控制流(bit-exact 留 M8)
# M8 paged kernel 有 block 读写接口后, 补 fill_block(A,pattern) → write_block(A) → read_block(new_id)==pattern
# 现在这条 case 抓"CoW 触发 + 新块分配 + ref_cnt/计数对", 抓不到"漏 cow_copy"
def test_block_manager_cow_control_flow():
    num_blocks = 4
    block_size = 16
    kv_slot_bytes = 512
    dev = xstar_cpp.Device.CUDA

    manager = xstar_cpp.BlockManager(num_blocks, block_size, kv_slot_bytes, dev)
    A = manager.alloc(1)[0]
    assert manager.block_ref_cnt(A) == 1
    manager.fork([A])
    assert manager.block_ref_cnt(A) == 2
    before = manager.num_allocated()
    assert before == 1
    new_id = manager.write_block(A)
    assert new_id != A
    assert manager.block_ref_cnt(A) == 1
    assert manager.block_ref_cnt(new_id) == 1
    assert manager.num_allocated() == before + 1


# ref_cnt==1 不 CoW
def test_block_manager_cow_no_trigger():
    num_blocks = 4
    block_size = 16
    kv_slot_bytes = 512
    dev = xstar_cpp.Device.CUDA

    manager = xstar_cpp.BlockManager(num_blocks, block_size, kv_slot_bytes, dev)
    A = manager.alloc(1)[0]
    assert manager.block_ref_cnt(A) == 1
    before_free = manager.num_free()
    before_alloc = manager.num_allocated()
    new_id = manager.write_block(A)
    assert new_id == A
    assert manager.num_free() == before_free and manager.num_allocated() == before_alloc
    assert manager.block_ref_cnt(A) == 1

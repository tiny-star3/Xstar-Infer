import pytest
import sys
import numpy as np
import torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# --- fixtures: 共享准备 ---
@pytest.fixture
def weight_path(tmp_path):
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    p = tmp_path / "w.bin"
    arr.tofile(p)
    return p


@pytest.fixture
def weight_arr():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    return arr


# from_numpy(arr) → to_numpy 出来 == 原数组(往返一致)
def test_from_numpy_to_numpy_roundtrip(weight_arr):
    arr = weight_arr
    t = xstar_cpp.from_numpy(arr)
    out = xstar_cpp.to_numpy(t)
    assert np.array_equal(arr, out), f"arr={arr} out={out}"


# 改 to_numpy 返回的数组, 不影响原 Tensor(owned 内存独立,不别名)
def test_to_numpy_has_independent_memory(weight_arr):
    arr = weight_arr
    t = xstar_cpp.from_numpy(arr)
    out0 = xstar_cpp.to_numpy(t)
    out0.flat[:4] = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    out1 = xstar_cpp.to_numpy(t)
    assert not np.array_equal(out1, out0) and np.array_equal(
        arr, out1
    ), f"arr={arr} out0={out0} out1={out1}"


# 构造一个 bf16 Tensor,numel×2 == nbytes(验 bf16 的 dtype_size)
def test_bfloat16_nbytes():
    t = xstar_cpp.Tensor([4], xstar_cpp.DType.BFloat16)
    assert 8 == t.nbytes(), f"nbytes={t.nbytes()}"


# MMapFile.size() == 文件字节数
def test_mmap_size_matches_file(weight_path):
    path = weight_path
    mf = xstar_cpp.MMapFile(str(path))
    assert mf.size() == 96, f"size={mf.size()}"


# 视图的 shape/numel/nbytes 正确
def test_view_shape_numel_nbytes(weight_path):
    path = weight_path
    mf = xstar_cpp.MMapFile(str(path))
    t = xstar_cpp.make_weight_view(mf, 0, [2, 3, 4], xstar_cpp.DType.Float32)
    assert list(t.shape()) == [2, 3, 4], f"shape={t.shape()}"
    assert t.numel() == 24, f"numel={t.numel()}"
    assert t.nbytes() == 96, f"nbytes={t.nbytes()}"


# to_numpy(view) == 原数组(内容正确)
def test_view_content_matches_file(weight_path, weight_arr):
    path = weight_path
    arr = weight_arr
    mf = xstar_cpp.MMapFile(str(path))
    t = xstar_cpp.make_weight_view(mf, 0, [2, 3, 4], xstar_cpp.DType.Float32)
    out = xstar_cpp.to_numpy(t)
    assert np.array_equal(out, arr), f"view content={xstar_cpp.to_numpy(t)}"


# 零拷贝(别名法): 改底层文件后同一视图再读反映新内容(不等于旧读)
def test_view_is_zero_copy_aliased(weight_path, weight_arr):
    # 不碰裸指针, 原地改底层文件(同大小 r+b seek+write) → 同一个 t 再读应反映新内容
    path = weight_path
    arr = weight_arr
    mf = xstar_cpp.MMapFile(str(path))
    t = xstar_cpp.make_weight_view(mf, 0, [2, 3, 4], xstar_cpp.DType.Float32)
    out0 = xstar_cpp.to_numpy(t)
    arr.flat[:4] = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    with open(path, "r+b") as f:  # r+b 不截断, 原地覆写
        f.seek(0)
        f.write(arr.tobytes())  # 96 字节同大小
    out1 = xstar_cpp.to_numpy(t)  # 同一个 t 再读
    assert np.array_equal(out1, arr) and not np.array_equal(
        out1, out0
    ), f"out0={out0} arr={arr} out1={out1}"


# 越界 nbytes>文件 → 抛 RuntimeError
def test_out_of_bounds_raises(weight_path):
    path = weight_path
    mf = xstar_cpp.MMapFile(str(path))
    with pytest.raises(RuntimeError, match="weight view exceeds mmap region"):
        xstar_cpp.make_weight_view(mf, 0, [2, 3, 5], xstar_cpp.DType.Float32)


# 未对齐 offset → 抛 RuntimeError
def test_misaligned_offset_raises(weight_path):
    path = weight_path
    mf = xstar_cpp.MMapFile(str(path))
    with pytest.raises(RuntimeError, match="offset not aligned to dtype size"):
        xstar_cpp.make_weight_view(mf, 1, [2, 3, 4], xstar_cpp.DType.Float32)


# 非零 offset=48 视图内容正确
def test_nonzero_offset_view(weight_path, weight_arr):
    path = weight_path
    arr = weight_arr
    mf = xstar_cpp.MMapFile(str(path))
    # 跳前 48 字节(12 floats), 后 12 floats 当 [1,3,4]
    t = xstar_cpp.make_weight_view(mf, 48, [1, 3, 4], xstar_cpp.DType.Float32)
    tail = arr.flatten()[12:].reshape(1, 3, 4)
    out = xstar_cpp.to_numpy(t)
    assert np.array_equal(out, tail), f"out={out} tail={tail}"


# bf16 比特进出一致
def test_bfloat16_from_numpy_to_numpy_roundtrip():
    t_bf16 = torch.randn(8, 16, dtype=torch.bfloat16)
    bits = t_bf16.view(torch.uint16).numpy()
    ct = xstar_cpp.from_numpy_raw(bits, [8, 16], xstar_cpp.DType.BFloat16)
    out_bytes = xstar_cpp.to_numpy_raw(ct)
    out_bf16 = out_bytes.view(np.uint16).reshape(8, 16)
    assert np.array_equal(bits, out_bf16), f"out_bf16={out_bf16} bits={bits}"

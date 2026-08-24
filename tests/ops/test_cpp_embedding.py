import pytest
import sys
import numpy as np
import torch

from xstar.layers.embedding import Embedding
from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


@pytest.fixture
def weight_path(tmp_path):
    arr = np.arange(320, dtype=np.float32).reshape(10, 32)
    p = tmp_path / "w.bin"
    arr.tofile(p)
    return p


@pytest.fixture
def weight_arr():
    arr = np.arange(320, dtype=np.float32).reshape(10, 32)
    return arr


# rank1 乱序 id(抓 i vs ids[i] 的铁律 case)
# 源偏移用 i 不是 ids[i] 在顺序 id 下隐身(i==ids[i]), 只有乱序 id 才暴露
def test_embedding_rank1_f32_disordered_ids():
    ids = np.array([5, 3, 7, 0, 2])
    vocab = 10
    hidden = 32
    weight = torch.randn(vocab, hidden)

    ref_embedding = Embedding(vocab, hidden, device="cpu", dtype=torch.float32)
    ref_embedding.weight.data.copy_(weight)
    ref = ref_embedding(torch.from_numpy(ids))

    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.embedding(weight_t, ids)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# rank2 乱序 id(真实调度形状)
# 验"前导维展开 + 每个位置独立 gather"
def test_embedding_rank2_f32_disordered_ids():
    ids = np.array([[5, 3, 7, 0], [2, 8, 1, 4]])
    vocab = 10
    hidden = 32
    weight = torch.randn(vocab, hidden)

    ref_embedding = Embedding(vocab, hidden, device="cpu", dtype=torch.float32)
    ref_embedding.weight.data.copy_(weight)
    ref = ref_embedding(torch.from_numpy(ids))

    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.embedding(weight_t, ids)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# bf16 乱序 id
# bf16 权重表路径。注意 weight 进 C++ 走比特 view(uint16)
def test_embedding_bf16_disordered_ids():
    ids = np.array([[5, 3, 7, 0], [2, 8, 1, 4]])
    vocab = 10
    hidden = 32
    weight = torch.randn(vocab, hidden, dtype=torch.bfloat16)

    ref_embedding = Embedding(vocab, hidden, device="cpu", dtype=torch.bfloat16)
    ref_embedding.weight.data.copy_(weight)
    ref = ref_embedding(torch.from_numpy(ids))

    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.embedding(weight_t, ids)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# id=0 边界(vocab 第一行)
# vocab 第一行,边界端点
def test_embedding_id_zero_boundary():
    ids = np.array([0, 5, 3])
    vocab = 10
    hidden = 32
    weight = torch.randn(vocab, hidden)

    ref_embedding = Embedding(vocab, hidden, device="cpu", dtype=torch.float32)
    ref_embedding.weight.data.copy_(weight)
    ref = ref_embedding(torch.from_numpy(ids))

    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.embedding(weight_t, ids)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# id=vocab-1 边界(最后一行)
# vocab 最后一行, 另一端点。抓"off-by-one"(< 写成 <= 之类)
def test_embedding_id_last_boundary():
    ids = np.array([9, 5, 3])
    vocab = 10
    hidden = 32
    weight = torch.randn(vocab, hidden)

    ref_embedding = Embedding(vocab, hidden, device="cpu", dtype=torch.float32)
    ref_embedding.weight.data.copy_(weight)
    ref = ref_embedding(torch.from_numpy(ids))

    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.embedding(weight_t, ids)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# 越界 id 抛异常
# 验"先判再 gather"的越界检查
def test_embedding_out_of_range_raises():
    ids = np.array([10, -1, 3])
    vocab = 10
    hidden = 32
    weight = torch.randn(vocab, hidden)
    weight_t = torch_to_cpp(weight)

    with pytest.raises(RuntimeError, match="out-of-range index"):
        xstar_cpp.embedding(weight_t, ids)


# 零拷贝别名穿过 op(mmap 投资变现)
# 证明 embedding 读的是 mmap 活页(不是 from_numpy_raw 那种快照), 改文件 op 立刻反映
# weight tying 的地基:将来 lm_head 和 embed view 同一 offset 就 tied, 靠的就是这个零拷贝别名性质
def test_embedding_mmap_view_aliased(weight_path, weight_arr):
    vocab, hidden = weight_arr.shape
    ids = np.array([5, 3, 7, 0, 2])
    weight = torch.from_numpy(weight_arr)

    ref_embedding = Embedding(vocab, hidden, device="cpu", dtype=torch.float32)
    ref_embedding.weight.data.copy_(weight)
    ref0 = ref_embedding(torch.from_numpy(ids))

    mf = xstar_cpp.MMapFile(str(weight_path))
    weight_t = xstar_cpp.make_weight_view(
        mf, 0, [vocab, hidden], xstar_cpp.DType.Float32
    )
    y0_t = xstar_cpp.embedding(weight_t, ids)
    cpp0 = cpp_to_torch(y0_t, ref0.shape)

    weight_arr.flat[:4] = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    with open(weight_path, "r+b") as f:  # r+b 不截断, 原地覆写
        f.seek(0)
        f.write(weight_arr.tobytes())

    ref_embedding.weight.data.copy_(weight)
    ref1 = ref_embedding(torch.from_numpy(ids))

    y1_t = xstar_cpp.embedding(weight_t, ids)
    cpp1 = cpp_to_torch(y1_t, ref1.shape)

    assert (
        torch.equal(ref0, cpp0)
        and not torch.equal(cpp0, cpp1)
        and torch.equal(cpp1, ref1)
    ), f"cpp0={cpp0} ref0={ref0} cpp1={cpp1} ref1={ref1}"


# int32 id 安全宽化(probe 回归)
# binding 允许 int32 安全宽化、拒绝 float64 有损
def test_embedding_int32_ids_widened():
    ids = np.array([5, 3, 7, 0, 2], dtype=np.int32)
    vocab = 10
    hidden = 32
    weight = torch.randn(vocab, hidden)

    ref_embedding = Embedding(vocab, hidden, device="cpu", dtype=torch.float32)
    ref_embedding.weight.data.copy_(weight)
    ref = ref_embedding(torch.from_numpy(ids))

    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.embedding(weight_t, ids)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"

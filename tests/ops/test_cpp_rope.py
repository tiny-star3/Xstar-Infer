import pytest
import sys
import torch

from xstar.layers.rope import RoPE
from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# rank2: split-half 配对 + 基础索引正确性, positions=arange(最朴素的"能对上")
# f32 bit-exact(torch.equal), 验 split-half 配对 + 索引正确
def test_rope_f32_bit_exact_rank2():
    seq_len = 4
    theta = 10000.0
    dim = 8
    max_seq_len = 16

    x = torch.randn(seq_len, dim)
    positions = torch.arange(seq_len)

    ref_rope = RoPE(theta, dim, max_seq_len)
    ref = ref_rope(x, positions)

    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)
    x_t = torch_to_cpp(x)
    y_t = xstar_cpp.rope(x_t, cache_t, positions.numpy())
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# rank3: 前导维 heads 展开,每个头独立旋转,同一 position 跨头共享
# 多头形状 (heads, seq_len, head_dim), 验前导维展开 + 每头独立旋转
def test_rope_f32_bit_exact_rank3():
    seq_len = 4
    theta = 10000.0
    dim = 8
    max_seq_len = 16
    heads = 2

    x = torch.randn(heads, seq_len, dim)
    positions = torch.arange(seq_len)

    ref_rope = RoPE(theta, dim, max_seq_len)
    ref = ref_rope(x, positions)

    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)
    x_t = torch_to_cpp(x)
    y_t = xstar_cpp.rope(x_t, cache_t, positions.numpy())
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.equal(cpp, ref), "cpp={cpp} ref={ref}"


# bf16: bf16 路径 + cache 恒 f32、op 内部 downcast。关键: cache_t 仍是 DType.Float32, x_t 才是 bf16(走 view uint16 + from_numpy_raw)
# bf16 路径,cache 仍 f32 喂入(op 内部 downcast)
def test_rope_bf16_allclose():
    seq_len = 4
    theta = 10000.0
    dim = 8
    max_seq_len = 16

    x = torch.randn(seq_len, dim, dtype=torch.bfloat16)
    positions = torch.arange(seq_len)

    ref_rope = RoPE(theta, dim, max_seq_len)
    ref = ref_rope(x, positions)

    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)
    x_t = torch_to_cpp(x)
    y_t = xstar_cpp.rope(x_t, cache_t, positions.numpy())
    cpp = cpp_to_torch(y_t, ref.shape)

    # bf16 对拍原则: 用 rtol 主导 + 小 atol 兜底
    # bf16 尾数 7 位 → 1 ULP ≈ 2⁻⁷ ≈ 0.78% 相对误差, 所以 rtol=1e-2 ≈ 1.28 个 bf16 ULP, 容中间不 downcast 的末位分歧
    assert torch.allclose(cpp, ref, rtol=1e-2, atol=1e-2), f"cpp={cpp} ref={ref}"


# disordered: 铁律 case。positions=[3,0,7,1] 全错位(index 0→pos 3, index 1→pos 0,...), 抓 p = i vs p = positions[i]——arange 下 i==positions[i] 隐身, [3,0,7,1] 才暴露
# 乱序 positions 铁律(抓 i vs positions[i] 索引 bug)
def test_rope_f32_disordered_positions():
    seq_len = 4
    theta = 10000.0
    dim = 8
    max_seq_len = 16

    x = torch.randn(seq_len, dim)
    positions = torch.tensor([3, 0, 7, 1], dtype=torch.int64)

    ref_rope = RoPE(theta, dim, max_seq_len)
    ref = ref_rope(x, positions)

    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)
    x_t = torch_to_cpp(x)
    y_t = xstar_cpp.rope(x_t, cache_t, positions.numpy())
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# out_of_range: positions=[5,20,3],长度 seq_len=3 过 binding 长度校验, 但值 20 ≥ max_seq_len=16 → core 抛
# position >= max_seq_len 抛异常, 验越界检查
def test_rope_upper_bound_raises():
    seq_len = 3
    theta = 10000.0
    dim = 8
    max_seq_len = 16

    x = torch.randn(seq_len, dim)
    positions = torch.tensor([5, 20, 3], dtype=torch.int64)

    ref_rope = RoPE(theta, dim, max_seq_len)

    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)
    x_t = torch_to_cpp(x)
    with pytest.raises(RuntimeError, match="out-of-range position"):
        xstar_cpp.rope(x_t, cache_t, positions.numpy())


# out_of_range: positions=[5,8,-1],长度 seq_len=3 过 binding 长度校验, 但值 -1 < 0 → core 抛
# position < 0 抛异常, 验越界检查
def test_rope_lower_bound_raises():
    seq_len = 3
    theta = 10000.0
    dim = 8
    max_seq_len = 16

    x = torch.randn(seq_len, dim)
    positions = torch.tensor([5, 8, -1], dtype=torch.int64)

    ref_rope = RoPE(theta, dim, max_seq_len)

    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)
    x_t = torch_to_cpp(x)
    with pytest.raises(RuntimeError, match="out-of-range position"):
        xstar_cpp.rope(x_t, cache_t, positions.numpy())


# dim2: dim/2=1 退化, 索引 p*(dim/2)+j 退化成 p+0, 确认公式在退化情形不破。dim=2 时只有一个频率通道(freq=theta⁰=1, 角度=m), 仍可见旋转
# dim=2 边界(dim/2=1, 索引 p*(dim/2)+j 退化成 p+0)
def test_rope_dim2_boundary_f32():
    seq_len = 4
    theta = 10000.0
    dim = 2
    max_seq_len = 8

    x = torch.randn(seq_len, dim)
    positions = torch.arange(seq_len)

    ref_rope = RoPE(theta, dim, max_seq_len)
    ref = ref_rope(x, positions)

    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)
    x_t = torch_to_cpp(x)
    y_t = xstar_cpp.rope(x_t, cache_t, positions.numpy())
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"

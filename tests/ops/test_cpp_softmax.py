import pytest
import sys
import torch

from xstar.layers.attention import softmax
from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# dim=0(跨行 reduce,非 last-axis)
# 非 last-axis 的 row_start 正确性(stride≠1 的跨步 reduce)
def test_softmax_f32_dim0_nonlast():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dim = 0

    ref = softmax(x, dim)

    x_t = torch_to_cpp(x)
    cpp_t = xstar_cpp.softmax(x_t, dim)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"cpp={cpp} ref={ref}"


# 3-D dim=1(中间轴)
# row_start 在 dim 是中间轴时, row_idx 被 dim 轴之外的轴正确分配
def test_softmax_f32_dim1_middle_3d():
    x = torch.randn(2, 3, 4)
    dim = 1

    ref = softmax(x, dim)

    x_t = torch_to_cpp(x)
    cpp_t = xstar_cpp.softmax(x_t, dim)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"cpp={cpp} ref={ref}"


# dim=-1(负索引)
# 负索引归一化(dim += rank)
def test_softmax_f32_dim_neg1():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dim = -1

    ref = softmax(x, dim)

    x_t = torch_to_cpp(x)
    cpp_t = xstar_cpp.softmax(x_t, dim)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"cpp={cpp} ref={ref}"


# 全负大幅值
# max-init 不下溢 → 不产 NaN
def test_softmax_f32_all_negative_large():
    x = torch.tensor([[-1000.0, -1001.0, -1002.0]])
    dim = -1

    ref = softmax(x, dim)

    x_t = torch_to_cpp(x)
    cpp_t = xstar_cpp.softmax(x_t, dim)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    # allclose 对 NaN 行为是 False
    assert not torch.isnan(cpp).any()
    assert torch.allclose(cpp, ref, atol=1e-4), f"cpp={cpp} ref={ref}"


# bf16 中间轴
# bf16 路径(static_cast<float> 进、static_cast<bfloat16> 出) + 中间轴 row_start
def test_softmax_bf16_dim1_3d():
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    dim = 1

    ref = softmax(x, dim, torch.float32).to(torch.bfloat16)

    x_t = torch_to_cpp(x)
    cpp_t = xstar_cpp.softmax(x_t, dim)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, rtol=1e-2, atol=1e-2), f"cpp={cpp} ref={ref}"


# 越界 dim
# dim 校验
def test_softmax_dim_upper_bound_raises():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dim = 3

    x_t = torch_to_cpp(x)
    with pytest.raises(RuntimeError, match="dim out of range"):
        xstar_cpp.softmax(x_t, dim)


def test_softmax_dim_lower_bound_raises():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dim = -3

    x_t = torch_to_cpp(x)
    with pytest.raises(RuntimeError, match="dim out of range"):
        xstar_cpp.softmax(x_t, dim)

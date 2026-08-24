import pytest
import sys
import torch

from xstar.layers.linear import Linear
from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# rank-3 盯 k=shape().back()(rank-2 隐身) + 转置索引(out≠in) + bias 加法 + 前导维 collapse
def test_linear_f32_rank3_with_bias():
    in_features = 5
    out_featrues = 4
    x = torch.randn(2, 3, in_features)
    weight = torch.randn(out_featrues, in_features)
    bias = torch.randn(out_featrues)

    ref_linear = Linear(in_features, out_featrues, True)
    ref_linear.weight.data.copy_(weight)
    ref_linear.bias.data.copy_(bias)
    ref = ref_linear(x)

    x_t = torch_to_cpp(x)
    weight_t = torch_to_cpp(weight)
    bias_t = torch_to_cpp(bias)
    y_t = xstar_cpp.linear(x_t, weight_t, bias_t)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"cpp={cpp} ref={ref}"


# nullptr 路径(Qwen2 gate/up/down 没 bias)
def test_linear_f32_no_bias():
    in_features = 5
    out_featrues = 4
    x = torch.randn(2, in_features)
    weight = torch.randn(out_featrues, in_features)

    ref_linear = Linear(in_features, out_featrues, False)
    ref_linear.weight.data.copy_(weight)
    ref = ref_linear(x)

    x_t = torch_to_cpp(x)
    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.linear(x_t, weight_t, None)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"cpp={cpp} ref={ref}"


# bf16 + bias 在 f32 加再 downcast
def test_linear_bf16_rank3_with_bias():
    in_features = 5
    out_featrues = 4
    x = torch.randn(2, 3, in_features, dtype=torch.bfloat16)
    weight = torch.randn(out_featrues, in_features, dtype=torch.bfloat16)
    bias = torch.randn(out_featrues, dtype=torch.bfloat16)

    ref_linear = Linear(in_features, out_featrues, True, dtype=torch.bfloat16)
    ref_linear.weight.data.copy_(weight)
    ref_linear.bias.data.copy_(bias)
    ref = ref_linear(x)

    x_t = torch_to_cpp(x)
    weight_t = torch_to_cpp(weight)
    bias_t = torch_to_cpp(bias)
    y_t = xstar_cpp.linear(x_t, weight_t, bias_t)
    cpp = cpp_to_torch(y_t, ref.shape)

    # 累加链越长,1-ULP 级别的偏差越容易累积到肉眼可见
    # bf16 对拍原则: 用 rtol 主导 + 小 atol 兜底
    # bf16 尾数 7 位 → 1 ULP ≈ 2⁻⁷ ≈ 0.78% 相对误差, 所以 rtol=1e-2 ≈ 1.28 个 bf16 ULP, 容中间不 downcast 的末位分歧
    assert torch.allclose(cpp, ref, rtol=1e-2, atol=1e-2), f"cpp={cpp} ref={ref}"


# 验 bias.shape[0]==out
def test_linear_f32_bias_shape_mismatch_raises():
    in_features = 5
    out_featrues = 4
    x = torch.randn(2, in_features)
    weight = torch.randn(out_featrues, in_features)
    bias = torch.randn(3)

    x_t = torch_to_cpp(x)
    weight_t = torch_to_cpp(weight)
    bias_t = torch_to_cpp(bias)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        xstar_cpp.linear(x_t, weight_t, bias_t)

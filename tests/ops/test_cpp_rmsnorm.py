import pytest
import sys
import torch

from xstar.layers.rmsnorm import RMSNorm
from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# f32 普通,rank 2(最常见调度器入参形状)
# 基线。"普通 f32 输入算对"
def test_rmsnorm_rank2_f32():
    shape = (4, 32)
    hidden = 32
    eps = 1e-5
    x = torch.randn(*shape)
    weight = torch.randn(hidden)

    ref_rmsnorm = RMSNorm(hidden, eps, device="cpu", dtype=torch.float32)
    ref_rmsnorm.weight.data.copy_(weight)
    ref = ref_rmsnorm(x)

    x_t = torch_to_cpp(x)
    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.rmsnorm(x_t, weight_t, eps)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.allclose(cpp, ref, rtol=0, atol=1e-6), f"cpp={cpp} ref={ref}"


# bf16 普通,rank 2
# bf16 路径的 downcast RNE 落点对不对
def test_rmsnorm_rank2_bf16():
    shape = (4, 32)
    hidden = 32
    eps = 1e-5
    x = torch.randn(*shape, dtype=torch.bfloat16)
    weight = torch.randn(hidden, dtype=torch.bfloat16)

    ref_rmsnorm = RMSNorm(hidden, eps, device="cpu", dtype=torch.bfloat16)
    ref_rmsnorm.weight.data.copy_(weight)
    ref = ref_rmsnorm(x)

    x_t = torch_to_cpp(x)
    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.rmsnorm(x_t, weight_t, eps)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.allclose(cpp, ref, rtol=0, atol=1e-2), f"cpp={cpp} ref={ref}"


# rank 1(num_rows==1 边界)
# loop 从多行退化到一行,numel/hidden 在 num_rows=1 时算对
def test_rmsnorm_rank1_f32():
    shape = (16,)
    hidden = 16
    eps = 1e-5
    x = torch.randn(*shape)
    weight = torch.randn(hidden)

    ref_rmsnorm = RMSNorm(hidden, eps, device="cpu", dtype=torch.float32)
    ref_rmsnorm.weight.data.copy_(weight)
    ref = ref_rmsnorm(x)

    x_t = torch_to_cpp(x)
    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.rmsnorm(x_t, weight_t, eps)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.allclose(cpp, ref, rtol=0, atol=1e-6), f"cpp={cpp} ref={ref}"


# 全零行(数值边界)
# eps 位置写错(忘加/加错位置)时全零行爆 Inf; 这是 rmsnorm 最经典的数值 bug
def test_rmsnorm_x_all_zero_f32():
    shape = (4, 32)
    hidden = 32
    eps = 1e-5
    x = torch.zeros(*shape)
    weight = torch.randn(hidden)

    ref_rmsnorm = RMSNorm(hidden, eps, device="cpu", dtype=torch.float32)
    ref_rmsnorm.weight.data.copy_(weight)
    ref = ref_rmsnorm(x)

    x_t = torch_to_cpp(x)
    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.rmsnorm(x_t, weight_t, eps)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert not torch.isnan(cpp).any() and not torch.isinf(cpp).any()
    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# hidden=1(退化边界)
# numel/hidden、单元素累加 rms += xij*xij、单元素访问 x[i*hidden+j] 同时受压。循环边界写错(如 hidden-1)时普通 hidden 看不出, hidden=1 直接暴露
def test_rmsnorm_hidden_1_f32():
    shape = (4, 1)
    hidden = 1
    eps = 1e-5
    x = torch.randn(*shape)
    weight = torch.randn(hidden)

    ref_rmsnorm = RMSNorm(hidden, eps, device="cpu", dtype=torch.float32)
    ref_rmsnorm.weight.data.copy_(weight)
    ref = ref_rmsnorm(x)

    x_t = torch_to_cpp(x)
    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.rmsnorm(x_t, weight_t, eps)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.allclose(cpp, ref, rtol=0, atol=1e-6), f"cpp={cpp} ref={ref}"


# f32 bit-exact(求和顺序实验)
# 回答上一轮那个"朴素累加顺序 vs PyTorch reduction 顺序"问题。equal 过→顺序吻合; 不过但 max_diff 极小→测到顺序差, 是已知可解释。这个 case 的价值在你知道, 不在通过
# 朴素累加和 PyTorch CPU reduction 在 hidden=1024 下差约 3e-6
def test_rmsnorm_bit_exact_f32():
    shape = (4, 1024)
    hidden = 1024
    eps = 1e-5
    x = torch.randn(*shape)
    weight = torch.randn(hidden)

    ref_rmsnorm = RMSNorm(hidden, eps, device="cpu", dtype=torch.float32)
    ref_rmsnorm.weight.data.copy_(weight)
    ref = ref_rmsnorm(x)

    x_t = torch_to_cpp(x)
    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.rmsnorm(x_t, weight_t, eps)
    cpp = cpp_to_torch(y_t, ref.shape)

    max_diff = (cpp - ref).abs().max()
    print(f"max_diff = {max_diff}")
    assert max_diff < 1e-4


# weight 全1 解耦定位(定位工具)
# 把"归一化"和"scale"解耦。weight 全1 时输出=纯归一化结果。这个 case 平时是绿的, 它的价值在别的 case 红了之后——若 Case 1(随机 weight)红、Case 7(全1)绿 → bug 在 *weight 那步; 若 Case 7 也红 → 归一化本身就有问题。留它在套件里, 等于预置了一个故障分流器
def test_rmsnorm_weight_all_one_f32():
    shape = (4, 32)
    hidden = 32
    eps = 1e-5
    x = torch.randn(*shape)
    weight = torch.ones(hidden)

    ref_rmsnorm = RMSNorm(hidden, eps, device="cpu", dtype=torch.float32)
    ref_rmsnorm.weight.data.copy_(weight)
    ref = ref_rmsnorm(x)

    x_t = torch_to_cpp(x)
    weight_t = torch_to_cpp(weight)
    y_t = xstar_cpp.rmsnorm(x_t, weight_t, eps)
    cpp = cpp_to_torch(y_t, ref.shape)

    assert torch.allclose(cpp, ref, rtol=0, atol=1e-6), f"cpp={cpp} ref={ref}"

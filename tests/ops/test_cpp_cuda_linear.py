import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 基本正确性: BiasAdd 特化, transB=true, GEMM+epilogue 串起来
def test_cuda_linear_f32_with_bias():
    x = torch.randn(2, 3, 8)
    w = torch.randn(4, 8)
    bias = torch.randn(4)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)
    bias_cpu = torch_to_cpp(bias)
    cpp_cpu = xstar_cpp.linear(x_cpu, w_cpu, bias_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    w_cuda = xstar_cpp.to_cuda(w_cpu)
    bias_cuda = xstar_cpp.to_cuda(bias_cpu)
    cpp_cuda = xstar_cpp.linear(x_cuda, w_cuda, bias_cuda)

    expected = cpp_to_torch(cpp_cpu, [2, 3, 4])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), [2, 3, 4])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# Identity 特化(HAS_BIAS=false): if constexpr 剪枝, bias 代码不执行
def test_cuda_linear_f32_no_bias():
    x = torch.randn(2, 3, 8)
    w = torch.randn(4, 8)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)
    cpp_cpu = xstar_cpp.linear(x_cpu, w_cpu, None)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    w_cuda = xstar_cpp.to_cuda(w_cpu)
    cpp_cuda = xstar_cpp.linear(x_cuda, w_cuda, None)

    expected = cpp_to_torch(cpp_cpu, [2, 3, 4])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), [2, 3, 4])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# bf16 BiasAdd + toFloat/RNE 路径; borderline(CPU 两次 downcast vs GPU 一次, ~0.0156, 靠 rtol)
def test_cuda_linear_bf16_with_bias():
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    w = torch.randn(4, 8, dtype=torch.bfloat16)
    bias = torch.randn(4, dtype=torch.bfloat16)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)
    bias_cpu = torch_to_cpp(bias)
    cpp_cpu = xstar_cpp.linear(x_cpu, w_cpu, bias_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    w_cuda = xstar_cpp.to_cuda(w_cpu)
    bias_cuda = xstar_cpp.to_cuda(bias_cpu)
    cpp_cuda = xstar_cpp.linear(x_cuda, w_cuda, bias_cuda)

    expected = cpp_to_torch(cpp_cpu, [2, 3, 4])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), [2, 3, 4])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        rtol=1e-2,
        atol=1e-2,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# bf16 Identity: 对照 3, 隔离"bias 那步"是否引入额外误差
def test_cuda_linear_bf16_no_bias():
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    w = torch.randn(4, 8, dtype=torch.bfloat16)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)
    cpp_cpu = xstar_cpp.linear(x_cpu, w_cpu, None)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    w_cuda = xstar_cpp.to_cuda(w_cpu)
    cpp_cuda = xstar_cpp.linear(x_cuda, w_cuda, None)

    expected = cpp_to_torch(cpp_cpu, [2, 3, 4])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), [2, 3, 4])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        rtol=1e-2,
        atol=1e-2,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# rank2(x 拍平成 m=16): m 维多 M-tile(blockIdx.y>1), 压 M 方向 boundary
def test_cuda_linear_f32_rank2():
    x = torch.randn(16, 8)
    w = torch.randn(4, 8)
    bias = torch.randn(4)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)
    bias_cpu = torch_to_cpp(bias)
    cpp_cpu = xstar_cpp.linear(x_cpu, w_cpu, bias_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    w_cuda = xstar_cpp.to_cuda(w_cpu)
    bias_cuda = xstar_cpp.to_cuda(bias_cpu)
    cpp_cuda = xstar_cpp.linear(x_cuda, w_cuda, bias_cuda)

    expected = cpp_to_torch(cpp_cpu, [16, 4])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), [16, 4])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# n=40>BN=32 → 两个 N-tile(col=0 和 col=32); 压 bias 全局索引 bias[col+t] 的 col≠0 分支 + N 方向 boundary tile(n 不整除 BN)
def test_cuda_linear_f32_multitile_n():
    x = torch.randn(4, 8)
    w = torch.randn(40, 8)
    bias = torch.randn(40)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)
    bias_cpu = torch_to_cpp(bias)
    cpp_cpu = xstar_cpp.linear(x_cpu, w_cpu, bias_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    w_cuda = xstar_cpp.to_cuda(w_cpu)
    bias_cuda = xstar_cpp.to_cuda(bias_cpu)
    cpp_cuda = xstar_cpp.linear(x_cuda, w_cuda, bias_cuda)

    expected = cpp_to_torch(cpp_cpu, [4, 40])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), [4, 40])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"

import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 基础正确性
def test_softmax_f32_last_axis():
    dim = -1
    shape = (8, 896)

    x = torch.randn(shape)
    x_cpu = torch_to_cpp(x)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    cpp_cpu = xstar_cpp.softmax(x_cpu, dim)
    cpp_cuda = xstar_cpp.softmax(x_cuda, dim)

    expected = cpp_to_torch(cpp_cpu, x.shape)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), x.shape)

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-6,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 3D 折叠 + inner_size>1 索引
def test_softmax_f32_dim0():
    dim = 0
    shape = (896, 16)

    x = torch.randn(shape)
    x_cpu = torch_to_cpp(x)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    cpp_cpu = xstar_cpp.softmax(x_cpu, dim)
    cpp_cuda = xstar_cpp.softmax(x_cuda, dim)

    expected = cpp_to_torch(cpp_cpu, x.shape)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), x.shape)

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-6,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 中间轴 + outer>1 + inner>1
def test_softmax_f32_dim1_rank3():
    dim = 1
    shape = (4, 32, 16)

    x = torch.randn(shape)
    x_cpu = torch_to_cpp(x)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    cpp_cpu = xstar_cpp.softmax(x_cpu, dim)
    cpp_cuda = xstar_cpp.softmax(x_cuda, dim)

    expected = cpp_to_torch(cpp_cpu, x.shape)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), x.shape)

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-6,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 值全负情况, max 处理
def test_softmax_f32_all_negative():
    dim = -1
    shape = (8, 896)

    x = -torch.randn(shape).abs() - 0.1
    x_cpu = torch_to_cpp(x)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    cpp_cpu = xstar_cpp.softmax(x_cpu, dim)
    cpp_cuda = xstar_cpp.softmax(x_cuda, dim)

    expected = cpp_to_torch(cpp_cpu, x.shape)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), x.shape)

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-6,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# grid-stride × online rescale 叠加
def test_softmax_f32_large_dim_grid_stride():
    dim = -1
    shape = (8, 4096)

    x = torch.randn(shape)
    x_cpu = torch_to_cpp(x)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    cpp_cpu = xstar_cpp.softmax(x_cpu, dim)
    cpp_cuda = xstar_cpp.softmax(x_cuda, dim)

    expected = cpp_to_torch(cpp_cpu, x.shape)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), x.shape)

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-6,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# bf16 上转/下转 + (T) 构造
def test_softmax_bf16_last_axis():
    dim = -1
    shape = (8, 896)

    x = torch.randn(shape, dtype=torch.bfloat16)
    x_cpu = torch_to_cpp(x)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    cpp_cpu = xstar_cpp.softmax(x_cpu, dim)
    cpp_cuda = xstar_cpp.softmax(x_cuda, dim)

    expected = cpp_to_torch(cpp_cpu, x.shape)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), x.shape)

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        rtol=1e-2,
        atol=1e-2,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# GPU 内存不泄漏(同 M2 no_leak, 循环内 del 强制每轮净零)
def test_softmax_no_leak():
    dim = -1
    shape = (8, 896)

    x = torch.randn(shape)
    x_cpu = torch_to_cpp(x)

    free0 = xstar_cpp.cuda_free_bytes()

    for _ in range(100):
        x_cuda = xstar_cpp.to_cuda(x_cpu)
        cpp_cuda = xstar_cpp.softmax(x_cuda, dim)

        del x_cuda, cpp_cuda

    free1 = xstar_cpp.cuda_free_bytes()

    assert free1 >= free0 - 1024 * 1024

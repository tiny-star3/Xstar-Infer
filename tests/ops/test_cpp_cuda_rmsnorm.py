import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# f32 基本路径, rank-1(num_rows=1,单 block)
def test_rmsnorm_f32_rank1():
    hidden = 896
    eps = 1e-06

    x = torch.randn(hidden)
    w = torch.randn(hidden)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    w_cuda = xstar_cpp.to_cuda(w_cpu)
    cpp_cpu = xstar_cpp.rmsnorm(x_cpu, w_cpu, eps)
    cpp_gpu = xstar_cpp.rmsnorm(x_cuda, w_cuda, eps)
    expected = cpp_to_torch(cpp_cpu, x.shape)
    gpu = cpp_to_torch(xstar_cpp.to_cpu(cpp_gpu), x.shape)

    diff = (gpu - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        gpu,
        expected,
        atol=1e-6,
    ), f"cpp_gpu={gpu} cpp_cpu={expected}"


# 多行(blockIdx.x 当 row, grid>1)
def test_rmsnorm_f32_rank2():
    batch = 8
    hidden = 896
    eps = 1e-06

    x = torch.randn(batch, hidden)
    w = torch.randn(hidden)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    w_cuda = xstar_cpp.to_cuda(w_cpu)
    cpp_cpu = xstar_cpp.rmsnorm(x_cpu, w_cpu, eps)
    cpp_gpu = xstar_cpp.rmsnorm(x_cuda, w_cuda, eps)
    expected = cpp_to_torch(cpp_cpu, x.shape)
    gpu = cpp_to_torch(xstar_cpp.to_cpu(cpp_gpu), x.shape)

    diff = (gpu - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        gpu,
        expected,
        atol=1e-6,
    ), f"cpp_gpu={gpu} cpp_cpu={expected}"


# rank-3(leading dims 扁平成 num_rows=6)
def test_rmsnorm_f32_rank3():
    batch = 8
    hidden = 896
    eps = 1e-06

    x = torch.randn(2, batch, hidden)
    w = torch.randn(hidden)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    w_cuda = xstar_cpp.to_cuda(w_cpu)
    cpp_cpu = xstar_cpp.rmsnorm(x_cpu, w_cpu, eps)
    cpp_gpu = xstar_cpp.rmsnorm(x_cuda, w_cuda, eps)
    expected = cpp_to_torch(cpp_cpu, x.shape)
    gpu = cpp_to_torch(xstar_cpp.to_cpu(cpp_gpu), x.shape)

    diff = (gpu - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        gpu,
        expected,
        atol=1e-6,
    ), f"cpp_gpu={gpu} cpp_cpu={expected}"


# bf16 upcast/downcast 路径
def test_rmsnorm_bf16_rank2():
    batch = 8
    hidden = 896
    eps = 1e-06

    x = torch.randn(batch, hidden, dtype=torch.bfloat16)
    w = torch.randn(hidden, dtype=torch.bfloat16)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    w_cuda = xstar_cpp.to_cuda(w_cpu)
    cpp_cpu = xstar_cpp.rmsnorm(x_cpu, w_cpu, eps)
    cpp_gpu = xstar_cpp.rmsnorm(x_cuda, w_cuda, eps)
    expected = cpp_to_torch(cpp_cpu, x.shape)
    gpu = cpp_to_torch(xstar_cpp.to_cpu(cpp_gpu), x.shape)

    diff = (gpu - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        gpu,
        expected,
        rtol=1e-2,
        atol=1e-2,
    ), f"cpp_gpu={gpu} cpp_cpu={expected}"


# device 检查
def test_rmsnorm_device_mismatch():
    hidden = 896
    eps = 1e-06

    x = torch.randn(hidden)
    w = torch.randn(hidden)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)
    x_cuda = xstar_cpp.to_cuda(x_cpu)

    with pytest.raises(RuntimeError, match="device mismatch"):
        xstar_cpp.rmsnorm(x_cuda, w_cpu, eps)


# GPU 内存不泄漏(同 M2 no_leak, 循环内 del 强制每轮净零)
def test_rmsnorm_no_leak():
    hidden = 896
    eps = 1e-06

    x = torch.randn(hidden)
    w = torch.randn(hidden)

    x_cpu = torch_to_cpp(x)
    w_cpu = torch_to_cpp(w)

    free0 = xstar_cpp.cuda_free_bytes()

    for _ in range(100):
        x_cuda = xstar_cpp.to_cuda(x_cpu)
        w_cuda = xstar_cpp.to_cuda(w_cpu)

        cpp_gpu = xstar_cpp.rmsnorm(x_cuda, w_cuda, eps)

        del x_cuda, w_cuda, cpp_gpu

    free1 = xstar_cpp.cuda_free_bytes()

    assert free1 >= free0 - 1024 * 1024

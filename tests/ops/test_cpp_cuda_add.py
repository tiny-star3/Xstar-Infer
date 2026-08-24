import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# f32 bit-exact(IEEE 单加法)
def test_cuda_add_f32():
    seq = 5
    hidden = 64

    a = torch.randn(seq, hidden)
    b = torch.randn(seq, hidden)
    expected = a + b
    a_cpu = torch_to_cpp(a)
    b_cpu = torch_to_cpp(b)
    a_cuda = xstar_cpp.to_cuda(a_cpu)
    b_cuda = xstar_cpp.to_cuda(b_cpu)
    cpp_cpu = xstar_cpp.add(a_cpu, b_cpu)
    cpp_cuda = xstar_cpp.add(a_cuda, b_cuda)

    cpu = cpp_to_torch(cpp_cpu, expected.shape)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), expected.shape)

    assert torch.equal(cpu, expected), f"cpu={cpu} expected={expected}"
    assert torch.equal(cuda, expected), f"cuda={cuda} expected={expected}"


# bf16 bit-exact + upcast/downcast 路径对
def test_cuda_add_bf16():
    seq = 5
    hidden = 64

    a = torch.randn(seq, hidden, dtype=torch.bfloat16)
    b = torch.randn(seq, hidden, dtype=torch.bfloat16)
    expected = (a.float() + b.float()).to(torch.bfloat16)
    a_cpu = torch_to_cpp(a)
    b_cpu = torch_to_cpp(b)
    a_cuda = xstar_cpp.to_cuda(a_cpu)
    b_cuda = xstar_cpp.to_cuda(b_cpu)
    cpp_cpu = xstar_cpp.add(a_cpu, b_cpu)
    cpp_cuda = xstar_cpp.add(a_cuda, b_cuda)

    cpu = cpp_to_torch(cpp_cpu, expected.shape)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), expected.shape)

    assert torch.equal(cpu, expected), f"cpu={cpu} expected={expected}"
    assert torch.equal(cuda, expected), f"cuda={cuda} expected={expected}"


# grid-stride stride 算术 + 多 pass 覆盖
def test_cuda_add_grid_stride():
    seq = 512
    hidden = 896

    a = torch.randn(seq, hidden)
    b = torch.randn(seq, hidden)
    expected = a + b
    a_cpu = torch_to_cpp(a)
    b_cpu = torch_to_cpp(b)
    a_cuda = xstar_cpp.to_cuda(a_cpu)
    b_cuda = xstar_cpp.to_cuda(b_cpu)
    cpp_cpu = xstar_cpp.add(a_cpu, b_cpu)
    cpp_cuda = xstar_cpp.add(a_cuda, b_cuda)

    cpu = cpp_to_torch(cpp_cpu, expected.shape)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), expected.shape)

    assert torch.equal(cpu, expected), f"cpu={cpu} expected={expected}"
    assert torch.equal(cuda, expected), f"cuda={cuda} expected={expected}"


# 三条 precondition
def test_cuda_add_throws():
    seq = 5
    hidden = 64

    a = torch.randn(seq, hidden)
    b = torch.randn(seq, hidden - 1)
    a_cpu = torch_to_cpp(a)
    b_cpu = torch_to_cpp(b)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        xstar_cpp.add(a_cpu, b_cpu)

    a = torch.randn(seq, hidden, dtype=torch.float)
    b = torch.randn(seq, hidden, dtype=torch.bfloat16)
    a_cpu = torch_to_cpp(a)
    b_cpu = torch_to_cpp(b)
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        xstar_cpp.add(a_cpu, b_cpu)

    a = torch.randn(seq, hidden)
    b = torch.randn(seq, hidden)
    a_cpu = torch_to_cpp(a)
    a_cuda = xstar_cpp.to_cuda(a_cpu)
    b_cpu = torch_to_cpp(b)
    with pytest.raises(RuntimeError, match="device mismatch"):
        xstar_cpp.add(a_cuda, b_cpu)

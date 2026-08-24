import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# Q 路 transpose 正确 + 三方 bit-exact
def test_cuda_head_split_f32_q():
    heads = 14
    seq = 5
    head_dim = 64

    x = torch.arange(heads * seq * head_dim, dtype=torch.float32).reshape(
        seq, heads * head_dim
    )

    expected = x.view(seq, heads, head_dim).transpose(0, 1).contiguous()

    x_cpu = torch_to_cpp(x)
    cpp_cpu = xstar_cpp.head_split(x_cpu, heads)
    cpu = cpp_to_torch(cpp_cpu, expected.shape)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    cpp_cuda = xstar_cpp.head_split(x_cuda, heads)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), expected.shape)

    assert torch.equal(cpu, expected), f"cpu={cpu} expected={expected}"
    assert torch.equal(cuda, expected), f"cuda={cuda} expected={expected}"


# bf16 bit-exact(byte copy,无 downcast)
def test_cuda_head_split_bf16_q():
    heads = 14
    seq = 5
    head_dim = 64

    x = torch.arange(heads * seq * head_dim, dtype=torch.bfloat16).reshape(
        seq, heads * head_dim
    )

    expected = x.view(seq, heads, head_dim).transpose(0, 1).contiguous()

    x_cpu = torch_to_cpp(x)
    cpp_cpu = xstar_cpp.head_split(x_cpu, heads)
    cpu = cpp_to_torch(cpp_cpu, expected.shape)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    cpp_cuda = xstar_cpp.head_split(x_cuda, heads)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), expected.shape)

    assert torch.equal(cpu, expected), f"cpu={cpu} expected={expected}"
    assert torch.equal(cuda, expected), f"cuda={cuda} expected={expected}"


# K/V 路(heads 少、seq 多)
def test_cuda_head_split_kv_gqa():
    heads = 2
    seq = 20
    head_dim = 64

    x = torch.arange(heads * seq * head_dim, dtype=torch.float32).reshape(
        seq, heads * head_dim
    )

    expected = x.view(seq, heads, head_dim).transpose(0, 1).contiguous()

    x_cpu = torch_to_cpp(x)
    cpp_cpu = xstar_cpp.head_split(x_cpu, heads)
    cpu = cpp_to_torch(cpp_cpu, expected.shape)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    cpp_cuda = xstar_cpp.head_split(x_cuda, heads)
    cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), expected.shape)

    assert torch.equal(cpu, expected), f"cpu={cpu} expected={expected}"
    assert torch.equal(cuda, expected), f"cuda={cuda} expected={expected}"


# k guard + blockDim loop 在各种 head_dim 下对
# 72: blockDim.x=64 时 t=0 覆盖 0-63、t=1 覆盖 64-71(8 活 56 idle)→ 测 k<head_dim guard 尾部
# 1: 只有 k=0 活,其余 idle → 极端 guard(像 rmsnorm 测 dim_size 到 1)
# 2: k=0, 1 活 → 最小非平凡
# 512: blockDim.x=256 → 2 pass 循环 → 测 loop 正确(不是只跑一遍)
def test_cuda_head_split_boundary_head_dim():
    heads = 4
    seq = 3
    head_dims = [72, 1, 2, 512]

    for head_dim in head_dims:
        x = torch.arange(heads * seq * head_dim, dtype=torch.float32).reshape(
            seq, heads * head_dim
        )

        expected = x.view(seq, heads, head_dim).transpose(0, 1).contiguous()

        x_cpu = torch_to_cpp(x)
        cpp_cpu = xstar_cpp.head_split(x_cpu, heads)
        cpu = cpp_to_torch(cpp_cpu, expected.shape)
        x_cuda = xstar_cpp.to_cuda(x_cpu)
        cpp_cuda = xstar_cpp.head_split(x_cuda, heads)
        cuda = cpp_to_torch(xstar_cpp.to_cpu(cpp_cuda), expected.shape)

        assert torch.equal(cpu, expected), f"cpu={cpu} expected={expected}"
        assert torch.equal(cuda, expected), f"cuda={cuda} expected={expected}"


# precondition 校验
def test_cuda_head_split_throws():
    heads = 5
    seq = 20
    head_dim = 64

    x = torch.arange(seq * head_dim, dtype=torch.float32).reshape(seq, head_dim)
    x_cpu = torch_to_cpp(x)
    with pytest.raises(RuntimeError, match="head_dim not integral"):
        xstar_cpp.head_split(x_cpu, heads)

    x = torch.arange(head_dim, dtype=torch.float32).reshape(head_dim)
    x_cpu = torch_to_cpp(x)
    with pytest.raises(RuntimeError, match="rank mismatch"):
        xstar_cpp.head_split(x_cpu, heads)

    x = torch.arange(heads * seq * head_dim, dtype=torch.float32).reshape(
        seq, heads * head_dim
    )
    x_cpu = torch_to_cpp(x)
    with pytest.raises(RuntimeError, match="heads not positive"):
        xstar_cpp.head_split(x_cpu, 0)

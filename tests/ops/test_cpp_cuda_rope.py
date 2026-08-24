import pytest
import sys
import torch
import numpy as np

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 多 head position 共享 + f32
def test_cuda_rope_f32_multi_head():
    num_outer = 14
    seq = 5
    dim = 64
    max_seq = 20

    positions = np.array([3, 7, 12, 0, 5])
    cache = torch.randn(2, max_seq, dim // 2)
    x = torch.randn(num_outer, seq, dim)

    cache_cpu = torch_to_cpp(cache)
    x_cpu = torch_to_cpp(x)
    cache_cuda = xstar_cpp.to_cuda(cache_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    out_cpu = xstar_cpp.rope(x_cpu, cache_cpu, positions)
    out_cuda = xstar_cpp.rope(x_cuda, cache_cuda, positions)

    expected = cpp_to_torch(out_cpu, [num_outer, seq, dim])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [num_outer, seq, dim])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda, expected, atol=1e-6
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# bf16
def test_cuda_rope_bf16_allclose():
    num_outer = 14
    seq = 5
    dim = 64
    max_seq = 20

    positions = np.array([3, 7, 12, 0, 5])
    cache = torch.randn(2, max_seq, dim // 2)
    x = torch.randn(num_outer, seq, dim, dtype=torch.bfloat16)

    cache_cpu = torch_to_cpp(cache)
    x_cpu = torch_to_cpp(x)
    cache_cuda = xstar_cpp.to_cuda(cache_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    out_cpu = xstar_cpp.rope(x_cpu, cache_cpu, positions)
    out_cuda = xstar_cpp.rope(x_cuda, cache_cuda, positions)

    expected = cpp_to_torch(out_cpu, [num_outer, seq, dim])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [num_outer, seq, dim])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda, expected, rtol=1e-2, atol=3e-2
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# ch 维和 h 维的边界 guard
def test_cuda_rope_boundary():
    num_outers = [14, 18]
    seq = 4
    dims = [64, 72]
    max_seq = 20

    for num_outer, dim in zip(num_outers, dims):
        positions = np.random.randint(0, max_seq, (seq,))
        cache = torch.randn(2, max_seq, dim // 2)
        x = torch.randn(num_outer, seq, dim)

        cache_cpu = torch_to_cpp(cache)
        x_cpu = torch_to_cpp(x)
        cache_cuda = xstar_cpp.to_cuda(cache_cpu)
        x_cuda = xstar_cpp.to_cuda(x_cpu)
        out_cpu = xstar_cpp.rope(x_cpu, cache_cpu, positions)
        out_cuda = xstar_cpp.rope(x_cuda, cache_cuda, positions)

        expected = cpp_to_torch(out_cpu, [num_outer, seq, dim])
        cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [num_outer, seq, dim])

        diff = (cuda - expected).abs().max().item()
        print(diff)
        assert torch.allclose(
            cuda, expected, atol=1e-6
        ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 长 seq 正确性(s 循环/并行跑完 512 不崩)
def test_cuda_rope_long_seq():
    num_outer = 14
    seq = 512
    dim = 64
    max_seq = 512

    positions = np.arange(max_seq)
    cache = torch.randn(2, max_seq, dim // 2)
    x = torch.randn(num_outer, seq, dim)

    cache_cpu = torch_to_cpp(cache)
    x_cpu = torch_to_cpp(x)
    cache_cuda = xstar_cpp.to_cuda(cache_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    out_cpu = xstar_cpp.rope(x_cpu, cache_cpu, positions)
    out_cuda = xstar_cpp.rope(x_cuda, cache_cuda, positions)

    expected = cpp_to_torch(out_cpu, [num_outer, seq, dim])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [num_outer, seq, dim])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda, expected, atol=1e-6
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 越界检查在 memcpy 之前(host 侧)
# 用 10**9 而非 10:OOB 偏移必崩 illegal address(CHECK_CUDA 抛的也是 RuntimeError), 靠 match 区分"host 先拦"vs"kernel 先崩"; id=10 读 padding 静默, 两种顺序都抛 RuntimeError, 测不出顺序
def test_cuda_rope_out_of_range():
    num_outer = 2
    seq = 3
    dim = 64
    max_seq = 10

    positions = np.array([3, 10**9, 5])
    cache = torch.randn(2, max_seq, dim // 2)
    x = torch.randn(num_outer, seq, dim)

    cache_cpu = torch_to_cpp(cache)
    x_cpu = torch_to_cpp(x)
    cache_cuda = xstar_cpp.to_cuda(cache_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)

    with pytest.raises(RuntimeError, match="out-of-range position"):
        xstar_cpp.rope(x_cuda, cache_cuda, positions)

    positions = np.array([3, -(10**9), 5])

    with pytest.raises(RuntimeError, match="out-of-range position"):
        xstar_cpp.rope(x_cuda, cache_cuda, positions)


# d_positions 每次 alloc 必须 free
def test_cuda_rope_no_leak():
    num_outer = 14
    seq = 2000
    dim = 64
    max_seq = 2000

    positions = np.arange(seq)
    cache = torch.randn(2, max_seq, dim // 2)
    x = torch.randn(num_outer, seq, dim)

    cache_cpu = torch_to_cpp(cache)
    x_cpu = torch_to_cpp(x)
    cache_cuda = xstar_cpp.to_cuda(cache_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    out_cpu = xstar_cpp.rope(x_cpu, cache_cpu, positions)

    free0 = xstar_cpp.cuda_free_bytes()

    for _ in range(100):
        out_cuda = xstar_cpp.rope(x_cuda, cache_cuda, positions)

        expected = cpp_to_torch(out_cpu, [num_outer, seq, dim])
        cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [num_outer, seq, dim])

        diff = (cuda - expected).abs().max().item()
        print(diff)
        assert torch.allclose(
            cuda, expected, atol=1e-6
        ), f"cpp_cuda={cuda} cpp_cpu={expected}"

        del out_cuda

    free1 = xstar_cpp.cuda_free_bytes()

    assert free1 >= free0 - 1024 * 1024

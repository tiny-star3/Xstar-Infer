import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# +intermediate 偏移(gate/up 符号区分)
def test_cuda_mlp_f32_gate_up_distinct():
    hidden = 8
    intermediate = 16
    num_rows = 2

    x = torch.randn(num_rows, hidden)
    gate_up_weight = torch.empty(2 * intermediate, hidden)
    gate_up_weight[:intermediate, :] = abs(torch.randn(intermediate, hidden))
    gate_up_weight[intermediate:, :] = -abs(torch.randn(intermediate, hidden))
    down_weight = torch.randn(hidden, intermediate)

    x_cpu = torch_to_cpp(x)
    gate_up_weight_cpu = torch_to_cpp(gate_up_weight)
    down_weight_cpu = torch_to_cpp(down_weight)
    out_cpu = xstar_cpp.mlp(x_cpu, gate_up_weight_cpu, down_weight_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    gate_up_weight_cuda = xstar_cpp.to_cuda(gate_up_weight_cpu)
    down_weight_cuda = xstar_cpp.to_cuda(down_weight_cpu)
    out_cuda = xstar_cpp.mlp(x_cuda, gate_up_weight_cuda, down_weight_cuda)

    expected = cpp_to_torch(out_cpu, [num_rows, hidden])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [num_rows, hidden])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# store guard 裁剪 inter<BN
def test_cuda_mlp_f32_single_tile():
    hidden = 8
    intermediate = 16
    num_rows = 2

    x = torch.randn(num_rows, hidden)
    gate_up_weight = torch.randn(2 * intermediate, hidden)
    down_weight = torch.randn(hidden, intermediate)

    x_cpu = torch_to_cpp(x)
    gate_up_weight_cpu = torch_to_cpp(gate_up_weight)
    down_weight_cpu = torch_to_cpp(down_weight)
    out_cpu = xstar_cpp.mlp(x_cpu, gate_up_weight_cpu, down_weight_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    gate_up_weight_cuda = xstar_cpp.to_cuda(gate_up_weight_cpu)
    down_weight_cuda = xstar_cpp.to_cuda(down_weight_cpu)
    out_cuda = xstar_cpp.mlp(x_cuda, gate_up_weight_cuda, down_weight_cuda)

    expected = cpp_to_torch(out_cpu, [num_rows, hidden])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [num_rows, hidden])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 跨 tile col≠0 的 up 行偏移
def test_cuda_mlp_f32_multi_tile():
    hidden = 8
    intermediate = 36
    num_rows = 2

    x = torch.randn(num_rows, hidden)
    gate_up_weight = torch.randn(2 * intermediate, hidden)
    down_weight = torch.randn(hidden, intermediate)

    x_cpu = torch_to_cpp(x)
    gate_up_weight_cpu = torch_to_cpp(gate_up_weight)
    down_weight_cpu = torch_to_cpp(down_weight)
    out_cpu = xstar_cpp.mlp(x_cpu, gate_up_weight_cpu, down_weight_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    gate_up_weight_cuda = xstar_cpp.to_cuda(gate_up_weight_cpu)
    down_weight_cuda = xstar_cpp.to_cuda(down_weight_cpu)
    out_cuda = xstar_cpp.mlp(x_cuda, gate_up_weight_cuda, down_weight_cuda)

    expected = cpp_to_torch(out_cpu, [num_rows, hidden])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [num_rows, hidden])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# row 方向末 tile guard
def test_cuda_mlp_f32_multi_row():
    hidden = 8
    intermediate = 16
    num_rows = 5

    x = torch.randn(num_rows, hidden)
    gate_up_weight = torch.randn(2 * intermediate, hidden)
    down_weight = torch.randn(hidden, intermediate)

    x_cpu = torch_to_cpp(x)
    gate_up_weight_cpu = torch_to_cpp(gate_up_weight)
    down_weight_cpu = torch_to_cpp(down_weight)
    out_cpu = xstar_cpp.mlp(x_cpu, gate_up_weight_cpu, down_weight_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    gate_up_weight_cuda = xstar_cpp.to_cuda(gate_up_weight_cpu)
    down_weight_cuda = xstar_cpp.to_cuda(down_weight_cpu)
    out_cuda = xstar_cpp.mlp(x_cuda, gate_up_weight_cuda, down_weight_cuda)

    expected = cpp_to_torch(out_cpu, [num_rows, hidden])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [num_rows, hidden])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# bf16 down-linear 累加顺序差 → 逐位 ULP 抖动 (非 bug, f32 干净证)
def test_cuda_mlp_bf16():
    hidden = 8
    intermediate = 16
    num_rows = 2

    x = torch.randn(num_rows, hidden, dtype=torch.bfloat16)
    gate_up_weight = torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16)
    down_weight = torch.randn(hidden, intermediate, dtype=torch.bfloat16)

    x_cpu = torch_to_cpp(x)
    gate_up_weight_cpu = torch_to_cpp(gate_up_weight)
    down_weight_cpu = torch_to_cpp(down_weight)
    out_cpu = xstar_cpp.mlp(x_cpu, gate_up_weight_cpu, down_weight_cpu)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    gate_up_weight_cuda = xstar_cpp.to_cuda(gate_up_weight_cpu)
    down_weight_cuda = xstar_cpp.to_cuda(down_weight_cpu)
    out_cuda = xstar_cpp.mlp(x_cuda, gate_up_weight_cuda, down_weight_cuda)

    expected = cpp_to_torch(out_cpu, [num_rows, hidden])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [num_rows, hidden])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        rtol=2e-2,
        atol=5e-1,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"

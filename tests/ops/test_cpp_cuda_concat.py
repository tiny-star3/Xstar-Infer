import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# rank-1(无 stride 循环迭代), axis 维求和
def test_concat_rank1_axis0():
    x1 = torch.randn(3)
    x2 = torch.randn(3)
    axis = 0

    ref = torch.cat([x1, x2], dim=axis)

    x1_cpu = torch_to_cpp(x1)
    x2_cpu = torch_to_cpp(x2)
    x1_cuda = xstar_cpp.to_cuda(x1_cpu)
    x2_cuda = xstar_cpp.to_cuda(x2_cpu)
    cpp_cuda = xstar_cpp.concat([x1_cuda, x2_cuda], axis)
    cpp_cpu = xstar_cpp.to_cpu(cpp_cuda)
    cpp = cpp_to_torch(cpp_cpu, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# axis=0 映射; out 偏移 idx*dtype_size; in_coord[0] 变、in_coord[1]=out_coord[1]
def test_concat_axis0_2d():
    x1 = torch.randn(2, 3)
    x2 = torch.randn(2, 3)
    axis = 0

    ref = torch.cat([x1, x2], dim=axis)

    x1_cpu = torch_to_cpp(x1)
    x2_cpu = torch_to_cpp(x2)
    x1_cuda = xstar_cpp.to_cuda(x1_cpu)
    x2_cuda = xstar_cpp.to_cuda(x2_cpu)
    cpp_cuda = xstar_cpp.concat([x1_cuda, x2_cuda], axis)
    cpp_cpu = xstar_cpp.to_cpu(cpp_cuda)
    cpp = cpp_to_torch(cpp_cpu, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# axis=last; stride 循环(axis 维用 d_axis_sizes[k]、其余用 d_out_shape[i])
def test_concat_axis_last_2d():
    x1 = torch.randn(2, 3)
    x2 = torch.randn(2, 3)
    axis = 1

    ref = torch.cat([x1, x2], dim=axis)

    x1_cpu = torch_to_cpp(x1)
    x2_cpu = torch_to_cpp(x2)
    x1_cuda = xstar_cpp.to_cuda(x1_cpu)
    x2_cuda = xstar_cpp.to_cuda(x2_cpu)
    cpp_cuda = xstar_cpp.concat([x1_cuda, x2_cuda], axis)
    cpp_cpu = xstar_cpp.to_cpu(cpp_cuda)
    cpp = cpp_to_torch(cpp_cpu, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# 负 axis 归一; rank-3 stride
def test_concat_axis_negative():
    x1 = torch.randn(2, 3, 4)
    x2 = torch.randn(2, 3, 4)
    axis = -1

    ref = torch.cat([x1, x2], dim=axis)

    x1_cpu = torch_to_cpp(x1)
    x2_cpu = torch_to_cpp(x2)
    x1_cuda = xstar_cpp.to_cuda(x1_cpu)
    x2_cuda = xstar_cpp.to_cuda(x2_cpu)
    cpp_cuda = xstar_cpp.concat([x1_cuda, x2_cuda], axis)
    cpp_cpu = xstar_cpp.to_cpu(cpp_cuda)
    cpp = cpp_to_torch(cpp_cpu, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# N>2 扫描(扫 d_axis_sizes 过 input 0), 验通用不是 2-input 特化
def test_concat_n_inputs():
    x1 = torch.randn(2, 2)
    x2 = torch.randn(2, 2)
    x3 = torch.randn(2, 2)
    axis = 0

    ref = torch.cat([x1, x2, x3], dim=axis)

    x1_cpu = torch_to_cpp(x1)
    x2_cpu = torch_to_cpp(x2)
    x3_cpu = torch_to_cpp(x3)
    x1_cuda = xstar_cpp.to_cuda(x1_cpu)
    x2_cuda = xstar_cpp.to_cuda(x2_cpu)
    x3_cuda = xstar_cpp.to_cuda(x3_cpu)
    cpp_cuda = xstar_cpp.concat([x1_cuda, x2_cuda, x3_cuda], axis)
    cpp_cpu = xstar_cpp.to_cpu(cpp_cuda)
    cpp = cpp_to_torch(cpp_cpu, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# dtype_size=2 路径(memcpy 拷 dtype_size 字节); bf16 raw bit 处理
def test_concat_bf16():
    x1 = torch.randn(4, 4, dtype=torch.bfloat16)
    x2 = torch.randn(4, 4, dtype=torch.bfloat16)
    axis = 0

    ref = torch.cat([x1, x2], dim=axis)

    x1_cpu = torch_to_cpp(x1)
    x2_cpu = torch_to_cpp(x2)
    x1_cuda = xstar_cpp.to_cuda(x1_cpu)
    x2_cuda = xstar_cpp.to_cuda(x2_cpu)
    cpp_cuda = xstar_cpp.concat([x1_cuda, x2_cuda], axis)
    cpp_cpu = xstar_cpp.to_cpu(cpp_cuda)
    cpp = cpp_to_torch(cpp_cpu, ref.shape)

    assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"


# GPU-only 契约
def test_concat_rejects_cpu():
    x1 = torch.randn(4, 4)
    x2 = torch.randn(4, 4)
    axis = 0

    x1_cpu = torch_to_cpp(x1)
    x2_cpu = torch_to_cpp(x2)

    with pytest.raises(RuntimeError, match="GPU-only"):
        xstar_cpp.concat([x1_cpu, x2_cpu], axis)


# 三个临时 buffer(d_ptrs/d_axis_sizes/d_out_shape)真 free 了
def test_concat_no_leak():
    x1 = torch.randn(1024)
    x2 = torch.randn(1024)
    axis = 0

    ref = torch.cat([x1, x2], dim=axis)

    x1_cpu = torch_to_cpp(x1)
    x2_cpu = torch_to_cpp(x2)
    free0 = xstar_cpp.cuda_free_bytes()

    for _ in range(100):
        x1_cuda = xstar_cpp.to_cuda(x1_cpu)
        x2_cuda = xstar_cpp.to_cuda(x2_cpu)
        cpp_cuda = xstar_cpp.concat([x1_cuda, x2_cuda], axis)
        cpp_cpu = xstar_cpp.to_cpu(cpp_cuda)
        cpp = cpp_to_torch(cpp_cpu, ref.shape)

        assert torch.equal(cpp, ref), f"cpp={cpp} ref={ref}"

        del x1_cuda, x2_cuda, cpp_cuda, cpp_cpu, cpp

    free1 = xstar_cpp.cuda_free_bytes()

    assert free1 >= free0 - 1024 * 1024

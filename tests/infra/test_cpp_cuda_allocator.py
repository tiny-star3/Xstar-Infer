import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# owned ctor 传 Device::CUDA 不崩, 且元数据(shape/dtype/nbytes)正确
# 证明 cuda_alloc 在构造里被调用、Tensor 元数据路径对
def test_cuda_alloc_nbytes():
    x_cuda = xstar_cpp.Tensor([4, 8], xstar_cpp.DType.Float32, xstar_cpp.Device.CUDA)

    assert (
        x_cuda.nbytes() == 128
        and x_cuda.dtype() == xstar_cpp.DType.Float32
        and list(x_cuda.shape()) == [4, 8]
    ), f"x_cuda.nbytes()={x_cuda.nbytes()} x_cuda.dtype()={x_cuda.dtype()} list(x_cuda.shape())={list(x_cuda.shape())}"


# H2D → D2H 往返逐 byte 一致
def test_h2d_d2h_roundtrip_bitexact():
    x = torch.randn(4, 8)
    x_cpu = torch_to_cpp(x)
    x_cuda = xstar_cpp.to_cuda(x_cpu)
    x_cpu2 = xstar_cpp.to_cpu(x_cuda)
    x2 = cpp_to_torch(x_cpu2, x.shape)

    assert torch.equal(x, x2), f"x={x} x2={x2}"


# 循环 alloc/free,显存不降
# cuda_free 不查返回值, 如果 cudaFree 失败内存泄漏 —— 这个测试抓"free 没生效"
# M7 pool 后此测试需改
def test_cuda_free_no_leak():
    free0 = xstar_cpp.cuda_free_bytes()

    for _ in range(100):
        xstar_cpp.Tensor([1024], xstar_cpp.DType.Float32, xstar_cpp.Device.CUDA)

    free1 = xstar_cpp.cuda_free_bytes()

    # 换台机器、有其他 GPU 进程、driver 内部元数据波动, cudaMemGetInfo 两次采样就不精确相等
    # 工业 leak 测试用容差:assert free1 >= free0 - tolerance, tolerance 给个绝对量(几 MB,driver 波动量级)
    assert free1 >= free0 - 1024 * 1024


# release() 按 device 分流 —— CPU 走 std::free、CUDA 走 cuda_free, 两条都不崩
def test_release_branches_by_device():
    free0 = xstar_cpp.cuda_free_bytes()

    for _ in range(100):
        xstar_cpp.Tensor([1024], xstar_cpp.DType.Float32, xstar_cpp.Device.CPU)

    free1 = xstar_cpp.cuda_free_bytes()

    # 不抛异常、不 segfault(隐式: 循环 CPU tensor, CPU 析构走 std::free 不碰 GPU allocator, 故 free 不变)
    # 换台机器、有其他 GPU 进程、driver 内部元数据波动, cudaMemGetInfo 两次采样就不精确相等
    # 工业 leak 测试用容差:assert free1 >= free0 - tolerance, tolerance 给个绝对量(几 MB,driver 波动量级)
    assert free1 >= free0 - 1024 * 1024


# 测"单向抛错"策略真生效 —— to_cuda 收到已是 CUDA 的 tensor 抛 RuntimeError
def test_to_cuda_rejects_cuda_input():
    x_cuda = xstar_cpp.Tensor([1024], xstar_cpp.DType.Float32, xstar_cpp.Device.CUDA)
    with pytest.raises(RuntimeError, match="already on CUDA"):
        xstar_cpp.to_cuda(x_cuda)

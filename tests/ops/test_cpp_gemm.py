import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# no-transpose 基础路径；lda=k/ldb=n/ldc=n 三者全不等时任何 ld 混淆都暴露成 O(1) 错
def test_gemm_f32_no_transpose():
    A = torch.randn(3, 5)
    B = torch.randn(5, 2)
    transB = False

    ref = A @ B

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)
    cpp_t = xstar_cpp.gemm(A_t, B_t, transB)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert cpp.shape == (3, 2) and torch.allclose(
        cpp, ref, atol=1e-4
    ), f"cpp={cpp} ref={ref}"


# ldb = transB ? k : n 的 true 分支（ldb=k=5）+ 转置索引 B[kk + j*ldb]
# 这条路径目前无 Tensor-op 消费者（matmul 走 false，linear/attention 直接调 kernel），全靠它兜底；k≠n 是命门，k==n 会让 true/false 分支同值、bug 静默通过
def test_gemm_f32_transpose():
    A = torch.randn(3, 5)
    B = torch.randn(2, 5)
    transB = True

    ref = A @ B.T

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)
    cpp_t = xstar_cpp.gemm(A_t, B_t, transB)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert cpp.shape == (3, 2) and torch.allclose(
        cpp, ref, atol=1e-4
    ), f"cpp={cpp} ref={ref}"


# bf16 cast pair 在 no-transpose 下成立
def test_gemm_bf16_no_transpose():
    A = torch.randn(3, 5, dtype=torch.bfloat16)
    B = torch.randn(5, 2, dtype=torch.bfloat16)
    transB = False

    ref = A @ B

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)
    cpp_t = xstar_cpp.gemm(A_t, B_t, transB)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, rtol=1e-2, atol=1e-2), f"cpp={cpp} ref={ref}"


# bf16 + 转置索引组合；kernel 在 f32 累加最后才 RNE 降回 bf16，比 torch CPU bf16（逐步降）更接近真值，所以 1e-2 有富余
def test_gemm_bf16_transpose():
    A = torch.randn(3, 5, dtype=torch.bfloat16)
    B = torch.randn(2, 5, dtype=torch.bfloat16)
    transB = True

    ref = A @ B.T

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)
    cpp_t = xstar_cpp.gemm(A_t, B_t, transB)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, rtol=1e-2, atol=1e-2), f"cpp={cpp} ref={ref}"


# n=1 时 j 循环退化到一次、ldc=1，ld 边界 bug 藏身处；再叠 transB=true 转置索引，探测力最强。边界压在 bug-prone 的 transB=true 上最值（k=5≠n=1 仍成立）
def test_gemm_f32_transpose_n1():
    A = torch.randn(3, 5)
    B = torch.randn(1, 5)
    transB = True

    ref = A @ B.T

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)
    cpp_t = xstar_cpp.gemm(A_t, B_t, transB)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert cpp.shape == (3, 1) and torch.allclose(
        cpp, ref, atol=1e-4
    ), f"cpp={cpp} ref={ref}"


# 校验 dtype
def test_gemm_dtype_mismatch_raises():
    A = torch.randn(3, 5)
    B = torch.randn(5, 2, dtype=torch.bfloat16)
    transB = False

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)

    with pytest.raises(RuntimeError, match="dtype mismatch"):
        xstar_cpp.gemm(A_t, B_t, transB)


# 校验 rank
def test_gemm_rank_mismatch_raises():
    A = torch.randn(2, 3, 5)
    B = torch.randn(5, 2)
    transB = False

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)

    with pytest.raises(RuntimeError, match="rank mismatch"):
        xstar_cpp.gemm(A_t, B_t, transB)


# 校验 inner-dim
def test_gemm_inner_dim_mismatch_no_transpose_raises():
    A = torch.randn(3, 4)
    B = torch.randn(5, 2)
    transB = False

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)

    with pytest.raises(RuntimeError, match="inner-dim mismatch"):
        xstar_cpp.gemm(A_t, B_t, transB)


# 校验 inner-dim
def test_gemm_inner_dim_mismatch_transpose_raises():
    A = torch.randn(3, 4)
    B = torch.randn(2, 5)
    transB = True

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)

    with pytest.raises(RuntimeError, match="inner-dim mismatch"):
        xstar_cpp.gemm(A_t, B_t, transB)

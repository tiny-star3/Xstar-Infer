import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 基本正确性: cooperative load、k-loop 跨 2 tile 累加、acc 存取
def test_cuda_gemm_f32_square():
    M = 16
    K = 16
    N = 16
    tranB = False

    A = torch.randn(M, K)
    B = torch.randn(K, N)
    A_cpu = torch_to_cpp(A)
    B_cpu = torch_to_cpp(B)
    A_cuda = xstar_cpp.to_cuda(A_cpu)
    B_cuda = xstar_cpp.to_cuda(B_cpu)
    C_cpu = xstar_cpp.gemm(A_cpu, B_cpu, tranB)
    C_cuda = xstar_cpp.gemm(A_cuda, B_cuda, tranB)

    expected = cpp_to_torch(C_cpu, [M, N])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(C_cuda), [M, N])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 多 block(4 个):锁 tile 起点坐标
def test_cuda_gemm_f32_multiblock():
    M = 64
    K = 64
    N = 16
    tranB = False

    A = torch.randn(M, K)
    B = torch.randn(K, N)
    A_cpu = torch_to_cpp(A)
    B_cpu = torch_to_cpp(B)
    A_cuda = xstar_cpp.to_cuda(A_cpu)
    B_cuda = xstar_cpp.to_cuda(B_cpu)
    C_cpu = xstar_cpp.gemm(A_cpu, B_cpu, tranB)
    C_cuda = xstar_cpp.gemm(A_cuda, B_cuda, tranB)

    expected = cpp_to_torch(C_cpu, [M, N])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(C_cuda), [M, N])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 三轴全非倍数 boundary + M≠K≠N(防维度互换)
def test_cuda_gemm_f32_nondiv_boundary():
    M = 40
    K = 20
    N = 56
    tranB = False

    A = torch.randn(M, K)
    B = torch.randn(K, N)
    A_cpu = torch_to_cpp(A)
    B_cpu = torch_to_cpp(B)
    A_cuda = xstar_cpp.to_cuda(A_cpu)
    B_cuda = xstar_cpp.to_cuda(B_cpu)
    C_cpu = xstar_cpp.gemm(A_cpu, B_cpu, tranB)
    C_cuda = xstar_cpp.gemm(A_cuda, B_cuda, tranB)

    expected = cpp_to_torch(C_cpu, [M, N])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(C_cuda), [M, N])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# dtype 路径(toFloat+RNE 下转), 容差比 f32 大
def test_cuda_gemm_bf16_multiblock():
    M = 64
    K = 64
    N = 16
    tranB = False

    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)
    A_cpu = torch_to_cpp(A)
    B_cpu = torch_to_cpp(B)
    A_cuda = xstar_cpp.to_cuda(A_cpu)
    B_cuda = xstar_cpp.to_cuda(B_cpu)
    C_cpu = xstar_cpp.gemm(A_cpu, B_cpu, tranB)
    C_cuda = xstar_cpp.gemm(A_cuda, B_cuda, tranB)

    expected = cpp_to_torch(C_cpu, [M, N])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(C_cuda), [M, N])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        rtol=1e-2,
        atol=1e-2,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# dtype 路径(toFloat+RNE 下转), 容差比 f32 大 + 三轴全非倍数 boundary + M≠K≠N(防维度互换)
def test_cuda_gemm_bf16_nondiv_boundary():
    M = 40
    K = 20
    N = 56
    tranB = False

    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)
    A_cpu = torch_to_cpp(A)
    B_cpu = torch_to_cpp(B)
    A_cuda = xstar_cpp.to_cuda(A_cpu)
    B_cuda = xstar_cpp.to_cuda(B_cpu)
    C_cpu = xstar_cpp.gemm(A_cpu, B_cpu, tranB)
    C_cuda = xstar_cpp.gemm(A_cuda, B_cuda, tranB)

    expected = cpp_to_torch(C_cpu, [M, N])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(C_cuda), [M, N])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        rtol=1e-2,
        atol=1e-2,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# transB = true, 基本正确性: cooperative load、k-loop 跨 2 tile 累加、acc 存取
def test_cuda_gemm_f32_transb_square():
    M = 16
    K = 16
    N = 16
    tranB = True

    A = torch.randn(M, K)
    B = torch.randn(K, N).T.contiguous()
    A_cpu = torch_to_cpp(A)
    B_cpu = torch_to_cpp(B)
    A_cuda = xstar_cpp.to_cuda(A_cpu)
    B_cuda = xstar_cpp.to_cuda(B_cpu)
    C_cpu = xstar_cpp.gemm(A_cpu, B_cpu, tranB)
    C_cuda = xstar_cpp.gemm(A_cuda, B_cuda, tranB)

    expected = cpp_to_torch(C_cpu, [M, N])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(C_cuda), [M, N])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# transB = true, 多 block(4 个):锁 tile 起点坐标
def test_cuda_gemm_f32_transb_multiblock():
    M = 64
    K = 64
    N = 16
    tranB = True

    A = torch.randn(M, K)
    B = torch.randn(K, N).T.contiguous()
    A_cpu = torch_to_cpp(A)
    B_cpu = torch_to_cpp(B)
    A_cuda = xstar_cpp.to_cuda(A_cpu)
    B_cuda = xstar_cpp.to_cuda(B_cpu)
    C_cpu = xstar_cpp.gemm(A_cpu, B_cpu, tranB)
    C_cuda = xstar_cpp.gemm(A_cuda, B_cuda, tranB)

    expected = cpp_to_torch(C_cpu, [M, N])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(C_cuda), [M, N])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# transB = true, 三轴全非倍数 boundary + M≠K≠N(防维度互换)
def test_cuda_gemm_f32_transb_nondiv_boundary():
    M = 40
    K = 20
    N = 56
    tranB = True

    A = torch.randn(M, K)
    B = torch.randn(K, N).T.contiguous()
    A_cpu = torch_to_cpp(A)
    B_cpu = torch_to_cpp(B)
    A_cuda = xstar_cpp.to_cuda(A_cpu)
    B_cuda = xstar_cpp.to_cuda(B_cpu)
    C_cpu = xstar_cpp.gemm(A_cpu, B_cpu, tranB)
    C_cuda = xstar_cpp.gemm(A_cuda, B_cuda, tranB)

    expected = cpp_to_torch(C_cpu, [M, N])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(C_cuda), [M, N])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-5,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# transB = true, dtype 路径(toFloat+RNE 下转), 容差比 f32 大
def test_cuda_gemm_bf16_transb_multiblock():
    M = 64
    K = 64
    N = 16
    tranB = True

    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16).T.contiguous()
    A_cpu = torch_to_cpp(A)
    B_cpu = torch_to_cpp(B)
    A_cuda = xstar_cpp.to_cuda(A_cpu)
    B_cuda = xstar_cpp.to_cuda(B_cpu)
    C_cpu = xstar_cpp.gemm(A_cpu, B_cpu, tranB)
    C_cuda = xstar_cpp.gemm(A_cuda, B_cuda, tranB)

    expected = cpp_to_torch(C_cpu, [M, N])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(C_cuda), [M, N])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        rtol=1e-2,
        atol=1e-2,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# transB = true, dtype 路径(toFloat+RNE 下转), 容差比 f32 大 + 三轴全非倍数 boundary + M≠K≠N(防维度互换)
def test_cuda_gemm_bf16_transb_nondiv_boundary():
    M = 40
    K = 20
    N = 56
    tranB = True

    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16).T.contiguous()
    A_cpu = torch_to_cpp(A)
    B_cpu = torch_to_cpp(B)
    A_cuda = xstar_cpp.to_cuda(A_cpu)
    B_cuda = xstar_cpp.to_cuda(B_cpu)
    C_cpu = xstar_cpp.gemm(A_cpu, B_cpu, tranB)
    C_cuda = xstar_cpp.gemm(A_cuda, B_cuda, tranB)

    expected = cpp_to_torch(C_cpu, [M, N])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(C_cuda), [M, N])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        rtol=1e-2,
        atol=1e-2,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"

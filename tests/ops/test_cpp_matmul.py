import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 非方阵盯 A 步长(方阵 k==n 隐身)
def test_matmul_f32_nonsquare():
    A = torch.randn(2, 3)
    B = torch.randn(3, 4)

    ref = A @ B

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)
    C_t = xstar_cpp.matmul(A_t, B_t)
    cpp = cpp_to_torch(C_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"cpp={cpp} ref={ref}"


# bf16 路径(f32 累加 + 末尾 downcast)
def test_matmul_bf16_nonsquare():
    A = torch.randn(2, 3, dtype=torch.bfloat16)
    B = torch.randn(3, 4, dtype=torch.bfloat16)

    ref = A @ B

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)
    C_t = xstar_cpp.matmul(A_t, B_t)
    cpp = cpp_to_torch(C_t, ref.shape)

    # 累加链越长,1-ULP 级别的偏差越容易累积到肉眼可见
    # bf16 对拍原则: 用 rtol 主导 + 小 atol 兜底
    # bf16 尾数 7 位 → 1 ULP ≈ 2⁻⁷ ≈ 0.78% 相对误差, 所以 rtol=1e-2 ≈ 1.28 个 bf16 ULP, 容中间不 downcast 的末位分歧
    assert torch.allclose(cpp, ref, rtol=1e-2, atol=1e-2), f"cpp={cpp} ref={ref}"


# 验内维检查
def test_matmul_f32_inner_dim_mismatch_raises():
    A = torch.randn(2, 3)
    B = torch.randn(4, 5)

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)
    with pytest.raises(RuntimeError, match="inner-dim mismatch"):
        xstar_cpp.matmul(A_t, B_t)


# 验 2-D 契约
def test_matmul_f32_not2d_raises():
    A = torch.randn(2, 3, 4)
    B = torch.randn(3, 4)

    A_t = torch_to_cpp(A)
    B_t = torch_to_cpp(B)
    with pytest.raises(RuntimeError, match="rank mismatch"):
        xstar_cpp.matmul(A_t, B_t)

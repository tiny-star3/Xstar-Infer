import pytest
import sys
import torch

from xstar.layers.mlp import SwiGLU
from xstar.layers.linear import Linear
from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 基础路径 + gate/up 顺序
def test_mlp_f32_rank2():
    leaddim = (2,)
    hidden = 8
    intermediate = 6

    x = torch.randn(*leaddim, hidden)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)

    ref_mlp = SwiGLU(hidden, intermediate)
    ref_mlp.gate_up_proj.weight.data.copy_(gate_up_w)
    ref_mlp.down_proj.weight.data.copy_(down_w)
    ref = ref_mlp(x)

    x_t = torch_to_cpp(x)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cpp_t = xstar_cpp.mlp(x_t, gate_up_w_t, down_w_t)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert cpp.shape == (*leaddim, hidden) and torch.allclose(
        cpp, ref, atol=1e-4
    ), f"cpp={cpp} ref={ref}"


# 前导维折叠（num_rows=6），rank-2 隐身铁律
def test_mlp_f32_rank3():
    leaddim = (2, 3)
    hidden = 8
    intermediate = 6

    x = torch.randn(*leaddim, hidden)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)

    ref_mlp = SwiGLU(hidden, intermediate)
    ref_mlp.gate_up_proj.weight.data.copy_(gate_up_w)
    ref_mlp.down_proj.weight.data.copy_(down_w)
    ref = ref_mlp(x)

    x_t = torch_to_cpp(x)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cpp_t = xstar_cpp.mlp(x_t, gate_up_w_t, down_w_t)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert cpp.shape == (*leaddim, hidden) and torch.allclose(
        cpp, ref, atol=1e-4
    ), f"cpp={cpp} ref={ref}"


# f32 case 用 SwiGLU（天然对齐），bf16 case 用本函数（契约对齐）
# bf16 参考: silu*up 走 f32 中间域、降一次 bf16, 对齐 C++ 的契约 (C++ 刻意 f32 中间域, 更精); GEMM 仍用 torch F.linear 独立比对.
# 两路径 silu*up 同值 → down GEMM 的抵消放大的是 0 → bit-exact, 故 bf16 仍能在 1e-2 紧容差下抓逻辑 bug (与 attention 的 softmax 走 f32 同一哲学).
# (此前试过用 torch 默认 bf16 SwiGLU 当参考 + 放容差: down GEMM 抵消把 downcast 分歧放大成重尾无界, 2000 seed 需 ≥0.24 且无上界——那是测"契约分歧"非"正确性", 不可用.)
def SwiGLU_hand(x, gate_up_w, down_w):
    hidden = x.shape[-1]
    intermediate = down_w.shape[-1]

    gate_up_proj = Linear(hidden, 2 * intermediate, bias=False, dtype=torch.bfloat16)
    down_proj = Linear(intermediate, hidden, bias=False, dtype=torch.bfloat16)
    gate_up_proj.weight.data.copy_(gate_up_w)
    down_proj.weight.data.copy_(down_w)
    # 一次投影得到 2*intermediate_size 维度
    gate_up = gate_up_proj(x)
    # 拆分为 gate (对应原 gate_proj) 和 value (对应原 up_proj)
    gate, value = gate_up.float().chunk(2, dim=-1)
    act = ((gate) * torch.sigmoid(gate) * value).to(torch.bfloat16)
    return down_proj(act)


# bf16 sigmoid 精度
def test_mlp_bf16_rank2():
    leaddim = (2,)
    hidden = 8
    intermediate = 6

    x = torch.randn(*leaddim, hidden, dtype=torch.bfloat16)
    gate_up_w = torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16)
    down_w = torch.randn(hidden, intermediate, dtype=torch.bfloat16)

    ref = SwiGLU_hand(x, gate_up_w, down_w)

    x_t = torch_to_cpp(x)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cpp_t = xstar_cpp.mlp(x_t, gate_up_w_t, down_w_t)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, rtol=1e-2, atol=1e-2), f"cpp={cpp} ref={ref}"


# bf16 + rank 折叠
def test_mlp_bf16_rank3():
    leaddim = (2, 3)
    hidden = 8
    intermediate = 6

    x = torch.randn(*leaddim, hidden, dtype=torch.bfloat16)
    gate_up_w = torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16)
    down_w = torch.randn(hidden, intermediate, dtype=torch.bfloat16)

    ref = SwiGLU_hand(x, gate_up_w, down_w)

    x_t = torch_to_cpp(x)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cpp_t = xstar_cpp.mlp(x_t, gate_up_w_t, down_w_t)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, rtol=1e-2, atol=1e-2), f"cpp={cpp} ref={ref}"


# [gate; up] 顺序，写反必红（差异 2.69 >> 1e-4）
# gate=1 落 silu 非线性区、up=10 落线性区，silu(1)·10 ≠ silu(10)·1，交换律失效
def test_mlp_gate_up_order():
    leaddim = (2,)
    hidden = 8
    intermediate = 6

    x = torch.ones(*leaddim, hidden)
    gate_up_w = torch.ones(2 * intermediate, hidden)
    # gate 半区
    gate_up_w[:6, :] = 0.125
    # up 半区
    gate_up_w[6:, :] = 1.25
    down_w = torch.randn(hidden, intermediate)

    ref_mlp = SwiGLU(hidden, intermediate)
    ref_mlp.gate_up_proj.weight.data.copy_(gate_up_w)
    ref_mlp.down_proj.weight.data.copy_(down_w)
    ref = ref_mlp(x)

    x_t = torch_to_cpp(x)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cpp_t = xstar_cpp.mlp(x_t, gate_up_w_t, down_w_t)
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"cpp={cpp} ref={ref}"


# 证 even-half 校验 load-bearing
# 没有它，奇数 2*intermediate 从 line 52 的整数除法溜过去
def test_mlp_odd_gate_up_raises():
    leaddim = (2,)
    hidden = 8

    x = torch.randn(*leaddim, hidden)
    gate_up_w = torch.randn(5, hidden)
    down_w = torch.randn(hidden, 2)

    x_t = torch_to_cpp(x)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)

    with pytest.raises(RuntimeError, match="gate_up out must be even"):
        xstar_cpp.mlp(x_t, gate_up_w_t, down_w_t)


# 触发类型不匹配
def test_mlp_dtype_mismatch_raises():
    leaddim = (2,)
    hidden = 8
    intermediate = 6

    x = torch.randn(*leaddim, hidden)
    gate_up_w = torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16)
    down_w = torch.randn(hidden, intermediate)

    x_t = torch_to_cpp(x)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)

    with pytest.raises(RuntimeError, match="dtype mismatch"):
        xstar_cpp.mlp(x_t, gate_up_w_t, down_w_t)


# 只触发 hidden 不匹配
def test_mlp_shape_mismatch_raises():
    leaddim = (2,)
    hidden = 8
    intermediate = 6

    x = torch.randn(*leaddim, hidden)
    gate_up_w = torch.randn(2 * intermediate, hidden + 1)
    down_w = torch.randn(hidden, intermediate)

    x_t = torch_to_cpp(x)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        xstar_cpp.mlp(x_t, gate_up_w_t, down_w_t)

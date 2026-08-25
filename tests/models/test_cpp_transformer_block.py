import pytest
import sys
import torch
import numpy as np

from xstar.layers.transformer_block import TransformerBlock
from xstar.layers.rope import RoPE
from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 合法 f32 block 端到端对拍 Python TransformerBlock, 顺带抓 head_split 顺序错、ln2 用错 x、残差方向反
def test_block_forward_f32_parity():
    hidden = 896
    num_heads = 14
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 4864
    seq = 4
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 512

    # 权重/x 用 0.02 尺度而非默认 std=1.0: std=1.0 下 6+ 算子复合使残差流爆到 ~1e5,
    # f32 每数 ~7 位有效数字,两个等价实现各累积一路浮点误差,相对差被放大到 ~1e-2,
    # 既让 allclose(atol=1e-4) 误红,又会把真实装配 bug(head_split 顺序错/ln2 用错 x) 藏进噪声里
    # 0.02 让输出回到 O(1), atol=1e-4 落在 ulp 噪声之上、bug 差异之下, 测试才既不误报又不漏报
    # 注意:这是调输入工况让噪声回归 ulp 级,不是放宽容差赌 seed
    x = torch.randn(seq, num_heads * head_dim) * 0.02
    positions = np.array([0, 1, 2, 3])
    ln1_w = torch.randn(hidden) * 0.02
    ln2_w = torch.randn(hidden) * 0.02
    q_w = torch.randn(num_heads * head_dim, hidden) * 0.02
    q_b = torch.randn(num_heads * head_dim) * 0.02
    k_w = torch.randn(num_key_value_heads * head_dim, hidden) * 0.02
    k_b = torch.randn(num_key_value_heads * head_dim) * 0.02
    v_w = torch.randn(num_key_value_heads * head_dim, hidden) * 0.02
    v_b = torch.randn(num_key_value_heads * head_dim) * 0.02
    o_w = torch.randn(hidden, num_heads * head_dim) * 0.02
    gate_up_w = torch.randn(2 * intermediate, hidden) * 0.02
    down_w = torch.randn(hidden, intermediate) * 0.02
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)
    ref_block = TransformerBlock(
        hidden, num_heads, num_key_value_heads, intermediate, eps, ref_rope
    )
    ref_block.input_layernorm.weight.data.copy_(ln1_w)
    ref_block.post_attention_layernorm.weight.data.copy_(ln2_w)
    ref_block.attn.q_proj.weight.data.copy_(q_w)
    ref_block.attn.q_proj.bias.data.copy_(q_b)
    ref_block.attn.k_proj.weight.data.copy_(k_w)
    ref_block.attn.k_proj.bias.data.copy_(k_b)
    ref_block.attn.v_proj.weight.data.copy_(v_w)
    ref_block.attn.v_proj.bias.data.copy_(v_b)
    ref_block.attn.o_proj.weight.data.copy_(o_w)
    ref_block.mlp.gate_up_proj.weight.data.copy_(gate_up_w)
    ref_block.mlp.down_proj.weight.data.copy_(down_w)
    ref = ref_block(x, attention_mask=mask, token_positions=torch.from_numpy(positions))

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)
    cpp_t = xstar_cpp.transformer_block(
        x_t,
        num_heads,
        ln1_w_t,
        ln2_w_t,
        eps,
        q_w_t,
        q_b_t,
        k_w_t,
        k_b_t,
        v_w_t,
        v_b_t,
        o_w_t,
        gate_up_w_t,
        down_w_t,
        cache_t,
        positions,
        mask,
    )
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"cpp={cpp} ref={ref}"


# bf16 block 对拍,量累积误差
# bf16 定位为"路径不回归"测试,非装配-bug 捕捉器:
# bf16 下 C++ 少中间 downcast("我更精") 与 PyTorch reference 每步 downcast 的 round 噪声, 经 down_proj cancellation 放大后, 与装配 bug 的输出差异同量级, 无法靠容差分离
# 实测 20 seed (wstd=0.02): 正常 max_abs ~2.4e-4; 注入"post_attention_layernorm 误用原始 x 而非 attention 残差 (ln2 feed 错张量)"这一装配错时 max_abs ~4.9e-4 —— 窗口窄且重叠
# 装配 bug 由 case 1 f32 兜底: f32 无 downcast 分歧, 正常噪声 ~1e-8 vs 上述 ln2 错张量 >1e-4, 窗口 4 量级
# 本 case 只验证: bf16 路径跑通 (dtype/bridge 转换无 crash)、数值在 round 噪声量级、不回归
# atol=1e-2 实测稳定 (20/20 seed 过, max_rel 0.001-0.003), 足够"不回归"判定, 不试图更紧
def test_block_forward_bf16_parity():
    hidden = 896
    num_heads = 14
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 4864
    seq = 4
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 512

    # 权重/x 用 0.02 尺度而非默认 std=1.0: std=1.0 下 6+ 算子复合使残差流爆到 ~1e5,
    # f32 每数 ~7 位有效数字,两个等价实现各累积一路浮点误差,相对差被放大到 ~1e-2,
    # 既让 allclose(atol=1e-4) 误红,又会把真实装配 bug(head_split 顺序错/ln2 用错 x) 藏进噪声里
    # 0.02 让输出回到 O(1), atol=1e-4 落在 ulp 噪声之上、bug 差异之下, 测试才既不误报又不漏报
    # 注意:这是调输入工况让噪声回归 ulp 级,不是放宽容差赌 seed
    x = torch.randn(seq, num_heads * head_dim, dtype=torch.bfloat16) * 0.02
    positions = np.array([0, 1, 2, 3])
    ln1_w = torch.randn(hidden, dtype=torch.bfloat16) * 0.02
    ln2_w = torch.randn(hidden, dtype=torch.bfloat16) * 0.02
    q_w = torch.randn(num_heads * head_dim, hidden, dtype=torch.bfloat16) * 0.02
    q_b = torch.randn(num_heads * head_dim, dtype=torch.bfloat16) * 0.02
    k_w = (
        torch.randn(num_key_value_heads * head_dim, hidden, dtype=torch.bfloat16) * 0.02
    )
    k_b = torch.randn(num_key_value_heads * head_dim, dtype=torch.bfloat16) * 0.02
    v_w = (
        torch.randn(num_key_value_heads * head_dim, hidden, dtype=torch.bfloat16) * 0.02
    )
    v_b = torch.randn(num_key_value_heads * head_dim, dtype=torch.bfloat16) * 0.02
    o_w = torch.randn(hidden, num_heads * head_dim, dtype=torch.bfloat16) * 0.02
    gate_up_w = torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16) * 0.02
    down_w = torch.randn(hidden, intermediate, dtype=torch.bfloat16) * 0.02
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)
    ref_block = TransformerBlock(
        hidden,
        num_heads,
        num_key_value_heads,
        intermediate,
        eps,
        ref_rope,
        dtype=torch.bfloat16,
    )
    ref_block.input_layernorm.weight.data.copy_(ln1_w)
    ref_block.post_attention_layernorm.weight.data.copy_(ln2_w)
    ref_block.attn.q_proj.weight.data.copy_(q_w)
    ref_block.attn.q_proj.bias.data.copy_(q_b)
    ref_block.attn.k_proj.weight.data.copy_(k_w)
    ref_block.attn.k_proj.bias.data.copy_(k_b)
    ref_block.attn.v_proj.weight.data.copy_(v_w)
    ref_block.attn.v_proj.bias.data.copy_(v_b)
    ref_block.attn.o_proj.weight.data.copy_(o_w)
    ref_block.mlp.gate_up_proj.weight.data.copy_(gate_up_w)
    ref_block.mlp.down_proj.weight.data.copy_(down_w)
    ref = ref_block(x, attention_mask=mask, token_positions=torch.from_numpy(positions))

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)
    cpp_t = xstar_cpp.transformer_block(
        x_t,
        num_heads,
        ln1_w_t,
        ln2_w_t,
        eps,
        q_w_t,
        q_b_t,
        k_w_t,
        k_b_t,
        v_w_t,
        v_b_t,
        o_w_t,
        gate_up_w_t,
        down_w_t,
        cache_t,
        positions,
        mask,
    )
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, rtol=1e-2, atol=1e-2), f"cpp={cpp} ref={ref}"


# positions 透传给 rope(非 0..seq-1)
def test_block_non_contiguous_positions():
    hidden = 896
    num_heads = 14
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 4864
    seq = 4
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 512

    # 权重/x 用 0.02 尺度而非默认 std=1.0: std=1.0 下 6+ 算子复合使残差流爆到 ~1e5,
    # f32 每数 ~7 位有效数字,两个等价实现各累积一路浮点误差,相对差被放大到 ~1e-2,
    # 既让 allclose(atol=1e-4) 误红,又会把真实装配 bug(head_split 顺序错/ln2 用错 x) 藏进噪声里
    # 0.02 让输出回到 O(1), atol=1e-4 落在 ulp 噪声之上、bug 差异之下, 测试才既不误报又不漏报
    # 注意:这是调输入工况让噪声回归 ulp 级,不是放宽容差赌 seed
    x = torch.randn(seq, num_heads * head_dim) * 0.02
    positions = np.array([3, 1, 0, 2])
    ln1_w = torch.randn(hidden) * 0.02
    ln2_w = torch.randn(hidden) * 0.02
    q_w = torch.randn(num_heads * head_dim, hidden) * 0.02
    q_b = torch.randn(num_heads * head_dim) * 0.02
    k_w = torch.randn(num_key_value_heads * head_dim, hidden) * 0.02
    k_b = torch.randn(num_key_value_heads * head_dim) * 0.02
    v_w = torch.randn(num_key_value_heads * head_dim, hidden) * 0.02
    v_b = torch.randn(num_key_value_heads * head_dim) * 0.02
    o_w = torch.randn(hidden, num_heads * head_dim) * 0.02
    gate_up_w = torch.randn(2 * intermediate, hidden) * 0.02
    down_w = torch.randn(hidden, intermediate) * 0.02
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)
    ref_block = TransformerBlock(
        hidden, num_heads, num_key_value_heads, intermediate, eps, ref_rope
    )
    ref_block.input_layernorm.weight.data.copy_(ln1_w)
    ref_block.post_attention_layernorm.weight.data.copy_(ln2_w)
    ref_block.attn.q_proj.weight.data.copy_(q_w)
    ref_block.attn.q_proj.bias.data.copy_(q_b)
    ref_block.attn.k_proj.weight.data.copy_(k_w)
    ref_block.attn.k_proj.bias.data.copy_(k_b)
    ref_block.attn.v_proj.weight.data.copy_(v_w)
    ref_block.attn.v_proj.bias.data.copy_(v_b)
    ref_block.attn.o_proj.weight.data.copy_(o_w)
    ref_block.mlp.gate_up_proj.weight.data.copy_(gate_up_w)
    ref_block.mlp.down_proj.weight.data.copy_(down_w)
    ref = ref_block(x, attention_mask=mask, token_positions=torch.from_numpy(positions))

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)
    cpp_t = xstar_cpp.transformer_block(
        x_t,
        num_heads,
        ln1_w_t,
        ln2_w_t,
        eps,
        q_w_t,
        q_b_t,
        k_w_t,
        k_b_t,
        v_w_t,
        v_b_t,
        o_w_t,
        gate_up_w_t,
        down_w_t,
        cache_t,
        positions,
        mask,
    )
    cpp = cpp_to_torch(cpp_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"cpp={cpp} ref={ref}"


# rank≠2 拦截
def test_block_rejects_rank3():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(2, seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="block rank mismatch"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# num_heads<=0 守卫在除法前拦截,不 SIGFPE
def test_block_rejects_zero_num_heads():
    hidden = 16
    num_heads = 0
    num_key_value_heads = 2
    head_dim = 4
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="num_heads must be positive"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# head_dim<=0 守卫夹在两除法间
def test_block_rejects_zero_head_dim():
    hidden = 16
    num_heads = 20
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="head_dim must be positive"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# head_dim>0 过守卫,但 % 抓到整数截断
def test_block_rejects_nonintegral_head_dim():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim - 1, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="head_dim not integral"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# num_kv 整数性
def test_block_rejects_nonintegral_num_kv():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim - 1, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim - 1, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="num_kv not integral"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# rep 整数性
def test_block_rejects_nonintegral_rep():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 3
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="rep not integral"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# k/v proj out 不等
def test_block_rejects_kv_out_mismatch():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim + 4, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="k/v proj out mismatch"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# q_proj in ≠ hidden
def test_block_rejects_hidden_mismatch():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden + 4)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="hidden mismatch"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# o_proj in ≠ nh*hd
def test_block_rejects_o_proj_in_mismatch():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim + 4)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="o_proj in mismatch"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# o_proj out ≠ hidden
def test_block_rejects_o_proj_out_mismatch():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden + 4, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="o_proj out mismatch"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# ln1 权重 ≠ hidden
def test_block_rejects_ln1_weight_mismatch():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden + 4)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="ln weight mismatch"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# ln2 单独错也 throw
def test_block_rejects_ln2_weight_mismatch():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden + 4)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="ln weight mismatch"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# block 已删 cache 校验,defer 给 rope——rank 错由 rope 抛
def test_block_defers_cache_rank_check_to_rope():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(
        ref_rope._freq_cis_cache.reshape(2 * max_seq_len, head_dim // 2)
    )

    with pytest.raises(RuntimeError, match="rank mismatch"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# cache shape[2] ≠ head_dim/2 由 rope 抛
def test_block_defers_cache_shape_check_to_rope():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(seq, num_heads * head_dim)
    positions = np.array([0, 1, 2])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None
    rope = torch.randn(2, max_seq_len, head_dim // 2 + 1)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(rope)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )


# wrapper 的长度守卫, load-bearing(C++ rope 不验 positions 长度, 越界读 UB)
def test_block_rejects_positions_length_mismatch():
    hidden = 16
    num_heads = 4
    num_key_value_heads = 2
    head_dim = hidden // num_heads
    intermediate = 32
    seq = 3
    eps = 1e-6
    theta = 1000000.0
    max_seq_len = 8

    x = torch.randn(2, seq, num_heads * head_dim)
    positions = np.array([0, 1])
    ln1_w = torch.randn(hidden)
    ln2_w = torch.randn(hidden)
    q_w = torch.randn(num_heads * head_dim, hidden)
    q_b = torch.randn(num_heads * head_dim)
    k_w = torch.randn(num_key_value_heads * head_dim, hidden)
    k_b = torch.randn(num_key_value_heads * head_dim)
    v_w = torch.randn(num_key_value_heads * head_dim, hidden)
    v_b = torch.randn(num_key_value_heads * head_dim)
    o_w = torch.randn(hidden, num_heads * head_dim)
    gate_up_w = torch.randn(2 * intermediate, hidden)
    down_w = torch.randn(hidden, intermediate)
    mask = None

    ref_rope = RoPE(theta, head_dim, max_seq_len)

    x_t = torch_to_cpp(x)
    ln1_w_t = torch_to_cpp(ln1_w)
    ln2_w_t = torch_to_cpp(ln2_w)
    q_w_t = torch_to_cpp(q_w)
    q_b_t = torch_to_cpp(q_b)
    k_w_t = torch_to_cpp(k_w)
    k_b_t = torch_to_cpp(k_b)
    v_w_t = torch_to_cpp(v_w)
    v_b_t = torch_to_cpp(v_b)
    o_w_t = torch_to_cpp(o_w)
    gate_up_w_t = torch_to_cpp(gate_up_w)
    down_w_t = torch_to_cpp(down_w)
    cache_t = torch_to_cpp(ref_rope._freq_cis_cache)

    with pytest.raises(RuntimeError, match="positions length != seq_len"):
        xstar_cpp.transformer_block(
            x_t,
            num_heads,
            ln1_w_t,
            ln2_w_t,
            eps,
            q_w_t,
            q_b_t,
            k_w_t,
            k_b_t,
            v_w_t,
            v_b_t,
            o_w_t,
            gate_up_w_t,
            down_w_t,
            cache_t,
            positions,
            mask,
        )

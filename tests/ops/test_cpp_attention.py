import pytest
import sys
import torch
from einops import rearrange, einsum
import math

from xstar.layers.attention import softmax
from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


def ref_attention(Q, K, V, attention_mask=None):
    num_heads = Q.shape[0]
    num_key_value_heads = K.shape[0]
    seq = Q.shape[1]
    assert num_heads % num_key_value_heads == 0
    K = K[:, None, :, :].expand(
        -1,
        num_heads // num_key_value_heads,
        -1,
        -1,
    )
    V = V[:, None, :, :].expand(
        -1,
        num_heads // num_key_value_heads,
        -1,
        -1,
    )
    K = rearrange(
        K,
        "k_heads rep ... -> (k_heads rep) ...",
        k_heads=num_key_value_heads,
    )
    V = rearrange(
        V,
        "v_heads rep ... -> (v_heads rep) ...",
        v_heads=num_key_value_heads,
    )

    qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(
        Q.shape[-1]
    )
    if attention_mask is not None:
        qk = qk + attention_mask
    else:
        # 根据当前的 sequence_length 实时生成。它能完美适配变长输入，且通过 (None,) * len(batch_dims) 这种写法，能够自动处理任意数量的 Batch 维度（即支持多维广播）
        # Construct causal mask
        iota = torch.arange(seq, device=Q.device)
        qi = rearrange(iota, "query -> query 1")
        kj = rearrange(iota, "key   -> 1   key")
        # 生成了一个下三角矩阵, 当 query_index >= key_index 时为 True（可见）, 当 query_index < key_index 时为 False（不可见，即未来信息）
        causal_mask = qi >= kj  # (query, key)
        qk = torch.where(causal_mask, qk, float("-inf"))

    qk = softmax(qk, -1, torch.float32).to(Q.dtype)

    attn_output = einsum(qk, V, "... queries keys, ... keys d_v -> ... queries d_v")

    # Concatenate the attention output from all heads.
    # (sequence_length, num_heads * d_v).
    attn_output = rearrange(
        attn_output, "heads seq d_v -> seq (heads d_v)"
    ).contiguous()

    return attn_output


# GQA + causal, seq≠hd
# GQA 索引(h/rep)、causal 分支、Q@K^T/scores@V 两 reduce、合头
def test_attention_f32_gqa_causal():
    num_heads = 4
    num_key_value_heads = 2
    seq = 3
    head_dim = 4

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(num_key_value_heads, seq, head_dim)
    V = torch.randn(num_key_value_heads, seq, head_dim)
    mask = None

    ref = ref_attention(Q, K, V, mask)

    Q_t = torch_to_cpp(Q)
    K_t = torch_to_cpp(K)
    V_t = torch_to_cpp(V)
    O_t = xstar_cpp.attention(Q_t, K_t, V_t, mask)
    cpp = cpp_to_torch(O_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"ref={ref} cpp={cpp}"


# GQA + additive zeros mask(全可见)
# additive mask 分支(mask≠None 走 qk+=mask, bypass causal)
def test_attention_f32_gqa_nomask():
    num_heads = 4
    num_key_value_heads = 2
    seq = 3
    head_dim = 4

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(num_key_value_heads, seq, head_dim)
    V = torch.randn(num_key_value_heads, seq, head_dim)
    mask = torch.zeros(seq, seq)

    ref = ref_attention(Q, K, V, mask)

    Q_t = torch_to_cpp(Q)
    K_t = torch_to_cpp(K)
    V_t = torch_to_cpp(V)
    mask_t = torch_to_cpp(mask)
    O_t = xstar_cpp.attention(Q_t, K_t, V_t, mask_t)
    cpp = cpp_to_torch(O_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"ref={ref} cpp={cpp}"


# 含 -inf 的非标准 additive mask
# additive mask 含 -inf 时 softmax 正确归零 + 部分可见, 兼测 "mask 被真正应用, 不是被忽略退回 causal"
def test_attention_f32_additive_mask_with_inf():
    num_heads = 4
    num_key_value_heads = 2
    seq = 3
    head_dim = 4

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(num_key_value_heads, seq, head_dim)
    V = torch.randn(num_key_value_heads, seq, head_dim)
    mask = torch.zeros(seq, seq)
    # -inf 落在对角线下方, causal 永不藏下三角
    mask[2, 0] = float("-inf")

    ref = ref_attention(Q, K, V, mask)

    Q_t = torch_to_cpp(Q)
    K_t = torch_to_cpp(K)
    V_t = torch_to_cpp(V)
    mask_t = torch_to_cpp(mask)
    O_t = xstar_cpp.attention(Q_t, K_t, V_t, mask_t)
    cpp = cpp_to_torch(O_t, ref.shape)

    assert not torch.isnan(cpp).any()
    assert torch.allclose(cpp, ref, atol=1e-4), f"ref={ref} cpp={cpp}"


# bf16 + causal
# bf16 两 reduce + -inf 写进 bf16 再读回(因果 mask 的 -inf 走 static_cast<bfloat16>(-inf), softmax 读回 float 是否还是 -inf、exp(-inf)=0)
def test_attention_bf16_gqa_causal():
    num_heads = 4
    num_key_value_heads = 2
    seq = 3
    head_dim = 4

    Q = torch.randn(num_heads, seq, head_dim, dtype=torch.bfloat16)
    K = torch.randn(num_key_value_heads, seq, head_dim, dtype=torch.bfloat16)
    V = torch.randn(num_key_value_heads, seq, head_dim, dtype=torch.bfloat16)
    mask = None

    ref = ref_attention(Q, K, V, mask)

    Q_t = torch_to_cpp(Q)
    K_t = torch_to_cpp(K)
    V_t = torch_to_cpp(V)
    O_t = xstar_cpp.attention(Q_t, K_t, V_t, mask)
    cpp = cpp_to_torch(O_t, ref.shape)

    assert not torch.isnan(cpp).any()
    assert torch.allclose(cpp, ref, rtol=1e-2, atol=1e-2), f"ref={ref} cpp={cpp}"


# f32 在 seq > hd 下的 qk stride(h*seq*seq 索引)/ scores@V 索引 —— 真实模型 seq>>hd 的正常方向
def test_attention_f32_seq_gt_head_dim():
    num_heads = 4
    num_key_value_heads = 2
    seq = 6
    head_dim = 4

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(num_key_value_heads, seq, head_dim)
    V = torch.randn(num_key_value_heads, seq, head_dim)
    mask = None

    ref = ref_attention(Q, K, V, mask)

    Q_t = torch_to_cpp(Q)
    K_t = torch_to_cpp(K)
    V_t = torch_to_cpp(V)
    O_t = xstar_cpp.attention(Q_t, K_t, V_t, mask)
    cpp = cpp_to_torch(O_t, ref.shape)

    assert torch.allclose(cpp, ref, atol=1e-4), f"ref={ref} cpp={cpp}"


# bf16 Q@K^T 段若误用 size=hd 的 buffer 而 j 跑 seq, seq > hd 时越界
def test_attention_bf16_seq_gt_head_dim():
    num_heads = 4
    num_key_value_heads = 2
    seq = 6
    head_dim = 4

    Q = torch.randn(num_heads, seq, head_dim, dtype=torch.bfloat16)
    K = torch.randn(num_key_value_heads, seq, head_dim, dtype=torch.bfloat16)
    V = torch.randn(num_key_value_heads, seq, head_dim, dtype=torch.bfloat16)
    mask = None

    ref = ref_attention(Q, K, V, mask)

    Q_t = torch_to_cpp(Q)
    K_t = torch_to_cpp(K)
    V_t = torch_to_cpp(V)
    O_t = xstar_cpp.attention(Q_t, K_t, V_t, mask)
    cpp = cpp_to_torch(O_t, ref.shape)

    assert not torch.isnan(cpp).any()
    assert torch.allclose(cpp, ref, rtol=1e-2, atol=1e-2), f"ref={ref} cpp={cpp}"


# rep 整除校验
def test_attention_rep_not_integral_raises():
    num_heads = 3
    num_key_value_heads = 2
    seq = 3
    head_dim = 4

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(num_key_value_heads, seq, head_dim)
    V = torch.randn(num_key_value_heads, seq, head_dim)
    mask = None

    Q_t = torch_to_cpp(Q)
    K_t = torch_to_cpp(K)
    V_t = torch_to_cpp(V)
    with pytest.raises(RuntimeError, match="rep not integral"):
        xstar_cpp.attention(Q_t, K_t, V_t, mask)


# mask shape 校验(必须(seq,seq)方阵, 且 seq == V.shape[1])
def test_attention_mask_shape_mismatch_raises():
    num_heads = 4
    num_key_value_heads = 2
    seq = 3
    head_dim = 4

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(num_key_value_heads, seq, head_dim)
    V = torch.randn(num_key_value_heads, seq, head_dim)
    mask = torch.zeros(seq, seq + 1)

    Q_t = torch_to_cpp(Q)
    K_t = torch_to_cpp(K)
    V_t = torch_to_cpp(V)
    mask_t = torch_to_cpp(mask)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        xstar_cpp.attention(Q_t, K_t, V_t, mask_t)


# 钉死 "外部 mask 藏掉对角线 → NaN 是 caller 契约" 这个边界
def test_attention_fully_masked_row_is_nan():
    num_heads = 4
    num_key_value_heads = 2
    seq = 3
    head_dim = 4

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(num_key_value_heads, seq, head_dim)
    V = torch.randn(num_key_value_heads, seq, head_dim)
    mask = torch.full((seq, seq), fill_value=float("-inf"))

    ref = ref_attention(Q, K, V, mask)

    Q_t = torch_to_cpp(Q)
    K_t = torch_to_cpp(K)
    V_t = torch_to_cpp(V)
    mask_t = torch_to_cpp(mask)
    O_t = xstar_cpp.attention(Q_t, K_t, V_t, mask_t)
    cpp = cpp_to_torch(O_t, ref.shape)

    assert torch.isnan(cpp).all() and torch.isnan(ref).all()

import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 逐 head gemm transB=true, causal 分支(i<col→-inf), GQA 索引 h/rep=2, scores@V
def test_cuda_attention_f32_gqa_causal():
    num_heads = 4
    kv = 2
    seq = 4
    head_dim = 8

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(kv, seq, head_dim)
    V = torch.randn(kv, seq, head_dim)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    out_cpu = xstar_cpp.attention(Q_cpu, K_cpu, V_cpu, mask)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    out_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask)

    expected = cpp_to_torch(out_cpu, [seq, num_heads * head_dim])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, num_heads * head_dim])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-6,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# additive mask 分支(scale+add, 每 (i,j) 都走), mask 广播跨 heads, mask 指针非空走 if(mask)
def test_cuda_attention_f32_zeros_mask():
    num_heads = 4
    kv = 2
    seq = 4
    head_dim = 8

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(kv, seq, head_dim)
    V = torch.randn(kv, seq, head_dim)
    mask = torch.zeros(seq, seq)
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    mask_cpu = torch_to_cpp(mask)
    out_cpu = xstar_cpp.attention(Q_cpu, K_cpu, V_cpu, mask_cpu)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    mask_cuda = xstar_cpp.to_cuda(mask_cpu)
    out_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask_cuda)

    expected = cpp_to_torch(out_cpu, [seq, num_heads * head_dim])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, num_heads * head_dim])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-6,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# mask 含 -inf, softmax 对该位归零, 验证"mask 被真正应用不是被忽略退回 causal"(若退回 causal,query2→key0 本就被藏, 看不出; 但 query2→key0 是下三角,causal 下可见, additive -inf 下不可见 —— 两种结果不同, 能区分)
def test_cuda_attention_f32_additive_mask_with_inf():
    num_heads = 4
    kv = 2
    seq = 4
    head_dim = 8

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(kv, seq, head_dim)
    V = torch.randn(kv, seq, head_dim)
    mask = torch.zeros(seq, seq)
    # -inf 落在对角线下方, causal 永不藏下三角
    mask[2, 0] = float("-inf")
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    mask_cpu = torch_to_cpp(mask)
    out_cpu = xstar_cpp.attention(Q_cpu, K_cpu, V_cpu, mask_cpu)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    mask_cuda = xstar_cpp.to_cuda(mask_cpu)
    out_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask_cuda)

    expected = cpp_to_torch(out_cpu, [seq, num_heads * head_dim])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, num_heads * head_dim])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert not torch.isnan(cuda).any()
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-6,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# seq>hd(真实模型方向), Q@K^T 的 qk 是 (4,8,8), scores@V 的 lda=seq=8, 压 h seq seq 指针偏移和 scores@V 的 strided C(ldc=num_heads*head_dim=16)
def test_cuda_attention_f32_seq_gt_head_dim():
    num_heads = 4
    kv = 2
    seq = 8
    head_dim = 4

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(kv, seq, head_dim)
    V = torch.randn(kv, seq, head_dim)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    out_cpu = xstar_cpp.attention(Q_cpu, K_cpu, V_cpu, mask)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    out_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask)

    expected = cpp_to_torch(out_cpu, [seq, num_heads * head_dim])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, num_heads * head_dim])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        atol=1e-6,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# bf16 causal, 只 scale 一次 downcast(kernel L31), 不落偏差窗口; -FLT_MAX 写进 bf16(probe 证 = 0xFF80 = -inf), softmax 读回 f32 是否 -inf、exp(-inf)=0
def test_cuda_attention_bf16_gqa_causal():
    num_heads = 4
    kv = 2
    seq = 4
    head_dim = 8

    Q = torch.randn(num_heads, seq, head_dim, dtype=torch.bfloat16)
    K = torch.randn(kv, seq, head_dim, dtype=torch.bfloat16)
    V = torch.randn(kv, seq, head_dim, dtype=torch.bfloat16)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    out_cpu = xstar_cpp.attention(Q_cpu, K_cpu, V_cpu, mask)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    out_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask)

    expected = cpp_to_torch(out_cpu, [seq, num_heads * head_dim])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, num_heads * head_dim])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert not torch.isnan(cuda).any()
    assert torch.allclose(
        cuda,
        expected,
        rtol=1e-2,
        atol=1e-2,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 唯一落在偏差窗口的 case —— kernel 一次 downcast(qk*scale+mask)vs CPU 两次(scale 落 bf16, 再 +mask 落 bf16)
def test_cuda_attention_bf16_additive_mask():
    num_heads = 4
    kv = 2
    seq = 4
    head_dim = 8

    Q = torch.randn(num_heads, seq, head_dim, dtype=torch.bfloat16)
    K = torch.randn(kv, seq, head_dim, dtype=torch.bfloat16)
    V = torch.randn(kv, seq, head_dim, dtype=torch.bfloat16)
    mask = torch.randn(seq, seq, dtype=torch.bfloat16)
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    mask_cpu = torch_to_cpp(mask)
    out_cpu = xstar_cpp.attention(Q_cpu, K_cpu, V_cpu, mask_cpu)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    mask_cuda = xstar_cpp.to_cuda(mask_cpu)
    out_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask_cuda)

    expected = cpp_to_torch(out_cpu, [seq, num_heads * head_dim])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, num_heads * head_dim])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.allclose(
        cuda,
        expected,
        rtol=1e-2,
        atol=2e-2,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# bf16 + seq>hd, 验证 bf16 下 qk buffer 大小按 seq(seq=8)不是 hd, 越界探针
def test_cuda_attention_bf16_seq_gt_head_dim():
    num_heads = 4
    kv = 2
    seq = 8
    head_dim = 4

    Q = torch.randn(num_heads, seq, head_dim, dtype=torch.bfloat16)
    K = torch.randn(kv, seq, head_dim, dtype=torch.bfloat16)
    V = torch.randn(kv, seq, head_dim, dtype=torch.bfloat16)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    out_cpu = xstar_cpp.attention(Q_cpu, K_cpu, V_cpu, mask)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    out_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask)

    expected = cpp_to_torch(out_cpu, [seq, num_heads * head_dim])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, num_heads * head_dim])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert not torch.isnan(cuda).any()
    assert torch.allclose(
        cuda,
        expected,
        rtol=1e-2,
        atol=1e-2,
    ), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 全 -inf mask → softmax 0/0 = NaN, 钉死"外部 mask 藏掉对角线 → NaN 是 caller 契约"
def test_cuda_attention_fully_masked_row_is_nan():
    num_heads = 4
    kv = 2
    seq = 4
    head_dim = 8

    Q = torch.randn(num_heads, seq, head_dim)
    K = torch.randn(kv, seq, head_dim)
    V = torch.randn(kv, seq, head_dim)
    mask = torch.full((seq, seq), float("-inf"))
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    mask_cpu = torch_to_cpp(mask)
    out_cpu = xstar_cpp.attention(Q_cpu, K_cpu, V_cpu, mask_cpu)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    mask_cuda = xstar_cpp.to_cuda(mask_cpu)
    out_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask_cuda)

    expected = cpp_to_torch(out_cpu, [seq, num_heads * head_dim])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, num_heads * head_dim])

    assert torch.isnan(cuda).all() and torch.isnan(expected).all()

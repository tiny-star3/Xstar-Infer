import pytest
import sys
import torch

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 单头 / 无 mask / seq=128(两块)/ f32
def test_cuda_fa2_f32_multi_block_causal():
    num_heads = 1
    num_kv_heads = 1
    seq = 128
    head_dim = 64
    dtype = torch.float32

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    ref_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask)
    fa2_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask)

    ref = cpp_to_torch(xstar_cpp.to_cpu(ref_cuda), [seq, num_heads * head_dim])
    fa2 = cpp_to_torch(xstar_cpp.to_cpu(fa2_cuda), [seq, num_heads * head_dim])

    diff = (fa2 - ref).abs().max()
    print(diff)
    assert torch.allclose(
        fa2,
        ref,
        atol=1e-6,
    ), f"fa2={fa2} ref={ref}"


# seq=100,验 key+query 边界 guard + store guard
def test_cuda_fa2_f32_seq_not_divisible():
    num_heads = 1
    num_kv_heads = 1
    seq = 100
    head_dim = 64
    dtype = torch.float32

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    ref_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask)
    fa2_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask)

    ref = cpp_to_torch(xstar_cpp.to_cpu(ref_cuda), [seq, num_heads * head_dim])
    fa2 = cpp_to_torch(xstar_cpp.to_cpu(fa2_cuda), [seq, num_heads * head_dim])

    diff = (fa2 - ref).abs().max()
    print(diff)
    assert torch.allclose(
        fa2,
        ref,
        atol=1e-6,
    ), f"fa2={fa2} ref={ref}"


# nh=4 nkv=2 seq=128,验 kv_head 索引
def test_cuda_fa2_f32_gqa_multi_block():
    num_heads = 4
    num_kv_heads = 2
    seq = 128
    head_dim = 64
    dtype = torch.float32

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    ref_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask)
    fa2_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask)

    ref = cpp_to_torch(xstar_cpp.to_cpu(ref_cuda), [seq, num_heads * head_dim])
    fa2 = cpp_to_torch(xstar_cpp.to_cpu(fa2_cuda), [seq, num_heads * head_dim])

    diff = (fa2 - ref).abs().max()
    print(diff)
    assert torch.allclose(
        fa2,
        ref,
        atol=1e-6,
    ), f"fa2={fa2} ref={ref}"


# mask=randn, 验 if 分支(不叠 causal)
def test_cuda_fa2_f32_additive_mask():
    num_heads = 1
    num_kv_heads = 1
    seq = 128
    head_dim = 64
    dtype = torch.float32

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = torch.randn(seq, seq, dtype=dtype)
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    mask_cpu = torch_to_cpp(mask)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    mask_cuda = xstar_cpp.to_cuda(mask_cpu)
    ref_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask_cuda)
    fa2_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask_cuda)

    ref = cpp_to_torch(xstar_cpp.to_cpu(ref_cuda), [seq, num_heads * head_dim])
    fa2 = cpp_to_torch(xstar_cpp.to_cpu(fa2_cuda), [seq, num_heads * head_dim])

    diff = (fa2 - ref).abs().max()
    print(diff)
    assert torch.allclose(
        fa2,
        ref,
        atol=1e-6,
    ), f"fa2={fa2} ref={ref}"


# mask 对角下三角放 -inf,验 mask 真应用非退回 causal
def test_cuda_fa2_f32_additive_mask_inf():
    num_heads = 1
    num_kv_heads = 1
    seq = 128
    head_dim = 64
    dtype = torch.float32

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = torch.zeros(seq, seq, dtype=dtype)
    # -inf 落在对角线下方, causal 永不藏下三角
    mask[2, 0] = float("-inf")
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    mask_cpu = torch_to_cpp(mask)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    mask_cuda = xstar_cpp.to_cuda(mask_cpu)
    ref_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask_cuda)
    fa2_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask_cuda)

    ref = cpp_to_torch(xstar_cpp.to_cpu(ref_cuda), [seq, num_heads * head_dim])
    fa2 = cpp_to_torch(xstar_cpp.to_cpu(fa2_cuda), [seq, num_heads * head_dim])

    diff = (fa2 - ref).abs().max()
    print(diff)
    assert torch.allclose(
        fa2,
        ref,
        atol=1e-6,
    ), f"fa2={fa2} ref={ref}"


# bf16 + seq=128,两 dtype
def test_cuda_fa2_bf16_multi_block_causal():
    num_heads = 1
    num_kv_heads = 1
    seq = 128
    head_dim = 64
    dtype = torch.bfloat16

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    ref_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask)
    fa2_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask)

    ref = cpp_to_torch(xstar_cpp.to_cpu(ref_cuda), [seq, num_heads * head_dim])
    fa2 = cpp_to_torch(xstar_cpp.to_cpu(fa2_cuda), [seq, num_heads * head_dim])

    diff = (fa2 - ref).abs().max()
    print(diff)
    assert torch.allclose(
        fa2,
        ref,
        rtol=1e-2,
        atol=2e-2,
    ), f"fa2={fa2} ref={ref}"


# 唯一可能进偏差窗口的(FA2 f32+mask vs M5 bf16 两次 downcast)
def test_cuda_fa2_bf16_additive_mask():
    num_heads = 1
    num_kv_heads = 1
    seq = 128
    head_dim = 64
    dtype = torch.bfloat16

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = torch.randn(seq, seq, dtype=dtype)
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    mask_cpu = torch_to_cpp(mask)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    mask_cuda = xstar_cpp.to_cuda(mask_cpu)
    ref_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask_cuda)
    fa2_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask_cuda)

    ref = cpp_to_torch(xstar_cpp.to_cpu(ref_cuda), [seq, num_heads * head_dim])
    fa2 = cpp_to_torch(xstar_cpp.to_cpu(fa2_cuda), [seq, num_heads * head_dim])

    diff = (fa2 - ref).abs().max()
    print(diff)
    assert torch.allclose(
        fa2,
        ref,
        rtol=1e-2,
        atol=3e-2,
    ), f"fa2={fa2} ref={ref}"


# decode 免 mask 分支, 多块
def test_cuda_fa2_decode_f32_multi_block():
    num_heads = 1
    num_kv_heads = 1
    seq = 128
    head_dim = 64
    dtype = torch.float32

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    Q_decode = Q[:, -1:, :]
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    Q_decode_cpu = torch_to_cpp(Q_decode)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    Q_decode_cuda = xstar_cpp.to_cuda(Q_decode_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    prefill_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask)
    decode_cuda = xstar_cpp.attention_fa2(Q_decode_cuda, K_cuda, V_cuda, mask)

    prefill = cpp_to_torch(xstar_cpp.to_cpu(prefill_cuda), [seq, num_heads * head_dim])
    decode = cpp_to_torch(xstar_cpp.to_cpu(decode_cuda), [1, num_heads * head_dim])

    diff = (decode - prefill[-1:, :]).abs().max()
    print(diff)
    assert torch.allclose(
        decode,
        prefill[-1:, :],
        atol=1e-6,
    ), f"decode={decode} prefill={prefill[-1:, :]}"


# 末块 key 边界 (key 64..99 界内、100..127 OOB)
def test_cuda_fa2_decode_f32_boundary():
    num_heads = 1
    num_kv_heads = 1
    seq = 100
    head_dim = 64
    dtype = torch.float32

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    Q_decode = Q[:, -1:, :]
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    Q_decode_cpu = torch_to_cpp(Q_decode)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    Q_decode_cuda = xstar_cpp.to_cuda(Q_decode_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    prefill_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask)
    decode_cuda = xstar_cpp.attention_fa2(Q_decode_cuda, K_cuda, V_cuda, mask)

    prefill = cpp_to_torch(xstar_cpp.to_cpu(prefill_cuda), [seq, num_heads * head_dim])
    decode = cpp_to_torch(xstar_cpp.to_cpu(decode_cuda), [1, num_heads * head_dim])

    diff = (decode - prefill[-1:, :]).abs().max()
    print(diff)
    assert torch.allclose(
        decode,
        prefill[-1:, :],
        atol=1e-6,
    ), f"decode={decode} prefill={prefill[-1:, :]}"


# decode 下 kv_head 索引
def test_cuda_fa2_decode_f32_gqa():
    num_heads = 4
    num_kv_heads = 2
    seq = 128
    head_dim = 64
    dtype = torch.float32

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    Q_decode = Q[:, -1:, :]
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    Q_decode_cpu = torch_to_cpp(Q_decode.contiguous())
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    Q_decode_cuda = xstar_cpp.to_cuda(Q_decode_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    prefill_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask)
    decode_cuda = xstar_cpp.attention_fa2(Q_decode_cuda, K_cuda, V_cuda, mask)

    prefill = cpp_to_torch(xstar_cpp.to_cpu(prefill_cuda), [seq, num_heads * head_dim])
    decode = cpp_to_torch(xstar_cpp.to_cpu(decode_cuda), [1, num_heads * head_dim])

    diff = (decode - prefill[-1:, :]).abs().max()
    print(diff)
    assert torch.allclose(
        decode,
        prefill[-1:, :],
        atol=1e-6,
    ), f"decode={decode} prefill={prefill[-1:, :]}"


# bf16 decode
def test_cuda_fa2_decode_bf16():
    num_heads = 1
    num_kv_heads = 1
    seq = 128
    head_dim = 64
    dtype = torch.bfloat16

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    Q_decode = Q[:, -1:, :]
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    Q_decode_cpu = torch_to_cpp(Q_decode)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    Q_decode_cuda = xstar_cpp.to_cuda(Q_decode_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    prefill_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask)
    decode_cuda = xstar_cpp.attention_fa2(Q_decode_cuda, K_cuda, V_cuda, mask)

    prefill = cpp_to_torch(xstar_cpp.to_cpu(prefill_cuda), [seq, num_heads * head_dim])
    decode = cpp_to_torch(xstar_cpp.to_cpu(decode_cuda), [1, num_heads * head_dim])

    diff = (decode - prefill[-1:, :]).abs().max()
    print(diff)
    assert torch.allclose(
        decode,
        prefill[-1:, :],
        atol=1e-6,
    ), f"decode={decode} prefill={prefill[-1:, :]}"


# long-seq, qb=0..3 多块 skip
def test_cuda_fa2_f32_long_seq_causal():
    num_heads = 1
    num_kv_heads = 1
    seq = 256
    head_dim = 64
    dtype = torch.float32

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    ref_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask)
    fa2_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask)

    ref = cpp_to_torch(xstar_cpp.to_cpu(ref_cuda), [seq, num_heads * head_dim])
    fa2 = cpp_to_torch(xstar_cpp.to_cpu(fa2_cuda), [seq, num_heads * head_dim])

    diff = (fa2 - ref).abs().max()
    print(diff)
    assert torch.allclose(
        fa2,
        ref,
        atol=1e-6,
    ), f"fa2={fa2} ref={ref}"


# long-seq, GQA
def test_cuda_fa2_f32_gqa_long_seq_causal():
    num_heads = 4
    num_kv_heads = 2
    seq = 256
    head_dim = 64
    dtype = torch.float32

    Q = torch.randn(num_heads, seq, head_dim, dtype=dtype)
    K = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    V = torch.randn(num_kv_heads, seq, head_dim, dtype=dtype)
    mask = None
    Q_cpu = torch_to_cpp(Q)
    K_cpu = torch_to_cpp(K)
    V_cpu = torch_to_cpp(V)
    Q_cuda = xstar_cpp.to_cuda(Q_cpu)
    K_cuda = xstar_cpp.to_cuda(K_cpu)
    V_cuda = xstar_cpp.to_cuda(V_cpu)
    ref_cuda = xstar_cpp.attention(Q_cuda, K_cuda, V_cuda, mask)
    fa2_cuda = xstar_cpp.attention_fa2(Q_cuda, K_cuda, V_cuda, mask)

    ref = cpp_to_torch(xstar_cpp.to_cpu(ref_cuda), [seq, num_heads * head_dim])
    fa2 = cpp_to_torch(xstar_cpp.to_cpu(fa2_cuda), [seq, num_heads * head_dim])

    diff = (fa2 - ref).abs().max()
    print(diff)
    assert torch.allclose(
        fa2,
        ref,
        atol=1e-6,
    ), f"fa2={fa2} ref={ref}"

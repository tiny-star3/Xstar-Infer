import pytest
import sys
import torch
import numpy as np

from tests.bridge import torch_to_cpp, cpp_to_torch

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp


# 真实模型方向(Qwen2.5 hidden=896)+ 896 不整除 256
def test_cuda_embedding_f32_hidden_896():
    vocab = 20
    hidden = 896
    seq = 5

    ids = np.random.randint(0, vocab, (seq,))
    weight = torch.randn(vocab, hidden)
    weight_cpu = torch_to_cpp(weight)
    out_cpu = xstar_cpp.embedding(weight_cpu, ids)
    weight_cuda = xstar_cpp.to_cuda(weight_cpu)
    out_cuda = xstar_cpp.embedding(weight_cuda, ids)

    expected = cpp_to_torch(out_cpu, [seq, hidden])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, hidden])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.equal(cuda, expected), f"cpp_cuda={cuda} cpp_cpu={expected}"


# bf16 字节搬移 bit-exact
def test_cuda_embedding_bf16_hidden_896():
    vocab = 20
    hidden = 896
    seq = 5

    ids = np.random.randint(0, vocab, (seq,))
    weight = torch.randn(vocab, hidden, dtype=torch.bfloat16)
    weight_cpu = torch_to_cpp(weight)
    out_cpu = xstar_cpp.embedding(weight_cpu, ids)
    weight_cuda = xstar_cpp.to_cuda(weight_cpu)
    out_cuda = xstar_cpp.embedding(weight_cuda, ids)

    expected = cpp_to_torch(out_cpu, [seq, hidden])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, hidden])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.equal(cuda, expected), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 四个不同的边界 regime
# hidden=257: col=ceil(257/256)=2, 257%2=1≠0 → 尾线程截断
# hidden=4:col=1, threads 0-3 各搬 1, threads 4-255 全 idle(252 个 idle)
# hidden=255: col=1, threads 0-254 活, thread 255 idle(1 个 idle)
# hidden=512:col=2, 512%2=0 且 512=256*2 → 无 idle 无截断, thread 255 tx*col=510 搬 2 正好
def test_cuda_embedding_boundary_hidden():
    vocab = 20
    hiddens = [257, 4, 255, 512]
    seq = 5

    for hidden in hiddens:
        ids = np.random.randint(0, vocab, (seq,))
        weight = torch.randn(vocab, hidden, dtype=torch.bfloat16)
        weight_cpu = torch_to_cpp(weight)
        out_cpu = xstar_cpp.embedding(weight_cpu, ids)
        weight_cuda = xstar_cpp.to_cuda(weight_cpu)
        out_cuda = xstar_cpp.embedding(weight_cuda, ids)

        expected = cpp_to_torch(out_cpu, [seq, hidden])
        cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, hidden])

        diff = (cuda - expected).abs().max().item()
        print(diff)
        assert torch.equal(cuda, expected), f"cpp_cuda={cuda} cpp_cpu={expected}"


# ids_shape 多维
def test_cuda_embedding_ids_2d():
    vocab = 15
    hidden = 64
    seq = 5

    ids = np.random.randint(0, vocab, (2, seq))
    weight = torch.randn(vocab, hidden, dtype=torch.bfloat16)
    weight_cpu = torch_to_cpp(weight)
    out_cpu = xstar_cpp.embedding(weight_cpu, ids)
    weight_cuda = xstar_cpp.to_cuda(weight_cpu)
    out_cuda = xstar_cpp.embedding(weight_cuda, ids)

    expected = cpp_to_torch(out_cpu, [2, seq, hidden])
    cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [2, seq, hidden])

    diff = (cuda - expected).abs().max().item()
    print(diff)
    assert torch.equal(cuda, expected), f"cpp_cuda={cuda} cpp_cpu={expected}"


# 越界检查在 memcpy 之前(host 侧)
# 用 10**9 而非 10:OOB 偏移必崩 illegal address(CHECK_CUDA 抛的也是 RuntimeError), 靠 match 区分"host 先拦"vs"kernel 先崩"; id=10 读 padding 静默, 两种顺序都抛 RuntimeError, 测不出顺序
def test_cuda_embedding_out_of_range():
    vocab = 10
    hidden = 8

    # 高越界
    ids = np.array([0, 10**9, 2], dtype=np.int64)
    weight = torch.randn(vocab, hidden)
    weight_cpu = torch_to_cpp(weight)
    weight_cuda = xstar_cpp.to_cuda(weight_cpu)

    with pytest.raises(RuntimeError, match="out-of-range index"):
        xstar_cpp.embedding(weight_cuda, ids)

    # 负越界
    ids = np.array([0, -(10**9), 2], dtype=np.int64)

    with pytest.raises(RuntimeError, match="out-of-range index"):
        xstar_cpp.embedding(weight_cuda, ids)


# d_ids 每次 alloc 必须每次 free
def test_cuda_embedding_no_leak():
    vocab = 20
    hidden = 896
    seq = 2000

    ids = np.random.randint(0, vocab, (seq,))
    weight = torch.randn(vocab, hidden)
    weight_cpu = torch_to_cpp(weight)
    out_cpu = xstar_cpp.embedding(weight_cpu, ids)
    expected = cpp_to_torch(out_cpu, [seq, hidden])
    weight_cuda = xstar_cpp.to_cuda(weight_cpu)

    free0 = xstar_cpp.cuda_free_bytes()

    for _ in range(100):
        out_cuda = xstar_cpp.embedding(weight_cuda, ids)

        cuda = cpp_to_torch(xstar_cpp.to_cpu(out_cuda), [seq, hidden])

        diff = (cuda - expected).abs().max().item()
        print(diff)
        assert torch.equal(cuda, expected), f"cpp_cuda={cuda} cpp_cpu={expected}"

        del out_cuda, cuda

    free1 = xstar_cpp.cuda_free_bytes()

    assert free1 >= free0 - 1024 * 1024

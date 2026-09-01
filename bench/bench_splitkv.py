import pytest
import sys
import torch
import os
import math

from tests.bridge import torch_to_cpp, cpp_to_torch
from tests.harness.oracle_qwen2 import load_reference_model

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp

# 本地模型路径
qwen2_model_path = "~/models/Qwen2.5-0.5B"
qwen2_model_path = os.path.expanduser(qwen2_model_path)

CONTEXTS = [512, 2048, 8192, 32000]
BATCHES = [1, 4]
SPLITS = [1, 2, 4, 8, 16]  # 1 = 非 split 基线（num_splits=1 强制走 else）
J = 10  # 计时 decode 步数
WARMUP = 3


def setup():
    py_model, _ = load_reference_model(qwen2_model_path, device="cuda")
    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    rope_cache = py_model.model.positional_encoder._freq_cis_cache
    rope_cache_cpu = torch_to_cpp(rope_cache.cpu())
    rope_cache_cuda = xstar_cpp.to_cuda(rope_cache_cpu)

    return w, cfg, rope_cache_cuda


def bench_cell(w, cfg, rope_cache_cuda, batch, context, S):
    # 建独立 pool: num_blocks = batch * math.ceil((context + J) / 16)（+1 余量）
    block_size = 16
    nkv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    dtype_size = 2
    kv_slot_bytes = nkv * head_dim * dtype_size * 2
    num_layers = cfg.num_hidden_layers
    num_blocks = batch * math.ceil((context + WARMUP + J) / block_size) + 1
    max_seq_len = cfg.max_position_embeddings

    device = xstar_cpp.Device.CUDA
    dtype = xstar_cpp.DType.BFloat16

    bm = xstar_cpp.BlockManager(
        num_blocks, block_size, kv_slot_bytes, device, num_layers
    )
    # 建 batch 个 PagedKVCache
    kv = [
        xstar_cpp.PagedKVCache(nkv, head_dim, max_seq_len, block_size, dtype, device)
        for _ in range(batch)
    ]
    # dummy prefill: segs = [arange(offset_i, offset_i+context)], num_splits=-1 (prefill 不碰 split, -1 自动政策无所谓)
    input_ids = torch.arange(batch * context)
    cu_seqlens_q_host = [i * context for i in range(batch + 1)]
    multi_prefill_cuda = xstar_cpp.qwen2_forward_multi(
        w,
        cfg,
        rope_cache_cuda,
        bm,
        kv,
        False,
        input_ids.numpy(),
        cu_seqlens_q_host,
    )
    # warmup: WARMUP 步 decode(num_splits=S, 固定 next_ids)
    next_ids = torch.full((batch,), 666)
    decode_cu_seqlens_q = list(range(batch + 1))
    for _ in range(WARMUP):
        xstar_cpp.qwen2_forward_multi(
            w,
            cfg,
            rope_cache_cuda,
            bm,
            kv,
            True,
            next_ids.numpy(),
            decode_cu_seqlens_q,
            num_splits=S,
        )
    # torch.cuda.Event 计时 J 步 decode, 返回 per_step_ms + 最后一步 logits
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    logits_cuda = None
    for _ in range(J):
        logits_cuda = xstar_cpp.qwen2_forward_multi(
            w,
            cfg,
            rope_cache_cuda,
            bm,
            kv,
            True,
            next_ids.numpy(),
            decode_cu_seqlens_q,
            num_splits=S,
        )
    end.record()
    torch.cuda.synchronize()
    per_step_ms = start.elapsed_time(end) / J
    logits = cpp_to_torch(xstar_cpp.to_cpu(logits_cuda), [batch, cfg.vocab_size])
    return per_step_ms, logits


def main():
    w, cfg, rope_cache_cuda = setup()
    for batch in BATCHES:
        for context in CONTEXTS:
            base_ms, base_logits = bench_cell(
                w, cfg, rope_cache_cuda, batch, context, S=1
            )
            row = {}
            for S in SPLITS[1:]:
                ms, logits = bench_cell(w, cfg, rope_cache_cuda, batch, context, S)
                assert torch.equal(
                    logits.argmax(-1), base_logits.argmax(-1)
                )  # 每格对拍
                row[S] = ms
            row[1] = base_ms
            # 输出 speedup = base_ms / row[S], 打印表
            print(
                f"batch={batch} context={context}: "
                + "  ".join(
                    f"S={s}:{row[s]:.3f}ms(speedup {base_ms/row[s]:.2f}x)" for s in row
                )
            )

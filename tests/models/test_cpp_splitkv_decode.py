import pytest
import sys
import torch
import os
import math

from tests.bridge import torch_to_cpp, cpp_to_torch
from tests.harness.oracle_qwen2 import load_reference_model, reference

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp

# 本地模型路径
qwen2_model_path = "~/models/Qwen2.5-0.5B"
qwen2_model_path = os.path.expanduser(qwen2_model_path)


# 单请求 vs 多请求 decode 对拍, 多请求长 context 触发 split 路径, 单请求永远不走 split
def test_qwen2_forward_multi_matches_single_splitkv_decode():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    rope_cache = py_model.model.positional_encoder._freq_cis_cache
    rope_cache_cpu = torch_to_cpp(rope_cache.cpu())
    rope_cache_cuda = xstar_cpp.to_cuda(rope_cache_cpu)
    dtype = xstar_cpp.DType.BFloat16
    device = xstar_cpp.Device.CUDA

    lens = [7, 17, 50, 600]
    segs = [
        torch.arange(101, 101 + 7),
        torch.arange(201, 201 + 17),
        torch.arange(301, 301 + 50),
        torch.arange(401, 401 + 600),
    ]
    input_ids_concat = torch.cat(segs)
    cu_seqlens_q_host = [0, 7, 24, 74, 674]

    block_size = 16
    nkv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    dtype_size = 2
    kv_slot_bytes = nkv * head_dim * dtype_size * 2
    num_layers = cfg.num_hidden_layers
    max_seq_len = cfg.max_position_embeddings
    num_blocks = math.ceil(max_seq_len / block_size)
    mask = None

    bm_multi = xstar_cpp.BlockManager(
        num_blocks, block_size, kv_slot_bytes, device, num_layers
    )
    kv_multi = [
        xstar_cpp.PagedKVCache(nkv, head_dim, max_seq_len, block_size, dtype, device)
        for _ in range(4)
    ]

    bm_single = [
        xstar_cpp.BlockManager(
            num_blocks, block_size, kv_slot_bytes, device, num_layers
        )
        for _ in range(4)
    ]
    kv_single = [
        xstar_cpp.PagedKVCache(nkv, head_dim, max_seq_len, block_size, dtype, device)
        for _ in range(4)
    ]

    multi_prefill_cuda = xstar_cpp.qwen2_forward_multi(
        w,
        cfg,
        rope_cache_cuda,
        bm_multi,
        kv_multi,
        False,
        input_ids_concat.numpy(),
        cu_seqlens_q_host,
    )
    multi_prefill = cpp_to_torch(
        xstar_cpp.to_cpu(multi_prefill_cuda), [len(input_ids_concat), cfg.vocab_size]
    )

    single_prefill_cuda = [
        xstar_cpp.qwen2_forward_paged(
            w,
            cfg,
            rope_cache_cuda,
            bm_single[i],
            kv_single[i],
            False,
            segs[i].numpy(),
            mask,
        )
        for i in range(4)
    ]
    single_prefill = torch.cat(
        [
            cpp_to_torch(
                xstar_cpp.to_cpu(single_logit_cuda), [len(segs[i]), cfg.vocab_size]
            )
            for i, single_logit_cuda in enumerate(single_prefill_cuda)
        ],
        dim=0,
    )

    assert torch.equal(
        multi_prefill, single_prefill
    ), f"multi_prefill={multi_prefill} single_prefill={single_prefill}"

    # 固定值 decode 一轮
    next_ids = torch.tensor([108, 218, 351, 471])

    multi_decode_cuda = xstar_cpp.qwen2_forward_multi(
        w,
        cfg,
        rope_cache_cuda,
        bm_multi,
        kv_multi,
        True,
        next_ids.numpy(),
        [0, 1, 2, 3, 4],
    )
    multi_decode = cpp_to_torch(
        xstar_cpp.to_cpu(multi_decode_cuda), [len(next_ids), cfg.vocab_size]
    )

    single_decode_cuda = [
        xstar_cpp.qwen2_forward_paged(
            w,
            cfg,
            rope_cache_cuda,
            bm_single[i],
            kv_single[i],
            True,
            next_ids[i].reshape(1).numpy(),
            mask,
        )
        for i in range(4)
    ]
    single_decode = torch.cat(
        [
            cpp_to_torch(
                xstar_cpp.to_cpu(single_logit_cuda),
                [len(next_ids[i].reshape(1)), cfg.vocab_size],
            )
            for i, single_logit_cuda in enumerate(single_decode_cuda)
        ],
        dim=0,
    )

    max_diff = (multi_decode.float() - single_decode.float().cpu()).abs().max()
    print(max_diff)
    assert torch.equal(multi_decode.argmax(-1), single_decode.argmax(-1))

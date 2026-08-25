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


# prefill 对拍(paged prefill logits == kvcache prefill logits)
def test_paged_prefill_matches_kv_cache():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    dtype = xstar_cpp.DType.BFloat16
    device = xstar_cpp.Device.CUDA
    kv = xstar_cpp.KVCache(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.max_position_embeddings,
        cfg.hidden_size // cfg.num_attention_heads,
        dtype,
        device,
    )
    rope_cache = py_model.model.positional_encoder._freq_cis_cache
    rope_cache_cpu = torch_to_cpp(rope_cache.cpu())
    rope_cache_cuda = xstar_cpp.to_cuda(rope_cache_cpu)

    block_size = 16
    nkv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    dtype_size = 2
    kv_slot_bytes = nkv * head_dim * dtype_size * 2
    num_layers = cfg.num_hidden_layers
    max_seq_len = cfg.max_position_embeddings
    num_blocks = math.ceil(max_seq_len / block_size)
    bm = xstar_cpp.BlockManager(
        num_blocks, block_size, kv_slot_bytes, device, num_layers
    )
    paged_kv = xstar_cpp.PagedKVCache(
        nkv, head_dim, max_seq_len, block_size, dtype, device
    )

    prompt = "你好，你是谁？"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.squeeze()
    mask = None

    kv_cache_prefill_logits_cuda = xstar_cpp.qwen2_forward_incremental(
        w, cfg, rope_cache_cuda, kv, False, input_ids.numpy(), mask
    )
    paged_prefill_logits_cuda = xstar_cpp.qwen2_forward_paged(
        w, cfg, rope_cache_cuda, bm, paged_kv, False, input_ids.numpy(), mask
    )
    kv_cache_prefill_logits = cpp_to_torch(
        xstar_cpp.to_cpu(kv_cache_prefill_logits_cuda), [len(input_ids), cfg.vocab_size]
    )
    paged_prefill_logits = cpp_to_torch(
        xstar_cpp.to_cpu(paged_prefill_logits_cuda), [len(input_ids), cfg.vocab_size]
    )
    max_diff = (
        (paged_prefill_logits.float() - kv_cache_prefill_logits.float()).abs().max()
    )
    print(max_diff)
    assert torch.equal(
        paged_prefill_logits, kv_cache_prefill_logits
    ), f"paged_prefill_logits={paged_prefill_logits} kv_cache_prefill_logits={kv_cache_prefill_logits}"


# decode 1 步对拍
def test_paged_decode_one_step_matches_kv_cache():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    dtype = xstar_cpp.DType.BFloat16
    device = xstar_cpp.Device.CUDA
    kv = xstar_cpp.KVCache(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.max_position_embeddings,
        cfg.hidden_size // cfg.num_attention_heads,
        dtype,
        device,
    )
    rope_cache = py_model.model.positional_encoder._freq_cis_cache
    rope_cache_cpu = torch_to_cpp(rope_cache.cpu())
    rope_cache_cuda = xstar_cpp.to_cuda(rope_cache_cpu)

    block_size = 16
    nkv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    dtype_size = 2
    kv_slot_bytes = nkv * head_dim * dtype_size * 2
    num_layers = cfg.num_hidden_layers
    max_seq_len = cfg.max_position_embeddings
    num_blocks = math.ceil(max_seq_len / block_size)
    bm = xstar_cpp.BlockManager(
        num_blocks, block_size, kv_slot_bytes, device, num_layers
    )
    paged_kv = xstar_cpp.PagedKVCache(
        nkv, head_dim, max_seq_len, block_size, dtype, device
    )

    prompt = "你好，你是谁？"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.squeeze()
    mask = None

    kv_cache_prefill_logits_cuda = xstar_cpp.qwen2_forward_incremental(
        w, cfg, rope_cache_cuda, kv, False, input_ids.numpy(), mask
    )
    paged_prefill_logits_cuda = xstar_cpp.qwen2_forward_paged(
        w, cfg, rope_cache_cuda, bm, paged_kv, False, input_ids.numpy(), mask
    )
    kv_cache_prefill_logits = cpp_to_torch(
        xstar_cpp.to_cpu(kv_cache_prefill_logits_cuda), [len(input_ids), cfg.vocab_size]
    )

    kv_cache_next_id = kv_cache_prefill_logits[-1].argmax(-1, keepdim=True)

    kv_cache_decode_logits_cuda = xstar_cpp.qwen2_forward_incremental(
        w, cfg, rope_cache_cuda, kv, True, kv_cache_next_id.numpy(), mask
    )
    paged_decode_logits_cuda = xstar_cpp.qwen2_forward_paged(
        w, cfg, rope_cache_cuda, bm, paged_kv, True, kv_cache_next_id.numpy(), mask
    )
    kv_cache_decode_logits = cpp_to_torch(
        xstar_cpp.to_cpu(kv_cache_decode_logits_cuda),
        [len(kv_cache_next_id), cfg.vocab_size],
    )
    paged_decode_logits = cpp_to_torch(
        xstar_cpp.to_cpu(paged_decode_logits_cuda),
        [len(kv_cache_next_id), cfg.vocab_size],
    )

    max_diff = (
        (paged_decode_logits.float() - kv_cache_decode_logits[-1:, :].float())
        .abs()
        .max()
    )
    print(max_diff)
    assert torch.equal(
        paged_decode_logits, kv_cache_decode_logits[-1:, :]
    ), f"paged_decode_logits={paged_decode_logits} kv_cache_decode_logits={kv_cache_decode_logits[-1:, :]}"


# 多步 greedy 整链(钝感 prompt, max_diff 量级 + 敏感 prompt argmax match)
# cursor 不变量
def test_paged_greedy_matches_kv_cache():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    rope_cache = py_model.model.positional_encoder._freq_cis_cache
    rope_cache_cpu = torch_to_cpp(rope_cache.cpu())
    rope_cache_cuda = xstar_cpp.to_cuda(rope_cache_cpu)
    dtype = xstar_cpp.DType.BFloat16
    device = xstar_cpp.Device.CUDA

    prompts = ["你好，你是谁？", "The capital of France is"]
    for prompt in prompts:
        kv = xstar_cpp.KVCache(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.max_position_embeddings,
            cfg.hidden_size // cfg.num_attention_heads,
            dtype,
            device,
        )

        block_size = 16
        nkv = cfg.num_key_value_heads
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        dtype_size = 2
        kv_slot_bytes = nkv * head_dim * dtype_size * 2
        num_layers = cfg.num_hidden_layers
        max_seq_len = cfg.max_position_embeddings
        num_blocks = math.ceil(max_seq_len / block_size)
        bm = xstar_cpp.BlockManager(
            num_blocks, block_size, kv_slot_bytes, device, num_layers
        )
        paged_kv = xstar_cpp.PagedKVCache(
            nkv, head_dim, max_seq_len, block_size, dtype, device
        )

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.squeeze()
        mask = None

        kv_cache_prefill_logits_cuda = xstar_cpp.qwen2_forward_incremental(
            w, cfg, rope_cache_cuda, kv, False, input_ids.numpy(), mask
        )
        paged_prefill_logits_cuda = xstar_cpp.qwen2_forward_paged(
            w, cfg, rope_cache_cuda, bm, paged_kv, False, input_ids.numpy(), mask
        )
        kv_cache_prefill_logits = cpp_to_torch(
            xstar_cpp.to_cpu(kv_cache_prefill_logits_cuda),
            [len(input_ids), cfg.vocab_size],
        )

        kv_cache_next_id = kv_cache_prefill_logits[-1].argmax(-1, keepdim=True)

        # greedy 自回归 20 步, 对比
        steps = 20
        for i in range(1, steps + 1):
            kv_cache_decode_logits_cuda = xstar_cpp.qwen2_forward_incremental(
                w, cfg, rope_cache_cuda, kv, True, kv_cache_next_id.numpy(), mask
            )
            paged_decode_logits_cuda = xstar_cpp.qwen2_forward_paged(
                w,
                cfg,
                rope_cache_cuda,
                bm,
                paged_kv,
                True,
                kv_cache_next_id.numpy(),
                mask,
            )
            assert kv.cursor() == len(input_ids) + i
            assert paged_kv.cursor() == len(input_ids) + i
            assert len(paged_kv.block_table()) == math.ceil((len(input_ids) + i) / 16)
            assert bm.num_allocated() == math.ceil((len(input_ids) + i) / 16)
            kv_cache_decode_logits = cpp_to_torch(
                xstar_cpp.to_cpu(kv_cache_decode_logits_cuda),
                [len(kv_cache_next_id), cfg.vocab_size],
            )
            paged_decode_logits = cpp_to_torch(
                xstar_cpp.to_cpu(paged_decode_logits_cuda),
                [len(kv_cache_next_id), cfg.vocab_size],
            )

            kv_cache_next_id = kv_cache_decode_logits[-1].argmax(-1, keepdim=True)
            paged_next_id = paged_decode_logits[-1].argmax(-1, keepdim=True)

            max_diff = (paged_next_id.float() - kv_cache_next_id.float()).abs().max()
            match = (paged_next_id == kv_cache_next_id.cpu()).item()
            print(
                f"step {i} max_diff={max_diff} kv_cache_next_id={kv_cache_next_id.item()} paged_next_id={paged_next_id.item()} match={match}"
            )
            assert torch.equal(
                paged_decode_logits, kv_cache_decode_logits
            ), f"paged_decode_logits={paged_decode_logits} kv_cache_decode_logits={kv_cache_decode_logits}"


# 多步 greedy 整链(钝感 prompt, max_diff 量级 + 敏感 prompt argmax match)
# paged vs HF Qwen2.5-0.5B
def test_paged_greedy_matches_hf():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    rope_cache = py_model.model.positional_encoder._freq_cis_cache
    rope_cache_cpu = torch_to_cpp(rope_cache.cpu())
    rope_cache_cuda = xstar_cpp.to_cuda(rope_cache_cpu)
    dtype = xstar_cpp.DType.BFloat16
    device = xstar_cpp.Device.CUDA

    prompts = ["你好，你是谁？", "The capital of France is"]
    for prompt in prompts:
        block_size = 16
        nkv = cfg.num_key_value_heads
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        dtype_size = 2
        kv_slot_bytes = nkv * head_dim * dtype_size * 2
        num_layers = cfg.num_hidden_layers
        max_seq_len = cfg.max_position_embeddings
        num_blocks = math.ceil(max_seq_len / block_size)
        bm = xstar_cpp.BlockManager(
            num_blocks, block_size, kv_slot_bytes, device, num_layers
        )
        paged_kv = xstar_cpp.PagedKVCache(
            nkv, head_dim, max_seq_len, block_size, dtype, device
        )

        ref_input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        paged_input_ids = ref_input_ids.clone().squeeze()
        mask = None

        paged_prefill_logits_cuda = xstar_cpp.qwen2_forward_paged(
            w,
            cfg,
            rope_cache_cuda,
            bm,
            paged_kv,
            False,
            paged_input_ids.numpy(),
            mask,
        )
        paged_prefill_logits = cpp_to_torch(
            xstar_cpp.to_cpu(paged_prefill_logits_cuda),
            [len(paged_input_ids), cfg.vocab_size],
        )

        paged_next_id = paged_prefill_logits[-1].argmax(-1, keepdim=True)
        paged_input_ids = torch.cat([paged_input_ids, paged_next_id], dim=-1)

        ref_input_ids = ref_input_ids.to("cuda")
        ref_logits = reference("lm", ref_input_ids, ctx={"py_model": py_model})
        ref_next_id = ref_logits[:, -1].argmax(-1, keepdim=True)
        ref_input_ids = torch.cat([ref_input_ids.to("cuda"), ref_next_id], dim=-1)

        max_diff = (paged_prefill_logits.float() - ref_logits.float().cpu()).abs().max()
        match = (paged_next_id == ref_next_id.cpu()).item()
        print(
            f"prefill max_diff={max_diff} ref_next_id={ref_next_id.item()} paged_next_id={paged_next_id.item()} match={match}"
        )

        # greedy 自回归 20 步, 对比
        steps = 20
        for i in range(1, steps + 1):
            ref_logits = reference("lm", ref_input_ids, ctx={"py_model": py_model})
            ref_next_id = ref_logits[:, -1].argmax(-1, keepdim=True)
            ref_input_ids = torch.cat([ref_input_ids.to("cuda"), ref_next_id], dim=-1)

            paged_decode_logits_cuda = xstar_cpp.qwen2_forward_paged(
                w,
                cfg,
                rope_cache_cuda,
                bm,
                paged_kv,
                True,
                paged_next_id.numpy(),
                mask,
            )
            paged_decode_logits = cpp_to_torch(
                xstar_cpp.to_cpu(paged_decode_logits_cuda),
                [len(paged_next_id), cfg.vocab_size],
            )

            paged_next_id = paged_decode_logits[-1].argmax(-1, keepdim=True)
            paged_input_ids = torch.cat([paged_input_ids, paged_next_id], dim=-1)

            max_diff = (
                (paged_decode_logits.float() - ref_logits.float().cpu()).abs().max()
            )
            match = (paged_next_id == ref_next_id.cpu()).item()
            print(
                f"step {i} max_diff={max_diff} ref_next_id={ref_next_id.item()} paged_next_id={paged_next_id.item()} match={match}"
            )

        print("paged: " + tokenizer.decode(paged_input_ids))
        print("ref: " + tokenizer.decode(ref_input_ids.squeeze()))


# 测试多请求和单请求, 固定 token
def test_qwen2_forward_multi_fixed_matches_single_paged():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    rope_cache = py_model.model.positional_encoder._freq_cis_cache
    rope_cache_cpu = torch_to_cpp(rope_cache.cpu())
    rope_cache_cuda = xstar_cpp.to_cuda(rope_cache_cpu)
    dtype = xstar_cpp.DType.BFloat16
    device = xstar_cpp.Device.CUDA

    lens = [7, 17, 50, 70]
    segs = [
        torch.arange(101, 101 + 7),
        torch.arange(201, 201 + 17),
        torch.arange(301, 301 + 50),
        torch.arange(401, 401 + 70),
    ]
    input_ids_concat = torch.cat(segs)
    cu_seqlens_q_host = [0, 7, 24, 74, 144]

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


# 测试多请求和单请求, 真实 token
def test_qwen2_forward_multi_real_matches_single_paged():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    rope_cache = py_model.model.positional_encoder._freq_cis_cache
    rope_cache_cpu = torch_to_cpp(rope_cache.cpu())
    rope_cache_cuda = xstar_cpp.to_cuda(rope_cache_cpu)
    dtype = xstar_cpp.DType.BFloat16
    device = xstar_cpp.Device.CUDA

    prompts = [
        ["你好"],
        ["你好，请问你是谁？"],
        ["请详细介绍一下人工智能的发展历史，从图灵和达特茅斯会议讲起。"],
        [
            "The theory of computation is the branch that deals with how efficiently problems can be solved."
        ],
    ]
    segs = [
        tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
        for prompt in prompts
    ]
    input_ids_concat = torch.cat(segs)
    cu_seqlens_q_host = [0]
    for seg in segs:
        cu_seqlens_q_host.append(len(seg) + cu_seqlens_q_host[-1])

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

    multi_output_ids = segs.copy()
    single_output_ids = segs.copy()

    multi_decode = multi_prefill
    # 每段最后 token 的行号 = cu_seqlens_q_host[1:] - 1
    last_rows = torch.tensor(cu_seqlens_q_host[1:]) - 1
    multi_next_ids = multi_decode[last_rows].argmax(-1)

    single_decode = single_prefill
    single_next_ids = single_decode[last_rows].argmax(-1)
    # greedy 自回归 20 步, 对比
    step = 20
    for i in range(step):
        for j, next_id in enumerate(multi_next_ids):
            multi_output_ids[j] = torch.cat(
                [multi_output_ids[j], next_id.reshape(1)], dim=-1
            )
        for j, next_id in enumerate(single_next_ids):
            single_output_ids[j] = torch.cat(
                [single_output_ids[j], next_id.reshape(1)], dim=-1
            )

        multi_decode_cuda = xstar_cpp.qwen2_forward_multi(
            w,
            cfg,
            rope_cache_cuda,
            bm_multi,
            kv_multi,
            True,
            multi_next_ids.numpy(),
            [0, 1, 2, 3, 4],
        )
        multi_decode = cpp_to_torch(
            xstar_cpp.to_cpu(multi_decode_cuda), [len(multi_next_ids), cfg.vocab_size]
        )

        single_decode_cuda = [
            xstar_cpp.qwen2_forward_paged(
                w,
                cfg,
                rope_cache_cuda,
                bm_single[i],
                kv_single[i],
                True,
                single_next_ids[i].reshape(1).numpy(),
                mask,
            )
            for i in range(4)
        ]
        single_decode = torch.cat(
            [
                cpp_to_torch(
                    xstar_cpp.to_cpu(single_logit_cuda),
                    [len(single_next_ids[i].reshape(1)), cfg.vocab_size],
                )
                for i, single_logit_cuda in enumerate(single_decode_cuda)
            ],
            dim=0,
        )

        max_diff = (multi_decode.float() - single_decode.float().cpu()).abs().max()
        multi_next_ids = multi_decode.argmax(-1)
        single_next_ids = single_decode.argmax(-1)
        match = (multi_next_ids == single_next_ids).all().item()
        print(
            f"step {i} max_diff={max_diff} multi_next_ids={multi_next_ids.tolist()} single_next_ids={single_next_ids.tolist()} match={match}"
        )

    for j in range(4):
        print(f"multi  seq{j}: " + tokenizer.decode(multi_output_ids[j]))
        print(f"single seq{j}: " + tokenizer.decode(single_output_ids[j]))

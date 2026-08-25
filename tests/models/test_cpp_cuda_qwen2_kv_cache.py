import pytest
import sys
import torch
import os

from tests.bridge import torch_to_cpp, cpp_to_torch
from tests.harness.oracle_qwen2 import load_reference_model, reference

sys.path.insert(0, "xstar_cpp_py")
import xstar_cpp

# 本地模型路径
qwen2_model_path = "~/models/Qwen2.5-0.5B"
qwen2_model_path = os.path.expanduser(qwen2_model_path)


# prefill 对拍(incremental prefill logits == non-incremental)
def test_kv_cache_prefill_matches_non_incremental():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    kv = xstar_cpp.KVCache(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.max_position_embeddings,
        cfg.hidden_size // cfg.num_attention_heads,
        xstar_cpp.DType.BFloat16,
        xstar_cpp.Device.CUDA,
    )
    rope_cache = py_model.model.positional_encoder._freq_cis_cache
    rope_cache_cpu = torch_to_cpp(rope_cache.cpu())
    rope_cache_cuda = xstar_cpp.to_cuda(rope_cache_cpu)

    prompt = "你好，你是谁？"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.squeeze()
    positions = None
    mask = None

    non_inc_logits_cuda = xstar_cpp.qwen2_forward(
        w, cfg, rope_cache_cuda, input_ids.numpy(), positions, mask
    )
    inc_prefill_logits_cuda = xstar_cpp.qwen2_forward_incremental(
        w, cfg, rope_cache_cuda, kv, False, input_ids.numpy(), mask
    )
    non_inc_logits = cpp_to_torch(
        xstar_cpp.to_cpu(non_inc_logits_cuda), [len(input_ids), cfg.vocab_size]
    )
    inc_prefill_logits = cpp_to_torch(
        xstar_cpp.to_cpu(inc_prefill_logits_cuda), [len(input_ids), cfg.vocab_size]
    )
    max_diff = (inc_prefill_logits.float() - non_inc_logits.float()).abs().max()
    print(max_diff)
    assert torch.equal(
        inc_prefill_logits, non_inc_logits
    ), f"inc_prefill_logits={inc_prefill_logits} non_inc_logits={non_inc_logits}"


# decode 1 步对拍
def test_kv_cache_decode_one_step_matches_non_incremental():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    kv = xstar_cpp.KVCache(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.max_position_embeddings,
        cfg.hidden_size // cfg.num_attention_heads,
        xstar_cpp.DType.BFloat16,
        xstar_cpp.Device.CUDA,
    )
    rope_cache = py_model.model.positional_encoder._freq_cis_cache
    rope_cache_cpu = torch_to_cpp(rope_cache.cpu())
    rope_cache_cuda = xstar_cpp.to_cuda(rope_cache_cpu)

    prompt = "你好，你是谁？"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.squeeze()
    positions = None
    mask = None

    non_inc_logits_cuda = xstar_cpp.qwen2_forward(
        w, cfg, rope_cache_cuda, input_ids.numpy(), positions, mask
    )
    inc_prefill_logits_cuda = xstar_cpp.qwen2_forward_incremental(
        w, cfg, rope_cache_cuda, kv, False, input_ids.numpy(), mask
    )
    non_inc_logits = cpp_to_torch(
        xstar_cpp.to_cpu(non_inc_logits_cuda), [len(input_ids), cfg.vocab_size]
    )

    non_inc_next_id = non_inc_logits[-1].argmax(-1, keepdim=True)
    non_inc_input_id = torch.cat([input_ids, non_inc_next_id], dim=-1)

    non_inc_logits_cuda = xstar_cpp.qwen2_forward(
        w, cfg, rope_cache_cuda, non_inc_input_id.numpy(), positions, mask
    )
    inc_decode_logits_cuda = xstar_cpp.qwen2_forward_incremental(
        w, cfg, rope_cache_cuda, kv, True, non_inc_next_id.numpy(), mask
    )
    non_inc_logits = cpp_to_torch(
        xstar_cpp.to_cpu(non_inc_logits_cuda), [len(non_inc_input_id), cfg.vocab_size]
    )
    inc_decode_logits = cpp_to_torch(
        xstar_cpp.to_cpu(inc_decode_logits_cuda), [len(non_inc_next_id), cfg.vocab_size]
    )

    max_diff = (inc_decode_logits.float() - non_inc_logits[-1:, :].float()).abs().max()
    print(max_diff)
    assert torch.equal(
        inc_decode_logits, non_inc_logits[-1:, :]
    ), f"inc_decode_logits={inc_decode_logits} non_inc_logits={non_inc_logits[-1:, :]}"


# 多步 greedy 整链(钝感 prompt, max_diff 量级 + 敏感 prompt argmax match)
# cursor 不变量
def test_kv_cache_greedy_matches_non_incremental():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    rope_cache = py_model.model.positional_encoder._freq_cis_cache
    rope_cache_cpu = torch_to_cpp(rope_cache.cpu())
    rope_cache_cuda = xstar_cpp.to_cuda(rope_cache_cpu)

    prompts = ["你好，你是谁？", "The capital of France is"]
    for prompt in prompts:
        kv = xstar_cpp.KVCache(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.max_position_embeddings,
            cfg.hidden_size // cfg.num_attention_heads,
            xstar_cpp.DType.BFloat16,
            xstar_cpp.Device.CUDA,
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.squeeze()
        positions = None
        mask = None

        non_inc_logits_cuda = xstar_cpp.qwen2_forward(
            w, cfg, rope_cache_cuda, input_ids.numpy(), positions, mask
        )
        inc_prefill_logits_cuda = xstar_cpp.qwen2_forward_incremental(
            w, cfg, rope_cache_cuda, kv, False, input_ids.numpy(), mask
        )
        assert kv.cursor() == len(input_ids)
        non_inc_logits = cpp_to_torch(
            xstar_cpp.to_cpu(non_inc_logits_cuda), [len(input_ids), cfg.vocab_size]
        )
        inc_prefill_logits = cpp_to_torch(
            xstar_cpp.to_cpu(inc_prefill_logits_cuda), [len(input_ids), cfg.vocab_size]
        )

        non_inc_next_id = non_inc_logits[-1].argmax(-1, keepdim=True)
        non_inc_input_id = torch.cat([input_ids, non_inc_next_id], dim=-1)

        # greedy 自回归 20 步, 对比
        steps = 20
        for i in range(1, steps + 1):
            non_inc_logits_cuda = xstar_cpp.qwen2_forward(
                w, cfg, rope_cache_cuda, non_inc_input_id.numpy(), positions, mask
            )
            inc_decode_logits_cuda = xstar_cpp.qwen2_forward_incremental(
                w, cfg, rope_cache_cuda, kv, True, non_inc_next_id.numpy(), mask
            )
            assert kv.cursor() == len(input_ids) + i
            non_inc_logits = cpp_to_torch(
                xstar_cpp.to_cpu(non_inc_logits_cuda),
                [len(non_inc_input_id), cfg.vocab_size],
            )
            inc_decode_logits = cpp_to_torch(
                xstar_cpp.to_cpu(inc_decode_logits_cuda),
                [len(non_inc_next_id), cfg.vocab_size],
            )

            non_inc_next_id = non_inc_logits[-1].argmax(-1, keepdim=True)
            inc_decode_next_id = inc_decode_logits[-1].argmax(-1, keepdim=True)
            non_inc_input_id = torch.cat([non_inc_input_id, non_inc_next_id], dim=-1)

            max_diff = (
                (inc_decode_logits.float() - non_inc_logits[-1:, :].float()).abs().max()
            )
            match = (inc_decode_next_id == non_inc_next_id.cpu()).item()
            print(
                f"step {i} max_diff={max_diff} inc_decode_next_id={inc_decode_next_id.item()} non_inc_next_id={non_inc_next_id.item()} match={match}"
            )
            assert torch.equal(
                inc_decode_logits, non_inc_logits[-1:, :]
            ), f"inc_decode_logits={inc_decode_logits} non_inc_logits={non_inc_logits[-1:, :]}"

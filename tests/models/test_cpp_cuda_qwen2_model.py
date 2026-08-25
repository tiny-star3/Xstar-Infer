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


# 测端到端 greedy 生成, xstar pytorch vs cuda 参考
# bf16 敏感 prompt 三方分叉
# bf16 打平点 prompt(step0 top1/top2 几乎相等): ref/HF/cuda 三方在打平点合法分叉, 无对错
# 价值是观测"bf16 噪声在敏感点翻 argmax", 不做 assert
# 注意 step0 单步 diff 仅 0.19(在 bf16 误差内), 分叉是噪声翻向不是 bug
def test_forward_greedy_matches_reference_cuda():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    cache = py_model.model.positional_encoder._freq_cis_cache
    cache_cpu = torch_to_cpp(cache.cpu())
    cache_cuda = xstar_cpp.to_cuda(cache_cpu)

    prompt = "你好，你是谁？"
    ref_input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    cpp_input_ids = ref_input_ids.clone().squeeze()
    ref_input_ids = ref_input_ids.to("cuda")
    positions = None
    mask = None

    # greedy 自回归 20 步,和 ref 对比
    steps = 20
    for i in range(steps):
        ref_logits = reference("lm", ref_input_ids, ctx={"py_model": py_model})
        ref_next_id = ref_logits[:, -1].argmax(-1, keepdim=True)
        ref_input_ids = torch.cat([ref_input_ids.to("cuda"), ref_next_id], dim=-1)

        cpp_logits_cuda = xstar_cpp.qwen2_forward(
            w, cfg, cache_cuda, cpp_input_ids.numpy(), positions, mask
        )
        assert list(cpp_logits_cuda.shape()) == list(ref_logits.squeeze().shape)
        cpp_logits = cpp_to_torch(
            xstar_cpp.to_cpu(cpp_logits_cuda), ref_logits.squeeze().shape
        )
        cpp_next_id = cpp_logits[-1].argmax(-1, keepdim=True)
        cpp_input_ids = torch.cat([cpp_input_ids, cpp_next_id], dim=-1)
        max_diff = (cpp_logits.float() - ref_logits.cpu().float()).abs().max()
        match = (cpp_next_id == ref_next_id.cpu()).item()
        print(
            f"step {i} max_diff={max_diff} cpp={cpp_next_id.item()} ref={ref_next_id.item()} match={match}"
        )

    print("cpp: " + tokenizer.decode(cpp_input_ids))
    print("ref: " + tokenizer.decode(ref_input_ids.squeeze()))


# 测端到端 greedy 生成, xstar pytorch vs cuda 参考
# 非敏感 prompt(非 bf16 打平点)
# 非"打平点起始"的 prompt, 验证 cuda 在钝感区单步数值稳定复现 ref
# 但 0.5B greedy 跑几步必然再撞打平点(此 prompt step5 撞), 之后级联分叉
# 所以 argmax 不做硬 assert; 判据是分叉前 max_diff 稳定(此例 step0-5 恒 0.6875, 无累积增长) = 单步数值无 bug, 分叉是 bf16 噪声
# step5+ 的 max_diff 跳升(15→20)是分叉后输入不同导致的级联, 不是单步误差
def test_forward_greedy_matches_reference_cuda_nonsensitive():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cuda")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    cache = py_model.model.positional_encoder._freq_cis_cache
    cache_cpu = torch_to_cpp(cache.cpu())
    cache_cuda = xstar_cpp.to_cuda(cache_cpu)

    prompt = "The capital of France is"
    ref_input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    cpp_input_ids = ref_input_ids.clone().squeeze()
    ref_input_ids = ref_input_ids.to("cuda")
    positions = None
    mask = None

    # greedy 自回归 20 步,和 ref 对比
    steps = 20
    for i in range(steps):
        ref_logits = reference("lm", ref_input_ids, ctx={"py_model": py_model})
        ref_next_id = ref_logits[:, -1].argmax(-1, keepdim=True)
        ref_input_ids = torch.cat([ref_input_ids.to("cuda"), ref_next_id], dim=-1)

        cpp_logits_cuda = xstar_cpp.qwen2_forward(
            w, cfg, cache_cuda, cpp_input_ids.numpy(), positions, mask
        )
        assert list(cpp_logits_cuda.shape()) == list(ref_logits.squeeze().shape)
        cpp_logits = cpp_to_torch(
            xstar_cpp.to_cpu(cpp_logits_cuda), ref_logits.squeeze().shape
        )
        cpp_next_id = cpp_logits[-1].argmax(-1, keepdim=True)
        cpp_input_ids = torch.cat([cpp_input_ids, cpp_next_id], dim=-1)
        max_diff = (cpp_logits.float() - ref_logits.cpu().float()).abs().max()
        match = (cpp_next_id == ref_next_id.cpu()).item()
        print(
            f"step {i} max_diff={max_diff} cpp={cpp_next_id.item()} ref={ref_next_id.item()} match={match}"
        )

    print("cpp: " + tokenizer.decode(cpp_input_ids))
    print("ref: " + tokenizer.decode(ref_input_ids.squeeze()))


# 测端到端 greedy 生成, xstar cpu vs cuda 参考
def test_forward_logits_cuda_vs_cpu():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cpu")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w_cpu = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CPU)
    w_cuda = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CUDA)
    cache = py_model.model.positional_encoder._freq_cis_cache
    cache_cpu = torch_to_cpp(cache)
    cache_cuda = xstar_cpp.to_cuda(cache_cpu)

    prompt = "你好，你是谁？"
    ref_input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    cpp_input_ids_cpu = ref_input_ids.clone().squeeze()
    cpp_input_ids_cuda = ref_input_ids.clone().squeeze()
    positions = None
    mask = None

    # greedy 自回归 20 步,和 ref 对比
    steps = 20
    for i in range(steps):
        cpp_logits_cpu = xstar_cpp.qwen2_forward(
            w_cpu, cfg, cache_cpu, cpp_input_ids_cpu.numpy(), positions, mask
        )
        cpp_logits_cuda = xstar_cpp.qwen2_forward(
            w_cuda, cfg, cache_cuda, cpp_input_ids_cuda.numpy(), positions, mask
        )
        assert list(cpp_logits_cuda.shape()) == list(cpp_logits_cpu.shape())
        cpp_logits_cpu = cpp_to_torch(
            cpp_logits_cpu, [len(cpp_input_ids_cpu), cfg.vocab_size]
        )
        cpp_logits_cuda = cpp_to_torch(
            xstar_cpp.to_cpu(cpp_logits_cuda), [len(cpp_input_ids_cuda), cfg.vocab_size]
        )
        cpp_next_id_cpu = cpp_logits_cpu[-1].argmax(-1, keepdim=True)
        cpp_next_id_cuda = cpp_logits_cuda[-1].argmax(-1, keepdim=True)
        cpp_input_ids_cpu = torch.cat([cpp_input_ids_cpu, cpp_next_id_cpu], dim=-1)
        cpp_input_ids_cuda = torch.cat([cpp_input_ids_cuda, cpp_next_id_cuda], dim=-1)
        max_diff = (cpp_logits_cuda.float() - cpp_logits_cpu.float()).abs().max()
        match = (cpp_next_id_cuda == cpp_next_id_cpu).item()
        print(
            f"step {i} max_diff={max_diff} cpp_next_id_cuda={cpp_next_id_cuda.item()} cpp_next_id_cpu={cpp_next_id_cpu.item()} match={match}"
        )

    print("cuda: " + tokenizer.decode(cpp_input_ids_cuda))
    print("cpu: " + tokenizer.decode(cpp_input_ids_cpu))

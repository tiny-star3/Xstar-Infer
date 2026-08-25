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


# 测端到端 greedy 生成 vs xstar-Py 参考
def test_forward_greedy_matches_reference():
    py_model, tokenizer = load_reference_model(qwen2_model_path, device="cpu")

    cfg = xstar_cpp.parse_config_json(open(qwen2_model_path + "/config.json").read())
    mf = xstar_cpp.MMapFile(qwen2_model_path + "/model.safetensors")
    w = xstar_cpp.load_qwen2_weights(mf, cfg, xstar_cpp.Device.CPU)
    cache = py_model.model.positional_encoder._freq_cis_cache
    cache_t = torch_to_cpp(cache)

    prompt = "你好，你是谁？"
    ref_input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    cpp_input_ids = ref_input_ids.clone().squeeze()
    positions = None
    mask = None

    # greedy 自回归 20 步,和 ref 对比
    steps = 20
    for i in range(steps):
        ref_logits = reference("lm", ref_input_ids, ctx={"py_model": py_model})
        ref_next_id = ref_logits[:, -1].argmax(-1, keepdim=True)
        ref_input_ids = torch.cat([ref_input_ids, ref_next_id], dim=-1)

        cpp_logits_t = xstar_cpp.qwen2_forward(
            w, cfg, cache_t, cpp_input_ids.numpy(), positions, mask
        )
        assert list(cpp_logits_t.shape()) == list(ref_logits.squeeze().shape)
        cpp_logits = cpp_to_torch(cpp_logits_t, ref_logits.squeeze().shape)
        cpp_next_id = cpp_logits[-1].argmax(-1, keepdim=True)
        cpp_input_ids = torch.cat([cpp_input_ids, cpp_next_id], dim=-1)
        max_diff = (cpp_logits.float() - ref_logits.float()).abs().max()
        match = (cpp_next_id == ref_next_id).item()
        print(
            f"step {i} max_diff={max_diff} cpp={cpp_next_id.item()} ref={ref_next_id.item()} match={match}"
        )

    print("cpp: " + tokenizer.decode(cpp_input_ids))
    print("ref: " + tokenizer.decode(ref_input_ids.squeeze()))

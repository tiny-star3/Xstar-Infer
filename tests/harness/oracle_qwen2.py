import torch
import math
from dataclasses import dataclass

import transformers
from xstar.models.qwen2 import Qwen2ForCausalLM

# ── 容差表(纯数据, 无逻辑) ──
# 容差带:每层一个 (atol, judge_mode)。一处定义,oracle 和调用方都引用。
# judge_mode 三种:
#   "equal"     —— 无计算/镜像计算层,bit-exact(torch.equal)
#   "allclose"  —— bf16 噪声层,带 atol 的 allclose
#   "amplified" —— 归一化放大的层,额外看相对误差是否守恒(放大幅对误差不是 bug)
# coherence(生成连贯性)不在表里,那是端到端验收,不走 judge。
TOLERANCES: dict[str, tuple[float | None, str]] = {
    "embedding": (None, "equal"),
    "rmsnorm": (1e-6, "allclose"),
    "rope": (1e-2, "allclose"),
    "attention": (2e-2, "allclose"),
    "mlp": (5e-3, "allclose"),
    "block": (0.2, "amplified"),
}


# ── reference(产参考输出) ──
def load_reference_model(model_path: str, device: str) -> tuple:
    """
    Lazy 加载 HF 模型作权重来源, 返回 tokenizer + PyTorch reference 模型
    PyTorch 模型权重从 HF 拷贝(gate_up 融合在此处理, weight tying 已在 Qwen2ForCausalLM 构造时绑定, 拷 embed_tokens 即同步 lm_head)
    拷完释放 HF
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
    )
    qwen2_model = (
        transformers.Qwen2ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )

    py_model = Qwen2ForCausalLM(
        qwen2_model.config.vocab_size,
        qwen2_model.config.max_position_embeddings,
        qwen2_model.config.hidden_size,
        qwen2_model.config.num_hidden_layers,
        qwen2_model.config.num_attention_heads,
        qwen2_model.config.num_key_value_heads,
        qwen2_model.config.intermediate_size,
        qwen2_model.config.rms_norm_eps,
        qwen2_model.config.rope_theta,
        device,
        torch.bfloat16,
    ).eval()
    py_model.embed_tokens.weight.data.copy_(qwen2_model.model.embed_tokens.weight)
    for i in range(qwen2_model.config.num_hidden_layers):
        py_model.model.layers[i].input_layernorm.weight.data.copy_(
            qwen2_model.model.layers[i].input_layernorm.weight
        )
        py_model.model.layers[i].post_attention_layernorm.weight.data.copy_(
            qwen2_model.model.layers[i].post_attention_layernorm.weight
        )
        py_model.model.layers[i].attn.q_proj.weight.data.copy_(
            qwen2_model.model.layers[i].self_attn.q_proj.weight
        )
        py_model.model.layers[i].attn.q_proj.bias.data.copy_(
            qwen2_model.model.layers[i].self_attn.q_proj.bias
        )
        py_model.model.layers[i].attn.k_proj.weight.data.copy_(
            qwen2_model.model.layers[i].self_attn.k_proj.weight
        )
        py_model.model.layers[i].attn.k_proj.bias.data.copy_(
            qwen2_model.model.layers[i].self_attn.k_proj.bias
        )
        py_model.model.layers[i].attn.v_proj.weight.data.copy_(
            qwen2_model.model.layers[i].self_attn.v_proj.weight
        )
        py_model.model.layers[i].attn.v_proj.bias.data.copy_(
            qwen2_model.model.layers[i].self_attn.v_proj.bias
        )
        py_model.model.layers[i].attn.o_proj.weight.data.copy_(
            qwen2_model.model.layers[i].self_attn.o_proj.weight
        )
        py_model.model.layers[i].mlp.gate_up_proj.weight.data.copy_(
            torch.cat(
                (
                    qwen2_model.model.layers[i].mlp.gate_proj.weight,
                    qwen2_model.model.layers[i].mlp.up_proj.weight,
                ),
                dim=0,
            )
        )
        py_model.model.layers[i].mlp.down_proj.weight.data.copy_(
            qwen2_model.model.layers[i].mlp.down_proj.weight
        )
    py_model.model.ln_final.weight.data.copy_(qwen2_model.model.norm.weight)

    # 及时释放 HF 模型
    del qwen2_model
    torch.cuda.empty_cache()

    return py_model, tokenizer


def reference(
    layer: str,  # "embedding" / "rmsnorm" / "rope" / "attention" / "mlp" / "block" / "model" / "lm"
    input: torch.Tensor,  # 该层的输入(共模: C++ 和参考侧喂同一份)
    ctx: dict,  # 上下文: py_model, layer_idx, attention_mask, token_positions...
) -> torch.Tensor:
    """
    给定层名 + 输入,产出该层的期望输出(PyTorch reference)
    纯产出,不做任何判定。共模输入由调用方准备
    """
    py_model = ctx["py_model"]
    layer_idx = ctx.get("layer_idx")
    attention_mask = ctx.get("attention_mask")
    token_positions = ctx.get("token_positions")

    with torch.no_grad():
        if layer == "embedding":
            output = py_model.embed_tokens(input)
        elif layer == "rmsnorm":
            norm_kind = ctx.get("norm_kind")
            if norm_kind == "input":
                output = py_model.model.layers[layer_idx].input_layernorm(input)
            elif norm_kind == "post":
                output = py_model.model.layers[layer_idx].post_attention_layernorm(
                    input
                )
            else:
                raise ValueError(f"unknown norm_kind: {norm_kind}")
        elif layer == "rope":
            # input 是 Q/K (调用方先做好 q_proj+reshape)
            output = py_model.model.positional_encoder(input, token_positions)
        elif layer == "attention":
            output = py_model.model.layers[layer_idx].attn(
                input, attention_mask, token_positions
            )
        elif layer == "mlp":
            output = py_model.model.layers[layer_idx].mlp(input)
        elif layer == "block":
            output = py_model.model.layers[layer_idx](
                input, attention_mask, token_positions
            )
        elif layer == "model":
            output = py_model.model(input, attention_mask, token_positions)
        elif layer == "lm":
            output = py_model(input, attention_mask, token_positions)
        else:
            raise ValueError(f"unknown layer: {layer}")

    return output


# ── judge(判定) ──
# Judgement:
#   passed: bool                —— 这层 pass/fail
#   max_diff: float             —— |mine - ref| 的最大值
#   relative_err: float | None  —— max_diff / max(|ref|), 只在 amplified 模式算, 其余模式为 None, report 据此决定是否显示 rel_err 项
#   mode: str                   —— 这次用的是哪种 judge_mode(回显,方便打印)
#   layer: str                  —— 哪一层(回显)
#   note: str | None            —— 失败原因(shape/dtype 不匹配); 成功或无诊断时为 None
@dataclass
class Judgement:
    passed: bool
    max_diff: float
    relative_err: float | None
    mode: str
    layer: str
    note: str | None


def judge(
    cpp: torch.Tensor,  # 输出(C++ 侧)
    ref: torch.Tensor,  # reference(...) 产出的期望输出
    layer: str,  # 查容差表用
) -> Judgement:
    """
    查容差表 → 按该层的 judge_mode 比对 cpp 与 ref → 返回带诊断的 Judgement

    先把 ref 搬到 cpp 的 device, 再在 float32 上算 max_diff(诊断用,所有模式共享)
    - equal:    要求 cpp/ref dtype 一致,否则直接判 FAIL; 比特级 torch.equal
    - allclose: float32 上 torch.allclose(rtol=0, atol=atol)
    - amplified:passed 判据同 allclose(宽 atol);额外算 relative_err 作诊断, 供人判断"绝对 diff 大是因为 RMSNorm 放大(相对误差守恒)还是真 bug"

    未知层(含走 coherence 的 model/lm)不在容差表,抛 ValueError
    shape 不匹配直接判 FAIL(max_diff=inf, note 记录两边 shape);
    equal 的 dtype 不匹配判 FAIL(note 记录两边 dtype)。
    """
    atol, mode = TOLERANCES.get(layer, (None, None))
    # 未知层(不在容差表, 含走 coherence 的 model/lm)不该进 judge
    if mode is None:
        raise ValueError(f"unknown layer: {layer}")
    if cpp.shape != ref.shape:
        return Judgement(
            False,
            float("inf"),
            None,
            mode,
            layer,
            f"shape mismatch: cpp={tuple(cpp.shape)} ref={tuple(ref.shape)}",
        )

    ref = ref.to(cpp.device)
    max_diff = (cpp.float() - ref.float()).abs().max().item()

    if mode == "equal":
        if ref.dtype != cpp.dtype:
            j = Judgement(
                False,
                max_diff,
                None,
                mode,
                layer,
                f"dtype mismatch: cpp={cpp.dtype} ref={ref.dtype}",
            )
        else:
            j = Judgement(torch.equal(cpp, ref), max_diff, None, mode, layer, None)
    elif mode == "allclose":
        j = Judgement(
            torch.allclose(cpp.float(), ref.float(), rtol=0, atol=atol),
            max_diff,
            None,
            mode,
            layer,
            None,
        )
    elif mode == "amplified":
        relative_err = max_diff / max(ref.float().abs().max().item(), 1e-12)
        j = Judgement(
            torch.allclose(cpp.float(), ref.float(), rtol=0, atol=atol),
            max_diff,
            relative_err,
            mode,
            layer,
            None,
        )

    return j


def report(j: "Judgement") -> str:
    """
    格式化为可打印文本:主信息一行(状态、层名左对齐 10 宽、max_diff、mode、可选 rel_err)
        失败时 note 另起一行
        max_diff 为 inf(shape 不匹配)时显示 N/A

    示例:
      [PASS] attention   max_diff=0.013700  mode=allclose
      [FAIL] mlp         max_diff=0.180000  mode=allclose
      [PASS] block       max_diff=0.190000  mode=amplified  rel_err=0.051000
      [FAIL] embedding   max_diff=N/A       mode=equal
      note=shape mismatch: cpp=(1, 8, 896) ref=(1, 8, 14, 64)
    """
    status = "[PASS]" if j.passed else "[FAIL]"
    return (
        f"{status} {j.layer:<10}"
        + (
            f"  max_diff=N/A"
            if math.isinf(j.max_diff)
            else f"  max_diff={j.max_diff:.6f}"
        )
        + f"  mode={j.mode}"
        + (f"  rel_err={j.relative_err:.6f}" if j.relative_err is not None else "")
        + (f"\nnote={j.note}" if j.note is not None else "")
    )

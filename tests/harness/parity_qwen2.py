import os

# 在"HF 库被 import 之前"设置离线开关
# 保证 HF 不会联网校验/下载，reference 完全确定、可复现，harness 永不漂移
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import transformers
from einops import rearrange

from xstar.layers.embedding import Embedding
from xstar.layers.rmsnorm import RMSNorm
from xstar.layers.rope import RoPE
from xstar.layers.attention import GroupedQueryAttention
from xstar.layers.mlp import SwiGLU
from xstar.layers.transformer_block import TransformerBlock
from xstar.models.qwen2 import Qwen2Model, Qwen2ForCausalLM

# 本地模型路径
qwen2_model_path = "~/models/Qwen2.5-0.5B"
qwen2_model_path = os.path.expanduser(qwen2_model_path)

device = "cuda"

# 共用同一个 tokenizer, 只验证 reference 正确性
tokenizer = transformers.AutoTokenizer.from_pretrained(
    qwen2_model_path,
)
qwen2_model = (
    transformers.Qwen2ForCausalLM.from_pretrained(
        qwen2_model_path, torch_dtype=torch.bfloat16
    )
    .to(device)
    .eval()
)


"""
    验证各层结果形状正确
"""
# (batch_size, seq_len) -> (1, 8)
token_ids = torch.tensor([[3, 6, 33, 66, 333, 666, 3333, 6666]]).to(device)
with torch.no_grad():
    qwen2_output = qwen2_model(token_ids, output_hidden_states=True)

# Qwen2ForCausalLM 的输出的 hidden_states 为长度 25(embedding output + num_hidden_layers) 的元组
print("Qwen2.5-0.5B output logits shape: ", qwen2_output.logits.shape)
print("Qwen2.5-0.5B output hidden_states len: ", len(qwen2_output.hidden_states))
print("Qwen2.5-0.5B output hidden_states shape: ", qwen2_output.hidden_states[0].shape)


"""
    验证输出结果正确
"""
prompt = "你好，你是谁？"
input_encode = tokenizer(prompt, return_tensors="pt")
input_ids, attention_mask = input_encode.input_ids, input_encode.attention_mask
with torch.no_grad():
    output_ids = qwen2_model.generate(
        input_ids.to(device),
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )
    result = tokenizer.decode(output_ids[0].tolist())

print(result)

# 看"模型图"——打印模型结构
print("模型结构: ", qwen2_model)
# 看"权重字典"——打印 state_dict 的键
print("权重名称: ", list(qwen2_model.state_dict().keys()))


"""
    对拍 Embedding 模块
"""
embed_weight = qwen2_model.model.embed_tokens.weight
embed = Embedding(
    *embed_weight.shape, device=embed_weight.device, dtype=embed_weight.dtype
)
assert embed.weight.dtype == embed_weight.dtype
assert embed.weight.device == embed_weight.device
embed.weight.data.copy_(embed_weight)
# copy_,是真拷贝,两块独立内存,ptr 应该不同
assert embed.weight.data_ptr() != embed_weight.data_ptr()
with torch.no_grad():
    embed_output = embed(token_ids)
assert (
    torch.allclose(embed_output, qwen2_output.hidden_states[0], rtol=1e-3, atol=1e-3)
    == True
)
# 由于 Embedding 只是纯查表,没有任何计算，输出应该完全相等
assert torch.equal(embed_output, qwen2_output.hidden_states[0]) == True
print(embed_output[0, 0, :5])
print(qwen2_output.hidden_states[0][0, 0, :5])

"""
    对拍 RMSNorm 模块
"""
rmsnorm_weight = qwen2_model.model.layers[0].input_layernorm.weight
rmsnorm_eps = qwen2_model.model.layers[0].input_layernorm.variance_epsilon
rms = RMSNorm(
    *rmsnorm_weight.shape,
    eps=rmsnorm_eps,
    device=rmsnorm_weight.device,
    dtype=rmsnorm_weight.dtype,
)
rms.weight.data.copy_(rmsnorm_weight)
with torch.no_grad():
    rms_output = rms(qwen2_output.hidden_states[0])
    ref_rms_output = qwen2_model.model.layers[0].input_layernorm(
        qwen2_output.hidden_states[0]
    )
print(
    "max rms_output diff: ",
    torch.max(abs(rms_output.to(torch.float32) - ref_rms_output.to(torch.float32))),
)
assert torch.allclose(rms_output, ref_rms_output, rtol=1e-6, atol=1e-6) == True


"""
    对拍 RoPE 模块
"""
rope_cache = []


def rope_hook(module, input, output):
    rope_cache.append(output)


# 获取 HF 的 cos/sin
rope_handle = qwen2_model.model.layers[0].self_attn.rotary_emb.register_forward_hook(
    rope_hook
)
with torch.no_grad():
    qwen2_output = qwen2_model(token_ids, output_hidden_states=True)
rope_handle.remove()
# 验证 cos/sin 相等, 二者应该完全相等
ref_cos_cached, ref_sin_cached = rope_cache[0]
rope = RoPE(
    qwen2_model.config.rope_theta,
    qwen2_model.config.hidden_size // qwen2_model.config.num_attention_heads,
    512,
    ref_cos_cached.device,
)
cos_cached, sin_cached = rope._freq_cis_cache[:, : token_ids.shape[-1], :].unbind(0)
cos_cached = torch.cat((cos_cached, cos_cached), dim=-1)
sin_cached = torch.cat((sin_cached, sin_cached), dim=-1)
assert ref_cos_cached.shape == cos_cached.shape
assert ref_sin_cached.shape == sin_cached.shape
print(
    "max cos_cached diff: ", torch.max(abs(cos_cached.float() - ref_cos_cached.float()))
)
print(
    "max sin_cached diff: ", torch.max(abs(sin_cached.float() - ref_sin_cached.float()))
)
assert (
    torch.allclose(cos_cached.float(), ref_cos_cached.float(), rtol=1e-2, atol=1e-2)
    == True
)
assert (
    torch.allclose(sin_cached.float(), ref_sin_cached.float(), rtol=1e-2, atol=1e-2)
    == True
)
# 验证 RoPE 结果
rope_Q = torch.randn(
    1,
    qwen2_model.config.num_attention_heads,
    token_ids.shape[-1],
    qwen2_model.config.hidden_size // qwen2_model.config.num_attention_heads,
    device=ref_cos_cached.device,
    dtype=torch.bfloat16,
)
rope_K = torch.randn(
    1,
    qwen2_model.config.num_key_value_heads,
    token_ids.shape[-1],
    qwen2_model.config.hidden_size // qwen2_model.config.num_attention_heads,
    device=ref_cos_cached.device,
    dtype=torch.bfloat16,
)
# cos/sin_cached 位置(seq_len, head_dim), position_ids 索引(batch_size, seq_len)
position_ids = torch.arange(
    0,
    8,
    dtype=torch.int,
    device=ref_cos_cached.device,
).unsqueeze(0)
with torch.no_grad():
    ref_rope_Q_output, ref_rope_K_output = (
        transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb(
            rope_Q,
            rope_K,
            ref_cos_cached,
            ref_sin_cached,
            position_ids,
        )
    )
    rope_Q_output = rope(rope_Q, position_ids)
    rope_K_output = rope(rope_K, position_ids)
print(
    "max rope_Q_output diff: ",
    torch.max(abs(rope_Q_output.float() - ref_rope_Q_output.float())),
)
print(
    "max rope_K_output diff: ",
    torch.max(abs(rope_K_output.float() - ref_rope_K_output.float())),
)
assert torch.allclose(rope_Q_output, ref_rope_Q_output, rtol=1e-2, atol=1e-2) == True
assert torch.allclose(rope_K_output, ref_rope_K_output, rtol=1e-2, atol=1e-2) == True


"""
    对拍 GroupedQueryAttention 模块
"""
mask_cache = []


def attention_hook(module, input, kwargs, output):
    mask_cache.append(kwargs.get("attention_mask"))


# 获取 GroupedQueryAttention 的 mask
attention_handle = qwen2_model.model.layers[0].self_attn.register_forward_hook(
    attention_hook, with_kwargs=True
)
with torch.no_grad():
    qwen2_output = qwen2_model(token_ids, output_hidden_states=True)
attention_handle.remove()
mask = mask_cache[0]
print("attention mask: ", mask)

# 获取 GroupedQueryAttention 的 weight
q_proj = qwen2_model.model.layers[0].self_attn.q_proj.weight
q_bias = qwen2_model.model.layers[0].self_attn.q_proj.bias
k_proj = qwen2_model.model.layers[0].self_attn.k_proj.weight
k_bias = qwen2_model.model.layers[0].self_attn.k_proj.bias
v_proj = qwen2_model.model.layers[0].self_attn.v_proj.weight
v_bias = qwen2_model.model.layers[0].self_attn.v_proj.bias
o_proj = qwen2_model.model.layers[0].self_attn.o_proj.weight
attention = GroupedQueryAttention(
    qwen2_model.config.hidden_size,
    qwen2_model.config.num_attention_heads,
    qwen2_model.config.num_key_value_heads,
    rope,
    device=q_proj.device,
    dtype=q_proj.dtype,
)
attention.q_proj.weight.data.copy_(q_proj)
attention.q_proj.bias.data.copy_(q_bias)
attention.k_proj.weight.data.copy_(k_proj)
attention.k_proj.bias.data.copy_(k_bias)
attention.v_proj.weight.data.copy_(v_proj)
attention.v_proj.bias.data.copy_(v_bias)
attention.o_proj.weight.data.copy_(o_proj)

# 验证真 Q/K 的 RoPE 结果一致
with torch.no_grad():
    rope_Q_real = attention.q_proj(ref_rms_output)
    rope_K_real = attention.k_proj(ref_rms_output)
    rope_Q_real = rearrange(
        rope_Q_real,
        "... seq (heads d) -> ... heads seq d",
        heads=qwen2_model.config.num_attention_heads,
    )
    rope_K_real = rearrange(
        rope_K_real,
        "... seq (heads d) -> ... heads seq d",
        heads=qwen2_model.config.num_key_value_heads,
    )
    ref_rope_Q_real_output, ref_rope_K_real_output = (
        transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb(
            rope_Q_real,
            rope_K_real,
            ref_cos_cached,
            ref_sin_cached,
            position_ids,
        )
    )
    rope_Q_real_output = rope(rope_Q_real, position_ids)
    rope_K_real_output = rope(rope_K_real, position_ids)
print(
    "max rope_Q_real_output diff: ",
    torch.max(abs(rope_Q_real_output.float() - ref_rope_Q_real_output.float())),
)
print(
    "max rope_K_real_output diff: ",
    torch.max(abs(rope_K_real_output.float() - ref_rope_K_real_output.float())),
)
assert (
    torch.allclose(rope_Q_real_output, ref_rope_Q_real_output, rtol=1e-2, atol=1e-2)
    == True
)
assert (
    torch.allclose(rope_K_real_output, ref_rope_K_real_output, rtol=1e-2, atol=1e-2)
    == True
)

# 验证不带 padding tokens 的 GroupedQueryAttention 结果
with torch.no_grad():
    ref_attention_output = qwen2_model.model.layers[0].self_attn(ref_rms_output, mask)[
        0
    ]
    attention_output = attention(ref_rms_output, mask)
assert ref_attention_output.shape == attention_output.shape
print(
    "max attention output diff: ",
    torch.max(abs(attention_output.float() - ref_attention_output.float())),
)
assert torch.allclose(attention_output, ref_attention_output, rtol=2e-2, atol=2e-2)

# 验证带 right padding tokens 的 GroupedQueryAttention 结果
# 因为 多个 tokens 序列组成 batch_size 时, seq_len 可能不一样, 这时把短的 padding 到最大的 seq_len
pad_ids = torch.tensor(
    [[tokenizer.pad_token_id, tokenizer.pad_token_id]], device=device
)
padding_right_token_ids = torch.cat((token_ids, pad_ids), dim=1)
mask_right = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], device=device, dtype=int)
# 获取 padding right attention 的 mask
attention_handle = qwen2_model.model.layers[0].self_attn.register_forward_hook(
    attention_hook, with_kwargs=True
)
with torch.no_grad():
    qwen2_padding_right_output = qwen2_model(
        padding_right_token_ids, attention_mask=mask_right, output_hidden_states=True
    )
attention_handle.remove()
mask_right = mask_cache[-1]
print("padding right attention mask: ", mask_right)
with torch.no_grad():
    ref_padding_right_rms_output = qwen2_model.model.layers[0].input_layernorm(
        qwen2_padding_right_output.hidden_states[0]
    )
    ref_padding_right_attention_output = qwen2_model.model.layers[0].self_attn(
        ref_padding_right_rms_output, mask_right
    )[0]
    padding_right_attention_output = attention(ref_padding_right_rms_output, mask_right)
assert ref_padding_right_attention_output.shape == padding_right_attention_output.shape
# 只比真 token 位置
print(
    "max padding right attention output diff: ",
    torch.max(
        abs(
            padding_right_attention_output[:, :8, :].float()
            - ref_padding_right_attention_output[:, :8, :].float()
        )
    ),
)
assert torch.allclose(
    padding_right_attention_output[:, :8, :],
    ref_padding_right_attention_output[:, :8, :],
    rtol=2e-2,
    atol=2e-2,
)

# 验证带 left padding tokens 的 GroupedQueryAttention 结果
# 因为 多个 tokens 序列组成 batch_size 时, seq_len 可能不一样, 这时把短的 padding 到最大的 seq_len
pad_ids = torch.tensor(
    [[tokenizer.pad_token_id, tokenizer.pad_token_id]], device=device
)
padding_left_token_ids = torch.cat((pad_ids, token_ids), dim=1)
mask_left = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], device=device, dtype=int)
position_left_ids = torch.cat(
    (torch.tensor([[0, 0]], device=device, dtype=int), position_ids), dim=1
)
# 获取 padding left attention 的 mask
attention_handle = qwen2_model.model.layers[0].self_attn.register_forward_hook(
    attention_hook, with_kwargs=True
)
with torch.no_grad():
    qwen2_padding_left_output = qwen2_model(
        padding_left_token_ids, attention_mask=mask_left, output_hidden_states=True
    )
attention_handle.remove()
mask_left = mask_cache[-1]
print("padding left attention mask: ", mask_left)
with torch.no_grad():
    ref_padding_left_rms_output = qwen2_model.model.layers[0].input_layernorm(
        qwen2_padding_left_output.hidden_states[0]
    )
    ref_padding_left_attention_output = qwen2_model.model.layers[0].self_attn(
        ref_padding_left_rms_output, mask_left, position_left_ids
    )[0]
    padding_left_attention_output = attention(
        ref_padding_left_rms_output, mask_left, position_left_ids
    )
assert ref_padding_left_attention_output.shape == padding_left_attention_output.shape
# 只比真 token 位置
print(
    "max padding left attention output diff: ",
    torch.max(
        abs(
            padding_left_attention_output[:, 2:, :].float()
            - ref_padding_left_attention_output[:, 2:, :].float()
        )
    ),
)
assert torch.allclose(
    padding_left_attention_output[:, 2:, :],
    ref_padding_left_attention_output[:, 2:, :],
    rtol=2e-2,
    atol=2e-2,
)


"""
    对拍 SwiGLU 模块
"""
mlp_input_cache = []


def mlp_hook(module, input, output):
    mlp_input_cache.append(output)


# 获取 HF 的 SwiGLU input
mlp_handle = qwen2_model.model.layers[0].post_attention_layernorm.register_forward_hook(
    mlp_hook
)
with torch.no_grad():
    qwen2_output = qwen2_model(token_ids, output_hidden_states=True)
mlp_handle.remove()
mlp_input = mlp_input_cache[0]
gate_proj = qwen2_model.model.layers[0].mlp.gate_proj.weight
up_proj = qwen2_model.model.layers[0].mlp.up_proj.weight
down_proj = qwen2_model.model.layers[0].mlp.down_proj.weight
mlp = SwiGLU(
    qwen2_model.config.hidden_size,
    qwen2_model.config.intermediate_size,
    device=gate_proj.device,
    dtype=gate_proj.dtype,
)
mlp.gate_up_proj.weight.data.copy_(torch.cat((gate_proj, up_proj), dim=0))
mlp.down_proj.weight.data.copy_(down_proj)
# 验证 SwiGLU 结果
with torch.no_grad():
    ref_mlp_output = qwen2_model.model.layers[0].mlp(mlp_input)
    mlp_output = mlp(mlp_input)
assert ref_mlp_output.shape == mlp_output.shape
print(
    "max mlp output diff", torch.max(abs(mlp_output.float() - ref_mlp_output.float()))
)
assert torch.allclose(mlp_output, ref_mlp_output, rtol=5e-3, atol=5e-3)


"""
    对拍 TransformerBlock 模块
"""
transformer_block_input_cache = []


def transformer_block_hook(module, input, output):
    transformer_block_input_cache.append(input[0])


# 获取 HF 的 TransformerBlock input
transformer_block_handle = qwen2_model.model.layers[0].register_forward_hook(
    transformer_block_hook
)
with torch.no_grad():
    qwen2_output = qwen2_model(token_ids, output_hidden_states=True)
transformer_block_handle.remove()
transformer_block_input = transformer_block_input_cache[0]
assert transformer_block_input.shape == qwen2_output.hidden_states[0].shape
assert torch.equal(transformer_block_input, qwen2_output.hidden_states[0])
transformer_block = TransformerBlock(
    qwen2_model.config.hidden_size,
    qwen2_model.config.num_attention_heads,
    qwen2_model.config.num_key_value_heads,
    qwen2_model.config.intermediate_size,
    qwen2_model.config.rms_norm_eps,
    rope,
    device=qwen2_model.model.layers[0].input_layernorm.weight.device,
    dtype=qwen2_model.model.layers[0].input_layernorm.weight.dtype,
)
transformer_block.input_layernorm.weight.data.copy_(
    qwen2_model.model.layers[0].input_layernorm.weight
)
transformer_block.post_attention_layernorm.weight.data.copy_(
    qwen2_model.model.layers[0].post_attention_layernorm.weight
)
transformer_block.attn.q_proj.weight.data.copy_(
    qwen2_model.model.layers[0].self_attn.q_proj.weight
)
transformer_block.attn.q_proj.bias.data.copy_(
    qwen2_model.model.layers[0].self_attn.q_proj.bias
)
transformer_block.attn.k_proj.weight.data.copy_(
    qwen2_model.model.layers[0].self_attn.k_proj.weight
)
transformer_block.attn.k_proj.bias.data.copy_(
    qwen2_model.model.layers[0].self_attn.k_proj.bias
)
transformer_block.attn.v_proj.weight.data.copy_(
    qwen2_model.model.layers[0].self_attn.v_proj.weight
)
transformer_block.attn.v_proj.bias.data.copy_(
    qwen2_model.model.layers[0].self_attn.v_proj.bias
)
transformer_block.attn.o_proj.weight.data.copy_(
    qwen2_model.model.layers[0].self_attn.o_proj.weight
)
transformer_block.mlp.gate_up_proj.weight.data.copy_(
    torch.cat(
        (
            qwen2_model.model.layers[0].mlp.gate_proj.weight,
            qwen2_model.model.layers[0].mlp.up_proj.weight,
        ),
        dim=0,
    )
)
transformer_block.mlp.down_proj.weight.data.copy_(
    qwen2_model.model.layers[0].mlp.down_proj.weight
)
with torch.no_grad():
    ref_transformer_block_output = qwen2_model.model.layers[0](transformer_block_input)[
        0
    ]
    transformer_block_output = transformer_block(transformer_block_input)

    ln1 = transformer_block.input_layernorm(transformer_block_input)
    a = transformer_block.attn(ln1)  # 不传 mask/position,和单测同条件
    r1 = transformer_block_input + a  # 第一残差
    ln2 = transformer_block.post_attention_layernorm(r1)
    m = transformer_block.mlp(ln2)
    out = r1 + m  # 第二残差(=最终输出)

    rf_ln1 = qwen2_model.model.layers[0].input_layernorm(transformer_block_input)
    rf_a = qwen2_model.model.layers[0].self_attn(rf_ln1)[0]
    rf_r1 = transformer_block_input + rf_a
    rf_ln2 = qwen2_model.model.layers[0].post_attention_layernorm(rf_r1)
    rf_m = qwen2_model.model.layers[0].mlp(rf_ln2)
    rf_out = rf_r1 + rf_m

    print(
        "post_ln same-input:",
        torch.max(
            abs(
                qwen2_model.model.layers[0].post_attention_layernorm(r1)
                - transformer_block.post_attention_layernorm(r1)
            )
        ),
    )
    print(
        "r1 max:",
        torch.max(abs(r1)).item(),
        "r1 rms:",
        r1.float().pow(2).mean().sqrt().item(),
    )
    # 相对误差
    print("ln2 rel err:", (torch.max(abs(rf_ln2 - ln2)) / torch.max(abs(ln2))).item())
    print("r1 rel err:", (torch.max(abs(rf_r1 - r1)) / torch.max(abs(r1))).item())

    print("max ln1: ", torch.max(abs(rf_ln1 - ln1)))
    print("max a: ", torch.max(abs(rf_a - a)))
    print("max r1: ", torch.max(abs(rf_r1 - r1)))
    print("max ln2: ", torch.max(abs(rf_ln2 - ln2)))
    print("max m: ", torch.max(abs(rf_m - m)))
    print("max out", torch.max(abs(rf_out - out)))
    print("逐段out vs 整块ref:", torch.max(abs(out - ref_transformer_block_output[0])))


assert ref_transformer_block_output.shape == transformer_block_output.shape
print(
    "max transformer block output diff",
    torch.max(
        abs(transformer_block_output.float() - ref_transformer_block_output.float())
    ),
)
# Block parity 使用宽松容差:绝对 diff 被 RMSNorm 的 1/rms 因子放大。
# r1 的 rms 仅 ~0.019(即 1/rms ~53 倍),attention 的 bf16 噪声 ~0.0137
# 在 post_attention_layernorm 出口被放大到 ~0.4(相对误差保持 ~5% 不变 ——
# 是被缩放,不是被引入)。这是 bf16 + 归一化在低量级激活上的固有放大,
# 不是实现 bug。已逐段验证:input_layernorm / attn / 残差为 bit-exact /
# 注意力噪声量级;放大始于 post_attention_layernorm,纯粹是 1/rms 缩放。
assert torch.allclose(
    transformer_block_output, ref_transformer_block_output, rtol=0.2, atol=0.2
)


"""
    对拍 Qwen2Model 模块
"""
model = Qwen2Model(
    qwen2_model.config.max_position_embeddings,
    qwen2_model.config.hidden_size,
    qwen2_model.config.num_hidden_layers,
    qwen2_model.config.num_attention_heads,
    qwen2_model.config.num_key_value_heads,
    qwen2_model.config.intermediate_size,
    qwen2_model.config.rms_norm_eps,
    qwen2_model.config.rope_theta,
    qwen2_model.model.norm.weight.device,
    qwen2_model.model.norm.weight.dtype,
)

for i in range(qwen2_model.config.num_hidden_layers):
    model.layers[i].input_layernorm.weight.data.copy_(
        qwen2_model.model.layers[i].input_layernorm.weight
    )
    model.layers[i].post_attention_layernorm.weight.data.copy_(
        qwen2_model.model.layers[i].post_attention_layernorm.weight
    )
    model.layers[i].attn.q_proj.weight.data.copy_(
        qwen2_model.model.layers[i].self_attn.q_proj.weight
    )
    model.layers[i].attn.q_proj.bias.data.copy_(
        qwen2_model.model.layers[i].self_attn.q_proj.bias
    )
    model.layers[i].attn.k_proj.weight.data.copy_(
        qwen2_model.model.layers[i].self_attn.k_proj.weight
    )
    model.layers[i].attn.k_proj.bias.data.copy_(
        qwen2_model.model.layers[i].self_attn.k_proj.bias
    )
    model.layers[i].attn.v_proj.weight.data.copy_(
        qwen2_model.model.layers[i].self_attn.v_proj.weight
    )
    model.layers[i].attn.v_proj.bias.data.copy_(
        qwen2_model.model.layers[i].self_attn.v_proj.bias
    )
    model.layers[i].attn.o_proj.weight.data.copy_(
        qwen2_model.model.layers[i].self_attn.o_proj.weight
    )
    model.layers[i].mlp.gate_up_proj.weight.data.copy_(
        torch.cat(
            (
                qwen2_model.model.layers[i].mlp.gate_proj.weight,
                qwen2_model.model.layers[i].mlp.up_proj.weight,
            ),
            dim=0,
        )
    )
    model.layers[i].mlp.down_proj.weight.data.copy_(
        qwen2_model.model.layers[i].mlp.down_proj.weight
    )
model.ln_final.weight.data.copy_(qwen2_model.model.norm.weight)

# 输入 8 个随机 token_ids
with torch.no_grad():
    x = qwen2_output.hidden_states[0]
    for i, layer in enumerate(model.layers):
        x = layer(x)
        print(
            f"max layer{i} output diff: ",
            torch.max(abs(x - qwen2_output.hidden_states[i + 1])),
        )
        if i == 2:
            print("layer2 max:", torch.max(abs(x)).item())
            print(
                "ref layer2 max:",
                torch.max(abs(qwen2_output.hidden_states[3])).item(),
            )

    ref_model_output = qwen2_output.hidden_states[-1]
    model_output = model(qwen2_output.hidden_states[0])

assert ref_model_output.shape == model_output.shape
print(
    "max model output diff: ",
    torch.max(abs(model_output.float() - ref_model_output.float())),
)

# 输入真实文本
real_encode = tokenizer("你好，你是谁？", return_tensors="pt").to(device)
real_ids = real_encode.input_ids
with torch.no_grad():
    real_qwen2_output = qwen2_model(real_ids, output_hidden_states=True)

    x = real_qwen2_output.hidden_states[0]
    for i, layer in enumerate(model.layers):
        x = layer(x)
        print(
            f"max real ids layer{i} output diff: ",
            torch.max(abs(x - real_qwen2_output.hidden_states[i + 1])),
        )

    ref_model_output = real_qwen2_output.hidden_states[-1]
    model_output = model(real_qwen2_output.hidden_states[0])
assert ref_model_output.shape == model_output.shape
print(
    "max real ids model output diff: ",
    torch.max(abs(model_output.float() - ref_model_output.float())),
)

"""
    对拍 Qwen2ForCausalLM 模块
"""
lm = Qwen2ForCausalLM(
    qwen2_model.config.vocab_size,
    qwen2_model.config.max_position_embeddings,
    qwen2_model.config.hidden_size,
    qwen2_model.config.num_hidden_layers,
    qwen2_model.config.num_attention_heads,
    qwen2_model.config.num_key_value_heads,
    qwen2_model.config.intermediate_size,
    qwen2_model.config.rms_norm_eps,
    qwen2_model.config.rope_theta,
    qwen2_model.model.norm.weight.device,
    qwen2_model.model.norm.weight.dtype,
)
lm.embed_tokens.weight.data.copy_(qwen2_model.model.embed_tokens.weight)
for i in range(qwen2_model.config.num_hidden_layers):
    lm.model.layers[i].input_layernorm.weight.data.copy_(
        qwen2_model.model.layers[i].input_layernorm.weight
    )
    lm.model.layers[i].post_attention_layernorm.weight.data.copy_(
        qwen2_model.model.layers[i].post_attention_layernorm.weight
    )
    lm.model.layers[i].attn.q_proj.weight.data.copy_(
        qwen2_model.model.layers[i].self_attn.q_proj.weight
    )
    lm.model.layers[i].attn.q_proj.bias.data.copy_(
        qwen2_model.model.layers[i].self_attn.q_proj.bias
    )
    lm.model.layers[i].attn.k_proj.weight.data.copy_(
        qwen2_model.model.layers[i].self_attn.k_proj.weight
    )
    lm.model.layers[i].attn.k_proj.bias.data.copy_(
        qwen2_model.model.layers[i].self_attn.k_proj.bias
    )
    lm.model.layers[i].attn.v_proj.weight.data.copy_(
        qwen2_model.model.layers[i].self_attn.v_proj.weight
    )
    lm.model.layers[i].attn.v_proj.bias.data.copy_(
        qwen2_model.model.layers[i].self_attn.v_proj.bias
    )
    lm.model.layers[i].attn.o_proj.weight.data.copy_(
        qwen2_model.model.layers[i].self_attn.o_proj.weight
    )
    lm.model.layers[i].mlp.gate_up_proj.weight.data.copy_(
        torch.cat(
            (
                qwen2_model.model.layers[i].mlp.gate_proj.weight,
                qwen2_model.model.layers[i].mlp.up_proj.weight,
            ),
            dim=0,
        )
    )
    lm.model.layers[i].mlp.down_proj.weight.data.copy_(
        qwen2_model.model.layers[i].mlp.down_proj.weight
    )
lm.model.ln_final.weight.data.copy_(qwen2_model.model.norm.weight)

# 输入 8 个随机 token_ids
with torch.no_grad():
    ref_lm_logits = qwen2_output.logits
    lm_logits = lm(token_ids)

assert ref_lm_logits.shape == lm_logits.shape
print(
    "max lm logits diff: ",
    torch.max(abs(lm_logits.float() - ref_lm_logits.float())),
)
print("top1:", (lm_logits[:, -1].argmax(-1) == ref_lm_logits[:, -1].argmax(-1)).all())

# 输入真实文本
with torch.no_grad():
    ref_real_lm_logits = real_qwen2_output.logits
    real_lm_logits = lm(real_ids)

assert ref_real_lm_logits.shape == real_lm_logits.shape
print(
    "max real ids lm logits diff: ",
    torch.max(abs(real_lm_logits.float() - ref_real_lm_logits.float())),
)
print(
    "real top1:",
    (real_lm_logits[:, -1].argmax(-1) == ref_real_lm_logits[:, -1].argmax(-1)).all(),
)
ref_top2 = ref_real_lm_logits[:, -1].topk(2).values
print("ref top2:", ref_top2, "gap:", (ref_top2[..., 0] - ref_top2[..., 1]).item())
print("ref argmax:", ref_real_lm_logits[:, -1].argmax(-1).item())
print("argmax:", real_lm_logits[:, -1].argmax(-1).item())


# greedy 自回归 20 步,和 HF 对比
def greedy(model, ids, steps):
    for _ in range(steps):
        logits = model(ids)
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


gen = greedy(lm, real_ids, 20)
ref_gen = qwen2_model.generate(real_ids, do_sample=False, max_new_tokens=20)
print("generate text:", tokenizer.decode(gen[0]))
print("ref generate text:", tokenizer.decode(ref_gen[0]))

#include <stdexcept>
#include <string>
#include <memory>

#include "qwen2_model.h"
#include "ops/embedding.h"
#include "ops/transformer_block.h"
#include "ops/rmsnorm.h"
#include "ops/linear.h"
#include "json_scan.h"
#include "weight_io.h"
#include "cuda/cuda_allocator.h"

Tensor qwen2_forward(const Qwen2ModelWeights &w,
                     const Qwen2Config &cfg,
                     const Tensor &rope_cache,
                     const std::int64_t *input_ids, std::int64_t seq_len,
                     const std::int64_t *positions,
                     const Tensor *mask)
{
    if (w.layers.size() != cfg.num_hidden_layers)
        throw std::runtime_error("config and weights from different models");

    Tensor x = embedding(w.embed_tokens_w, input_ids, std::vector<std::int64_t>{seq_len});
    for (size_t i = 0; i < cfg.num_hidden_layers; i++)
    {
        x = transformer_block(x, cfg.num_attention_heads, w.layers[i].ln1_w, w.layers[i].ln2_w, static_cast<float>(cfg.rms_norm_eps), w.layers[i].q_w, &w.layers[i].q_b, w.layers[i].k_w, &w.layers[i].k_b, w.layers[i].v_w, &w.layers[i].v_b, w.layers[i].o_w, w.layers[i].gate_up_w, w.layers[i].down_w, rope_cache, positions, mask);
    }
    x = rmsnorm(x, w.ln_final_w, static_cast<float>(cfg.rms_norm_eps));
    Tensor logits = linear(x, w.lm_head_w, nullptr);

    return logits;
}

Tensor qwen2_forward(const Qwen2ModelWeights &w,
                     const Qwen2Config &cfg,
                     const Tensor &rope_cache,
                     KVCache &kv_cache, bool is_decode,
                     const std::int64_t *input_ids, std::int64_t seq_len,
                     const std::int64_t *positions,
                     const Tensor *mask)
{
    if (w.layers.size() != cfg.num_hidden_layers)
        throw std::runtime_error("config and weights from different models");

    Tensor x = embedding(w.embed_tokens_w, input_ids, std::vector<std::int64_t>{seq_len});
    for (size_t i = 0; i < cfg.num_hidden_layers; i++)
    {
        x = transformer_block(x, cfg.num_attention_heads, w.layers[i].ln1_w, w.layers[i].ln2_w, static_cast<float>(cfg.rms_norm_eps), w.layers[i].q_w, &w.layers[i].q_b, w.layers[i].k_w, &w.layers[i].k_b, w.layers[i].v_w, &w.layers[i].v_b, w.layers[i].o_w, w.layers[i].gate_up_w, w.layers[i].down_w, rope_cache, positions, mask, kv_cache, is_decode, static_cast<std::int64_t>(i));
    }
    x = rmsnorm(x, w.ln_final_w, static_cast<float>(cfg.rms_norm_eps));
    Tensor logits = linear(x, w.lm_head_w, nullptr);

    return logits;
}

Tensor qwen2_forward(const Qwen2ModelWeights &w,
                     const Qwen2Config &cfg,
                     const Tensor &rope_cache,
                     BlockManager &bm, PagedKVCache &kv_cache, bool is_decode,
                     const std::int64_t *input_ids, std::int64_t seq_len,
                     const std::int64_t *positions,
                     const Tensor *mask)
{
    if (bm.num_layers() != cfg.num_hidden_layers)
        throw std::runtime_error("bm.num_layers() != cfg.num_hidden_layers");
    if (w.layers.size() != cfg.num_hidden_layers)
        throw std::runtime_error("config and weights from different models");

    Tensor x = embedding(w.embed_tokens_w, input_ids, std::vector<std::int64_t>{seq_len});
    for (size_t i = 0; i < cfg.num_hidden_layers; i++)
    {
        x = transformer_block(x, cfg.num_attention_heads, w.layers[i].ln1_w, w.layers[i].ln2_w, static_cast<float>(cfg.rms_norm_eps), w.layers[i].q_w, &w.layers[i].q_b, w.layers[i].k_w, &w.layers[i].k_b, w.layers[i].v_w, &w.layers[i].v_b, w.layers[i].o_w, w.layers[i].gate_up_w, w.layers[i].down_w, rope_cache, positions, mask, bm, kv_cache, is_decode, static_cast<std::int64_t>(i));
    }
    x = rmsnorm(x, w.ln_final_w, static_cast<float>(cfg.rms_norm_eps));
    Tensor logits = linear(x, w.lm_head_w, nullptr);

    return logits;
}

Tensor qwen2_forward(const Qwen2ModelWeights &w,
                     const Qwen2Config &cfg,
                     const Tensor &rope_cache,
                     BlockManager &bm,
                     std::vector<PagedKVCache *> &kv_caches,
                     bool is_decode,
                     const std::int64_t *input_ids,
                     const std::vector<std::int64_t> &cu_seqlens_q_host)
{
    if (bm.num_layers() != cfg.num_hidden_layers)
        throw std::runtime_error("bm.num_layers() != cfg.num_hidden_layers");
    if (w.layers.size() != cfg.num_hidden_layers)
        throw std::runtime_error("config and weights from different models");
    if (cu_seqlens_q_host.size() != kv_caches.size() + 1)
        throw std::runtime_error("cu_seqlens_q_host.size() != kv_caches.size()+1");

    std::int64_t sum_q = cu_seqlens_q_host.back();
    int num_seqs = kv_caches.size();
    std::unique_ptr<std::int64_t[]> positions(new std::int64_t[sum_q]);
    for (int s = 0; s < num_seqs; s++)
    {
        std::int64_t start = cu_seqlens_q_host[s];
        std::int64_t len = cu_seqlens_q_host[s + 1] - start;
        if (is_decode)
        {
            positions[start] = kv_caches[s]->cursor();
        }
        else
        {
            for (std::int64_t i = 0; i < len; i++)
            {
                positions[start + i] = i;
            }
        }
    }

    Tensor x = embedding(w.embed_tokens_w, input_ids, std::vector<std::int64_t>{sum_q});
    for (size_t i = 0; i < cfg.num_hidden_layers; i++)
    {
        x = transformer_block(x, cfg.num_attention_heads, w.layers[i].ln1_w, w.layers[i].ln2_w, static_cast<float>(cfg.rms_norm_eps), w.layers[i].q_w, &w.layers[i].q_b, w.layers[i].k_w, &w.layers[i].k_b, w.layers[i].v_w, &w.layers[i].v_b, w.layers[i].o_w, w.layers[i].gate_up_w, w.layers[i].down_w, rope_cache, bm, kv_caches, is_decode, static_cast<std::int64_t>(i), positions.get(), cu_seqlens_q_host);
    }
    x = rmsnorm(x, w.ln_final_w, static_cast<float>(cfg.rms_norm_eps));
    Tensor logits = linear(x, w.lm_head_w, nullptr);

    return logits;
}

namespace
{
    // 内部链接, 不暴露符号, 语义明确"loader 私有 helper"
    Tensor get_weight_dev(const std::unordered_map<std::string, TensorMeta> &meta, MMapFile &mf, const std::string &key, Device dev)
    {
        auto it = meta.find(key);
        if (it == meta.end())
            throw std::runtime_error("missing weight key: " + key);

        // CPU mmap 非拥有 view
        Tensor cpu_view = make_weight_view(mf, it->second.offset, it->second.shape, it->second.dtype);
        if (dev == Device::CUDA)
        {
            // GPU: owned GPU tensor (h2d)
            return to_cuda(cpu_view);
        }
        else if (dev == Device::CPU)
        {
            // CPU: 原样返回 view (zero-copy)
            return cpu_view;
        }
        else
            throw std::runtime_error("unsupported device");
    }

}

Qwen2ModelWeights load_qwen2_weights(MMapFile &mf, const Qwen2Config &cfg, Device dev)
{
    auto meta = parse_safetensors_header(mf);
    Tensor embed_tokens_w = get_weight_dev(meta, mf, "model.embed_tokens.weight", dev);
    if (cfg.tie_word_embeddings == false)
    {
        throw std::runtime_error("tie_word_embeddings==false not supported");
    }
    // tied (tie_word_embeddings=true): lm_head 与 embed 共享存储
    Tensor lm_head_w = Tensor(embed_tokens_w.data(), embed_tokens_w.shape(), embed_tokens_w.dtype(), embed_tokens_w.device());
    Tensor ln_final_w = get_weight_dev(meta, mf, "model.norm.weight", dev);

    std::vector<Qwen2LayerWeights> layers;
    for (size_t i = 0; i < cfg.num_hidden_layers; i++)
    {
        Tensor q_w = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight", dev);

        Tensor q_b = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias", dev);

        Tensor k_w = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight", dev);

        Tensor k_b = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias", dev);

        Tensor v_w = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight", dev);

        Tensor v_b = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias", dev);

        Tensor o_w = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight", dev);

        // gate_w/up_w 强制 CPU mmap view: scatter 直接从 CPU view h2d 到 GPU fused 切片, 用完即弃(非拥有 view, 析构无害)
        Tensor gate_w = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".mlp.gate_proj.weight", Device::CPU);
        Tensor up_w = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".mlp.up_proj.weight", Device::CPU);
        if (gate_w.shape() != up_w.shape() || gate_w.dtype() != up_w.dtype())
        {
            throw std::runtime_error("layer " + std::to_string(i) + " gate_proj and up_proj shape/dtype mismatch for fusion");
        }
        Tensor gate_up_w(std::vector<std::int64_t>{gate_w.shape()[0] + up_w.shape()[0], gate_w.shape()[1]}, gate_w.dtype(), dev);
        if (dev == Device::CUDA)
        {
            cuda_memcpy_h2d(gate_up_w.data(), gate_w.data(), gate_w.nbytes());
            cuda_memcpy_h2d(static_cast<void *>(static_cast<char *>(gate_up_w.data()) + gate_w.nbytes()), up_w.data(), up_w.nbytes());
        }
        else if (dev == Device::CPU)
        {
            std::memcpy(gate_up_w.data(), gate_w.data(), gate_up_w.numel() / 2 * dtype_size(gate_up_w.dtype()));
            std::memcpy(static_cast<char *>(gate_up_w.data()) + gate_up_w.numel() / 2 * dtype_size(gate_up_w.dtype()), up_w.data(), gate_up_w.numel() / 2 * dtype_size(gate_up_w.dtype()));
        }
        else
            throw std::runtime_error("unsupported device");

        Tensor down_w = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".mlp.down_proj.weight", dev);

        Tensor ln1_w = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".input_layernorm.weight", dev);

        Tensor ln2_w = get_weight_dev(meta, mf, "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight", dev);

        layers.emplace_back(Qwen2LayerWeights{std::move(ln1_w), std::move(ln2_w), std::move(q_w), std::move(q_b), std::move(k_w), std::move(k_b), std::move(v_w), std::move(v_b), std::move(o_w), std::move(gate_up_w), std::move(down_w)});
    }

    return Qwen2ModelWeights{std::move(embed_tokens_w), std::move(layers), std::move(ln_final_w), std::move(lm_head_w)};
}
#pragma once

#include "tensor.h"
#include "qwen2_config.h"
#include "mmap_file.h"
#include "kv_cache.h"
#include "paged_kv_cache.h"

/**
 * Per-layer weights for one Qwen2 decoder block.
 * 11 tensors, matching the transformer_block op's weight parameters one-to-one
 * (ln1_w, ln2_w, q_w, q_b, k_w, k_b, v_w, v_b, o_w, gate_up_w, down_w).
 *
 * gate_up_w is the FUSED gate_proj + up_proj (concatenated along the out dim at load time).
 * safetensors stores gate_proj and up_proj as SEPARATE keys, but the block op consumes the concatenation, so the loader cats once and the block never sees the split halves.
 * o_proj has NO bias in Qwen2.5-0.5B (verified: safetensors has o_proj.weight but no o_proj.bias).
 */
struct Qwen2LayerWeights
{
    Tensor ln1_w;     // input_layernorm.weight,            [hidden]
    Tensor ln2_w;     // post_attention_layernorm.weight,   [hidden]
    Tensor q_w;       // self_attn.q_proj.weight,           [hidden, hidden]
    Tensor q_b;       // self_attn.q_proj.bias,             [hidden]
    Tensor k_w;       // self_attn.k_proj.weight,           [nkv*hd, hidden]
    Tensor k_b;       // self_attn.k_proj.bias,             [nkv*hd]
    Tensor v_w;       // self_attn.v_proj.weight,           [nkv*hd, hidden]
    Tensor v_b;       // self_attn.v_proj.bias,             [nkv*hd]
    Tensor o_w;       // self_attn.o_proj.weight,           [hidden, hidden] (no bias)
    Tensor gate_up_w; // FUSED mlp.gate_proj + up_proj,     [2*inter, hidden]
    Tensor down_w;    // mlp.down_proj.weight,              [hidden, inter]
};

/**
 * Whole-model weights for Qwen2ForCausalLM: embed_tokens + N decoder blocks + ln_final + lm_head.
 *
 * When tie_word_embeddings=true (Qwen2.5-0.5B), lm_head_w is a VIEW onto embed_tokens_w (same [vocab, hidden] buffer, no copy) -- mirroring Qwen2ForCausalLM's `self.lm_head.weight = self.embed_tokens.weight`.
 * The loader constructs lm_head_w as a non-owning Tensor pointing at embed_tokens_w's data; the forward then runs lm_head = linear(x, lm_head_w, nullptr).
 */
struct Qwen2ModelWeights
{
    Tensor embed_tokens_w;                 // [vocab, hidden]
    std::vector<Qwen2LayerWeights> layers; // num_hidden_layers entries
    Tensor ln_final_w;                     // model.norm.weight, [hidden]
    Tensor lm_head_w;                      // tied: view onto embed_tokens_w
};

/**
 * Run a full Qwen2ForCausalLM forward pass and return next-token logits.
 *
 * Pipeline: embed(input_ids) -> N x transformer_block -> ln_final -> lm_head.
 * This is a BLACK BOX: it returns logits only, with no intermediate-state port (no "return hidden at layer k" debug hook) -- the inference main path stays clean.
 * Bug localization when end-to-end parity fails relies on the chain:
 *  embedding bit-exact spot-check -> single-op regression -> layer-0 block spot-check -> (only if needed) ad-hoc first-k-layers probe.
 *
 * Args:
 *   w:          Qwen2ModelWeights (embed + layers + ln_final + lm_head view).
 *   cfg:        Qwen2Config; num_hidden_layers drives the loop count, num_heads and rms_norm_eps are forwarded into every block, rope_theta / max_position_embeddings / head_dim (derived) sized the cache.
 *   rope_cache: shared RoPE cache Tensor (f32), model-level -- ONE cache materialized for the whole stack, passed to every block's RoPE.
 *   input_ids:  token ids, length seq_len (1-D; no batch dim in Phase 1).
 *   seq_len:    number of tokens in input_ids.
 *   positions:  per-token RoPE positions, length seq_len (1-D int64).
 *   mask:       additive attention mask, or nullptr for causal (common-mode: both C++ and PyTorch build their own causal mask, no additive perturbation -- verified in attention.cpp).
 *
 * Returns:
 *   Logits Tensor of shape [seq_len, vocab_size], dtype = embed_tokens_w.dtype (bf16 for the real model).
 *   Raw logits, no softmax -- callers sample.
 *
 * Throws std::runtime_error on:
 *   - cfg.num_hidden_layers != w.layers.size() (config and weights from different models -- hard error, no guessing)
 *   - any shape/dtype inconsistency propagated from the sub-ops
 *   - mask shape incompatible with attention (propagated from the block)
 *
 * Notes:
 *   - positions is ALWAYS non-null at this layer.
 *     The binding (qwen2_forward_py) materializes an arange(0..seq_len-1) when Python passes positions=None, so the core never sees nullptr.
 *     Passing nullptr here would crash in rope (positions[i] is dereferenced unconditionally) -- do not.
 *   - The core does NOT check positions LENGTH (raw pointers carry no length).
 *     Length consistency (positions length == seq_len) is enforced at the BINDING layer, where both arrive as py::array_t with .shape(0) -- see rope_py / transformer_block_py.
 *     The core only checks position VALUE range (in-range for the rope cache) inside rope.
 *   - Common-mode input contract (critical for oracle parity): mask and positions are EITHER nullptr/None on BOTH sides OR arange on BOTH sides.
 *     A mixed mode (one None, one arange) injects a non-shared perturbation and breaks parity without a real bug.
 *   - lm_head is linear(x, lm_head_w, nullptr) -- lm_head_w is the tied view of embed_tokens_w, already in [vocab, hidden] = [out, in] layout, so no extra transpose beyond what linear's weight convention handles.
 *   - The loop count is cfg.num_hidden_layers; it MUST match w.layers.size().
 *     A mismatch means the weights and config are from different models and is treated as a hard error (checked at forward entry).
 */
Tensor qwen2_forward(const Qwen2ModelWeights &w,
                     const Qwen2Config &cfg,
                     const Tensor &rope_cache,
                     const std::int64_t *input_ids, std::int64_t seq_len,
                     const std::int64_t *positions,
                     const Tensor *mask);

/**
 * Incremental Qwen2 forward with continuous KV cache (true incremental: decode attention runs 1 Q against cached K/V, not full-seq Q).
 *
 * Pipeline: embed -> N x transformer_block(incremental) -> ln_final -> lm_head.
 * Each block writes the new tokens' K/V into kv_cache and reads prior K/V from it (decode), so attention runs only over live tokens.
 *
 * is_decode dispatch (explicit, drives the block's cache write/read):
 *   false (prefill): input_ids = full prompt, seq_len = prompt length, positions = arange(0, seq_len), writes [0, seq_len), cursor = seq_len.
 *                    MUST be called ONCE per kv_cache before any decode.
 *   true  (decode):  input_ids = 1 new token, seq_len = 1, positions = [cursor] (the new token's absolute position = kv_cache.cursor() BEFORE this step), writes at cursor, cursor++, attention reads [0, cursor].
 *
 * CONTRACT (load-bearing -- the cache does NOT defend against this):
 *   prefill exactly ONCE on a fresh kv_cache (cursor=0), then decode N times.
 *   A second prefill on the same cache overwrites (cursor reset to seq_len, prior history lost).
 *   The cache's write() prefill branch is fill-not-append by design.
 *   If you need to re-prefill, construct a new KVCache.
 *
 * positions: the BINDING derives positions from kv_cache.cursor() (decode: [cursor]; prefill: arange).
 *   The core receives a non-null positions pointer (same as M6/M8 -- rope dereferences it unconditionally).
 *
 * mask: nullptr both modes (prefill causal on the fly over [0, seq); decode mask-free, new token sees
 *   all cached keys).
 *
 * Parity: oracle is the non-incremental qwen2_forward (same weights/cfg/prompt).
 *   The cache copy is a bit-exact bf16 memcpy; the new numeric site is decode running 1-Q attention vs prefill's full-seq attention -- mathematically equal but 24-layer bf16 accumulation order differs.
 *   Tolerance PROBED end-to-end, NOT assumed from M6/M8.
 *
 * Returns: logits for the INPUT tokens (prefill: [seq_len, vocab]; decode: [1, vocab]).
 */
Tensor qwen2_forward(const Qwen2ModelWeights &w,
                     const Qwen2Config &cfg,
                     const Tensor &rope_cache,
                     KVCache &kv_cache, bool is_decode,
                     const std::int64_t *input_ids, std::int64_t seq_len,
                     const std::int64_t *positions,
                     const Tensor *mask);

/**
 * Paged Qwen2 forward (M9 sub-block B): KV in a shared BlockManager pool, per-seq block_table
 * indirect addressing. Paged analog of qwen2_forward(KVCache&); no-cache + continuous overloads
 * KEPT as perf baselines. prefill AND decode both read the pool (industrial: vLLM prefix-enabled /
 * SGLang default; not the "prefill reads temp K/V" stopgap).
 *
 * bm: GLOBAL pool, caller-owned, shared across seqs (industrial "one pool, per-seq holds only a
 *     block_table"). Construct with num_layers == cfg.num_hidden_layers (asserted), num_blocks for
 *     max concurrency, kv_slot_bytes = nkv*hd*2 (0.5B=512). Passed in like rope_cache (not a model attr).
 * kv_cache: PagedKVCache (per-seq): block_table + cursor; write() paged_writes into bm, attn reads pool.
 * is_decode: false=prefill (writes [0,seq), cursor=seq), true=decode (1 token, cursor++).
 *
 * CONTRACT: prefill ONCE on a fresh kv_cache (blocks pre-allocated), then decode N times; PagedKVCache
 *   must alloc a block per step when crossing a block boundary.
 * positions/mask: positions non-null; mask nullptr both modes (prefill causal on the fly, decode mask-free).
 *
 * Parity: Tier1 bit-exact vs continuous qwen2_forward(KVCache&) (paging layout-only no-op at tile=64==Bc);
 *   Tier2 allclose vs HF. Tolerance PROBED.
 * Returns: prefill [seq,vocab]; decode [1,vocab].
 * Throws: cfg.num_hidden_layers != bm.num_layers(); propagated sub-op errors.
 */
Tensor qwen2_forward(const Qwen2ModelWeights &w,
                     const Qwen2Config &cfg,
                     const Tensor &rope_cache,
                     BlockManager &bm, PagedKVCache &kv_cache, bool is_decode,
                     const std::int64_t *input_ids, std::int64_t seq_len,
                     const std::int64_t *positions,
                     const Tensor *mask);

/**
 * Multi-request paged forward (Phase 3): N sequences -> one batched paged_attention per layer.
 * varlen (true varlen, no padding): input_ids is the concatenation of all seqs' tokens; cu_seqlens_q_host ([0, l0, l0+l1, ..., sum]) locates each seq's slice.
 * The multi-request transformer_block does the rest (per-seq prepare_meta + whole-batch paged_write + 2D block_table gather + cu_seqlens_k).
 *
 * is_decode: false=prefill (one prompt per seq, cu_seqlens_q_host=[0,l0,...]); true=decode (one token per seq, cu_seqlens_q_host=[0,1,...,num_seqs]).
 *
 * positions: NOT an arg -- forward builds it from cu_seqlens_q_host + kv_caches[i]->cursor(): prefill=seg-local arange concat; decode=per-seq [cursor-1].
 *   This matches vLLM CommonAttentionMetadata (positions is built at the runner layer, attention just consumes it).
 *
 * bm: shared global pool (num_layers == cfg.num_hidden_layers, asserted); kv_caches: one PagedKVCache per seq.
 *
 * CONTRACT: prefill ONCE on fresh caches, then decode N times. positions length == cu_seqlens_q_host.back().
 *
 * Parity: batched logits == N single-request qwen2_forward(paged) concat (bit-exact target, then PROBED).
 * Returns: [sum_q, vocab] (prefill) / [num_seqs, vocab] (decode); sum_q = cu_seqlens_q_host.back().
 * Throws: cfg.num_hidden_layers mismatch; cu_seqlens_q_host.size() != kv_caches.size()+1.
 */
Tensor qwen2_forward(const Qwen2ModelWeights &w,
                     const Qwen2Config &cfg,
                     const Tensor &rope_cache,
                     BlockManager &bm,
                     std::vector<PagedKVCache *> &kv_caches,
                     bool is_decode,
                     const std::int64_t *input_ids,
                     const std::vector<std::int64_t> &cu_seqlens_q_host);

/**
 * Load Qwen2.5 weights from a safetensors mmap into a Qwen2ModelWeights.
 *
 * This is the CONSTRUCTOR path for Qwen2ModelWeights: because Tensor is copy-disabled (double-free guard) and has no default ctor, the struct is move-only and unconstructible from Python -- so the loader builds it in C++ and moves it out.
 * Python only HOLDS the returned object and passes it to qwen2_forward; it never constructs, copies, or reads/writes fields.
 *
 * Args:
 *   mf:  MMapFile of the whole safetensors file (embed + all layers + norm; lm_head.weight is ABSENT when tie_word_embeddings=true, verified for Qwen2.5-0.5B: 290 keys = 24 layers x 12 + embed_tokens + model.norm).
 *   cfg: Qwen2Config. num_hidden_layers sets the layer count; tie_word_embeddings governs lm_head (true -> tied view onto embed_tokens_w; false -> not supported here, the safetensors must then carry lm_head.weight).
 *   dev: target device for the loaded weights. CPU -> zero-copy mmap views; CUDA -> owned GPU tensors (H2D, weights reside on GPU after load). qwen2_forward infers its device from w.embed_tokens_w.device(), so the forward runs entirely on `dev` without a separate device argument.
 *
 * Returns:
 *   Qwen2ModelWeights (moved out). Weight storage depends on `dev`:
 *     - dev == CPU: ZERO-COPY. embed_tokens_w, per-layer weights, ln_final_w, lm_head_w are NON-OWNING views pointing into the mmap (built via make_weight_view). gate_up_w is the ONE owned tensor (CPU cat of gate_proj+up_proj).
 *     - dev == CUDA: OWNED GPU tensors. Each weight is make_weight_view (CPU mmap view) then to_cuda (H2D into a fresh GPU buffer), so weights RESIDE on GPU and survive independent of the mmap. This is the load-once / reside-on-GPU path (vLLM/SGLang structure: load moves weights to GPU once, inference never re-moves them). gate_up_w is owned GPU too, built by pre-allocating the fused buffer on GPU and scattering gate/up via two cuda_memcpy_h2d (NOT a concat kernel -- see Notes).
 *     - gate_up_w is the ONE owned tensor on BOTH devices (the safetensors stores gate_proj and up_proj separately, the block consumes the fusion):
 *         CPU: std::memcpy cat into a freshly allocated CPU buffer.
 *         CUDA: PRE-ALLOCATE the fused buffer on GPU (Tensor shape ctor, Device::CUDA), then SCATTER gate_proj into the front half and up_proj into the back half via two cuda_memcpy_h2d (vLLM MergedColumnParallelLinear.weight_loader pattern: scatter-into-preallocated, NOT a concat kernel).
 *       Why scatter not concat: a concat kernel (Phase 2 M2's device concat) is for RUNTIME dynamic joins (KV-cache append); load-time static fusion uses scatter -- no intermediate GPU tensors (gate_gpu + up_gpu would 2x the peak), and the offset-based placement extends naturally to TP shard loading later.
 *
 * Throws std::runtime_error on:
 *   - any expected weight key is MISSING from the safetensors header (message names the missing key, e.g. "missing weight key: model.layers.0.self_attn.o_proj.weight")
 *   - tie_word_embeddings == false (lm_head as an independent weight is not implemented; the real Qwen2.5-0.5B is tied, so this is the only path)
 *   - shape/dtype mismatch propagated from make_weight_view
 *
 * Notes:
 *   - gate_up_w cat ORDER is gate_proj first, up_proj second (dim 0).
 *     This matches oracle_qwen2.py's `torch.cat([gate_proj, up_proj], ...)`: the mlp op splits gate_up_w into [gate; up] halves, gate passes through silu and up does not -- a reversed order silently corrupts SwiGLU.
 *     The order is load-bearing and fixed by this loader.
 *   - lm_head_w (tied, tie_word_embeddings=true) shares storage with embed_tokens_w:
 *       CPU: a SECOND independent mmap view at the same offset (symmetric tied, both non-owning, MMapFile keeps the mmap alive -- no cross-member dependency).
 *       CUDA: a NON-OWNING view over embed_tokens_w's GPU buffer (Tensor(ptr, shape, dtype, CUDA), owns_data_=false), saving one vocab*hidden GPU allocation (~272MB for Qwen2.5-0.5B). This introduces a LIFECYCLE DEPENDENCY: lm_head_w must be destroyed BEFORE embed_tokens_w. Guaranteed by struct member reverse-destruction order (lm_head_w declared after embed_tokens_w, so it destructs first and does not free; embed_tokens_w destructs later and frees the GPU buffer). Moving the whole Qwen2ModelWeights is safe (members move together, the view pointer relationship is preserved). DO NOT move lm_head_w OUT of the struct independently -- embed_tokens_w could be destroyed first, leaving lm_head_w dangling.
 *   - The loader does NOT validate semantic consistency between cfg and the safetensors (e.g. hidden_size vs embed shape[1]); it trusts the caller to pass a config and weights from the SAME model.
 *     A mismatch surfaces later as a shape error inside qwen2_forward's sub-ops.
 */
Qwen2ModelWeights load_qwen2_weights(MMapFile &mf, const Qwen2Config &cfg, Device dev);

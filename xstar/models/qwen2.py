from __future__ import annotations

import torch
from torch import nn

from xstar.layers.embedding import Embedding
from xstar.layers.transformer_block import TransformerBlock
from xstar.layers.rope import RoPE
from xstar.layers.rmsnorm import RMSNorm
from xstar.layers.linear import Linear


class Qwen2Model(nn.Module):
    """
    The Qwen2 decoder backbone: N pre-norm transformer blocks followed by a
    final RMSNorm. No embedding table and no language head -- this is the pure
    hidden-state stack; embeddings live in Qwen2ForCausalLM so the tied
    language head can share the embedding tensor directly.

    A single RoPE instance is built here from (rope_theta, head_dim,
    max_position_embeddings) and injected into every block, so the rotary
    cache is materialized once for the whole stack. The final norm is named
    ``ln_final`` here and corresponds to HF's ``model.norm``.
    """

    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        rope_theta: float | None = 10_000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct the decoder backbone.

        Args:
            max_position_embeddings: Size of the RoPE cache, an upper bound on
                sequence length -- not a fixed input length. Shorter inputs
                just index a prefix of the cache.
            hidden_size: Model dimensionality (d_model).
            num_hidden_layers: Number of stacked transformer blocks.
            num_heads: Number of query heads.
            num_key_value_heads: Number of KV heads (GQA when < num_heads).
            intermediate_size: Inner dimension of the SwiGLU MLP.
            rms_norm_eps: Epsilon for the RMSNorm layers.
            rope_theta: Base theta for RoPE. If ``None``, no positional encoder
                is built (blocks run without RoPE).
            device/dtype: Forwarded to every sub-module so all weights live on
                the same device/dtype as the reference (zero-conversion loading).
        """
        super(Qwen2Model, self).__init__()
        head_dim = hidden_size // num_heads
        self.positional_encoder = (
            RoPE(rope_theta, head_dim, max_position_embeddings, device=device)
            if rope_theta is not None
            else None
        )

        # 必须使用 ModuleList
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    num_key_value_heads,
                    intermediate_size,
                    rms_norm_eps,
                    self.positional_encoder,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.ln_final = RMSNorm(hidden_size, rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the decoder backbone: stack of pre-norm blocks then the final RMSNorm.

        Args:
            x: Hidden states of shape ``(..., seq_len, hidden_size)`` with
                arbitrary leading batch dims (typically the embedded tokens).
            attention_mask: Optional *additive* mask, forwarded to every block.
                If ``None``, each block builds its own causal mask.
            token_positions: Optional per-token positions, forwarded to RoPE.
                If ``None``, RoPE uses ``0..seq_len-1``.

        Returns:
            Hidden states of shape ``(..., seq_len, hidden_size)`` after the
            final RMSNorm. No language head is applied -- that is the caller's
            (Qwen2ForCausalLM) job.
        """
        for layer in self.layers:
            x = layer(x, attention_mask, token_positions)

        return self.ln_final(x)


class Qwen2ForCausalLM(nn.Module):
    """
    Qwen2 for causal language modeling: token embeddings + decoder backbone +
    a language head, with the head's weights tied to the embedding table.

    This is the top-level module parity-checked against HF's
    ``Qwen2ForCausalLM``. The embedding table lives here (not in Qwen2Model) so
    that ``lm_head.weight = embed_tokens.weight`` makes the head share the exact
    same tensor object as the embeddings -- matching Qwen2.5-0.5B's
    ``tie_word_embeddings=True`` and halving the embedding parameter memory.
    The backbone (Qwen2Model) holds the blocks and final norm only.
    """

    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        rope_theta: float | None = 10_000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct the causal LM.

        Args:
            vocab_size: Vocabulary size; both the embedding table's first dim
                and the language head's output dim.
            max_position_embeddings: Size of the RoPE cache, an upper bound on
                sequence length -- not a fixed input length. Shorter inputs
                just index a prefix of the cache.
            hidden_size: Model dimensionality (d_model).
            num_hidden_layers: Number of stacked transformer blocks.
            num_heads: Number of query heads.
            num_key_value_heads: Number of KV heads (GQA when < num_heads).
            intermediate_size: Inner dimension of the SwiGLU MLP.
            rms_norm_eps: Epsilon for the RMSNorm layers.
            rope_theta: Base theta for RoPE. If ``None``, no positional encoder
                is built (blocks run without RoPE).
            device/dtype: Forwarded to every sub-module so all weights live on
                the same device/dtype as the reference (zero-conversion loading).

        Notes:
            ``lm_head.weight`` is reassigned to ``embed_tokens.weight`` after
            construction, so the head holds no independent parameters -- load
            the embedding weights once and the head is correct automatically.
        """
        super(Qwen2ForCausalLM, self).__init__()
        self.embed_tokens = Embedding(
            vocab_size, hidden_size, device=device, dtype=dtype
        )
        self.model = Qwen2Model(
            max_position_embeddings,
            hidden_size,
            num_hidden_layers,
            num_heads,
            num_key_value_heads,
            intermediate_size,
            rms_norm_eps,
            rope_theta,
            device,
            dtype,
        )
        self.lm_head = Linear(
            hidden_size, vocab_size, bias=False, device=device, dtype=dtype
        )

        # 权重绑定 (Weight Tying)
        # Weight tying: the language head shares the token embedding weights instead
        # of holding an independent copy. After this assignment lm_head.weight IS
        # embed_tokens.weight (same tensor), so loading the embedding weights once is enough
        # -- the head updates automatically. This matches Qwen2.5-0.5B's
        # tie_word_embeddings=True and halves the embedding parameter memory.
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run a full forward pass and return next-token logits.

        Embeds the input ids, runs the decoder backbone, then applies the tied
        language head. The output is raw logits (no softmax) -- callers run
        softmax / sampling themselves at inference time.

        Args:
            x: Token ids of shape ``(..., seq_len)`` with arbitrary leading
                batch dims.
            attention_mask: Optional *additive* mask, forwarded through the
                backbone to every block. If ``None``, each block builds its own
                causal mask.
            token_positions: Optional per-token positions, forwarded to RoPE.
                If ``None``, RoPE uses ``0..seq_len-1``.

        Returns:
            Logits of shape ``(..., seq_len, vocab_size)``.
        """
        x = self.embed_tokens(x)
        x = self.model(x, attention_mask, token_positions)
        # Softmax 应该在推理（Inference）时由用户按需调用
        return self.lm_head(x)

from __future__ import annotations

import torch
from torch import nn
from xstar.layers.rope import RoPE
from xstar.layers.attention import GroupedQueryAttention
from xstar.layers.mlp import SwiGLU
from xstar.layers.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """
    A pre-norm Transformer decoder block with two residual sub-layers.

    Matches Qwen2DecoderLayer: a self-attention sub-layer and an MLP sub-layer,
    each preceded by its own RMSNorm (pre-norm). The RoPE instance is shared
    across all blocks -- built once in Qwen2Model and injected here, so the
    rotary cache is not duplicated per layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        positional_encoder: RoPE | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct the pre-norm decoder block.

        Args:
            hidden_size: Block input dimensionality (d_model).
            num_heads: Number of query heads in attention.
            num_key_value_heads: Number of KV heads (num_kv < num_heads
                triggers GQA; each KV head is shared by
                num_heads // num_key_value_heads query heads).
            intermediate_size: Inner dimension of the SwiGLU MLP.
            rms_norm_eps: Epsilon for the RMSNorm layers (read from the
                reference config, e.g. 1e-6).
            positional_encoder: Shared RoPE instance injected into attention.
                Built once in Qwen2Model and passed to every block, so the
                rotary cache lives once for the whole stack rather than once
                per layer.
            device/dtype: Forwarded to every sub-module so all weights live on
                the same device/dtype as the reference (zero conversion).
        """
        super(TransformerBlock, self).__init__()
        self.input_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, device=device, dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=rms_norm_eps, device=device, dtype=dtype
        )
        self.attn = GroupedQueryAttention(
            hidden_size,
            num_heads,
            num_key_value_heads,
            positional_encoder,
            device=device,
            dtype=dtype,
        )
        self.mlp = SwiGLU(hidden_size, intermediate_size, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pre-norm decoder block forward: two residual sub-layers.

            x = x + attn(input_layernorm(x))
            x = x + mlp(post_attention_layernorm(x))

        The second sub-layer normalizes the *post-attention* residual (the
        output of the first sub-layer), not the original input -- matching HF's
        Qwen2DecoderLayer, where ``post_attention_layernorm`` runs on the
        updated hidden states after the attention residual.

        Args:
            x: Hidden states of shape ``(..., seq_len, hidden_size)`` with
                arbitrary leading batch dims.
            attention_mask: Optional *additive* mask, forwarded to attention.
                If ``None``, attention builds its own causal mask.
            token_positions: Optional per-token positions of shape
                ``(..., seq_len)``, forwarded to attention (and thus RoPE). If
                ``None``, RoPE uses ``0..seq_len-1``.

        Returns:
            Hidden states of the same shape as ``x``.
        """
        # Pre-norm
        x = x + self.attn(self.input_layernorm(x), attention_mask, token_positions)
        return x + self.mlp(self.post_attention_layernorm(x))

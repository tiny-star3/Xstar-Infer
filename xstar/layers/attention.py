from __future__ import annotations

import warnings

import math
import torch
from torch import nn
from einops import rearrange, einsum, repeat
from xstar.layers.rope import RoPE
from xstar.layers.linear import Linear


def softmax(
    input: torch.Tensor, dim: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    Numerically stable softmax over the ``dim`` axis.

    Subtracts the per-axis max before exponentiation so the largest exponent
    is 0, which keeps ``exp`` from overflowing even for large logits. The max
    shift is a constant per slice so it cancels in the normalized sum and does
    not change the result.

    Args:
        input: Tensor of arbitrary shape.
        dim: The axis to normalize over (e.g. ``-1`` for the key axis in
            attention scores).
        dtype: Optional target dtype to cast to *before* computing. Attention
            passes ``torch.float32`` here so the max-shift, exp and sum run in
            full precision; the caller then downcasts the weights to bf16 before
            the weighted sum over V. If ``None``, ``input`` keeps its dtype.

    Returns:
        A tensor of the same shape as ``input`` with the values along ``dim``
        summing to 1.
    """
    if dtype is not None:
        input = input.to(dtype)
    input_max = torch.max(input, dim=dim, keepdim=True).values
    input_exp = torch.exp(input - input_max)
    return input_exp / torch.sum(input_exp, dim=dim, keepdim=True)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    Query heads share key/value heads in groups: each of the num_key_value_heads
    KV heads is reused by num_heads // num_key_value_heads query heads, cutting KV
    size vs multi-head attention while keeping query capacity. The shared RoPE is
    injected once and reused across layers to avoid duplicating the rotary cache.
    All projections are placed on the same device/dtype as the reference so weight
    loading is a zero-conversion copy.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        positional_encoder: RoPE | None = None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            hidden_size: Dimensionality of the block input (d_model)
            num_heads: Number of query heads
            num_key_value_heads: Number of key/value heads (num_kv < num_heads
                triggers GQA; each KV head is shared by
                num_heads // num_key_value_heads query heads)
            positional_encoder: Shared RoPE instance. Injected once and shared
                across all layers instead of each layer building its own, which
                avoids duplicating the rotary cache across blocks
            device/dtype: Forwarded to every projection so all weights live on
                the same device/dtype as the reference (zero conversion)
        """
        super(GroupedQueryAttention, self).__init__()
        # 所有层共享同一个旋转位置编码缓存。这不仅节省显存，还减少了重复计算
        if positional_encoder is None:
            warnings.warn("No positional encoder provided", stacklevel=2)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.num_key_value_heads = num_key_value_heads

        self.q_proj = Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.o_proj = Linear(
            num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.positional_encoder: RoPE | None = positional_encoder  # RoPE

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply grouped-query attention to the input hidden states.

        Args:
            x: Input hidden states of shape (..., seq_len, hidden_size) with arbitrary
                leading batch dims.
            attention_mask: Optional *additive* mask broadcastable to
                (..., num_heads, seq_len, seq_len). Hidden positions must hold a large
                negative value (finfo.min) so they zero out after softmax. If None, a
                causal mask is built on the fly -- the layer is always causal even
                without an external mask.
            token_positions: Optional per-token positions of shape (..., seq_len)
                forwarded to the positional encoder (RoPE). If None, RoPE uses
                0..seq_len-1.

        Returns:
            Output hidden states of shape (..., seq_len, hidden_size), the output
            projection applied to the concatenated multi-head attention output.

        Notes:
            RoPE is applied before KV repetition so each KV head is rotated once and
            then broadcast, rather than rotating every repeated copy. Softmax runs in
            float32, then the weights are downcast to bf16 and the weighted sum over V
            is computed in bf16, matching the reference attention path.

        """
        # 使用解包语法获取前导维度
        *batch_dims, seq_len, hidden_size = x.size()
        assert hidden_size == self.hidden_size

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, head_dim).
        Q = rearrange(Q, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        K = rearrange(
            K, "... seq (heads d) -> ... heads seq d", heads=self.num_key_value_heads
        )
        V = rearrange(
            V, "... seq (heads d) -> ... heads seq d", heads=self.num_key_value_heads
        )

        if self.positional_encoder is not None:  # RoPE is enabled
            Q = self.positional_encoder(Q, token_positions)
            K = self.positional_encoder(K, token_positions)

        # Repeat each KV head to match the number of query heads (GQA).
        # With num_heads query heads and num_key_value_heads KV heads, each KV head is reused by num_heads // num_key_value_heads query heads.
        # einops.repeat MATERIALIZES a new tensor (a real copy, NOT a view) with the rep axis inserted between k_heads and seq; rearrange then merges (k_heads rep) into the head axis so Q and K/V line up head-for-head, yielding head order (kv0, kv0, ..., kv1, kv1, ...) -- each KV head contiguous for `rep` slots.
        #
        # (expand + rearrange would be a zero-copy view alternative, but this path uses repeat; the materialization is the honest cost. The C++ port does NOT repeat at all -- it indexes the shared KV by h/rep, which is the zero-copy version of this step.)
        #
        # This must run *after* RoPE: rotary embeddings are per-position and identical for every copy of a shared KV head, so rotating once and broadcasting is cheaper (and matches the reference) than rotating every repeated copy.
        assert self.num_heads % self.num_key_value_heads == 0
        K = repeat(
            K,
            "... k_heads seq dim -> ... k_heads rep seq dim",
            rep=self.num_heads // self.num_key_value_heads,
        )
        V = repeat(
            V,
            "... v_heads seq dim -> ... v_heads rep seq dim",
            rep=self.num_heads // self.num_key_value_heads,
        )
        K = rearrange(
            K,
            "... k_heads rep seq dim -> ... (k_heads rep) seq dim",
            k_heads=self.num_key_value_heads,
        )
        V = rearrange(
            V,
            "... v_heads rep seq dim -> ... (v_heads rep) seq dim",
            v_heads=self.num_key_value_heads,
        )

        qk = einsum(
            Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
        ) / math.sqrt(Q.shape[-1])
        if attention_mask is not None:
            qk = qk + attention_mask
        else:
            # 根据当前的 sequence_length 实时生成。它能完美适配变长输入，且通过 (None,) * len(batch_dims) 这种写法，能够自动处理任意数量的 Batch 维度（即支持多维广播）
            # Construct causal mask
            iota = torch.arange(seq_len, device=x.device)
            qi = rearrange(iota, "query -> query 1")
            kj = rearrange(iota, "key   -> 1   key")
            # 生成了一个下三角矩阵, 当 query_index >= key_index 时为 True（可见）, 当 query_index < key_index 时为 False（不可见，即未来信息）
            causal_mask = qi >= kj  # (query, key)
            # 在 PyTorch 索引中，None 等同于 np.newaxis，作用是增加一个长度为 1 的新维度
            # 如果 len(batch_dims) 是 2, 这个表达式的结果就是 (None, None), 生成的索引元组是 (None, None, ...)
            causal_mask = causal_mask.__getitem__(
                (None,) * len(batch_dims) + (...,)
            )  # Add appropriate leading dimensions
            qk = torch.where(causal_mask, qk, float("-inf"))

        qk = softmax(qk, -1, torch.float32).to(V.dtype)

        attn_output = einsum(qk, V, "... queries keys, ... keys d_v -> ... queries d_v")

        # Concatenate the attention output from all heads.
        # (..., sequence_length, num_heads * d_v).
        attn_output = rearrange(
            attn_output, "... heads seq d_v -> ... seq (heads d_v)"
        ).contiguous()

        # Apply the output projection
        output = self.o_proj(attn_output)
        return output

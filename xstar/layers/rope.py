from __future__ import annotations

import torch
from torch import nn
from einops import einsum


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position by rotating each of the ``dim/2`` 2-D subspaces of the
    query/key vectors by an angle proportional to the token's position. Pairs
    are formed split-half style: the first and second halves are paired as
    ``(x_i, x_{i+d/2})``. This is mathematically equivalent to HF's
    ``rotate_half`` formulation (``x*cos + rotate_half(x)*sin``) -- it is just
    written in expanded form, computing the two output halves directly as
    ``x1*cos - x2*sin`` and ``x2*cos + x1*sin``.

    The cos/sin table is precomputed once in float32 and stored as a buffer of
    shape ``(2, max_seq_len, dim/2)`` (leading axis stacks ``[cos, sin]``). It
    is non-persistent because it is a deterministic function of
    ``(theta, dim, max_seq_len)`` and can be rebuilt. ``max_seq_len`` is an
    upper bound on the cache size, not a fixed input length -- shorter
    sequences just index a prefix.

    There is no ``dtype`` parameter: the cache stays float32 for precision and
    is downcast to the input dtype only at use time, once per forward.
    """

    def __init__(
        self,
        theta: float,
        dim: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """
        Construct the RoPE module and create buffers if needed
        Args:
            theta: float  Θ value for the RoPE
            dim: int  dimension of query and key vectors
            max_seq_len: int  Maximum sequence length that will be input
            device: torch.device | None = None  Device to store the buffer on
        """
        super(RoPE, self).__init__()
        # 使用 static method 在初始化时预先计算出所有可能的 cos 和 sin 值，并存储在 _freq_cis_cache 缓冲区（Buffer）中
        self.register_buffer(
            "_freq_cis_cache",
            RoPE._init_cache(max_seq_len, dim, theta, device),
            persistent=False,
        )

    @staticmethod
    def _init_cache(
        max_seq_len: int, dim: int, theta: float, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Build the cos/sin cache of shape ``(2, max_seq_len, dim/2)``.

        Rows are positions, columns are the ``dim/2`` frequency channels; the
        leading axis stacks ``[cos, sin]``.
        """

        assert dim % 2 == 0

        d = torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim
        freqs = torch.tensor(theta, device=device) ** -d
        t = torch.arange(max_seq_len, device=device, dtype=torch.int64)

        # 得到了每个位置、每个维度的旋转角度 mθi
        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply rotary position embeddings to ``x``.

        Args:
            x: Tensor of shape ``(..., seq_len, dim)`` with arbitrary leading
                batch dims (from attention, ``(..., heads, seq_len, head_dim)``).
            token_positions: Optional positions of shape ``(..., seq_len)``.
                If given, the cos/sin rows at these positions are gathered;
                otherwise positions ``0..seq_len-1`` are used.

        Returns:
            Tensor of the same shape as ``x`` with each split-half pair rotated
            by its position angle.
        """

        # RoPE 的数学原理是将 d 维向量看作 d/2 个二维平面的旋转
        # 分块式 (Split-Half)：将向量分成前半部分 x1 和后半部分 x2, 将 (x1 ,x1+d/2) 配对旋转
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        if token_positions is not None:
            # 根据传入的 token_positions（每个 Token 的位置 ID），从缓存中提取对应的 cos 和 sin
            cos, sin = (
                self._freq_cis_cache[:, token_positions, :].to(dtype=x.dtype).unbind(0)
            )
        else:
            seq_len = x.size(-2)
            cos, sin = self._freq_cis_cache[:, :seq_len, :].to(dtype=x.dtype).unbind(0)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = torch.concat((x1_rot, x2_rot), dim=-1)
        return result

    def extra_repr(self) -> str:
        return f"max_seq_len={self._freq_cis_cache.shape[1]}, dim/2={self._freq_cis_cache.shape[2]}"

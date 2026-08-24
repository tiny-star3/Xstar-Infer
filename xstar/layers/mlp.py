from __future__ import annotations

import torch
from torch import nn
from xstar.layers.linear import Linear


class SwiGLU(nn.Module):
    """
    SwiGLU activation MLP used by Qwen2: ``down(silu(gate(x)) * up(x))``.

    The gate and up projections are fused into a single ``Linear`` of width
    ``2 * intermediate_size`` and split along the last axis at runtime, so the
    block issues one GEMM instead of two. Both projections are bias-free.

    SwiGLU replaces the GeLU-style nonlinearity of the original GLU with SiLU
    (``x * sigmoid(x)``); here ``torch.sigmoid(gate) * gate`` is written
    explicitly instead of calling ``F.silu`` to make the activation literal and
    keep the path dependency-free for the C++/CUDA port.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            hidden_size: Block input dimensionality (d_model).
            intermediate_size: Inner dimension of the MLP (per-projection width
                before the 2x fusion).
            device/dtype: Forwarded to every projection so weights live on the
                same device/dtype as the reference (zero conversion).
        """
        super(SwiGLU, self).__init__()
        # 合并 gate_proj 和 up_proj, 执行一次大的矩阵乘法（GEMM）比执行两次小的矩阵乘法效率更高
        self.gate_up_proj = Linear(
            hidden_size, 2 * intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.down_proj = Linear(
            intermediate_size, hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SwiGLU MLP.

        Args:
            x: Hidden states of shape ``(..., hidden_size)``.

        Returns:
            Output of shape ``(..., hidden_size)``.

        Notes:
            ``chunk(2, dim=-1)`` splits the fused projection into ``gate`` then
            ``value`` -- the order must match the fused weight layout, which is
            ``[gate_proj ; up_proj]`` along the output axis.
        """
        # 一次投影得到 2*intermediate_size 维度
        gate_up_proj = self.gate_up_proj(x)
        # 拆分为 gate (对应原 gate_proj) 和 value (对应原 up_proj)
        gate, value = gate_up_proj.chunk(2, dim=-1)
        return self.down_proj((gate) * torch.sigmoid(gate) * value)

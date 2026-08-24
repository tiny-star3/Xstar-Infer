from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root-Mean-Square Layer Normalization (RMSNorm).

    Unlike LayerNorm, RMSNorm drops the mean subtraction and only rescales by
    the RMS of the activations, then applies a learned per-channel gain
    (``self.weight``, init to ones). Used in every pre-norm position in Qwen2.

    The reduction runs in float32 regardless of input dtype: the input is
    upcast before computing the squared mean and rsqrt, then downcast back
    before applying the gain. This matches the reference numerics -- the
    bf16 rsqrt would lose too much precision in the normalization scale.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct the RMSNorm module

        Args:
            hidden_size: int  Hidden dimension of the model
            eps: float = 1e-5  Epsilon value for numerical stability
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm along the last axis.

        Args:
            x: Hidden states of shape ``(..., hidden_size)`` with arbitrary
                leading batch dims.

        Returns:
            Normalized tensor of the same shape, scaled by ``self.weight`` and
            cast back to the input dtype.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # rsqrt 是“平方根倒数”
        # 在 GPU 指令集中，计算 rsqrt 然后进行一次乘法，比计算 sqrt 然后进行一次除法要快得多。除法在底层硬件中是非常昂贵的运算
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        result = self.weight * x.to(in_dtype)

        return result

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"

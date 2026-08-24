from __future__ import annotations

import torch
from torch import nn
from einops import einsum


class Linear(nn.Module):
    """
    A from-scratch linear layer (``y = xW^T + b``) built without
    ``nn.Linear``.

    The weight is allocated with ``torch.empty`` and filled later by an
    external ``copy_`` load, so the construction-time init value is irrelevant
    (matches the project's "build structure, load values" convention). The
    forward pass uses an ``einsum`` contraction instead of ``F.linear``: this
    makes the row/column layout explicit and keeps the implementation
    dependency-free, which matters once the same matmul is rewritten by hand in
    C++/CUDA for Phase 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a linear transformation module

        Args:
            in_features: int  final dimension of the input
            out_features: int  final dimension of the output
            bias: bool  If set to ``False``, the layer will not learn an additive bias. Default: ``True``
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super(Linear, self).__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation ``y = x W^T + b``.

        Args:
            x: Input of shape ``(..., in_features)`` with arbitrary leading
                batch dims.

        Returns:
            Output of shape ``(..., out_features)``. The bias is added only
            when present (``o_proj`` and most MLP projections pass
            ``bias=False``); otherwise a literal ``0`` is added, which is a
            no-op broadcast.
        """
        return einsum(
            x,
            self.weight,
            "... in_features, out_features in_features -> ... out_features",
        ) + (self.bias if self.bias is not None else 0)

    def extra_repr(self) -> str:
        return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"

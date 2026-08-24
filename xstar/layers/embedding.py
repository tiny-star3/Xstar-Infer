from __future__ import annotations

import torch
from torch import nn


class Embedding(nn.Module):
    """
    A token embedding table with direct weight indexing.

    Built deliberately thin instead of subclassing ``nn.Embedding``: the table
    is a single ``nn.Parameter`` of shape ``(vocab_size, hidden_size)`` allocated
    with ``torch.empty`` (the values are filled later by an external
    ``copy_`` load, so the construction-time init value is irrelevant), and the
    forward pass is a plain ``self.weight[token_ids]`` lookup.

    Exposing the table as ``self.weight`` (rather than hiding it behind
    ``nn.Embedding.weight``) makes weight tying direct: the language head can be
    bound to the exact same tensor via ``lm_head.weight = embed_tokens.weight``.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an embedding module

        Args:
            vocab_size: int  Size of the vocabulary
            hidden_size: int  Dimension of the embedding vectors, i.e., 𝑑model
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, device=device, dtype=dtype)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up the embedding vector for each token id.

        Args:
            token_ids: Integer tensor of shape ``(..., seq_len)`` with arbitrary
                leading batch dims. Each entry is a vocabulary index.

        Returns:
            Embedding tensor of shape ``(..., seq_len, hidden_size)`` gathered
            from ``self.weight``.

        Notes:
            ``token_ids.long()`` casts to int64 before indexing, since advanced
            indexing on ``self.weight`` requires an integer index tensor. The
            cast is a no-op if the input is already an integer type.
        """
        return self.weight[token_ids.long()]

    def extra_repr(self) -> str:
        return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"

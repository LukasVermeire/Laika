"""Shared utilities for spatial head modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Learned attention pooling: (B, L, C) -> (B, C)."""

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Linear(channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention pooling: ``(B, L, C) -> (B, C)``."""
        # x: (B, L, C)
        attn_logits = self.query(x)  # (B, L, 1)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, L, 1)
        pooled = (x * attn_weights).sum(dim=1)  # (B, C)
        return pooled


def apply_output_activation(
    out: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """Apply the output activation function.

    Parameters
    ----------
    out
        Output tensor.
    activation
        Activation name: ``"softplus"`` or ``"linear"``.

    Returns
    -------
    Activated tensor.
    """
    if activation == "softplus":
        return F.softplus(out)
    if activation != "linear":
        raise ValueError(
            f"Unknown output_activation: {activation!r}. "
            "Expected 'softplus' or 'linear'."
        )
    return out

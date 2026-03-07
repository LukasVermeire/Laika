"""Hypernetwork-style head inspired by Scooby."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_head
from .base import SpatialHead
from ._utils import AttentionPool, apply_output_activation


@register_head("hyperconv")
class HyperConvHead(SpatialHead):
    """Hypernetwork-style head inspired by Scooby.

    Cell embeddings are mapped via an MLP to per-cell linear projection weights,
    which are applied over sequence positions via einsum. This is position-aware
    (like cross-attention) but lightweight (like FiLM).

    Parameters
    ----------
    d_model
        Trunk projection dimension.
    hidden_dim
        Hidden dimension in the weight generator MLP.
    weight_dropout
        Dropout in weight generator MLP.
    pooling
        Pooling over sequence positions: ``"mean"`` or ``"attention"``.
    dropout
        General dropout rate.
    output_activation
        Output activation function. (``"softplus"`` or ``"linear"``)
    spatial_emb_dropout
        Element-wise dropout rate for cell embeddings.
    trunk_emb_dropout
        Element-wise dropout rate for trunk embeddings.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 256,
        weight_dropout: float = 0.2,
        pooling: str = "mean",
        dropout: float = 0.1,
        output_activation: str = "softplus",
        spatial_emb_dropout: float = 0.0,
        trunk_emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.weight_dropout = weight_dropout
        self.pooling = pooling
        self.dropout = dropout
        self.output_activation = output_activation
        self.spatial_emb_dropout = spatial_emb_dropout
        self.trunk_emb_dropout = trunk_emb_dropout
        self.spatial_emb_drop = nn.Dropout(spatial_emb_dropout)
        self.trunk_emb_drop = nn.Dropout(trunk_emb_dropout)

    def setup(
        self,
        input_seq_len: int,
        trunk_channels: int,
        spatial_emb_dim: int,
    ) -> None:
        """Build layers (called by Laika after model dimensions are known)."""
        # Trunk projection
        self.trunk_proj = nn.Linear(trunk_channels, self.d_model)
        self.trunk_norm = nn.LayerNorm(self.d_model)
        self.trunk_drop = nn.Dropout(self.dropout)

        # Weight generator: cell embedding -> projection weights + bias
        self.weight_generator = nn.Sequential(
            nn.Linear(spatial_emb_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.weight_dropout),
            nn.Linear(self.hidden_dim, self.d_model + 1),  # +1 for bias
        )

        # Zero-initialize the last linear layer (Scooby pattern)
        last_linear = self.weight_generator[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

        # Optional attention pooling over sequence positions
        if self.pooling == "attention":
            self.seq_attn_pool = AttentionPool(1)

    def forward(
        self, trunk_emb: torch.Tensor, cell_embs: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        trunk_emb
            (B, L, channels) pooled trunk embeddings.
        cell_embs
            (B, N, emb_dim) spatial cell embeddings.

        Returns
        -------
        (B, N) predictions.
        """
        # Element-wise embedding dropout
        cell_embs = self.spatial_emb_drop(cell_embs)
        trunk_emb = self.trunk_emb_drop(trunk_emb)

        # Trunk projection: (B, L, C) -> (B, L, d_model)
        x = self.trunk_proj(trunk_emb)
        x = self.trunk_norm(x)
        x = self.trunk_drop(x)

        # Weight generator: (B, N, emb_dim) -> (B, N, d_model + 1)
        w = self.weight_generator(cell_embs)
        weights = w[..., :-1]  # (B, N, d_model)
        biases = w[..., -1]  # (B, N)

        # Per-cell linear projection over positions:
        # (B, L, d_model) x (B, N, d_model) -> (B, N, L)
        scores = torch.einsum("bld, bnd -> bnl", x, weights) + biases.unsqueeze(-1)

        # Pool over sequence positions -> (B, N)
        if self.pooling == "mean":
            out = scores.mean(dim=-1)
        elif self.pooling == "attention":
            # (B, N, L) -> (B*N, L, 1) for AttentionPool
            B, N, L = scores.shape
            out = self.seq_attn_pool(
                scores.reshape(B * N, L, 1)
            ).reshape(B, N)  # (B, N)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling!r}")

        return apply_output_activation(out, self.output_activation)

    @property
    def name(self) -> str:
        return "hyperconv"

    @property
    def config(self) -> dict:
        return {
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "weight_dropout": self.weight_dropout,
            "pooling": self.pooling,
            "dropout": self.dropout,
            "output_activation": self.output_activation,
            "spatial_emb_dropout": self.spatial_emb_dropout,
            "trunk_emb_dropout": self.trunk_emb_dropout,
        }

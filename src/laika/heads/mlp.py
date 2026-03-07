"""MLP head with learned attention pooling."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import register_head
from ._utils import AttentionPool, apply_output_activation
from .base import SpatialHead


@register_head("mlp")
class MLPHead(SpatialHead):
    """MLP head with learned attention pooling.

    Parameters
    ----------
    d_model
        Projection dimension.
    hidden_dim
        Hidden dimension.
    dropout
        Dropout rate.
    output_activation
        Output activation function. ("softplus" or "linear")
    spatial_emb_dropout
        Element-wise dropout rate for cell embeddings.
    trunk_emb_dropout
        Element-wise dropout rate for trunk embeddings.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        output_activation: str = "softplus",
        spatial_emb_dropout: float = 0.0,
        trunk_emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
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
        # Trunk reduction: project channels then attention-pool over sequence
        self.trunk_proj = nn.Linear(trunk_channels, self.d_model)
        self.trunk_norm = nn.LayerNorm(self.d_model)
        self.attn_pool = AttentionPool(self.d_model)

        # Fusion MLP: gene_vector concat cell_emb -> scalar
        fusion_input_dim = self.d_model + spatial_emb_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self, trunk_emb: torch.Tensor, cell_embs: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        trunk_emb
            (B, pooled_seq_len, channels) pooled trunk embeddings.
        cell_embs
            (B, N, emb_dim) spatial cell embeddings.

        Returns
        -------
        (B, N) predictions.
        """
        B, N, _ = cell_embs.shape

        # Element-wise embedding dropout
        cell_embs = self.spatial_emb_drop(cell_embs)
        trunk_emb = self.trunk_emb_drop(trunk_emb)

        # Trunk reduction: (B, L, C) -> (B, d_model)
        x = self.trunk_proj(trunk_emb)  # (B, L, d_model)
        x = self.trunk_norm(x)  # (B, L, d_model)
        gene_vec = self.attn_pool(x)  # (B, d_model)

        # Broadcast gene vector to match cells: (B, d_model) -> (B, N, d_model)
        gene_vec = gene_vec.unsqueeze(1).expand(-1, N, -1)

        # Fusion: concat and MLP
        fused = torch.cat([gene_vec, cell_embs], dim=-1)  # (B, N, d_model + emb_dim)
        out = self.fusion(fused).squeeze(-1)  # (B, N)

        return apply_output_activation(out, self.output_activation)

    @property
    def name(self) -> str:
        return "mlp"

    @property
    def config(self) -> dict:
        return {
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "output_activation": self.output_activation,
            "spatial_emb_dropout": self.spatial_emb_dropout,
            "trunk_emb_dropout": self.trunk_emb_dropout,
        }

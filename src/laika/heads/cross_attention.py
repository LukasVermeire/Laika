"""Cross-attention head: cells attend to trunk sequence positions."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import register_head
from ._utils import apply_output_activation
from .base import SpatialHead


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        """
        Parameters
        ----------
        d_model
            Embedding dimension.
        num_heads
            Number of attention heads.
        ffn_dim
            FFN hidden dimension.
        dropout
            Dropout rate.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self, cell_repr: torch.Tensor, trunk_repr: torch.Tensor
    ) -> torch.Tensor:
        """Apply cross-attention.

        Parameters
        ----------
        cell_repr
            ``(B, N, d_model)`` queries.
        trunk_repr
            ``(B, L, d_model)`` keys/values.

        Returns
        -------
        torch.Tensor
            ``(B, N, d_model)`` updated queries.
        """
        # Pre-norm cross-attention
        x = self.norm1(cell_repr)
        x, _ = self.cross_attn(query=x, key=trunk_repr, value=trunk_repr)
        cell_repr = cell_repr + self.drop1(x)

        # Pre-norm FFN
        x = self.norm2(cell_repr)
        x = self.ffn(x)
        cell_repr = cell_repr + self.drop2(x)

        return cell_repr


@register_head("cross_attention")
class CrossAttentionHead(SpatialHead):
    """Cross-attention head: cells attend to trunk sequence positions.

    Parameters
    ----------
    d_model
        Projection dimension.
    num_heads
        Number of attention heads.
    num_layers
        Number of blocks.
    ffn_dim
        Hidden dimension in FFN.
    hidden_dim
        Output MLP hidden dimension.
    dropout
        Dropout rate.
    output_activation
        Output activation function. (``"softplus"`` or ``"linear"``)
    spatial_emb_dropout
        Element-wise dropout rate for cell embeddings.
    trunk_emb_dropout
        Element-wise dropout rate for trunk embeddings.
    use_absolute_pos_emb
        Add absolute positional embeddings to trunk embeddings.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        output_activation: str = "softplus",
        spatial_emb_dropout: float = 0.0,
        trunk_emb_dropout: float = 0.0,
        use_absolute_pos_emb: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.output_activation = output_activation
        self.spatial_emb_dropout = spatial_emb_dropout
        self.trunk_emb_dropout = trunk_emb_dropout
        self.use_absolute_pos_emb = use_absolute_pos_emb
        self.spatial_emb_drop = nn.Dropout(spatial_emb_dropout)
        self.trunk_emb_drop = nn.Dropout(trunk_emb_dropout)

    def setup(
        self,
        input_seq_len: int,
        trunk_channels: int,
        spatial_emb_dim: int,
    ) -> None:
        """Build layers (called by Laika after model dimensions are known)."""
        # Trunk projection: (B, L, 1536) -> (B, L, d_model)
        self.trunk_proj = nn.Linear(trunk_channels, self.d_model)
        self.trunk_norm = nn.LayerNorm(self.d_model)
        if self.use_absolute_pos_emb:
            self.pos_emb = nn.Embedding(input_seq_len, self.d_model)

        # Spatial projection: (B, N, emb_dim) -> (B, N, d_model)
        self.spatial_proj = nn.Linear(spatial_emb_dim, self.d_model)
        self.spatial_norm = nn.LayerNorm(self.d_model)

        # Cross-attention blocks
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
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
        B, L, _ = trunk_emb.shape

        # Element-wise embedding dropout
        cell_embs = self.spatial_emb_drop(cell_embs)
        trunk_emb = self.trunk_emb_drop(trunk_emb)

        # Project trunk + positional encoding
        trunk_repr = self.trunk_proj(trunk_emb)  # (B, L, d_model)
        trunk_repr = self.trunk_norm(trunk_repr)
        if self.use_absolute_pos_emb:
            positions = torch.arange(L, device=trunk_emb.device)
            trunk_repr = trunk_repr + self.pos_emb(positions)  # broadcast over B

        # Project spatial embeddings
        cell_repr = self.spatial_proj(cell_embs)  # (B, N, d_model)
        cell_repr = self.spatial_norm(cell_repr)

        # Cross-attention blocks
        for block in self.blocks:
            cell_repr = block(cell_repr, trunk_repr)

        # Output MLP
        out = self.output_mlp(cell_repr).squeeze(-1)  # (B, N)

        return apply_output_activation(out, self.output_activation)

    @property
    def name(self) -> str:
        return "cross_attention"

    @property
    def config(self) -> dict:
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "output_activation": self.output_activation,
            "spatial_emb_dropout": self.spatial_emb_dropout,
            "trunk_emb_dropout": self.trunk_emb_dropout,
            "use_absolute_pos_emb": self.use_absolute_pos_emb,
        }

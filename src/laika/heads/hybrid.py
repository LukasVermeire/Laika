"""Hybrid head: residual FiLM + cross-attention paths."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import register_head
from ._utils import AttentionPool, apply_output_activation
from .base import SpatialHead
from .cross_attention import CrossAttentionBlock
from .film import FiLMConditioner, FiLMLayer


@register_head("hybrid")
class HybridHead(SpatialHead):
    """Hybrid head: residual FiLM + cross-attention.

    Main path (FiLM): spatial embeddings flow directly, modulated by a
    gene-level DNA summary via attention pooling.

    Residual path (cross-attention): cells attend to specific trunk positions,
    scaled by a learnable residual weight (initialized small).

    Parameters
    ----------
    d_model
        Shared projection dimension for combining both paths.
    num_film_layers
        Number of FiLM modulation layers.
    film_hidden_dim
        Hidden dimension in FiLM layers.
    num_cross_attn_layers
        Number of cross-attention blocks (fewer than standalone).
    num_heads
        Number of attention heads.
    ffn_dim
        Cross-attention FFN hidden dimension.
    output_hidden_dim
        Output MLP hidden dimension.
    dropout
        Dropout rate.
    residual_init_scale
        Initial value of the learnable residual scale parameter.
    output_activation
        Output activation function. (``"softplus"`` or ``"linear"``)
    spatial_emb_dropout
        Element-wise dropout rate for cell embeddings.
    trunk_emb_dropout
        Element-wise dropout rate for trunk embeddings.
    use_absolute_pos_emb
        Add absolute positional embeddings to the cross-attention path.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_film_layers: int = 2,
        film_hidden_dim: int = 128,
        num_cross_attn_layers: int = 1,
        num_heads: int = 4,
        ffn_dim: int = 512,
        output_hidden_dim: int = 128,
        dropout: float = 0.1,
        residual_init_scale: float = 0.1,
        output_activation: str = "softplus",
        spatial_emb_dropout: float = 0.0,
        trunk_emb_dropout: float = 0.0,
        use_absolute_pos_emb: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_film_layers = num_film_layers
        self.film_hidden_dim = film_hidden_dim
        self.num_cross_attn_layers = num_cross_attn_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.output_hidden_dim = output_hidden_dim
        self.dropout = dropout
        self.residual_init_scale = residual_init_scale
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
        # --- FiLM path ---
        self.film_trunk_proj = nn.Linear(trunk_channels, self.d_model)
        self.film_trunk_norm = nn.LayerNorm(self.d_model)
        self.attn_pool = AttentionPool(self.d_model)

        self.conditioner = FiLMConditioner(
            spatial_emb_dim=spatial_emb_dim,
            hidden_dim=self.film_hidden_dim,
            num_layers=self.num_film_layers,
        )

        layer_dims = [self.d_model] + [self.film_hidden_dim] * self.num_film_layers
        self.film_layers = nn.ModuleList(
            [
                FiLMLayer(layer_dims[i], layer_dims[i + 1], self.dropout)
                for i in range(self.num_film_layers)
            ]
        )

        # Bridge: project FiLM output to d_model for addition with cross-attn
        self.film_out_proj = nn.Linear(self.film_hidden_dim, self.d_model)
        self.film_out_norm = nn.LayerNorm(self.d_model)

        # --- Cross-attention path ---
        self.ca_trunk_proj = nn.Linear(trunk_channels, self.d_model)
        self.ca_trunk_norm = nn.LayerNorm(self.d_model)
        if self.use_absolute_pos_emb:
            self.pos_emb = nn.Embedding(input_seq_len, self.d_model)

        self.spatial_proj = nn.Linear(spatial_emb_dim, self.d_model)
        self.spatial_norm = nn.LayerNorm(self.d_model)

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_cross_attn_layers)
            ]
        )

        # --- Combination ---
        self.residual_scale = nn.Parameter(
            torch.tensor(self.residual_init_scale)
        )

        # --- Output MLP ---
        self.output_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.output_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.output_hidden_dim, 1),
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
        _, N, _ = cell_embs.shape

        # Shared embedding dropout
        cell_embs = self.spatial_emb_drop(cell_embs)
        trunk_emb = self.trunk_emb_drop(trunk_emb)

        # --- FiLM path ---
        film_trunk = self.film_trunk_proj(trunk_emb)  # (B, L, d_model)
        film_trunk = self.film_trunk_norm(film_trunk)
        gene_vec = self.attn_pool(film_trunk)  # (B, d_model)

        film_params = self.conditioner(cell_embs)  # list of (gamma, beta)

        h = gene_vec.unsqueeze(1).expand(-1, N, -1)  # (B, N, d_model)
        for film_layer, (gamma, beta) in zip(self.film_layers, film_params):
            h = film_layer(h, gamma, beta)

        film_out = self.film_out_norm(self.film_out_proj(h))  # (B, N, d_model)

        # --- Cross-attention path ---
        ca_trunk = self.ca_trunk_proj(trunk_emb)  # (B, L, d_model)
        ca_trunk = self.ca_trunk_norm(ca_trunk)
        if self.use_absolute_pos_emb:
            positions = torch.arange(L, device=trunk_emb.device)
            ca_trunk = ca_trunk + self.pos_emb(positions)

        cell_repr = self.spatial_proj(cell_embs)  # (B, N, d_model)
        cell_repr = self.spatial_norm(cell_repr)

        for block in self.blocks:
            cell_repr = block(cell_repr, ca_trunk)

        ca_out = cell_repr  # (B, N, d_model)

        # --- Combine ---
        combined = film_out + self.residual_scale * ca_out  # (B, N, d_model)

        # --- Output ---
        out = self.output_mlp(combined).squeeze(-1)  # (B, N)

        return apply_output_activation(out, self.output_activation)

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def config(self) -> dict:
        return {
            "d_model": self.d_model,
            "num_film_layers": self.num_film_layers,
            "film_hidden_dim": self.film_hidden_dim,
            "num_cross_attn_layers": self.num_cross_attn_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "output_hidden_dim": self.output_hidden_dim,
            "dropout": self.dropout,
            "residual_init_scale": self.residual_init_scale,
            "output_activation": self.output_activation,
            "spatial_emb_dropout": self.spatial_emb_dropout,
            "trunk_emb_dropout": self.trunk_emb_dropout,
            "use_absolute_pos_emb": self.use_absolute_pos_emb,
        }

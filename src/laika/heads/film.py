"""FiLM head: spatial embeddings modulate the gene representation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_head
from .base import SpatialHead
from ._utils import AttentionPool, apply_output_activation


class FiLMConditioner(nn.Module):
    """Generate per-layer (gamma, beta) pairs from spatial embeddings."""

    def __init__(
        self,
        spatial_emb_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        """
        Parameters
        ----------
        spatial_emb_dim
            Input spatial embedding dimension.
        hidden_dim
            Hidden dimension for gamma/beta projections.
        num_layers
            Number of FiLM layers.
        """
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(spatial_emb_dim, hidden_dim),
            nn.GELU(),
        )
        # Per-layer gamma/beta projections
        self.gamma_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.beta_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        # Initialize gamma near 1, beta near 0 for stable training start
        for proj in self.gamma_projs:
            nn.init.ones_(proj.bias)
            nn.init.zeros_(proj.weight)
        for proj in self.beta_projs:
            nn.init.zeros_(proj.bias)
            nn.init.zeros_(proj.weight)

    def forward(
        self, cell_embs: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute per-layer (gamma, beta) pairs.

        Parameters
        ----------
        cell_embs
            ``(B, N, spatial_emb_dim)``

        Returns
        -------
        list of (gamma, beta) tuples, each ``(B, N, hidden_dim)``.
        """
        h = self.shared(cell_embs)  # (B, N, hidden_dim)
        return [
            (gamma_proj(h), beta_proj(h))
            for gamma_proj, beta_proj in zip(self.gamma_projs, self.beta_projs)
        ]


class FiLMLayer(nn.Module):
    """Single FiLM-modulated layer: LayerNorm -> Linear -> FiLM -> GELU -> Dropout."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        """
        Parameters
        ----------
        input_dim
            Input dimension.
        output_dim
            Output dimension.
        dropout
            Dropout rate.
        """
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM modulation.

        Parameters
        ----------
        h
            ``(B, N, input_dim)``
        gamma
            ``(B, N, output_dim)`` scale.
        beta
            ``(B, N, output_dim)`` shift.

        Returns
        -------
        torch.Tensor
            ``(B, N, output_dim)``
        """
        h = self.norm(h)
        h = self.linear(h)
        h = gamma * h + beta
        h = F.gelu(h)
        h = self.drop(h)
        return h


@register_head("film")
class FiLMHead(SpatialHead):
    """FiLM head: spatial embeddings modulate the gene representation.

    Parameters
    ----------
    d_model
        Projection dimension.
    hidden_dim
        Hidden dimension in FiLM layers.
    num_film_layers
        Number of layers.
    dropout
        Dropout rate.
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
        hidden_dim: int = 128,
        num_film_layers: int = 2,
        dropout: float = 0.1,
        output_activation: str = "softplus",
        spatial_emb_dropout: float = 0.0,
        trunk_emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_film_layers = num_film_layers
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

        # FiLM conditioning: spatial embeddings -> per-layer (gamma, beta)
        self.conditioner = FiLMConditioner(
            spatial_emb_dim=spatial_emb_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_film_layers,
        )

        # FiLM-modulated layers
        # First layer takes d_model input, subsequent layers take hidden_dim
        layer_dims = [self.d_model] + [self.hidden_dim] * self.num_film_layers
        self.film_layers = nn.ModuleList(
            [
                FiLMLayer(layer_dims[i], layer_dims[i + 1], self.dropout)
                for i in range(self.num_film_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, 1)

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

        # Generate FiLM parameters from spatial embeddings
        film_params = self.conditioner(cell_embs)  # list of (gamma, beta)

        # Broadcast gene vector: (B, d_model) -> (B, N, d_model)
        h = gene_vec.unsqueeze(1).expand(-1, N, -1)

        # Apply FiLM-modulated layers
        for film_layer, (gamma, beta) in zip(self.film_layers, film_params):
            h = film_layer(h, gamma, beta)

        # Output
        out = self.output_proj(h).squeeze(-1)  # (B, N)

        return apply_output_activation(out, self.output_activation)

    @property
    def name(self) -> str:
        return "film"

    @property
    def config(self) -> dict:
        return {
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "num_film_layers": self.num_film_layers,
            "dropout": self.dropout,
            "output_activation": self.output_activation,
            "spatial_emb_dropout": self.spatial_emb_dropout,
            "trunk_emb_dropout": self.trunk_emb_dropout,
        }

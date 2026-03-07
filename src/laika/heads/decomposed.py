"""Decomposed head: additive gene baseline + interaction term."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import get_head, register_head
from ._utils import AttentionPool, apply_output_activation
from .base import SpatialHead


@register_head("decomposed")
class DecomposedHead(SpatialHead):
    """Decomposed prediction head: y = mu(gene) + interaction(gene, cell).

    Two additive sub-networks:
    1. Gene baseline mu(gene): predicts per-gene mean expression from DNA alone.
    2. Interaction: a wrapped existing head capturing gene x cell residual.

    Auxiliary outputs (gene_baseline) are stored as attributes during forward
    for the trainer to access via ``get_auxiliaries()``.

    Parameters
    ----------
    interaction_head
        Name of the registered head to use for interaction (default ``"mlp"``).
    interaction_head_kwargs
        Keyword arguments forwarded to the interaction head constructor.
    gene_d_model
        Projection dimension for the gene baseline branch.
    spatial_emb_dropout
        Element-wise dropout rate for cell embeddings.
    trunk_emb_dropout
        Element-wise dropout rate for trunk embeddings.
    output_activation
        Output activation applied after summing both terms. (``"softplus"`` or ``"linear"``)
    """

    def __init__(
        self,
        interaction_head: str = "mlp",
        interaction_head_kwargs: dict | None = None,
        gene_d_model: int = 256,
        spatial_emb_dropout: float = 0.0,
        trunk_emb_dropout: float = 0.0,
        output_activation: str = "softplus",
    ):
        super().__init__()
        self.interaction_head_name = interaction_head
        self.interaction_head_kwargs = dict(interaction_head_kwargs or {})
        self.gene_d_model = gene_d_model
        self.spatial_emb_dropout = spatial_emb_dropout
        self.trunk_emb_dropout = trunk_emb_dropout
        self.output_activation = output_activation
        self.spatial_emb_drop = nn.Dropout(spatial_emb_dropout)
        self.trunk_emb_drop = nn.Dropout(trunk_emb_dropout)

        # Force linear activation on the interaction head — final activation
        # is applied after summing all three terms.
        self.interaction_head_kwargs["output_activation"] = "linear"
        # Avoid double-dropout: override interaction head's embedding dropout to 0
        self.interaction_head_kwargs["spatial_emb_dropout"] = 0.0
        self.interaction_head_kwargs["trunk_emb_dropout"] = 0.0

        self._interaction_head: SpatialHead = get_head(
            interaction_head, **self.interaction_head_kwargs
        )

        # Auxiliaries stored during forward
        self._gene_baseline: torch.Tensor | None = None

    def setup(
        self,
        input_seq_len: int,
        trunk_channels: int,
        spatial_emb_dim: int,
    ) -> None:
        """Build layers (called by Laika after model dimensions are known)."""
        # --- Gene baseline: trunk_emb -> scalar per gene ---
        self.gene_proj = nn.Linear(trunk_channels, self.gene_d_model)
        self.gene_norm = nn.LayerNorm(self.gene_d_model)
        self.gene_pool = AttentionPool(self.gene_d_model)
        self.gene_out = nn.Linear(self.gene_d_model, 1)

        # --- Interaction head ---
        self._interaction_head.setup(
            input_seq_len=input_seq_len,
            trunk_channels=trunk_channels,
            spatial_emb_dim=spatial_emb_dim,
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

        # Element-wise embedding dropout (applied once before all branches)
        cell_embs = self.spatial_emb_drop(cell_embs)
        trunk_emb = self.trunk_emb_drop(trunk_emb)

        # Gene baseline: (B, L, C) -> (B,)
        g = self.gene_proj(trunk_emb)       # (B, L, gene_d_model)
        g = self.gene_norm(g)               # (B, L, gene_d_model)
        g = self.gene_pool(g)               # (B, gene_d_model)
        gene_baseline = self.gene_out(g).squeeze(-1)  # (B,)

        # Interaction: (B, N)
        interaction = self._interaction_head(trunk_emb, cell_embs)  # (B, N)

        # Store auxiliaries for trainer
        self._gene_baseline = gene_baseline

        # Additive decomposition + final activation
        combined = gene_baseline.unsqueeze(1) + interaction
        return apply_output_activation(combined, self.output_activation)

    def get_auxiliaries(self) -> dict[str, torch.Tensor]:
        """Return cached auxiliary outputs from the last forward pass."""
        result = {}
        if self._gene_baseline is not None:
            result["gene_baseline"] = self._gene_baseline
        return result

    def clear_auxiliaries(self) -> None:
        """Clear cached auxiliary tensors to free memory."""
        self._gene_baseline = None

    @property
    def name(self) -> str:
        return "decomposed"

    @property
    def config(self) -> dict:
        return {
            "interaction_head": self.interaction_head_name,
            "interaction_head_kwargs": self.interaction_head_kwargs,
            "gene_d_model": self.gene_d_model,
            "spatial_emb_dropout": self.spatial_emb_dropout,
            "trunk_emb_dropout": self.trunk_emb_dropout,
            "output_activation": self.output_activation,
        }

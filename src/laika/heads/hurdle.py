"""Hurdle head: separate gate and expression branches."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import get_head, register_head
from ._utils import AttentionPool
from .base import SpatialHead


@register_head("hurdle")
class HurdleHead(SpatialHead):
    """Two-part (hurdle) prediction head: gate * expression.

    Separates "is the gene expressed?" (binary gate) from "how much?"
    (regression), letting each branch specialize. The final prediction is
    ``sigmoid(gate_logits) * softplus(expression)``.

    The gate branch auxiliary logits are exposed via ``get_auxiliaries()``
    so the trainer can apply a BCE loss weighted by ``zero_gate_lambda``.

    Parameters
    ----------
    inner_head
        Name of the registered head for the expression branch.
    inner_head_kwargs
        Keyword arguments forwarded to the inner head constructor.
    gate_hidden_dim
        Hidden dimension for the zero-gate network.
    zero_gate_lambda
        BCE loss weight for the gate branch (read by trainer).
    spatial_emb_dropout
        Element-wise dropout rate for cell embeddings.
    trunk_emb_dropout
        Element-wise dropout rate for trunk embeddings.
    """

    def __init__(
        self,
        inner_head: str = "cross_attention",
        inner_head_kwargs: dict | None = None,
        gate_hidden_dim: int = 128,
        zero_gate_lambda: float = 1.0,
        spatial_emb_dropout: float = 0.0,
        trunk_emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.inner_head_name = inner_head
        self.inner_head_kwargs = dict(inner_head_kwargs or {})
        self.gate_hidden_dim = gate_hidden_dim
        self.zero_gate_lambda = zero_gate_lambda
        self.spatial_emb_dropout = spatial_emb_dropout
        self.trunk_emb_dropout = trunk_emb_dropout
        self.spatial_emb_drop = nn.Dropout(spatial_emb_dropout)
        self.trunk_emb_drop = nn.Dropout(trunk_emb_dropout)

        # Force linear activation on inner head — we apply softplus ourselves
        self.inner_head_kwargs["output_activation"] = "linear"
        # Avoid double-dropout: override inner head's embedding dropout to 0
        self.inner_head_kwargs["spatial_emb_dropout"] = 0.0
        self.inner_head_kwargs["trunk_emb_dropout"] = 0.0

        self._inner_head: SpatialHead = get_head(inner_head, **self.inner_head_kwargs)

        # Auxiliaries stored during forward
        self._gate_logits: torch.Tensor | None = None

    def setup(
        self,
        input_seq_len: int,
        trunk_channels: int,
        spatial_emb_dim: int,
    ) -> None:
        """Build layers (called by Laika after model dimensions are known)."""
        self._inner_head.setup(
            input_seq_len=input_seq_len,
            trunk_channels=trunk_channels,
            spatial_emb_dim=spatial_emb_dim,
        )

        # Gate network
        self.gate_trunk_proj = nn.Linear(trunk_channels, self.gate_hidden_dim)
        self.gate_trunk_norm = nn.LayerNorm(self.gate_hidden_dim)
        self.gate_trunk_pool = AttentionPool(self.gate_hidden_dim)
        self.gate_cell_proj = nn.Linear(spatial_emb_dim, self.gate_hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * self.gate_hidden_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 1),
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
        (B, N) predictions: ``sigmoid(gate) * softplus(expression)``.
        """
        B, N, _ = cell_embs.shape

        # Element-wise embedding dropout (applied once before both branches)
        cell_embs = self.spatial_emb_drop(cell_embs)
        trunk_emb = self.trunk_emb_drop(trunk_emb)

        # Expression branch: (B, N), always >= 0
        expression = F.softplus(self._inner_head(trunk_emb, cell_embs))

        # Gate branch
        g = self.gate_trunk_pool(
            self.gate_trunk_norm(self.gate_trunk_proj(trunk_emb))
        )  # (B, gate_hidden_dim)
        c = self.gate_cell_proj(cell_embs)  # (B, N, gate_hidden_dim)
        # Broadcast g -> (B, N, gate_hidden_dim), concat with c
        g_expanded = g.unsqueeze(1).expand(-1, N, -1)  # (B, N, gate_hidden_dim)
        gate_input = torch.cat([g_expanded, c], dim=-1)  # (B, N, 2*gate_hidden_dim)
        gate_logits = self.gate_mlp(gate_input).squeeze(-1)  # (B, N)

        # Store for auxiliary loss
        self._gate_logits = gate_logits

        return torch.sigmoid(gate_logits) * expression

    def get_auxiliaries(self) -> dict[str, torch.Tensor]:
        """Return cached auxiliary outputs from the last forward pass."""
        result = {}
        if self._gate_logits is not None:
            result["zero_gate_logits"] = self._gate_logits
        return result

    def clear_auxiliaries(self) -> None:
        """Clear cached auxiliary tensors to free memory."""
        self._gate_logits = None

    @property
    def name(self) -> str:
        return "hurdle"

    @property
    def config(self) -> dict:
        return {
            "inner_head": self.inner_head_name,
            "inner_head_kwargs": self.inner_head_kwargs,
            "gate_hidden_dim": self.gate_hidden_dim,
            "zero_gate_lambda": self.zero_gate_lambda,
            "spatial_emb_dropout": self.spatial_emb_dropout,
            "trunk_emb_dropout": self.trunk_emb_dropout,
        }

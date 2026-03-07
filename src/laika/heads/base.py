"""Abstract base class for all spatial prediction heads."""

from __future__ import annotations

from abc import abstractmethod

import torch.nn as nn


class SpatialHead(nn.Module):
    """Abstract base for spatial prediction heads.

    Takes ``(trunk_emb, cell_embs)`` and returns ``(B, N)`` expression predictions.

    - ``trunk_emb``: ``(B, pooled_seq_len, channels)``
    - ``cell_embs``: ``(B, N, emb_dim)``
    """

    @abstractmethod
    def setup(
        self,
        input_seq_len: int,
        trunk_channels: int,
        spatial_emb_dim: int,
    ) -> None:
        """Initialize layers in-place.

        Parameters
        ----------
        input_seq_len
            Sequence length after pooling.
        trunk_channels
            Trunk output channel dimension.
        spatial_emb_dim
            Spatial cell embedding dimension.
        """
        ...

    @abstractmethod
    def forward(self, trunk_emb, cell_embs):
        """Forward pass: (trunk_emb, cell_embs) -> (B, N)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this head type."""
        ...

    @property
    @abstractmethod
    def config(self) -> dict:
        """Head initialization kwargs for reconstruction."""
        ...

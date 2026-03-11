"""Abstract base class for all cell encoder modules."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class BaseCellEncoder(nn.Module):
    """Abstract base for cell encoders.

    Takes expression vectors ``(B, N, n_genes)`` and produces cell embeddings
    ``(B, N, output_dim)``.

    All concrete encoders must call ``setup(n_input_genes, output_dim)`` before
    the first forward pass to initialise learnable layers.
    """

    @abstractmethod
    def setup(self, n_input_genes: int, output_dim: int) -> None:
        """Initialise layers in-place.

        Parameters
        ----------
        n_input_genes
            Number of input genes in the expression vector.
        output_dim
            Output embedding dimension (must match the head's ``spatial_emb_dim``).
        """
        ...

    @abstractmethod
    def forward(
        self,
        expression: torch.Tensor,
        gene_indices_to_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        expression
            ``(B, N, n_genes)`` expression vectors.
        gene_indices_to_mask
            ``(B,)`` index of the target gene to zero out per batch item.
            Pass ``-1`` to skip masking for a given item.

        Returns
        -------
        torch.Tensor
            ``(B, N, output_dim)`` cell embeddings.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this encoder type."""
        ...

    @property
    @abstractmethod
    def config(self) -> dict:
        """Encoder initialisation kwargs for reconstruction."""
        ...

    @property
    def output_dim(self) -> int | None:
        """Output embedding dimension set by ``setup()``, or ``None`` if not yet set."""
        return getattr(self, "_output_dim", None)

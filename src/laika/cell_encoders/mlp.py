"""MLP-based cell encoder — O(G) alternative to the transformer encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import register_cell_encoder
from .base import BaseCellEncoder


@register_cell_encoder("mlp")
class MLPCellEncoder(BaseCellEncoder):
    """MLP encoder for single-cell expression vectors.

    Applies LayerNorm over the expression vector then passes it through a
    stack of Linear → GELU → Dropout layers.  O(G) complexity makes it
    tractable for large gene sets (e.g. 8 000 genes) where the transformer's
    O(G²) self-attention would OOM.

    Parameters
    ----------
    d_hidden
        Hidden layer width.
    n_layers
        Number of hidden layers (each: Linear → GELU → Dropout).
    dropout
        Dropout rate applied after each activation.
    """

    def __init__(
        self,
        d_hidden: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._d_hidden = d_hidden
        self._n_layers = n_layers
        self._dropout = dropout

        # Populated by setup()
        self._n_input_genes: int | None = None
        self._output_dim: int | None = None

    def setup(self, n_input_genes: int, output_dim: int) -> None:
        """Initialise all learnable layers.

        Parameters
        ----------
        n_input_genes
            Number of input genes in the expression vector.
        output_dim
            Output embedding dimension.
        """
        self._n_input_genes = n_input_genes
        self._output_dim = output_dim

        self.layer_norm = nn.LayerNorm(n_input_genes)

        layers: list[nn.Module] = []
        in_dim = n_input_genes
        for _ in range(self._n_layers):
            layers += [
                nn.Linear(in_dim, self._d_hidden),
                nn.GELU(),
                nn.Dropout(self._dropout),
            ]
            in_dim = self._d_hidden

        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        expression: torch.Tensor,
        gene_indices_to_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        expression
            ``(B, N, n_genes)`` expression vectors where ``N`` is cells per gene.
        gene_indices_to_mask
            ``(B,)`` index of the target gene to zero out per batch item to
            prevent information leakage.  Use ``-1`` to skip masking.

        Returns
        -------
        torch.Tensor
            ``(B, N, output_dim)`` cell embeddings.
        """
        B, N, G = expression.shape

        # --- Mask target gene to prevent leakage ---
        if gene_indices_to_mask is not None:
            expression = expression.clone()
            for b_idx in range(B):
                idx = int(gene_indices_to_mask[b_idx].item())
                if idx >= 0:
                    expression[b_idx, :, idx] = 0.0

        # Reshape to (B*N, G) for batch processing
        x = expression.reshape(B * N, G)

        x = self.layer_norm(x)
        out = self.mlp(x)  # (B*N, output_dim)
        return out.reshape(B, N, self._output_dim)

    @property
    def name(self) -> str:
        """Short identifier."""
        return "mlp"

    @property
    def config(self) -> dict:
        """Initialisation kwargs for reconstruction."""
        return {
            "d_hidden": self._d_hidden,
            "n_layers": self._n_layers,
            "dropout": self._dropout,
        }

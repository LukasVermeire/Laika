"""Transformer-based cell encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import register_cell_encoder
from .base import BaseCellEncoder


@register_cell_encoder("transformer")
class TransformerCellEncoder(BaseCellEncoder):
    """Transformer encoder for single-cell expression vectors.

    Treats each gene as a token: expression value → per-gene embedding.
    A learnable CLS token is prepended and its output is projected to
    ``output_dim`` as the cell embedding.

    Parameters
    ----------
    d_model
        Internal embedding dimension.
    n_heads
        Number of attention heads (must divide ``d_model``).
    n_layers
        Number of transformer encoder layers.
    dropout
        Dropout rate applied inside the transformer.
    pool_mode
        Pooling strategy: ``"cls"`` uses the CLS token output;
        ``"mean"`` averages over gene token outputs.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        pool_mode: str = "cls",
    ):
        super().__init__()
        if pool_mode not in {"cls", "mean"}:
            raise ValueError(f"pool_mode must be 'cls' or 'mean', got {pool_mode!r}")
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._dropout = dropout
        self._pool_mode = pool_mode

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

        # Per-gene identity embedding (gene index → d_model)
        self.gene_embedding = nn.Embedding(n_input_genes, self._d_model)

        # Scalar expression value projected to d_model
        self.expression_proj = nn.Linear(1, self._d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self._d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Pre-transformer normalisation
        self.layer_norm = nn.LayerNorm(self._d_model)

        # Standard PyTorch transformer encoder (batch_first for (B, seq, d) layout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d_model,
            nhead=self._n_heads,
            dim_feedforward=self._d_model * 4,
            dropout=self._dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self._n_layers)

        # Final projection to output_dim
        self.output_proj = nn.Linear(self._d_model, output_dim)

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

        # Reshape to (B*N, G, 1) for token-level processing
        x = expression.reshape(B * N, G, 1)

        # Project scalar expression values: (B*N, G, d_model)
        token_values = self.expression_proj(x)

        # Add gene identity embeddings (broadcast over batch)
        gene_ids = torch.arange(G, device=expression.device)
        gene_embs = self.gene_embedding(gene_ids)       # (G, d_model)
        tokens = token_values + gene_embs.unsqueeze(0)  # (B*N, G, d_model)

        # Pre-transformer normalisation
        tokens = self.layer_norm(tokens)

        # Prepend CLS token: (B*N, G+1, d_model)
        cls = self.cls_token.expand(B * N, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Transformer encoder
        tokens = self.transformer(tokens)

        # Pooling
        if self._pool_mode == "cls":
            pooled = tokens[:, 0, :]        # (B*N, d_model) — CLS output
        else:
            pooled = tokens[:, 1:, :].mean(dim=1)  # (B*N, d_model) — mean of gene tokens

        # Project and reshape back to (B, N, output_dim)
        out = self.output_proj(pooled)  # (B*N, output_dim)
        return out.reshape(B, N, self._output_dim)

    @property
    def name(self) -> str:
        """Short identifier."""
        return "transformer"

    @property
    def config(self) -> dict:
        """Initialisation kwargs for reconstruction."""
        return {
            "d_model": self._d_model,
            "n_heads": self._n_heads,
            "n_layers": self._n_layers,
            "dropout": self._dropout,
            "pool_mode": self._pool_mode,
        }

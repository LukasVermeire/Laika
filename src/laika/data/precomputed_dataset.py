"""Dataset that serves precomputed trunk embeddings."""

from __future__ import annotations

import numpy as np
import torch

from ._base_dataset import BaseDataset


class GeneCentricDataset(BaseDataset):
    """Dataset that loads precomputed trunk embeddings for each gene.

    Parameters
    ----------
    genes
        List of gene names.
    precomputed_embeddings
        Mapping ``gene_name -> np.ndarray`` of shape ``(pooled_seq_len, channels)``.
    expression_matrix
        Expression matrix ``(n_genes, n_cells)``.
    spatial_embeddings
        Spatial cell embeddings ``(n_cells, emb_dim)``.
    cells_per_gene
        Cells to sample per gene.
    deterministic
        Deterministic ordering.
    gene_means
        Per-gene expression means ``(n_genes,)``.
    gene_stds
        Per-gene expression stds ``(n_genes,)``.
    normalize_targets
        Normalize targets.
    include_aux_targets
        Include auxiliary gene mean targets.
    gene_repeat_factor
        Repeat each gene this many times per epoch.
    """

    def __init__(
        self,
        genes: list[str],
        precomputed_embeddings: dict[str, np.ndarray],
        expression_matrix: np.ndarray,
        spatial_embeddings: np.ndarray,
        cells_per_gene: int | None = 4096,
        deterministic: bool = False,
        gene_means: np.ndarray | None = None,
        gene_stds: np.ndarray | None = None,
        normalize_targets: bool = False,
        include_aux_targets: bool = False,
        gene_repeat_factor: int = 1,
        encoder_expression_matrix: np.ndarray | None = None,
        encoder_gene_to_idx: dict[str, int] | None = None,
    ):
        super().__init__(
            genes=genes,
            expression_matrix=expression_matrix,
            spatial_embeddings=spatial_embeddings,
            cells_per_gene=cells_per_gene,
            deterministic=deterministic,
            gene_means=gene_means,
            gene_stds=gene_stds,
            normalize_targets=normalize_targets,
            include_aux_targets=include_aux_targets,
            gene_repeat_factor=gene_repeat_factor,
            encoder_expression_matrix=encoder_expression_matrix,
            encoder_gene_to_idx=encoder_gene_to_idx,
        )
        self.precomputed_embeddings = precomputed_embeddings

    def _get_gene_input(self, idx: int) -> dict:
        """Return ``{"trunk_emb": tensor}`` for gene at index ``idx``."""
        gene = self.genes[idx]
        trunk_emb = torch.from_numpy(
            self.precomputed_embeddings[gene].astype(np.float32)
        )
        return {"trunk_emb": trunk_emb}

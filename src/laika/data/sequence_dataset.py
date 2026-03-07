"""Dataset that fetches live DNA sequences per gene."""

from __future__ import annotations

import numpy as np
import torch

from ._base_dataset import BaseDataset
from .sequence_cache import GeneSequenceCache


class SequenceCentricDataset(BaseDataset):
    """Dataset that fetches one-hot encoded DNA sequences for each gene.

    Parameters
    ----------
    genes
        List of gene names.
    sequence_cache
        GeneSequenceCache.
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
        sequence_cache: GeneSequenceCache,
        expression_matrix: np.ndarray,
        spatial_embeddings: np.ndarray,
        cells_per_gene: int | None = 512,
        deterministic: bool = False,
        gene_means: np.ndarray | None = None,
        gene_stds: np.ndarray | None = None,
        normalize_targets: bool = False,
        include_aux_targets: bool = False,
        gene_repeat_factor: int = 1,
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
        )
        self.sequence_cache = sequence_cache

    def _get_gene_input(self, idx: int) -> dict:
        """Return ``{"onehot": tensor}`` for gene at index ``idx``, with optional stochastic shift."""
        gene = self.genes[idx]
        shift = 0
        if not self.deterministic and self.sequence_cache.max_stochastic_shift > 0:
            shift = np.random.randint(
                -self.sequence_cache.max_stochastic_shift,
                self.sequence_cache.max_stochastic_shift + 1,
            )
        onehot = torch.from_numpy(
            self.sequence_cache.get_onehot(gene, shift=shift).astype(np.float32)
        )
        return {"onehot": onehot}

"""Data module for head-only training with precomputed trunk embeddings."""

from __future__ import annotations

import numpy as np

from ._base_data_module import BaseDataModule
from .precomputed_dataset import GeneCentricDataset


class SpatialDataModule(BaseDataModule):
    """Data module for training with precomputed embeddings."""

    def make_dataset(
        self,
        genes: list[str],
        precomputed_embeddings: dict[str, np.ndarray],
        cells_per_gene: int | None = 4096,
        deterministic: bool = False,
        normalize_targets: bool = False,
        include_aux_targets: bool = False,
        gene_repeat_factor: int = 1,
    ) -> GeneCentricDataset:
        """Create a GeneCentricDataset.

        Parameters
        ----------
        genes
            List of genes.
        precomputed_embeddings
            Precomputed trunk embeddings.
        cells_per_gene
            Cells to sample per gene.
        deterministic
            Deterministic ordering.
        normalize_targets
            Normalize targets.
        include_aux_targets
            Include auxiliary gene/cell mean targets.
        gene_repeat_factor
            Repeat each gene this many times per epoch.

        Returns
        -------
        GeneCentricDataset
        """
        base_kwargs = self._build_dataset_base_kwargs(
            genes=genes,
            cells_per_gene=cells_per_gene,
            deterministic=deterministic,
            normalize_targets=normalize_targets,
            include_aux_targets=include_aux_targets,
            gene_repeat_factor=gene_repeat_factor,
        )

        return GeneCentricDataset(
            precomputed_embeddings=precomputed_embeddings,
            **base_kwargs,
        )

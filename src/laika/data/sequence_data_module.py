"""Data module for end-to-end fine-tuning with live DNA sequences."""

from __future__ import annotations

from ._base_data_module import BaseDataModule
from .sequence_dataset import SequenceCentricDataset


class SequenceDataModule(BaseDataModule):
    """Data module for end-to-end fine-tuning with live sequences."""

    def make_dataset(
        self,
        genes: list[str],
        cells_per_gene: int | None = 512,
        deterministic: bool = False,
        normalize_targets: bool = False,
        include_aux_targets: bool = False,
        gene_repeat_factor: int = 1,
    ) -> SequenceCentricDataset:
        """Create dataset.

        Parameters
        ----------
        genes
            List of genes.
        cells_per_gene
            Cells to sample per gene.
        deterministic
            Deterministic ordering.
        normalize_targets
            Normalize targets to zero-mean unit-std.
        include_aux_targets
            Include auxiliary gene/cell mean targets.
        gene_repeat_factor
            Repeat each gene this many times per epoch.

        Returns
        -------
        SequenceCentricDataset
        """
        base_kwargs = self._build_dataset_base_kwargs(
            genes=genes,
            cells_per_gene=cells_per_gene,
            deterministic=deterministic,
            normalize_targets=normalize_targets,
            include_aux_targets=include_aux_targets,
            gene_repeat_factor=gene_repeat_factor,
        )

        return SequenceCentricDataset(
            sequence_cache=self._sequence_cache,
            **base_kwargs,
        )

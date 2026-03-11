"""Base PyTorch Dataset for gene-centric spatial expression data."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset


def worker_init_fn(worker_id: int) -> None:
    """Reseed NumPy RNG per worker."""
    seed = torch.initial_seed() % (2**32) + worker_id
    np.random.seed(seed)


def collate_fn(batch: list[dict]) -> dict:
    """Stack batch items into tensors.

    Handles both ``trunk_emb`` (precomputed) and ``onehot`` (live-sequence) gene inputs,
    optional ``gene_mean_target``, and optional cell-encoder keys
    (``expression_vectors``, ``gene_encoder_idx``).
    """
    result = {
        "cell_embs": torch.stack([item["cell_embs"] for item in batch]),
        "targets": torch.stack([item["targets"] for item in batch]),
        "gene_name": [item["gene_name"] for item in batch],
    }
    # Add the gene-input key (either trunk_emb or onehot)
    if "trunk_emb" in batch[0]:
        result["trunk_emb"] = torch.stack([item["trunk_emb"] for item in batch])
    elif "onehot" in batch[0]:
        result["onehot"] = torch.stack([item["onehot"] for item in batch])
    # Auxiliary targets (optional)
    if "gene_mean_target" in batch[0]:
        result["gene_mean_target"] = torch.stack(
            [item["gene_mean_target"] for item in batch]
        )
    # Cell encoder data (optional)
    if "expression_vectors" in batch[0]:
        result["expression_vectors"] = torch.stack(
            [item["expression_vectors"] for item in batch]
        )
        result["gene_encoder_idx"] = torch.tensor(
            [item["gene_encoder_idx"] for item in batch], dtype=torch.long
        )
    return result


class BaseDataset(Dataset):
    """Base PyTorch Dataset where each item is one gene with many cells.

    Parameters
    ----------
    genes
        List of gene names.
    expression_matrix
        Expression matrix ``(n_genes, n_cells)``.
    spatial_embeddings
        Spatial cell embeddings ``(n_cells, emb_dim)``.
    cells_per_gene
        Cells to sample per gene. (``None`` = all)
    deterministic
        Deterministic cell sampling (for validation).
    gene_means
        Per-gene expression means ``(n_genes,)``.
    gene_stds
        Per-gene expression stds ``(n_genes,)``.
    normalize_targets
        Normalize targets to zero-mean unit-std.
    include_aux_targets
        Include per-gene mean expression as auxiliary target.
    gene_repeat_factor
        Repeat each gene this many times per epoch with fresh random cells.
    """

    def __init__(
        self,
        genes: list[str],
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
        self.genes = genes
        self.expression_matrix = expression_matrix
        self.spatial_embeddings = spatial_embeddings
        self.cells_per_gene = cells_per_gene
        self.deterministic = deterministic
        self.n_cells = spatial_embeddings.shape[0]
        self.normalize_targets = normalize_targets
        self.gene_means = gene_means  # (n_genes,) or None
        self.gene_stds = gene_stds    # (n_genes,) or None
        self.include_aux_targets = include_aux_targets
        self.gene_repeat_factor = gene_repeat_factor
        self.encoder_expression_matrix = encoder_expression_matrix  # (n_cells, n_encoder_genes) or None
        self.encoder_gene_to_idx = encoder_gene_to_idx or {}
        self._deterministic_seeds: list[int] | None = None
        if self.deterministic:
            self._deterministic_seeds = [
                int(hashlib.md5(gene.encode()).hexdigest(), 16) % (2**31)
                for gene in self.genes
            ]

    def __len__(self) -> int:
        """Total dataset length (genes × gene_repeat_factor)."""
        return len(self.genes) * self.gene_repeat_factor

    def _sample_cells(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Sample cells and return (cell_embs, targets, cell_indices) tensors."""
        n = self.cells_per_gene if self.cells_per_gene is not None else self.n_cells
        n = min(n, self.n_cells)

        if n >= self.n_cells:
            cell_indices = np.arange(self.n_cells)
        elif self.deterministic:
            # Fixed random permutation seeded per gene
            gene_seed = self._deterministic_seeds[idx]
            rng = np.random.RandomState(gene_seed)
            cell_indices = rng.choice(self.n_cells, size=n, replace=False)
        else:
            cell_indices = np.random.choice(self.n_cells, size=n, replace=False)

        cell_embs = torch.from_numpy(self.spatial_embeddings[cell_indices])
        raw_targets = self.expression_matrix[idx, cell_indices]
        if self.normalize_targets and self.gene_means is not None:
            raw_targets = (raw_targets - self.gene_means[idx]) / self.gene_stds[idx]
        targets = torch.from_numpy(raw_targets)
        return cell_embs, targets, cell_indices

    def _get_gene_input(self, idx: int) -> dict:
        """Return gene-specific input."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Return dict with gene input, cell_embs, targets, and gene_name."""
        idx = idx % len(self.genes)
        gene = self.genes[idx]
        gene_input = self._get_gene_input(idx)
        cell_embs, targets, cell_indices = self._sample_cells(idx)
        item = {
            **gene_input,
            "cell_embs": cell_embs,
            "targets": targets,
            "gene_name": gene,
        }
        if self.include_aux_targets:
            # Per-gene mean expression (scalar target for gene baseline)
            item["gene_mean_target"] = torch.tensor(
                self.expression_matrix[idx].mean(), dtype=torch.float32
            )
        if self.encoder_expression_matrix is not None:
            # Cell encoder data: expression vectors for the sampled cells
            encoder_expr = self.encoder_expression_matrix[cell_indices].astype(np.float32)
            item["expression_vectors"] = torch.from_numpy(encoder_expr)
            # Index of this gene in the encoder vocabulary (-1 = not present, no masking)
            item["gene_encoder_idx"] = self.encoder_gene_to_idx.get(gene, -1)
        return item

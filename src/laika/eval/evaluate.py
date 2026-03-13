"""Evaluation pipeline: prediction, metric computation, and result containers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from ..inference import Predictor
from .metrics import (
    global_correlations,
    per_cell_correlations,
    per_gene_correlations,
)


@dataclass
class EvalResults:
    """Container for evaluation metrics and raw data.

    Attributes
    ----------
    gene_names
        Gene names.
    y_true
        Ground truth expression ``(n_genes, n_cells)``. 
    y_pred
        Predicted expression ``(n_genes, n_cells)``.
    global_pearson
        Global Pearson R.
    global_spearman
        Global Spearman R.
    per_gene_pearson
        Pearson R per gene ``(n_genes,)``.
    per_gene_spearman
        Spearman R per gene ``(n_genes,)``.
    per_cell_pearson
        Pearson R per cell ``(n_cells,)``.
    per_cell_spearman
        Spearman R per cell ``(n_cells,)``.
    """

    gene_names: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    global_pearson: float = 0.0
    global_spearman: float = 0.0
    per_gene_pearson: np.ndarray = field(default_factory=lambda: np.array([]))
    per_gene_spearman: np.ndarray = field(default_factory=lambda: np.array([]))
    per_cell_pearson: np.ndarray = field(default_factory=lambda: np.array([]))
    per_cell_spearman: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> dict:
        """Return aggregate metrics (medians, NaN counts, n_genes, n_cells)."""
        return {
            "global_pearson": self.global_pearson,
            "global_spearman": self.global_spearman,
            "per_gene_pearson_median": float(np.nanmedian(self.per_gene_pearson)),
            "per_gene_spearman_median": float(np.nanmedian(self.per_gene_spearman)),
            "per_cell_pearson_median": float(np.nanmedian(self.per_cell_pearson)),
            "per_cell_spearman_median": float(np.nanmedian(self.per_cell_spearman)),
            "per_gene_nan_count": int(np.isnan(self.per_gene_pearson).sum()),
            "per_cell_nan_count": int(np.isnan(self.per_cell_pearson).sum()),
            "n_genes": len(self.gene_names),
            "n_cells": self.y_true.shape[1],
        }


def evaluate(
    predictor: Predictor,
    genes: list[str],
    expression_matrix: np.ndarray,
    spatial_embeddings: np.ndarray,
    precomputed_embeddings: dict[str, np.ndarray] | None = None,
    cell_indices: np.ndarray | None = None,
    cells_per_chunk: int = 1024,
    genes_per_chunk: int = 1,
    encoder_expression_matrix: np.ndarray | None = None,
    encoder_gene_to_idx: dict[str, int] | None = None,
) -> EvalResults:
    """Run evaluation.

    Parameters
    ----------
    predictor
        Predictor instance.
    genes
        List of genes.
    expression_matrix
        Ground truth expression.
    spatial_embeddings
        Cell embeddings.
    precomputed_embeddings
        Precomputed trunk embeddings.
    cell_indices
        Subset of cell indices.
    cells_per_chunk
        Max cells per forward pass.
    genes_per_chunk
        Max genes per forward pass.
    encoder_expression_matrix
        Expression matrix for the cell encoder ``(n_cells, n_encoder_genes)``.
        Required when the model has a cell encoder.
    encoder_gene_to_idx
        Mapping from encoder gene name to column index in ``encoder_expression_matrix``.

    Returns
    -------
    EvalResults
        Evaluation results.
    """
    logger.info(f"Evaluating {len(genes)} genes...")

    y_pred = predictor.predict(
        genes=genes,
        spatial_embeddings=spatial_embeddings,
        precomputed_embeddings=precomputed_embeddings,
        cell_indices=cell_indices,
        cells_per_chunk=cells_per_chunk,
        genes_per_chunk=genes_per_chunk,
        expression_matrix=encoder_expression_matrix,
        encoder_gene_to_idx=encoder_gene_to_idx,
    )

    y_true = expression_matrix
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: expression_matrix {y_true.shape} vs predictions {y_pred.shape}"
        )

    glob = global_correlations(y_true, y_pred)

    results = EvalResults(
        gene_names=list(genes),
        y_true=y_true,
        y_pred=y_pred,
        global_pearson=glob["pearson"],
        global_spearman=glob["spearman"],
        per_gene_pearson=per_gene_correlations(y_true, y_pred, "pearson"),
        per_gene_spearman=per_gene_correlations(y_true, y_pred, "spearman"),
        per_cell_pearson=per_cell_correlations(y_true, y_pred, "pearson"),
        per_cell_spearman=per_cell_correlations(y_true, y_pred, "spearman"),
    )

    s = results.summary()
    logger.info(
        f"Eval complete — "
        f"Global Pearson: {s['global_pearson']:.4f}, "
        f"Global Spearman: {s['global_spearman']:.4f} | "
        f"Per-gene Pearson median: {s['per_gene_pearson_median']:.4f}, "
        f"Per-cell Pearson median: {s['per_cell_pearson_median']:.4f} | "
        f"NaN genes: {s['per_gene_nan_count']}/{s['n_genes']}"
    )

    return results

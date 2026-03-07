"""Evaluation metric functions: per-gene, per-cell, and global correlations."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _row_correlations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "pearson",
) -> np.ndarray:
    """Compute correlation along axis 1."""
    if metric not in ("pearson", "spearman"):
        raise ValueError(f"metric must be 'pearson' or 'spearman', got '{metric}'")

    corr_fn = pearsonr if metric == "pearson" else spearmanr
    n_rows = y_true.shape[0]
    result = np.full(n_rows, np.nan, dtype=np.float64)

    for i in range(n_rows):
        t, p = y_true[i], y_pred[i]
        if np.std(t) < 1e-8 or np.std(p) < 1e-8:
            continue
        result[i] = corr_fn(t, p).statistic

    return result


def per_gene_correlations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "pearson",
) -> np.ndarray:
    """Correlation per gene (across cells).

    Parameters
    ----------
    y_true
        Ground truth ``(n_genes, n_cells)``.
    y_pred
        Predictions ``(n_genes, n_cells)``.
    metric
        Metric name. (``"pearson"`` or ``"spearman"``)

    Returns
    -------
    np.ndarray
        Per-gene correlations ``(n_genes,)``.
    """
    return _row_correlations(y_true, y_pred, metric)


def per_cell_correlations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "pearson",
) -> np.ndarray:
    """Correlation per cell (across genes).

    Parameters
    ----------
    y_true
        Ground truth ``(n_genes, n_cells)``.
    y_pred
        Predictions ``(n_genes, n_cells)``.
    metric
        Metric name. (``"pearson"`` or ``"spearman"``)

    Returns
    -------
    np.ndarray
        Per-cell correlations ``(n_cells,)``.
    """
    return _row_correlations(y_true.T, y_pred.T, metric)


def global_correlations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Global correlations.

    Parameters
    ----------
    y_true
        Ground truth ``(n_genes, n_cells)``.
    y_pred
        Predictions ``(n_genes, n_cells)``.

    Returns
    -------
    dict
        Global correlation metrics. ``{"pearson": float, "spearman": float}``
    """
    t = y_true.ravel()
    p = y_pred.ravel()
    return {
        "pearson": float(pearsonr(t, p).statistic),
        "spearman": float(spearmanr(t, p).statistic),
    }

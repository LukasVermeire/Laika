"""Evaluation module for Laika models."""

from .evaluate import EvalResults, evaluate
from .metrics import global_correlations, per_cell_correlations, per_gene_correlations
from .visualization import (
    plot_per_gene_correlations,
    plot_predicted_vs_true_for_genes,
    save_eval_plots,
)

__all__ = [
    "evaluate",
    "EvalResults",
    "per_gene_correlations",
    "per_cell_correlations",
    "global_correlations",
    "plot_per_gene_correlations",
    "plot_predicted_vs_true_for_genes",
    "save_eval_plots",
]

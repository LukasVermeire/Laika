"""Evaluation visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .evaluate import EvalResults


def _get_metric_values(results: EvalResults, metric: str) -> tuple[np.ndarray, str, str]:
    """Return per-gene metric values plus axis/title labels."""
    metric_key = metric.lower()
    if metric_key == "pearson":
        return (
            np.asarray(results.per_gene_pearson, dtype=np.float64),
            "Pearson R (per gene, across cells)",
            "Per-gene Pearson R",
        )
    if metric_key == "spearman":
        return (
            np.asarray(results.per_gene_spearman, dtype=np.float64),
            "Spearman R (per gene, across cells)",
            "Per-gene Spearman R",
        )
    raise ValueError(f"metric must be 'pearson' or 'spearman', got '{metric}'")


def plot_per_gene_correlations(
    results: EvalResults,
    save_path: str | Path,
    metric: str = "pearson",
    *,
    sort_desc: bool = True,
    dpi: int = 150,
) -> Path:
    """Plot per-gene correlation bars and save figure to disk.

    Parameters
    ----------
    results
        EvalResults containing per-gene metrics.
    save_path
        Output file path.
    metric
        Correlation metric. (``"pearson"`` or ``"spearman"``)
    sort_desc
        Sort genes descending by correlation.
    dpi
        Figure DPI.

    Returns
    -------
    Path
        Path to saved figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "plot_per_gene_correlations requires matplotlib. "
            "Install it with `pip install matplotlib`."
        ) from exc

    values, ylabel, title = _get_metric_values(results, metric)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D per-gene values, got shape {values.shape}")
    if len(values) == 0:
        raise ValueError("Cannot plot per-gene correlations: no genes in EvalResults.")
    if len(results.gene_names) != len(values):
        raise ValueError(
            "Mismatch between number of gene names and per-gene values: "
            f"{len(results.gene_names)} vs {len(values)}."
        )

    # Keep NaN values at the end when sorting descending (or start for ascending).
    fill_value = -np.inf if sort_desc else np.inf
    sort_ready = np.nan_to_num(values, nan=fill_value)
    order = np.argsort(sort_ready)
    if sort_desc:
        order = order[::-1]

    sorted_values = values[order]
    sorted_genes = [results.gene_names[int(i)] for i in order]

    fig_width = max(10.0, len(sorted_genes) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    colors = [
        "#9E9E9E" if np.isnan(v) else ("C0" if v >= 0 else "C3") for v in sorted_values
    ]
    positions = np.arange(len(sorted_genes))
    ax.bar(positions, sorted_values, color=colors)

    if len(sorted_genes) <= 200:
        ax.set_xticks(positions)
        ax.set_xticklabels(sorted_genes, rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"Genes (n={len(sorted_genes)})")

    ax.axhline(0.0, color="black", linewidth=0.8)
    median = float(np.nanmedian(sorted_values))
    if not np.isnan(median):
        ax.axhline(
            median,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label=f"Median: {median:.3f}",
        )
        ax.legend()

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_predicted_vs_true_for_genes(
    results: EvalResults,
    save_path: str | Path,
    *,
    genes: list[str] | None = None,
    max_cols: int = 5,
    point_alpha: float = 0.5,
    point_size: float = 10.0,
    independent_axes: bool = True,
    dpi: int = 150,
) -> Path:
    """Plot predicted-vs-ground-truth scatter per gene and save to disk.

    Parameters
    ----------
    results
        EvalResults with raw predictions.
    save_path
        Output file path.
    genes
        Genes to plot. (``None`` plots all genes)
    max_cols
        Max subplot columns.
    point_alpha
        Scatter point alpha.
    point_size
        Scatter point size.
    independent_axes
        Use independent axes per gene.
    dpi
        Figure DPI.

    Returns
    -------
    Path
        Path to saved figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "plot_predicted_vs_true_for_genes requires matplotlib. "
            "Install it with `pip install matplotlib`."
        ) from exc

    y_true = np.asarray(results.y_true)
    y_pred = np.asarray(results.y_pred)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(
            f"Expected 2D y_true/y_pred arrays, got {y_true.shape} and {y_pred.shape}."
        )
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}.")
    if len(results.gene_names) != y_true.shape[0]:
        raise ValueError(
            "Mismatch between number of gene names and y_true rows: "
            f"{len(results.gene_names)} vs {y_true.shape[0]}."
        )

    if genes is None:
        selected_genes = list(results.gene_names)
    else:
        selected_genes = [g for g in genes if g in results.gene_names]
        missing = sorted(set(genes).difference(selected_genes))
        if missing:
            missing_preview = ", ".join(missing[:5])
            if len(missing) > 5:
                missing_preview += ", ..."
            raise ValueError(f"Unknown gene names requested: {missing_preview}")

    if not selected_genes:
        raise ValueError("No genes selected for plotting.")
    if max_cols < 1:
        raise ValueError(f"max_cols must be >= 1, got {max_cols}.")

    gene_to_idx = {g: i for i, g in enumerate(results.gene_names)}
    selected_indices = [gene_to_idx[g] for g in selected_genes]
    has_per_gene_pearson = len(results.per_gene_pearson) == y_true.shape[0]

    n_genes = len(selected_genes)
    n_cols = min(max_cols, n_genes)
    n_rows = int(np.ceil(n_genes / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    for plot_idx, gene_idx in enumerate(selected_indices):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row][col]

        gene_name = results.gene_names[gene_idx]
        gt = y_true[gene_idx]
        pred = y_pred[gene_idx]

        valid = np.isfinite(gt) & np.isfinite(pred)
        if np.any(valid):
            gt_valid = gt[valid]
            pred_valid = pred[valid]
            ax.scatter(gt_valid, pred_valid, s=point_size, alpha=point_alpha)

            if independent_axes:
                x_lo, x_hi = float(gt_valid.min()), float(gt_valid.max())
                y_lo, y_hi = float(pred_valid.min()), float(pred_valid.max())
                if np.isclose(x_lo, x_hi):
                    pad = 1.0 if np.isclose(x_hi, 0.0) else 0.05 * abs(x_hi)
                    x_lo -= pad; x_hi += pad
                if np.isclose(y_lo, y_hi):
                    pad = 1.0 if np.isclose(y_hi, 0.0) else 0.05 * abs(y_hi)
                    y_lo -= pad; y_hi += pad
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_lo, y_hi)
            else:
                lower = float(min(gt_valid.min(), pred_valid.min()))
                upper = float(max(gt_valid.max(), pred_valid.max()))
                if np.isclose(lower, upper):
                    pad = 1.0 if np.isclose(upper, 0.0) else 0.05 * abs(upper)
                    lower -= pad
                    upper += pad
                ax.plot([lower, upper], [lower, upper], linestyle="--", color="gray", linewidth=1.0)
                ax.set_xlim(lower, upper)
                ax.set_ylim(lower, upper)
        else:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )

        if has_per_gene_pearson:
            pearson = float(results.per_gene_pearson[gene_idx])
            if np.isnan(pearson):
                ax.set_title(f"{gene_name}\nPearson: NaN")
            else:
                ax.set_title(f"{gene_name}\nPearson: {pearson:.3f}")
        else:
            ax.set_title(gene_name)

        if row == n_rows - 1:
            ax.set_xlabel("Ground truth")
        if col == 0:
            ax.set_ylabel("Prediction")

    for empty_idx in range(n_genes, n_rows * n_cols):
        row = empty_idx // n_cols
        col = empty_idx % n_cols
        axes[row][col].axis("off")

    fig.suptitle("Predicted vs ground truth (selected genes)", y=1.02)
    fig.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_eval_plots(
    results: EvalResults,
    output_dir: str | Path,
    *,
    prefix: str = "eval",
    dpi: int = 150,
) -> dict[str, Path]:
    """Save standard evaluation plots and return produced file paths.

    Parameters
    ----------
    results
        EvalResults to plot.
    output_dir
        Directory to save plots.
    prefix
        Filename prefix for all plots.
    dpi
        Figure DPI.

    Returns
    -------
    dict
        Mapping of plot name to saved file path.
    """
    out_dir = Path(output_dir)
    return {
        "per_gene_pearson": plot_per_gene_correlations(
            results,
            out_dir / f"{prefix}_per_gene_pearson.png",
            metric="pearson",
            dpi=dpi,
        ),
        "per_gene_spearman": plot_per_gene_correlations(
            results,
            out_dir / f"{prefix}_per_gene_spearman.png",
            metric="spearman",
            dpi=dpi,
        ),
    }

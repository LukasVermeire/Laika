"""Online metrics tracker for training/validation epochs."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


class MetricsTracker:
    """Tracks statistics for epoch metrics."""

    def __init__(
        self,
        reservoir_size: int = 50_000,
        *,
        track_per_gene: bool = True,
        track_spearman: bool = True,
        track_nonzero_pearson: bool = False,
    ):
        """
        Parameters
        ----------
        reservoir_size
            Max samples kept for global Spearman R estimation.
        track_per_gene
            Track per-gene Pearson/Spearman R.
        track_spearman
            Track Spearman R (global and per-gene).
        track_nonzero_pearson
            Track per-gene Pearson R on non-zero cells only.
        """
        self._reservoir_size = reservoir_size
        self._track_per_gene = track_per_gene
        self._track_spearman = track_spearman
        self._track_nonzero_pearson = track_nonzero_pearson
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        # Loss
        self._loss_sum: float = 0.0
        self._loss_count: int = 0

        # Global Pearson R
        self._n: int = 0
        self._mean_x: float = 0.0
        self._mean_y: float = 0.0
        self._cov_xy: float = 0.0
        self._var_x: float = 0.0
        self._var_y: float = 0.0

        # Global Spearman R (reservoir sampling)
        self._reservoir_true: np.ndarray | None = None
        self._reservoir_pred: np.ndarray | None = None
        self._seen: int = 0

        # Per-gene metrics
        self._gene_pearson_vals: list[float] = []
        self._gene_spearman_vals: list[float] = []
        self._gene_nz_pearson_vals: list[float] = []

    def update(
        self,
        loss: float,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        gene_names: list[str] | None = None,
    ) -> None:
        """Accumulate a batch of predictions.

        Parameters
        ----------
        loss
            Loss value.
        y_true
            Ground truth ``(n_genes, n_cells)``.
        y_pred
            Predicted values ``(n_genes, n_cells)``.
        gene_names
            Gene names ``(n_genes,)``.
        """
        # Per-gene metrics
        if self._track_per_gene and gene_names is not None and y_true.ndim == 2:
            for i in range(y_true.shape[0]):
                gt = y_true[i].astype(np.float64)
                pr = y_pred[i].astype(np.float64)
                # Pearson
                if gt.std() > 0 and pr.std() > 0:
                    r, _ = pearsonr(gt, pr)
                    self._gene_pearson_vals.append(float(r))
                else:
                    self._gene_pearson_vals.append(np.nan)
                # Spearman
                if self._track_spearman:
                    if gt.std() > 0 and pr.std() > 0:
                        rho, _ = spearmanr(gt, pr)
                        self._gene_spearman_vals.append(float(rho))
                    else:
                        self._gene_spearman_vals.append(np.nan)
                # Nonzero-only Pearson
                if self._track_nonzero_pearson:
                    nz_mask = gt > 0
                    if nz_mask.sum() >= 2:
                        gt_nz = gt[nz_mask]
                        pr_nz = pr[nz_mask]
                        if gt_nz.std() > 0 and pr_nz.std() > 0:
                            r_nz, _ = pearsonr(gt_nz, pr_nz)
                            self._gene_nz_pearson_vals.append(float(r_nz))
                        else:
                            self._gene_nz_pearson_vals.append(np.nan)
                    else:
                        self._gene_nz_pearson_vals.append(np.nan)

        # global metrics
        yt = y_true.ravel().astype(np.float64)
        yp = y_pred.ravel().astype(np.float64)
        n_new = len(yt)

        # loss — accumulate per batch (loss is already a reduced mean from the criterion)
        self._loss_sum += loss
        self._loss_count += 1

        # Global Pearson R
        self._welford_batch_update(yt, yp)

        # Global Spearman R (reservoir sampling)
        if self._track_spearman:
            self._reservoir_update(yt, yp)

    def _welford_batch_update(self, yt: np.ndarray, yp: np.ndarray) -> None:
        """Update Welford stats."""
        n_new = len(yt)
        if n_new == 0:
            return

        new_mean_x = yt.mean()
        new_mean_y = yp.mean()
        new_var_x = ((yt - new_mean_x) ** 2).sum()
        new_var_y = ((yp - new_mean_y) ** 2).sum()
        new_cov_xy = ((yt - new_mean_x) * (yp - new_mean_y)).sum()

        if self._n == 0:
            self._n = n_new
            self._mean_x = float(new_mean_x)
            self._mean_y = float(new_mean_y)
            self._var_x = float(new_var_x)
            self._var_y = float(new_var_y)
            self._cov_xy = float(new_cov_xy)
        else:
            old_n = self._n
            combined_n = old_n + n_new
            dx = float(new_mean_x) - self._mean_x
            dy = float(new_mean_y) - self._mean_y

            self._cov_xy += float(new_cov_xy) + dx * dy * old_n * n_new / combined_n
            self._var_x += float(new_var_x) + dx * dx * old_n * n_new / combined_n
            self._var_y += float(new_var_y) + dy * dy * old_n * n_new / combined_n
            self._mean_x += dx * n_new / combined_n
            self._mean_y += dy * n_new / combined_n
            self._n = combined_n

    def _reservoir_update(self, yt: np.ndarray, yp: np.ndarray) -> None:
        """Reservoir sampling for Spearman."""
        n = len(yt)

        if self._reservoir_true is None:
            # first batch
            if n <= self._reservoir_size:
                self._reservoir_true = yt.copy()
                self._reservoir_pred = yp.copy()
            else:
                indices = np.random.choice(n, self._reservoir_size, replace=False)
                self._reservoir_true = yt[indices].copy()
                self._reservoir_pred = yp[indices].copy()
            self._seen = n
            return

        prev_seen = self._seen
        self._seen += n

        # Determine how many new samples should replace reservoir entries
        current_len = len(self._reservoir_true)
        if current_len < self._reservoir_size:
            # Reservoir not yet full
            space = self._reservoir_size - current_len
            to_add = min(space, n)
            self._reservoir_true = np.concatenate(
                [self._reservoir_true, yt[:to_add]]
            )
            self._reservoir_pred = np.concatenate(
                [self._reservoir_pred, yp[:to_add]]
            )
            # Handle remaining via standard reservoir sampling
            for k in range(to_add, n):
                j = prev_seen + k
                r = np.random.randint(0, j + 1)
                if r < self._reservoir_size:
                    self._reservoir_true[r] = yt[k]
                    self._reservoir_pred[r] = yp[k]
        else:
            # Vectorised reservoir sampling for this batch
            indices_in_stream = np.arange(prev_seen, prev_seen + n)
            random_slots = np.array(
                [np.random.randint(0, j + 1) for j in indices_in_stream]
            )
            mask = random_slots < self._reservoir_size
            if mask.any():
                slots = random_slots[mask]
                self._reservoir_true[slots] = yt[mask]
                self._reservoir_pred[slots] = yp[mask]

    def compute(self) -> dict[str, float]:
        """Compute epoch-level metrics.

        Returns
        -------
        dict with keys ``"loss"``, ``"pearson_r"``, ``"spearman_r"``,
        ``"per_gene_pearson_r"``, ``"per_gene_spearman_r"``.
        """
        if self._loss_count == 0:
            return {
                "loss": 0.0,
                "pearson_r": 0.0,
                "spearman_r": 0.0,
                "per_gene_pearson_r": 0.0,
                "per_gene_spearman_r": 0.0,
            }

        # Loss
        loss = self._loss_sum / self._loss_count

        # Global Pearson R
        if self._n < 2 or self._var_x == 0.0 or self._var_y == 0.0:
            pearson_r = 0.0
        else:
            pearson_r = float(self._cov_xy / (self._var_x * self._var_y) ** 0.5)
            if np.isnan(pearson_r):
                pearson_r = 0.0

        # Global Spearman R from reservoir
        if not self._track_spearman:
            spearman_r_val = 0.0
        elif self._reservoir_true is None or len(self._reservoir_true) < 2:
            spearman_r_val = 0.0
        else:
            rho, _ = spearmanr(self._reservoir_true, self._reservoir_pred)
            spearman_r_val = float(rho) if not np.isnan(rho) else 0.0

        # Per-gene metrics (median across genes)
        if self._gene_pearson_vals:
            per_gene_pearson = float(np.nanmedian(self._gene_pearson_vals))
        else:
            per_gene_pearson = 0.0

        if self._gene_spearman_vals:
            per_gene_spearman = float(np.nanmedian(self._gene_spearman_vals))
        else:
            per_gene_spearman = 0.0

        result = {
            "loss": loss,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r_val,
            "per_gene_pearson_r": per_gene_pearson,
            "per_gene_spearman_r": per_gene_spearman,
        }

        if self._track_nonzero_pearson:
            if self._gene_nz_pearson_vals:
                result["nz_per_gene_pearson_r"] = float(
                    np.nanmedian(self._gene_nz_pearson_vals)
                )
            else:
                result["nz_per_gene_pearson_r"] = 0.0

        return result

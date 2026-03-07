"""Loss functions for Laika training."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

LOSS_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "mse": nn.MSELoss,
    "poisson": lambda: nn.PoissonNLLLoss(log_input=False),
    "huber": nn.HuberLoss,
}


def register_loss(name: str, factory: Callable[[], nn.Module]) -> None:
    """Register a named loss factory used by the Trainer."""
    if name in LOSS_REGISTRY:
        raise ValueError(f"Loss '{name}' is already registered.")
    LOSS_REGISTRY[name] = factory


def get_loss(name: str) -> nn.Module:
    """Build a registered loss by name."""
    if name not in LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name]()


def list_losses() -> list[str]:
    """Return names of all registered losses."""
    return list(LOSS_REGISTRY.keys())


class PearsonCorrelationLoss(nn.Module):
    """Differentiable Pearson correlation loss: ``1 - mean(per-gene Pearson R)``.
    """

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds, targets: (B, N) where B = genes, N = cells
        preds_centered = preds - preds.mean(dim=-1, keepdim=True)
        targets_centered = targets - targets.mean(dim=-1, keepdim=True)
        cov = (preds_centered * targets_centered).sum(dim=-1)
        preds_std = preds_centered.pow(2).sum(dim=-1).sqrt().clamp(min=1e-8)
        targets_std = targets_centered.pow(2).sum(dim=-1).sqrt().clamp(min=1e-8)
        pearson_r = cov / (preds_std * targets_std)
        return 1 - pearson_r.mean()


class CombinedLoss(nn.Module):
    """Combines a base loss with a Pearson correlation term.

    ``loss = base_lambda * base_loss(preds, targets) + corr_lambda * (1 - pearson_r)``
    """

    def __init__(
        self,
        base_loss: nn.Module,
        correlation_lambda: float,
        base_loss_lambda: float = 1.0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.correlation_lambda = correlation_lambda
        self.base_loss_lambda = base_loss_lambda
        self.correlation_loss = PearsonCorrelationLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.base_loss_lambda * self.base_loss(preds, targets)
            + self.correlation_lambda * self.correlation_loss(preds, targets)
        )


class WeightedMSELoss(nn.Module):
    """MSE loss that upweights non-zero target elements.

    Useful for zero-inflated data where standard MSE biases
    predictions toward zero.
    """

    def __init__(self, nonzero_weight: float = 3.0):
        super().__init__()
        self.nonzero_weight = nonzero_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.where(targets > 0, self.nonzero_weight, 1.0)
        return (weights * (preds - targets).pow(2)).mean()


class NonzeroMaskedPoissonLoss(nn.Module):
    """Poisson NLL loss computed only on entries where target > 0.

    For hurdle models: the gate handles zeros via BCE, this loss trains the
    expression branch only on informative (nonzero) cells.
    """

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        return torch.nn.functional.poisson_nll_loss(
            preds[mask], targets[mask], log_input=False
        )


class ListMLELoss(nn.Module):
    """Listwise ranking loss (ListMLE).

    Maximizes the likelihood of the ground-truth ranking
    under a Plackett-Luce model. For each gene (row), sorts
    cells by descending target expression and computes the
    negative log-likelihood of that permutation given predicted scores.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-10):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds, targets: (B, N) — B genes, N cells
        sorted_indices = targets.argsort(dim=-1, descending=True)
        sorted_preds = preds.gather(dim=-1, index=sorted_indices) / self.temperature

        # Plackett-Luce negative log-likelihood via reverse cumulative logsumexp
        max_val = sorted_preds.max(dim=-1, keepdim=True).values
        shifted = sorted_preds - max_val
        rev_cumsumexp = torch.logcumsumexp(
            shifted.flip(dims=[-1]), dim=-1
        ).flip(dims=[-1])
        nll = -torch.sum(shifted - rev_cumsumexp, dim=-1)
        return nll.mean()


class HybridRankingLoss(nn.Module):
    """Combines ListMLE ranking with magnitude loss on non-zeros."""

    def __init__(self, ranking_lambda: float = 1.0, magnitude_lambda: float = 0.5):
        super().__init__()
        self.ranking_loss = ListMLELoss()
        self.magnitude_loss = NonzeroMaskedPoissonLoss()
        self.ranking_lambda = ranking_lambda
        self.magnitude_lambda = magnitude_lambda

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.ranking_lambda * self.ranking_loss(preds, targets)
            + self.magnitude_lambda * self.magnitude_loss(preds, targets)
        )


LOSS_REGISTRY["pearson"] = PearsonCorrelationLoss
LOSS_REGISTRY["weighted_mse"] = WeightedMSELoss
LOSS_REGISTRY["nz_poisson"] = NonzeroMaskedPoissonLoss
LOSS_REGISTRY["list_mle"] = ListMLELoss
LOSS_REGISTRY["hybrid_ranking"] = HybridRankingLoss

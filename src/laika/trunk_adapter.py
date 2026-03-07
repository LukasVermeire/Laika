"""Thin adapter bridging a Keras trunk with PyTorch training loops."""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from .trunk import enable_trunk_lora
from .utils.keras_helpers import clear_trunk_losses, move_trunk_to_device


class TrunkAdapter:
    """Thin adapter around a Keras trunk when used from PyTorch training loops."""

    def __init__(self, trunk, device: torch.device):
        """
        Parameters
        ----------
        trunk
            Keras trunk model (output of ``load_borzoi_trunk``).
        device
            PyTorch device to place trunk variables on.
        """
        self._trunk = trunk
        self._device = device
        move_trunk_to_device(self._trunk, self._device)

    @property
    def trunk(self):
        """Underlying Keras trunk model."""
        return self._trunk

    def configure_train_mode(
        self,
        *,
        freeze: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
    ) -> dict[str, int] | None:
        """Configure trainable trunk parameters."""
        if freeze and use_lora:
            raise ValueError("freeze=True and use_lora=True are mutually exclusive.")

        if use_lora:
            stats = enable_trunk_lora(self._trunk, rank=lora_rank)
            logger.info(
                f"LoRA enabled: {stats['lora_layers']} layers adapted, "
                f"{stats['lora_params']:,} LoRA params "
                f"({stats['lora_params'] / stats['total_params']:.2%} of "
                f"{stats['total_params']:,} total)"
            )
        elif freeze:
            self._trunk.trainable = False
            stats = None
            logger.info("Trunk kept frozen during fine-tuning (freeze_trunk=True)")
        else:
            self._trunk.trainable = True
            stats = None
            logger.info("Trunk set to trainable=True")

        self._sync_requires_grad_flags()
        move_trunk_to_device(self._trunk, self._device)
        return stats

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return trainable torch parameters owned by the Keras trunk."""
        params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for var in self._trunk.variables:
            value = getattr(var, "_value", None)
            if value is None:
                continue
            if not isinstance(value, torch.nn.Parameter):
                raise RuntimeError(
                    "Could not map Keras variable to torch.nn.Parameter. "
                    "Expected KERAS_BACKEND=torch compatible variables."
                )
            if not value.requires_grad:
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            params.append(value)
        return params

    def forward(self, onehot: torch.Tensor, training: bool) -> torch.Tensor:
        """Run trunk forward pass.

        Parameters
        ----------
        onehot
            One-hot encoded DNA ``(B, seq_len, 4)``.
        training
            Training mode flag passed to Keras.
        """
        return self._trunk(onehot, training=training)

    def clear_losses(self) -> None:
        """Clear Keras internal losses accumulated during forward pass."""
        clear_trunk_losses(self._trunk)

    def save_weights(self, path: str) -> None:
        """Save trunk weights to HDF5 file."""
        self._trunk.save_weights(path)

    def load_weights(self, path: str) -> None:
        """Load trunk weights from HDF5 file."""
        self._trunk.load_weights(path)

    def freeze_for_inference(self) -> None:
        """Freeze trunk and sync requires_grad flags for inference."""
        self._trunk.trainable = False
        self._sync_requires_grad_flags()
        move_trunk_to_device(self._trunk, self._device)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable summary dict."""
        return {"device": str(self._device), "trainable": bool(self._trunk.trainable)}

    def _sync_requires_grad_flags(self) -> None:
        """Mirror Keras variable trainability to torch parameters."""
        for var in self._trunk.variables:
            value = getattr(var, "_value", None)
            if isinstance(value, torch.nn.Parameter):
                value.requires_grad_(bool(getattr(var, "trainable", False)))

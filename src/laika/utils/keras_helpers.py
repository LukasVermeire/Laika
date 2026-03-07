from __future__ import annotations

import torch
from loguru import logger


def move_trunk_to_device(trunk, device: torch.device) -> None:
    """Move Keras variables to PyTorch device."""
    try:
        for var in trunk.variables:
            var._value.data = var._value.data.to(device)
        logger.info(f"Trunk variables moved to {device}")
    except AttributeError:
        logger.warning(
            "Could not move trunk variables via var._value.data.to(device) — "
            "this is a private Keras 3 API that may have changed. "
            "Trunk may remain on CPU."
        )


def clear_trunk_losses(trunk) -> None:
    """Clear Keras internal losses."""
    try:
        trunk._clear_losses()
    except AttributeError:
        logger.warning(
            "Could not clear trunk losses via trunk._clear_losses() — "
            "this is a private Keras 3 API that may have changed. "
            "GPU memory may accumulate across batches."
        )

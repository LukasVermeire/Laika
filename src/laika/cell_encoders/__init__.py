"""Cell encoder registry: register/retrieve BaseCellEncoder implementations by name."""

from __future__ import annotations

from typing import Type

from .base import BaseCellEncoder

CELL_ENCODER_REGISTRY: dict[str, Type[BaseCellEncoder]] = {}


def register_cell_encoder(name: str):
    """Class decorator to register a cell encoder implementation.

    Parameters
    ----------
    name
        Registry key (e.g. ``"transformer"``).
    """

    def decorator(cls: Type[BaseCellEncoder]) -> Type[BaseCellEncoder]:
        if name in CELL_ENCODER_REGISTRY:
            raise ValueError(f"Cell encoder '{name}' is already registered.")
        CELL_ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def get_cell_encoder(name: str, **kwargs) -> BaseCellEncoder:
    """Instantiate a registered cell encoder by name.

    Parameters
    ----------
    name
        Registered encoder name like ``"transformer"``.
    **kwargs
        Encoder initialisation arguments forwarded to the constructor.

    Returns
    -------
    BaseCellEncoder
        Instantiated encoder (``setup()`` must be called before use).
    """
    if name not in CELL_ENCODER_REGISTRY:
        raise KeyError(
            f"Unknown cell encoder '{name}'. "
            f"Available: {list(CELL_ENCODER_REGISTRY.keys())}"
        )
    return CELL_ENCODER_REGISTRY[name](**kwargs)


def list_cell_encoders() -> list[str]:
    """Return names of all registered cell encoders."""
    return list(CELL_ENCODER_REGISTRY.keys())


# Auto-import concrete encoders so they self-register via the decorator.
from .transformer import TransformerCellEncoder  # noqa: F401, E402

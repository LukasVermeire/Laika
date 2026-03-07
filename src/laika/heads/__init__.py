"""Head registry: register/retrieve SpatialHead implementations by name."""

from __future__ import annotations

from typing import Type

from .base import SpatialHead

HEAD_REGISTRY: dict[str, Type[SpatialHead]] = {}


def register_head(name: str):
    """Class decorator to register a head implementation."""

    def decorator(cls: Type[SpatialHead]) -> Type[SpatialHead]:
        if name in HEAD_REGISTRY:
            raise ValueError(f"Head '{name}' is already registered.")
        HEAD_REGISTRY[name] = cls
        return cls

    return decorator


def get_head(name: str, **kwargs) -> SpatialHead:
    """Instantiate a registered head by name.

    Parameters
    ----------
    name
        Registered head name like ``"mlp"``.
    **kwargs
        Head initialization arguments.

    Returns
    -------
    Instantiated :class:`SpatialHead`.
    """
    if name not in HEAD_REGISTRY:
        raise KeyError(
            f"Unknown head '{name}'. Available: {list(HEAD_REGISTRY.keys())}"
        )
    return HEAD_REGISTRY[name](**kwargs)


def list_heads() -> list[str]:
    """Return names of all registered heads."""
    return list(HEAD_REGISTRY.keys())


# Auto-import concrete heads so they register themselves via the decorator.
from .mlp import MLPHead  # noqa: F401, E402
from .cross_attention import CrossAttentionHead  # noqa: F401, E402
from .film import FiLMHead  # noqa: F401, E402
from .decomposed import DecomposedHead  # noqa: F401, E402
from .hurdle import HurdleHead  # noqa: F401, E402
from .hybrid import HybridHead  # noqa: F401, E402
from .hyperconv import HyperConvHead  # noqa: F401, E402

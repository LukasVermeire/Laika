"""Borzoi trunk loading and LoRA utilities."""

from __future__ import annotations

import os

def load_borzoi_trunk(
    model_path: str,
    extraction_layer: str = "add_15",
):
    """Load the Borzoi Prime trunk as a Keras model.

    Parameters
    ----------
    model_path
        Path to a saved Keras model or CREsted model name.
    extraction_layer
        Name of the layer to extract as trunk output.

    Returns
    -------
    keras.Model
        Trunk model.
    """
    import keras
    # noinspection PyUnresolvedReferences
    # Import side-effects register CREsted custom layers for Keras deserialization.
    import crested.tl.zoo.utils
    from crested import get_model
    from crested.tl.zoo.utils import MultiheadAttention, AttentionPool1D

    custom_objects = {
        'MultiheadAttention': MultiheadAttention,
        'AttentionPool1D': AttentionPool1D,
    }

    if os.path.exists(model_path):
        full_model = keras.models.load_model(model_path, custom_objects=custom_objects)
    else:
        path, _ = get_model(model_path)
        full_model = keras.models.load_model(path, custom_objects=custom_objects)

    trunk = keras.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(extraction_layer).output,
        name=f"{full_model.name}_trunk",
    )
    return trunk


def enable_trunk_lora(trunk, rank: int = 8) -> dict[str, int]:
    """Enable LoRA on all Dense/EinsumDense layers in the trunk.

    Parameters
    ----------
    trunk
        Keras trunk model.
    rank
        Rank of the low-rank decomposition.

    Returns
    -------
    dict
        Stats with keys: ``lora_layers``, ``lora_params``,
        ``frozen_params``, ``total_params``.
    """
    import keras

    # Ensure child-layer trainable flags are honoured; setting Model.trainable=False
    # would force all descendants non-trainable (including LoRA weights).
    trunk.trainable = True

    visited: set[int] = set()
    lora_layers = 0
    all_layers = []

    def _recurse(layer):
        if id(layer) in visited:
            return
        visited.add(id(layer))
        all_layers.append(layer)

        for child in getattr(layer, "_flatten_layers", lambda: [])():
            if id(child) != id(layer):
                _recurse(child)

    _recurse(trunk)

    # Freeze everything first, then re-enable only LoRA-enabled projection layers.
    for layer in all_layers:
        if layer is not trunk:
            layer.trainable = False

    for layer in all_layers:
        if isinstance(layer, (keras.layers.Dense, keras.layers.EinsumDense)):
            layer.enable_lora(rank)
            layer.trainable = True
            lora_layers += 1

    if lora_layers == 0:
        raise RuntimeError(
            "No Dense/EinsumDense layers found in trunk — LoRA could not be applied."
        )

    lora_params = sum(v.numpy().size for v in trunk.trainable_variables)
    total_params = sum(v.numpy().size for v in trunk.variables)
    frozen_params = total_params - lora_params

    return {
        "lora_layers": lora_layers,
        "lora_params": lora_params,
        "frozen_params": frozen_params,
        "total_params": total_params,
    }


def get_trunk_dims(trunk) -> tuple[int, int]:
    """Return ``(seq_len, channels)`` from a loaded Keras trunk model."""
    shape = trunk.output_shape  # (None, seq_len, channels)
    return int(shape[1]), int(shape[2])

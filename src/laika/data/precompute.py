"""Trunk embedding precomputation and persistence utilities."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from .sequence_cache import GeneSequenceCache
from ..utils.keras_helpers import clear_trunk_losses


def precompute_trunk_embeddings(
    trunk,
    sequence_cache: GeneSequenceCache,
    genes: list[str],
    pool_factor: int,
    batch_size: int = 4,
    dtype: str = "float16",
) -> dict[str, np.ndarray]:
    """Precompute trunk embeddings for genes.

    Parameters
    ----------
    trunk
        Keras trunk model.
    sequence_cache
        GeneSequenceCache providing one-hot DNA.
    genes
        List of gene names.
    pool_factor
        Pooling factor.
    batch_size
        Batch size.
    dtype
        Storage dtype.

    Returns
    -------
    dict
        Mapping ``gene_name -> np.ndarray`` of shape
        ``(pooled_seq_len, channels)`` with the specified dtype.
    """
    import keras

    trunk_output_shape = trunk.output_shape  # (None, seq_len, channels)
    trunk_seq_len = trunk_output_shape[1]
    trunk_channels = trunk_output_shape[2]
    pooled_seq_len = trunk_seq_len // pool_factor

    bytes_per_gene = pooled_seq_len * trunk_channels * np.dtype(dtype).itemsize
    total_gb = bytes_per_gene * len(genes) / 1e9
    if total_gb > 8.0:
        logger.warning(
            f"Precomputed embeddings will require ~{total_gb:.1f} GB of RAM. "
            f"Consider increasing pool_factor (currently {pool_factor})."
        )
    logger.info(
        f"Precomputing trunk embeddings for {len(genes)} genes: "
        f"({trunk_seq_len} -> {pooled_seq_len}) x {trunk_channels}, "
        f"dtype={dtype}, estimated {total_gb:.2f} GB"
    )

    # Small pooling model for GPU-accelerated pooling
    if pool_factor > 1:
        pool_input = keras.layers.Input(shape=(trunk_seq_len, trunk_channels))
        pooled = keras.layers.AveragePooling1D(pool_size=pool_factor)(pool_input)
        pool_model = keras.Model(pool_input, pooled, name="trunk_pool_precompute")
    else:
        pool_model = None

    # Check if using torch backend
    _torch_backend = keras.backend.backend() == "torch"
    if _torch_backend:
        import torch

    embeddings: dict[str, np.ndarray] = {}

    pbar = tqdm(
        range(0, len(genes), batch_size),
        desc="Precomputing embeddings",
        unit="batch",
        total=(len(genes) + batch_size - 1) // batch_size,
    )
    for i in pbar:
        batch_genes = genes[i : i + batch_size]

        batch_onehot = np.stack(
            [sequence_cache.get_onehot(gene, shift=0) for gene in batch_genes]
        )

        if _torch_backend:
            # Use direct __call__ instead of .predict() - another OOM quickfix
            batch_tensor = torch.from_numpy(batch_onehot).to(
                dtype=torch.float32,
                device=next(v._value.device for v in trunk.variables),
            )
            with torch.no_grad():
                trunk_out = trunk(batch_tensor, training=False)
                if pool_model is not None:
                    trunk_out = pool_model(trunk_out, training=False)
                if isinstance(trunk_out, torch.Tensor):
                    trunk_out = trunk_out.cpu().numpy()
            del batch_tensor
        else:
            trunk_out = trunk.predict(batch_onehot, verbose=0)
            if pool_model is not None:
                trunk_out = pool_model.predict(trunk_out, verbose=0)

        for j, gene in enumerate(batch_genes):
            embeddings[gene] = trunk_out[j].astype(dtype)

        del batch_onehot, trunk_out
        clear_trunk_losses(trunk)
        if pool_model is not None:
            clear_trunk_losses(pool_model)
        gc.collect()
        if _torch_backend:
            torch.cuda.empty_cache()

        pbar.set_postfix({"genes": f"{min(i + batch_size, len(genes))}/{len(genes)}"})

    logger.info(
        f"Precomputation complete. {len(embeddings)} embeddings stored, "
        f"shape per gene: ({pooled_seq_len}, {trunk_channels})"
    )
    return embeddings


def save_precomputed_embeddings(
    embeddings: dict[str, np.ndarray],
    path: str | Path,
    pool_factor: int | None = None,
) -> None:
    """Save precomputed embeddings to file.

    Parameters
    ----------
    embeddings
        Embeddings dict.
    path
        Output path.
    pool_factor
        Optional pooling factor metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = dict(embeddings)
    if pool_factor is not None:
        save_dict["__meta_pool_factor__"] = np.array(pool_factor)
    np.savez_compressed(str(path), **save_dict)
    logger.info(f"Precomputed embeddings saved to {path} (pool_factor={pool_factor})")


def load_precomputed_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    """Load precomputed embeddings from file.

    Parameters
    ----------
    path
        Path to ``.npz`` file.

    Returns
    -------
    dict
        Mapping ``gene_name -> np.ndarray``.
    """
    path = Path(path)
    data = np.load(str(path))
    embeddings = {}
    for key in data.files:
        if key.startswith("__meta_"):
            logger.info(f"  metadata: {key} = {data[key]}")
        else:
            embeddings[key] = data[key]
    logger.info(f"Loaded {len(embeddings)} precomputed embeddings from {path}")
    return embeddings

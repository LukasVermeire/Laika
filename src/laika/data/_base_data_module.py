"""Abstract base data module: AnnData + Genome → train/val datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from anndata import AnnData
from loguru import logger
from scipy.sparse import issparse

from crested._genome import Genome

from ..config import DataModuleConfig, serialize_config_value
from ._gtf import _compute_gene_regions, _parse_gtf_gene_coords
from .sequence_cache import GeneSequenceCache


class BaseDataModule(ABC):
    """Base class for data modules orchestrating AnnData + Genome into datasets.

    Parameters
    ----------
    adata
        AnnData object with gene expression.
    genome
        Genome instance for DNA sequence fetching.
    gtf_path
        Path to a GTF annotation file.
    genes
        List of gene names to use.
    val_frac
        Fraction of genes held out for validation.
    val_genes
        Explicit list of validation genes.
    spatial_emb_key
        Key for spatial cell embeddings.
    seq_length
        DNA sequence window length.
    seed
        Random seed for train/val split.
    max_stochastic_shift
        Extra bases for stochastic shifting.
    sequence_cache_in_memory
        Keep fetched DNA windows in memory.
    sequence_cache_onehot
        Keep one-hot encoded DNA in memory (only if in-memory cache is enabled).
    """

    def __init__(
        self,
        adata: AnnData,
        genome: Genome,
        gtf_path: str | Path,
        genes: list[str] | None = None,
        val_frac: float = 0.1,
        val_genes: list[str] | None = None,
        spatial_emb_key: str = "spatial",
        seq_length: int = 524_288,
        seed: int = 42,
        max_stochastic_shift: int = 0,
        sequence_cache_in_memory: bool = True,
        sequence_cache_onehot: bool = True,
        encoder_genes: list[str] | None = None,
    ):
        self.adata = adata
        self.genome = genome
        self.gtf_path = gtf_path
        self.val_frac = val_frac
        self._val_genes_explicit = val_genes
        self.spatial_emb_key = spatial_emb_key
        self.seq_length = seq_length
        self.seed = seed
        self.max_stochastic_shift = max_stochastic_shift
        self.sequence_cache_in_memory = sequence_cache_in_memory
        self.sequence_cache_onehot = sequence_cache_onehot
        self._encoder_genes_requested = encoder_genes

        self._genes_requested = genes if genes is not None else list(adata.var_names)
        self._train_genes: list[str] | None = None
        self._val_genes: list[str] | None = None
        self._sequence_cache: GeneSequenceCache | None = None
        self._expression_matrix: np.ndarray | None = None
        self._spatial_embeddings: np.ndarray | None = None
        self._gene_to_idx: dict[str, int] | None = None
        self._gene_means: np.ndarray | None = None
        self._gene_stds: np.ndarray | None = None

        self._encoder_expression_matrix: np.ndarray | None = None
        self._encoder_gene_to_idx: dict[str, int] | None = None
        self._encoder_gene_names: list[str] | None = None

        self._is_setup = False

    def _check_setup(self) -> None:
        """Raise if ``setup()`` has not been called."""
        if not self._is_setup:
            raise RuntimeError("Call setup() first")

    @classmethod
    def from_config(
        cls,
        *,
        adata: AnnData,
        genome: Genome,
        genes: list[str] | None,
        config: DataModuleConfig,
        encoder_genes: list[str] | None = None,
    ):
        """Build a data module from a typed config object.

        Parameters
        ----------
        adata
            AnnData object.
        genome
            Genome instance.
        genes
            Target genes.
        config
            Data module config.
        encoder_genes
            Genes to use as cell encoder input.  Overrides
            ``config.encoder_genes`` when provided.  ``None`` falls back
            to ``config.encoder_genes``.
        """
        effective_encoder_genes = (
            encoder_genes if encoder_genes is not None else config.encoder_genes
        )
        return cls(
            adata=adata,
            genome=genome,
            gtf_path=config.gtf_path,
            genes=genes,
            val_frac=config.val_frac,
            val_genes=config.val_genes,
            spatial_emb_key=config.spatial_emb_key,
            seq_length=config.seq_length,
            seed=config.seed,
            max_stochastic_shift=config.max_stochastic_shift,
            sequence_cache_in_memory=config.sequence_cache_in_memory,
            sequence_cache_onehot=config.sequence_cache_onehot,
            encoder_genes=effective_encoder_genes,
        )

    def to_config_dict(self, include_runtime: bool = True) -> dict[str, Any]:
        """Return a serialisable config snapshot for logging."""
        cfg: dict[str, Any] = {
            "class_name": type(self).__name__,
            "gtf_path": self.gtf_path,
            "val_frac": self.val_frac,
            "val_genes": self._val_genes_explicit,
            "spatial_emb_key": self.spatial_emb_key,
            "seq_length": self.seq_length,
            "seed": self.seed,
            "max_stochastic_shift": self.max_stochastic_shift,
            "sequence_cache_in_memory": self.sequence_cache_in_memory,
            "sequence_cache_onehot": self.sequence_cache_onehot,
            "requested_genes_count": len(self._genes_requested),
        }
        if include_runtime:
            cfg["runtime"] = {
                "is_setup": self._is_setup,
                "n_cells": int(self.adata.n_obs),
                "n_genes_adata": int(self.adata.n_vars),
            }
            if self._is_setup:
                cfg["runtime"].update(
                    {
                        "train_genes_count": len(self._train_genes),
                        "val_genes_count": len(self._val_genes),
                        "spatial_emb_dim": int(self._spatial_embeddings.shape[1]),
                    }
                )
        return serialize_config_value(cfg)

    def setup(self) -> None:
        """Parse GTF, validate genes, split train/val, build caches."""
        logger.info(f"Setting up {type(self).__name__}...")

        gene_coords = _parse_gtf_gene_coords(self.gtf_path)
        logger.info(f"Parsed {len(gene_coords)} genes from GTF")

        valid_genes, skipped, gene_to_region = _compute_gene_regions(
            genes=self._genes_requested,
            gene_coords=gene_coords,
            genome=self.genome,
            seq_length=self.seq_length,
        )

        adata_genes = set(self.adata.var_names)
        valid_genes = [g for g in valid_genes if g in adata_genes]
        gene_to_region = {g: gene_to_region[g] for g in valid_genes}

        if not valid_genes:
            raise ValueError("No valid genes found after filtering.")
        logger.info(
            f"{len(valid_genes)} valid genes (skipped {len(skipped)} from GTF, "
            f"filtered to adata overlap)"
        )

        if self._val_genes_explicit is not None:
            self._val_genes = [g for g in self._val_genes_explicit if g in valid_genes]
            self._train_genes = [g for g in valid_genes if g not in set(self._val_genes)]
        else:
            rng = np.random.RandomState(self.seed)
            n_val = max(1, int(len(valid_genes) * self.val_frac))
            shuffled = rng.permutation(valid_genes).tolist()
            self._val_genes = shuffled[:n_val]
            self._train_genes = shuffled[n_val:]

        logger.info(
            f"Train/val split: {len(self._train_genes)} train, "
            f"{len(self._val_genes)} val genes"
        )

        self._sequence_cache = GeneSequenceCache(
            genome=self.genome,
            gene_to_region=gene_to_region,
            in_memory=self.sequence_cache_in_memory,
            max_stochastic_shift=self.max_stochastic_shift,
            cache_onehot=self.sequence_cache_onehot,
        )

        all_genes = self._train_genes + self._val_genes
        self._gene_to_idx = {g: i for i, g in enumerate(all_genes)}

        var_name_to_idx = {name: i for i, name in enumerate(self.adata.var_names)}
        adata_gene_indices = [var_name_to_idx[g] for g in all_genes]

        expr = self.adata.X[:, adata_gene_indices]
        if issparse(expr):
            expr = expr.toarray()
        self._expression_matrix = np.array(expr, dtype=np.float32).T  # (n_genes, n_cells)

        self._gene_means = self._expression_matrix.mean(axis=1)  # (n_genes,)
        self._gene_stds = self._expression_matrix.std(axis=1)    # (n_genes,)
        self._gene_stds[self._gene_stds < 1e-8] = 1.0  # avoid division by zero

        self._spatial_embeddings = np.array(
            self.adata.obsm[self.spatial_emb_key], dtype=np.float32
        )

        # Build cell encoder expression matrix when encoder genes are specified
        if self._encoder_genes_requested is not None:
            adata_gene_set = set(self.adata.var_names)
            enc_genes = [g for g in self._encoder_genes_requested if g in adata_gene_set]
            if not enc_genes:
                logger.warning(
                    "None of the requested encoder_genes found in adata.var_names; "
                    "cell encoder data will not be available."
                )
            else:
                enc_var_to_idx = {name: i for i, name in enumerate(self.adata.var_names)}
                enc_adata_indices = [enc_var_to_idx[g] for g in enc_genes]
                enc_expr = self.adata.X[:, enc_adata_indices]
                if issparse(enc_expr):
                    enc_expr = enc_expr.toarray()
                self._encoder_expression_matrix = np.array(enc_expr, dtype=np.float32)  # (n_cells, n_encoder_genes)
                self._encoder_gene_to_idx = {g: i for i, g in enumerate(enc_genes)}
                self._encoder_gene_names = enc_genes
                logger.info(
                    f"Cell encoder matrix built: {self._encoder_expression_matrix.shape} "
                    f"({len(enc_genes)} encoder genes)"
                )

        self._is_setup = True
        logger.info(
            f"{type(self).__name__} ready: expression {self._expression_matrix.shape}, "
            f"spatial embeddings {self._spatial_embeddings.shape}"
        )

    @property
    def train_genes(self) -> list[str]:
        self._check_setup()
        return self._train_genes

    @property
    def val_genes(self) -> list[str]:
        self._check_setup()
        return self._val_genes

    @property
    def sequence_cache(self) -> GeneSequenceCache:
        self._check_setup()
        return self._sequence_cache

    @property
    def spatial_emb_dim(self) -> int:
        self._check_setup()
        return self._spatial_embeddings.shape[1]

    @property
    def expression_matrix(self) -> np.ndarray:
        """Expression matrix ``(n_genes, n_cells)``."""
        self._check_setup()
        return self._expression_matrix

    @property
    def spatial_embeddings(self) -> np.ndarray:
        """Spatial cell embeddings ``(n_cells, emb_dim)``."""
        self._check_setup()
        return self._spatial_embeddings

    @property
    def gene_to_idx(self) -> dict[str, int]:
        """Mapping from gene name to row index."""
        self._check_setup()
        return self._gene_to_idx

    @property
    def gene_means(self) -> np.ndarray:
        """Per-gene expression means ``(n_genes,)``."""
        self._check_setup()
        return self._gene_means

    @property
    def gene_stds(self) -> np.ndarray:
        """Per-gene expression stds ``(n_genes,)``."""
        self._check_setup()
        return self._gene_stds

    @property
    def encoder_gene_names(self) -> list[str] | None:
        """Names of genes used as cell encoder input, or ``None`` if no encoder."""
        self._check_setup()
        return self._encoder_gene_names

    @property
    def encoder_gene_to_idx(self) -> dict[str, int] | None:
        """Mapping from encoder gene name to column index, or ``None`` if no encoder."""
        self._check_setup()
        return self._encoder_gene_to_idx

    @property
    def n_encoder_genes(self) -> int | None:
        """Number of encoder input genes, or ``None`` if no encoder."""
        self._check_setup()
        if self._encoder_gene_names is None:
            return None
        return len(self._encoder_gene_names)

    def get_gene_stats(self) -> dict[str, dict[str, float]]:
        """Return per-gene mean/std as a dictionary."""
        self._check_setup()
        all_genes = self._train_genes + self._val_genes
        return {
            gene: {
                "mean": float(self._gene_means[self._gene_to_idx[gene]]),
                "std": float(self._gene_stds[self._gene_to_idx[gene]]),
            }
            for gene in all_genes
        }

    def get_gene_stats_for_genes(self, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return mean/std arrays for subset of genes."""
        self._check_setup()
        gene_indices = [self._gene_to_idx[g] for g in genes]
        return self._gene_means[gene_indices], self._gene_stds[gene_indices]

    def get_expression_for_genes(self, genes: list[str]) -> np.ndarray:
        """Return expression sub-matrix for genes."""
        self._check_setup()
        gene_indices = [self._gene_to_idx[g] for g in genes]
        return self._expression_matrix[gene_indices]

    def _build_dataset_base_kwargs(
        self,
        genes: list[str],
        cells_per_gene: int | None,
        deterministic: bool,
        normalize_targets: bool,
        include_aux_targets: bool = False,
        gene_repeat_factor: int = 1,
    ) -> dict[str, Any]:
        self._check_setup()
        expression_sub = self.get_expression_for_genes(genes)
        gene_means, gene_stds = self.get_gene_stats_for_genes(genes)
        kwargs: dict[str, Any] = {
            "genes": genes,
            "expression_matrix": expression_sub,
            "spatial_embeddings": self._spatial_embeddings,
            "cells_per_gene": cells_per_gene,
            "deterministic": deterministic,
            "gene_means": gene_means,
            "gene_stds": gene_stds,
            "normalize_targets": normalize_targets,
            "gene_repeat_factor": gene_repeat_factor,
        }
        if include_aux_targets:
            kwargs["include_aux_targets"] = True
        if self._encoder_expression_matrix is not None:
            kwargs["encoder_expression_matrix"] = self._encoder_expression_matrix
            kwargs["encoder_gene_to_idx"] = self._encoder_gene_to_idx
        return kwargs

    @abstractmethod
    def make_dataset(self, genes: list[str], **kwargs):
        """Create a dataset for the given genes."""

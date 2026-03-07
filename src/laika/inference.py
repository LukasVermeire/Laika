"""Inference orchestrator for trained Laika models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from .model import Laika
from .trunk_adapter import TrunkAdapter


class Predictor:
    """Inference orchestrator for trained Laika models.

    Parameters
    ----------
    model
        Laika model architecture.
    head_weights
        Path to head checkpoint.
    trunk
        Path or model name for the trunk.
    trunk_weights
        Path to fine-tuned trunk weights.
    data_module
        Data module with sequence cache.
    device
        Device string.
    """

    @staticmethod
    def _load_gene_stats(run_dir: Path) -> dict[str, dict[str, float]] | None:
        """Load gene normalization stats from run directory if available."""
        import json
        stats_path = run_dir / "gene_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            logger.info(f"Loaded gene normalization stats ({len(stats)} genes)")
            return stats
        return None

    @staticmethod
    def _load_lora_config(run_dir: Path) -> dict | None:
        """Load LoRA sidecar config from run directory if available."""
        import json
        lora_path = run_dir / "lora_config.json"
        if lora_path.exists():
            with open(lora_path) as f:
                config = json.load(f)
            logger.info(f"Found LoRA config (rank={config['lora_rank']})")
            return config
        return None

    @classmethod
    def from_head_only(
        cls,
        run_dir: str | Path,
        model: Laika | None = None,
        device: str | None = None,
    ) -> Predictor:
        """Create a head-only predictor.

        Parameters
        ----------
        run_dir
            Run directory.
        model
            Laika model.
        device
            Device string.
        """
        run_dir = Path(run_dir)
        head_weights = run_dir / "head_only_best.pt"
        if not head_weights.exists():
            raise FileNotFoundError(f"No head_only_best.pt found in {run_dir}")
        weights_already_loaded = model is None
        if model is None:
            model = Laika.from_checkpoint(head_weights, map_location=device or "cpu")
        gene_stats = cls._load_gene_stats(run_dir)
        return cls(model=model,
                   head_weights=None if weights_already_loaded else head_weights,
                   device=device, gene_stats=gene_stats)

    @classmethod
    def from_finetune(
        cls,
        run_dir: str | Path,
        trunk: str | Path,
        data_module,
        model: Laika | None = None,
        device: str | None = None,
    ) -> Predictor:
        """Create a fine-tuning predictor.

        Parameters
        ----------
        run_dir
            Run directory.
        trunk
            Path or model name for the trunk.
        data_module
            Data module with sequence cache.
        model
            Laika model.
        device
            Device string.
        """
        run_dir = Path(run_dir)
        head_weights = run_dir / "finetune_best.pt"
        trunk_weights = run_dir / "finetune_best_trunk.weights.h5"
        if not head_weights.exists():
            raise FileNotFoundError(f"No finetune_best.pt found in {run_dir}")
        weights_already_loaded = model is None
        if model is None:
            model = Laika.from_checkpoint(head_weights, map_location=device or "cpu")
        gene_stats = cls._load_gene_stats(run_dir)

        # Check for LoRA sidecar config
        lora_config = cls._load_lora_config(run_dir)

        return cls(
            model=model,
            head_weights=None if weights_already_loaded else head_weights,
            trunk=trunk,
            trunk_weights=trunk_weights if trunk_weights.exists() else None,
            data_module=data_module,
            device=device,
            gene_stats=gene_stats,
            lora_config=lora_config,
        )

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        mode: str,
        trunk: str | Path | None = None,
        data_module=None,
        model: Laika | None = None,
        device: str | None = None,
    ) -> Predictor:
        """Create a predictor.

        Parameters
        ----------
        run_dir
            Run directory.
        mode
            Predictor mode. (``head_only`` or ``finetune``)
        trunk
            Path or model name for the trunk.
        data_module
            Data module.
        model
            Laika model.
        device
            Device string.
        """
        if mode == "head_only":
            return cls.from_head_only(
                run_dir, model=model, device=device
            )
        elif mode == "finetune":
            if trunk is None:
                raise ValueError("trunk is required for finetune mode")
            if data_module is None:
                raise ValueError("data_module is required for finetune mode")
            return cls.from_finetune(
                run_dir, trunk=trunk, data_module=data_module,
                model=model, device=device,
            )
        else:
            raise ValueError(f"mode must be 'head_only' or 'finetune', got {mode!r}")

    def __init__(
        self,
        model: Laika,
        head_weights: str | Path | None = None,
        trunk: str | Path | None = None,
        trunk_weights: str | Path | None = None,
        data_module=None,
        device: str | None = None,
        gene_stats: dict[str, dict[str, float]] | None = None,
        lora_config: dict | None = None,
    ):
        """
        Parameters
        ----------
        model
            Laika model instance.
        head_weights
            Path to head checkpoint.
        trunk
            Path or CREsted model name for the trunk.
        trunk_weights
            Path to fine-tuned trunk weights.
        data_module
            Data module with sequence cache (required for fine-tuning).
        device
            Device string. (``None`` auto-detects)
        gene_stats
            Per-gene normalization stats for denormalization.
        lora_config
            LoRA sidecar config dict.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load head weights (skip if already loaded, e.g. via from_checkpoint)
        self.model = model
        if head_weights is not None:
            self.model.load(head_weights, map_location=str(self.device))
        self.model.to(self.device)
        self.model.eval()

        self._gene_stats = gene_stats

        # Detect training mode
        self._is_finetune = trunk is not None

        if self._is_finetune:
            if data_module is None:
                raise ValueError(
                    "data_module is required for fine-tuning inference (trunk provided)."
                )
            if not hasattr(data_module, "sequence_cache"):
                raise ValueError(
                    "data_module must have a sequence_cache attribute. use setup()"
                )

            from .trunk import load_borzoi_trunk, enable_trunk_lora

            self._trunk_adapter = TrunkAdapter(load_borzoi_trunk(str(trunk)), self.device)
            if lora_config is not None:
                lora_stats = enable_trunk_lora(
                    self._trunk_adapter.trunk, rank=lora_config["lora_rank"]
                )
                logger.info(
                    f"LoRA re-enabled for inference: {lora_stats['lora_layers']} layers, "
                    f"{lora_stats['lora_params']:,} LoRA params"
                )
            if trunk_weights is not None:
                self._trunk_adapter.load_weights(str(trunk_weights))
                logger.info(f"Loaded fine-tuned trunk weights from {trunk_weights}")
            self._trunk_adapter.freeze_for_inference()
            self._data_module = data_module
            logger.info("Predictor ready (fine-tuning: live trunk)")
        else:
            self._trunk_adapter = None
            self._data_module = None
            logger.info("Predictor ready (head-only: precomputed embeddings)")

    def predict(
        self,
        genes: list[str],
        spatial_embeddings: np.ndarray,
        precomputed_embeddings: dict[str, np.ndarray] | None = None,
        cell_indices: np.ndarray | None = None,
        cells_per_chunk: int = 8192,
        genes_per_chunk: int = 32,
    ) -> np.ndarray:
        """Predict expression.

        Parameters
        ----------
        genes
            List of genes.
        spatial_embeddings
            Cell embeddings.
        precomputed_embeddings
            Dict mapping gene -> trunk embedding.
        cell_indices
            Subset of cell indices.
        cells_per_chunk
            Max cells processed per forward pass.
        genes_per_chunk
            Max genes processed per forward pass.

        Returns
        -------
        np.ndarray
            Predicted expression.
        """
        if not self._is_finetune and precomputed_embeddings is None:
            raise ValueError(
                "precomputed_embeddings is required for head-only inference "
                "(no trunk was provided to the Predictor). "
                "Load embeddings with laika.load_precomputed_embeddings()."
            )

        if cell_indices is None:
            cell_indices = np.arange(len(spatial_embeddings))
        if genes_per_chunk < 1:
            raise ValueError(f"genes_per_chunk must be >= 1, got {genes_per_chunk}")

        n_genes = len(genes)
        n_cells = len(cell_indices)
        predictions = np.zeros((n_genes, n_cells), dtype=np.float32)

        mode = "Fine-tuning (live trunk)" if self._is_finetune else "Head-only (precomputed)"
        logger.info(
            f"Predicting expression for {n_genes} genes x {n_cells} cells [{mode}]..."
        )

        spatial_all = torch.from_numpy(
            spatial_embeddings[cell_indices].astype(np.float32)
        )

        with torch.no_grad():
            for g_start in range(0, n_genes, genes_per_chunk):
                g_end = min(g_start + genes_per_chunk, n_genes)
                gene_chunk = genes[g_start:g_end]
                trunk_emb = self._get_trunk_embeddings_batch(
                    gene_chunk, precomputed_embeddings
                )
                n_gene_chunk = len(gene_chunk)

                for c_start in range(0, n_cells, cells_per_chunk):
                    c_end = min(c_start + cells_per_chunk, n_cells)
                    cell_chunk = spatial_all[c_start:c_end].to(self.device)
                    cell_chunk = cell_chunk.unsqueeze(0).expand(n_gene_chunk, -1, -1)

                    preds = self.model(trunk_emb, cell_chunk)

                    predictions[g_start:g_end, c_start:c_end] = preds.cpu().numpy()

                # Clean up trunk losses if using live trunk
                if self._is_finetune:
                    self._trunk_adapter.clear_losses()

        # Denormalize predictions if gene stats are available
        if self._gene_stats is not None:
            for g_idx, gene in enumerate(genes):
                if gene in self._gene_stats:
                    stats = self._gene_stats[gene]
                    predictions[g_idx] = predictions[g_idx] * stats["std"] + stats["mean"]

        logger.info("Prediction complete.")
        return predictions

    def _get_trunk_embeddings_batch(
        self,
        genes: list[str],
        precomputed_embeddings: dict[str, np.ndarray] | None,
    ) -> torch.Tensor:
        """Get a batch of trunk embeddings with shape ``(B, seq_len, channels)``."""
        if self._is_finetune:
            onehot_batch = np.stack(
                [self._data_module.sequence_cache.get_onehot(gene, shift=0) for gene in genes]
            ).astype(np.float32)
            onehot_tensor = torch.from_numpy(onehot_batch).to(self.device)
            trunk_out = self._trunk_adapter.forward(onehot_tensor, training=False)
            return trunk_out
        else:
            emb_batch = np.stack(
                [precomputed_embeddings[gene] for gene in genes]
            ).astype(np.float32)
            return torch.from_numpy(emb_batch).to(self.device)

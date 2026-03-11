"""Training orchestrator: head-only and fine-tuning loops for Laika models."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import (
    FinetuneConfig,
    HeadOnlyConfig,
    TrainerInitConfig,
    serialize_config_value,
)
from .data._base_dataset import collate_fn, worker_init_fn
from .losses import CombinedLoss, get_loss
from .data.precompute import (
    precompute_trunk_embeddings,
    save_precomputed_embeddings,
)
from .model import Laika
from .trunk_adapter import TrunkAdapter
from .utils.metrics import MetricsTracker

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def _resolve_loss(
    loss_fn: str | nn.Module,
    correlation_lambda: float = 0.0,
    base_loss_lambda: float = 1.0,
) -> nn.Module:
    if isinstance(loss_fn, nn.Module):
        base = loss_fn
    else:
        base = get_loss(loss_fn)

    if correlation_lambda > 0 or base_loss_lambda != 1.0:
        return CombinedLoss(base, correlation_lambda, base_loss_lambda)
    return base


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    *,
    factor: float = 0.5,
    plateau_patience: int = 5,
    min_lr: float = 1e-7,
    warmup_epochs: int = 0,
) -> Any | None:
    if scheduler_name == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
    elif scheduler_name == "plateau":
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=plateau_patience,
            min_lr=min_lr,
        )
    elif scheduler_name == "none":
        main_scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name!r}")

    if warmup_epochs > 0 and main_scheduler is not None:
        if isinstance(main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return main_scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

    return main_scheduler


class Trainer:
    """Trainer orchestrator.

    Parameters
    ----------
    model
        Laika model.
    save_dir
        Save directory.
    wandb_project
        W&B project name. (``None`` to disable)
    wandb_run_name
        W&B run name.
    wandb_group
        Optional W&B group.
    wandb_job_type
        Optional W&B job type.
    wandb_tags
        Optional W&B tags.
    wandb_notes
        Optional W&B notes.
    wandb_config
        Additional user configuration to store in W&B.
    device
        Device string. (``cuda`` or ``cpu``)
    """

    def __init__(
        self,
        model: Laika,
        save_dir: str = "laika_runs",
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        wandb_group: str | None = None,
        wandb_job_type: str | None = None,
        wandb_tags: list[str] | None = None,
        wandb_notes: str | None = None,
        wandb_config: dict[str, Any] | None = None,
        device: str | None = None,
    ):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        self._global_epoch: int = 0
        self._last_gate_accuracy: float | None = None
        self._run: Any | None = None
        self._phase_configs: dict[str, dict[str, Any]] = {}
        self._run_extra: dict[str, Any] = {}
        self._wandb_user_config = serialize_config_value(wandb_config or {})
        self._wandb_meta = {
            "project": wandb_project,
            "run_name": wandb_run_name,
            "group": wandb_group,
            "job_type": wandb_job_type,
            "tags": list(wandb_tags or []),
            "notes": wandb_notes,
        }
        if wandb_project is not None:
            if not _WANDB_AVAILABLE:
                raise ImportError(
                    "wandb is not installed. Run `pip install wandb`"
                )
            self._run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                group=wandb_group,
                job_type=wandb_job_type,
                tags=wandb_tags,
                notes=wandb_notes,
                dir=str(self.save_dir),
            )
            logger.info(f"W&B run initialised: {self._run.url}")
            self._sync_wandb_run_config()

    @classmethod
    def from_config(cls, model: Laika, config: TrainerInitConfig) -> "Trainer":
        """Build a trainer from a typed initialisation config."""
        return cls(
            model=model,
            save_dir=str(config.save_dir),
            wandb_project=config.wandb_project,
            wandb_run_name=config.wandb_run_name,
            wandb_group=config.wandb_group,
            wandb_job_type=config.wandb_job_type,
            wandb_tags=config.wandb_tags,
            wandb_notes=config.wandb_notes,
            wandb_config=config.wandb_config,
            device=config.device,
        )

    def _trainer_config_snapshot(self) -> dict[str, Any]:
        return serialize_config_value(
            {
                "save_dir": self.save_dir,
                "device": str(self.device),
                "wandb": self._wandb_meta,
            }
        )

    def _model_config_snapshot(self) -> dict[str, Any]:
        return serialize_config_value(
            {
                "class_name": type(self.model).__name__,
                "head_name": getattr(self.model, "_head_name", type(self.model.head).__name__),
                "head_config": self.model.head.config,
                "trunk_seq_len": self.model.trunk_seq_len,
                "pooled_seq_len": self.model.pooled_seq_len,
                "trunk_channels": self.model.trunk_channels,
                "spatial_emb_dim": self.model.spatial_emb_dim,
                "pool_factor": self.model.pool_factor,
            }
        )

    def _trunk_config_snapshot(self, trunk) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "class_name": type(trunk).__name__,
            "name": getattr(trunk, "name", None),
            "trainable": bool(getattr(trunk, "trainable", False)),
        }
        output_shape = getattr(trunk, "output_shape", None)
        if output_shape is not None:
            cfg["output_shape"] = output_shape
        return serialize_config_value(cfg)

    def _sync_wandb_run_config(
        self,
        *,
        phase_name: str | None = None,
        phase_config: HeadOnlyConfig | FinetuneConfig | None = None,
        data_module=None,
        trunk=None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._run is None:
            return

        if phase_name is not None and phase_config is not None:
            self._phase_configs[phase_name] = phase_config.to_dict()
        if extra:
            self._run_extra.update(serialize_config_value(extra))

        payload: dict[str, Any] = {
            "trainer": self._trainer_config_snapshot(),
            "model": self._model_config_snapshot(),
            "phases": dict(self._phase_configs),
        }
        if hasattr(data_module, "to_config_dict"):
            payload["data_module"] = data_module.to_config_dict(include_runtime=True)
        if trunk is not None:
            payload["trunk"] = self._trunk_config_snapshot(trunk)
        if self._wandb_user_config:
            payload["user"] = self._wandb_user_config
        if self._run_extra:
            payload["extra"] = dict(self._run_extra)

        wandb.config.update(
            {"laika": serialize_config_value(payload)},
            allow_val_change=True,
        )

    def update_run_config(self, extra: dict[str, Any]) -> None:
        """Attach extra structured config metadata to the W&B run."""
        self._sync_wandb_run_config(extra=extra)

    def _validate_phase_config(self, config) -> None:
        """Validate config against model head before a training phase."""
        aux_gene_lambda = config.aux_gene_loss_lambda
        if aux_gene_lambda > 0 and config.normalize_targets:
            raise ValueError(
                "Auxiliary gene loss (aux_gene_loss_lambda) "
                "is incompatible with normalize_targets=True because gene mean "
                "targets become trivially zero after normalization."
            )
        if config.normalize_targets and hasattr(self.model.head, "zero_gate_lambda"):
            raise ValueError(
                "HurdleHead is incompatible with normalize_targets=True. "
                "The hurdle output (sigmoid * softplus) is always >= 0 and "
                "cannot match normalized targets which can be negative."
            )

    def run_head_only(
        self,
        data_module,
        config: HeadOnlyConfig | None = None,
        trunk=None,
        precomputed_embeddings: dict[str, np.ndarray] | None = None,
    ) -> dict[str, list]:
        """Head-only training.

        Parameters
        ----------
        data_module
            Spatial data module.
        config
            Head-only configuration.
        trunk
            Keras trunk model.
        precomputed_embeddings
            Precomputed trunk embeddings.

        Returns
        -------
        dict
            Training history.
        """
        if config is None:
            config = HeadOnlyConfig()

        logger.info(
            "=== Head-only training with precomputed trunk embeddings ==="
        )

        aux_gene_lambda = config.aux_gene_loss_lambda
        include_aux = aux_gene_lambda > 0
        self._validate_phase_config(config)

        self._sync_wandb_run_config(
            phase_name="head_only",
            phase_config=config,
            data_module=data_module,
            trunk=trunk,
        )

        embeddings = self._auto_precompute(
            trunk=trunk,
            data_module=data_module,
            precomputed_embeddings=precomputed_embeddings,
            batch_size=config.precompute_batch_size,
            save_path=config.precomputed_save_path,
        )

        train_dataset = data_module.make_dataset(
            genes=data_module.train_genes,
            precomputed_embeddings=embeddings,
            cells_per_gene=config.cells_per_gene,
            deterministic=False,
            normalize_targets=config.normalize_targets,
            include_aux_targets=include_aux,
            gene_repeat_factor=config.gene_repeat_factor,
        )
        val_dataset = data_module.make_dataset(
            genes=data_module.val_genes,
            precomputed_embeddings=embeddings,
            cells_per_gene=config.val_cells_per_gene,
            deterministic=True,
            normalize_targets=config.normalize_targets,
            include_aux_targets=include_aux,
        )

        train_loader = self._make_loader(
            train_dataset, config.genes_per_batch, config.num_workers, shuffle=True
        )
        val_loader = self._make_loader(
            val_dataset, config.genes_per_batch, config.num_workers, shuffle=False
        )

        self._log_data_info(
            train_dataset, val_dataset, train_loader, val_loader,
            config.genes_per_batch, config.gradient_accumulation_steps,
            config.cells_per_gene,
        )

        # Save gene stats for inference denormalization
        if config.normalize_targets:
            self._save_gene_stats(data_module)

        self.model.to(self.device)

        criterion = _resolve_loss(
            config.loss_fn,
            config.correlation_loss_lambda,
            config.base_loss_lambda,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        lr_scheduler = _build_scheduler(
            optimizer,
            config.scheduler,
            config.epochs,
            factor=config.scheduler_factor,
            plateau_patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
            warmup_epochs=config.warmup_epochs,
        )
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {n_params:,}")
        if self._run:
            wandb.log({"HO/trainable_params": n_params})

        def forward_fn(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
            trunk_emb = batch["trunk_emb"].to(self.device, non_blocking=True)
            cell_embs = batch["cell_embs"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)
            expression_vectors = batch.get("expression_vectors")
            gene_encoder_idx = batch.get("gene_encoder_idx")
            if expression_vectors is not None:
                expression_vectors = expression_vectors.to(self.device, non_blocking=True)
                gene_encoder_idx = gene_encoder_idx.to(self.device, non_blocking=True)
            preds = self.model(trunk_emb, cell_embs, expression_vectors, gene_encoder_idx)
            return preds, targets

        history = self._training_loop(
            forward_fn=forward_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            epochs=config.epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            clip_grad_norm=config.clip_grad_norm,
            patience=config.patience,
            checkpoint_path=self.save_dir / "head_only_best.pt",
            phase_prefix="HO",
            all_params=list(self.model.parameters()),
            metrics_mode=config.metrics_mode,
            gc_frequency_steps=config.gc_frequency_steps,
            empty_cache_frequency_steps=config.empty_cache_frequency_steps,
            aux_gene_lambda=aux_gene_lambda,
        )

        # Load best model
        best_path = self.save_dir / "head_only_best.pt"
        if best_path.exists():
            self.model.load(best_path, map_location=str(self.device))
            logger.info("Loaded best head-only model weights")

        return history

    def run_finetune(
        self,
        data_module,
        trunk,
        config: FinetuneConfig | None = None,
    ) -> dict[str, list]:
        """End-to-end fine-tuning. (trunk + head)

        Parameters
        ----------
        data_module
            Sequence data module.
        trunk
            Keras trunk model.
        config
            Fine-tuning configuration.

        Returns
        -------
        dict
            Training history.
        """
        import keras

        if config is None:
            config = FinetuneConfig()

        logger.info("=== End-to-end fine-tuning (trunk + head) ===")

        aux_gene_lambda = config.aux_gene_loss_lambda
        include_aux = aux_gene_lambda > 0
        self._validate_phase_config(config)

        # Record finetune start for visual distinction in wandb
        if self._run:
            wandb.config.update(
                {"FT_start_step": self._global_epoch + 1}, allow_val_change=True
            )

        # require torch backend
        if keras.backend.backend() != "torch":
            raise RuntimeError(
                "Fine-tuning requires KERAS_BACKEND=torch. "
                "Set os.environ['KERAS_BACKEND'] = 'torch' before importing Keras."
            )

        # Load head-only checkpoint
        best_head_only_path = self.save_dir / "head_only_best.pt"
        if config.load_head_only_checkpoint and best_head_only_path.exists():
            self.model.load(best_head_only_path, map_location=str(self.device))
            logger.info(f"Loaded head-only weights from {best_head_only_path}")
        else:
            logger.info("Skipping head-only checkpoint load (not found or disabled)")

        trunk_adapter = TrunkAdapter(trunk, self.device)
        lora_stats = trunk_adapter.configure_train_mode(
            freeze=config.freeze_trunk,
            use_lora=config.use_lora,
            lora_rank=config.lora_rank,
        )
        trunk_params = trunk_adapter.trainable_parameters()
        self._sync_wandb_run_config(
            phase_name="finetune",
            phase_config=config,
            data_module=data_module,
            trunk=trunk,
            extra={
                "finetune": {
                    "freeze_trunk": config.freeze_trunk,
                    "use_lora": config.use_lora,
                    "lora_rank": config.lora_rank,
                    "lora_stats": lora_stats,
                }
            },
        )

        logger.info(
            f"Trunk trainable parameters: {sum(p.numel() for p in trunk_params):,}"
        )
        trunk_requires_grad = len(trunk_params) > 0
        if config.use_lora and not trunk_requires_grad:
            logger.warning(
                "LoRA requested but no trainable trunk parameters were found. "
                "The trunk will run without gradients."
            )

        self.model.to(self.device)

        criterion = _resolve_loss(
            config.loss_fn,
            config.correlation_loss_lambda,
            config.base_loss_lambda,
        )

        # Differential LR optimizer
        # Filter head params to only those requiring gradients, and exclude
        # any trunk params (which have their own group with a separate LR).
        trunk_param_ids = {id(p) for p in trunk_params}
        head_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in trunk_param_ids
        ]
        optimizer_groups = []
        if trunk_params:
            optimizer_groups.append({"params": trunk_params, "lr": config.trunk_lr})
        optimizer_groups.append({"params": head_params, "lr": config.head_lr})
        
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            weight_decay=config.weight_decay,
        )
        lr_scheduler = _build_scheduler(
            optimizer,
            config.scheduler,
            config.epochs,
            factor=config.scheduler_factor,
            plateau_patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
            warmup_epochs=config.warmup_epochs,
        )

        # Create datasets and loaders
        train_dataset = data_module.make_dataset(
            genes=data_module.train_genes,
            cells_per_gene=config.cells_per_gene,
            deterministic=False,
            normalize_targets=config.normalize_targets,
            include_aux_targets=include_aux,
            gene_repeat_factor=config.gene_repeat_factor,
        )
        val_dataset = data_module.make_dataset(
            genes=data_module.val_genes,
            cells_per_gene=config.val_cells_per_gene,
            deterministic=True,
            normalize_targets=config.normalize_targets,
            include_aux_targets=include_aux,
        )

        train_loader = self._make_loader(
            train_dataset, config.genes_per_batch, config.num_workers, shuffle=True
        )
        val_loader = self._make_loader(
            val_dataset, config.genes_per_batch, config.num_workers, shuffle=False
        )

        self._log_data_info(
            train_dataset, val_dataset, train_loader, val_loader,
            config.genes_per_batch, config.gradient_accumulation_steps,
            config.cells_per_gene,
        )

        # Save gene stats for inference denormalization
        if config.normalize_targets:
            self._save_gene_stats(data_module)

        n_head_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Head trainable parameters: {n_head_params:,}")

        if config.trunk_forward_training is None:
            # LoRA defaults to inference-mode forward to avoid attention-dropout
            # tensor blowups while still allowing gradients on LoRA weights.
            trunk_training = trunk_requires_grad and not config.use_lora
        else:
            trunk_training = bool(config.trunk_forward_training) and trunk_requires_grad

        logger.info(
            f"Trunk forward mode: training={trunk_training}, "
            f"requires_grad={trunk_requires_grad}"
        )
        forward_fn = self._make_trunk_forward(
            trunk_adapter,
            training=trunk_training,
            requires_grad=trunk_requires_grad,
        )
        forward_fn_eval = self._make_trunk_forward(
            trunk_adapter,
            training=False,
            requires_grad=False,
        )

        all_params = trunk_params + list(self.model.parameters())

        best_finetune_path = self.save_dir / "finetune_best.pt"
        best_trunk_path = self.save_dir / "finetune_best_trunk.weights.h5"

        history = self._training_loop(
            forward_fn=forward_fn,
            forward_fn_eval=forward_fn_eval,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            epochs=config.epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            clip_grad_norm=config.clip_grad_norm,
            patience=config.patience,
            checkpoint_path=best_finetune_path,
            phase_prefix="FT",
            all_params=all_params,
            trunk_adapter=trunk_adapter,
            trunk_checkpoint_path=best_trunk_path,
            metrics_mode=config.metrics_mode,
            gc_frequency_steps=config.gc_frequency_steps,
            empty_cache_frequency_steps=config.empty_cache_frequency_steps,
            aux_gene_lambda=aux_gene_lambda,
        )

        # Load best weights
        if best_finetune_path.exists():
            self.model.load(best_finetune_path, map_location=str(self.device))
            logger.info(f"Loaded best head weights from {best_finetune_path}")
        if best_trunk_path.exists():
            trunk_adapter.load_weights(str(best_trunk_path))
            logger.info(f"Loaded best trunk weights from {best_trunk_path}")

        # Save LoRA sidecar config for inference
        if config.use_lora:
            lora_config_path = self.save_dir / "lora_config.json"
            with open(lora_config_path, "w") as f:
                json.dump({"lora_rank": config.lora_rank}, f)
            logger.info(f"Saved LoRA config to {lora_config_path}")

        return history

    def finish(self) -> None:
        """Finalise W&B run."""
        if self._run is not None:
            self._run.finish()
            self._run = None

    def _save_gene_stats(self, data_module) -> None:
        """Save gene normalization stats."""
        gene_stats = data_module.get_gene_stats()
        gene_stats_path = self.save_dir / "gene_stats.json"
        with open(gene_stats_path, "w") as f:
            json.dump(gene_stats, f)
        logger.info(f"Saved gene normalization stats to {gene_stats_path}")

    def _make_trunk_forward(
        self,
        trunk_adapter: TrunkAdapter,
        training: bool,
        requires_grad: bool,
    ) -> Callable[[dict], tuple[torch.Tensor, torch.Tensor]]:
        def forward_fn(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
            onehot = batch["onehot"].to(self.device, non_blocking=True)
            cell_embs = batch["cell_embs"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)
            expression_vectors = batch.get("expression_vectors")
            gene_encoder_idx = batch.get("gene_encoder_idx")
            if expression_vectors is not None:
                expression_vectors = expression_vectors.to(self.device, non_blocking=True)
                gene_encoder_idx = gene_encoder_idx.to(self.device, non_blocking=True)
            if requires_grad:
                trunk_out = trunk_adapter.forward(onehot, training=training)
            else:
                with torch.no_grad():
                    trunk_out = trunk_adapter.forward(onehot, training=False)
            preds = self.model(trunk_out, cell_embs, expression_vectors, gene_encoder_idx)
            return preds, targets
        return forward_fn

    def _record_lr(
        self,
        history: dict[str, list],
        optimizer: torch.optim.Optimizer,
        is_finetune: bool,
        pbar,
        train_results: dict[str, float],
        val_results: dict[str, float],
    ) -> None:
        """Record learning rate and update progress bar."""
        base_postfix = {
            "loss": f"{train_results['loss']:.4f}",
            "v_loss": f"{val_results['loss']:.4f}",
            "r": f"{train_results['per_gene_pearson_r']:.3f}",
            "v_r": f"{val_results['per_gene_pearson_r']:.3f}",
        }
        if is_finetune:
            if len(optimizer.param_groups) > 1:
                trunk_lr = optimizer.param_groups[0]["lr"]
                head_lr = optimizer.param_groups[1]["lr"]
                history["trunk_lr"].append(trunk_lr)
                base_postfix["t_lr"] = f"{trunk_lr:.1e}"
            else:
                head_lr = optimizer.param_groups[0]["lr"]
            history["head_lr"].append(head_lr)
            base_postfix["h_lr"] = f"{head_lr:.1e}"
        else:
            lr = optimizer.param_groups[0]["lr"]
            history["lr"].append(lr)
            base_postfix["lr"] = f"{lr:.1e}"
        pbar.set_postfix(base_postfix)

    def _auto_precompute(
        self,
        trunk,
        data_module,
        precomputed_embeddings: dict[str, np.ndarray] | None,
        batch_size: int,
        save_path: str | None,
    ) -> dict[str, np.ndarray]:
        """Resolve embeddings."""
        if precomputed_embeddings is not None:
            logger.info(
                f"Using provided precomputed embeddings "
                f"({len(precomputed_embeddings)} genes)"
            )
            # Validate seq_len against model's expected pooled_seq_len
            sample_emb = next(iter(precomputed_embeddings.values()))
            actual_seq_len = sample_emb.shape[0]
            expected_seq_len = self.model.pooled_seq_len
            if actual_seq_len != expected_seq_len:
                raise ValueError(
                    f"Provided precomputed embeddings have seq_len {actual_seq_len}, "
                    f"but the model expects {expected_seq_len} "
                    f"(trunk_seq_len={self.model.trunk_seq_len}, "
                    f"pool_factor={self.model.pool_factor}). "
                    "This often happens when a cached .npz file was computed with "
                    "a different pool_factor. Delete the stale cache and re-run, "
                    "or rebuild the model to match."
                )
            return precomputed_embeddings

        if trunk is None:
            raise ValueError(
                "Either `trunk` or `precomputed_embeddings` must be provided."
            )

        all_genes = data_module.train_genes + data_module.val_genes
        logger.info(f"Auto-precomputing trunk embeddings for {len(all_genes)} genes...")

        t0 = time.monotonic()
        embeddings = precompute_trunk_embeddings(
            trunk=trunk,
            sequence_cache=data_module.sequence_cache,
            genes=all_genes,
            pool_factor=self.model.pool_factor,
            batch_size=batch_size,
        )
        elapsed = time.monotonic() - t0
        logger.info(f"Precomputation complete in {elapsed:.1f}s")

        # Auto-save
        actual_save_path = save_path or str(
            self.save_dir / "precomputed_embeddings.npz"
        )
        save_precomputed_embeddings(
            embeddings, actual_save_path, pool_factor=self.model.pool_factor
        )

        gc.collect()
        return embeddings

    def _make_loader(
        self,
        dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

    def _log_data_info(
        self,
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
        genes_per_batch: int,
        gradient_accumulation_steps: int,
        cells_per_gene: int,
    ) -> None:
        """Log dataset info."""
        repeat_factor = getattr(train_dataset, "gene_repeat_factor", 1)
        n_unique_genes = len(train_dataset.genes)
        train_info = f"Train: {len(train_dataset)} items"
        if repeat_factor > 1:
            train_info += f" ({n_unique_genes} genes x {repeat_factor} repeats)"
        else:
            train_info += f" ({n_unique_genes} genes)"
        train_info += f", {len(train_loader)} batches/epoch"
        logger.info(
            f"{train_info} | "
            f"Val: {len(val_dataset)} genes, {len(val_loader)} batches/epoch"
        )
        effective_batch = genes_per_batch * gradient_accumulation_steps
        logger.info(
            f"Effective batch: {effective_batch} genes "
            f"({genes_per_batch} x {gradient_accumulation_steps} accum), "
            f"{effective_batch * cells_per_gene} predictions/step"
        )

    def _compute_aux_loss(
        self,
        batch: dict,
        aux_gene_lambda: float,
    ) -> torch.Tensor | None:
        """Compute auxiliary losses (gene baseline, zero gate, etc.).

        Returns the total weighted auxiliary loss,
        or ``None`` if no auxiliary losses apply.
        """
        head = self.model.head
        if not hasattr(head, "get_auxiliaries"):
            return None

        auxiliaries = head.get_auxiliaries()
        if not auxiliaries:
            return None

        total = torch.tensor(0.0, device=self.device)
        found = False

        # Gene baseline (decomposed head)
        if "gene_baseline" in auxiliaries and aux_gene_lambda > 0:
            gene_target = batch["gene_mean_target"].to(self.device, non_blocking=True)
            total = total + aux_gene_lambda * nn.functional.mse_loss(
                auxiliaries["gene_baseline"], gene_target
            )
            found = True

        # Zero gate (hurdle head)
        if "zero_gate_logits" in auxiliaries:
            zero_gate_lambda = getattr(head, "zero_gate_lambda", 0.0)
            if zero_gate_lambda > 0:
                targets = batch["targets"].to(self.device, non_blocking=True)
                gate_target = (targets > 0).float()
                total = total + zero_gate_lambda * nn.functional.binary_cross_entropy_with_logits(
                    auxiliaries["zero_gate_logits"], gate_target
                )
                found = True

                # Gate accuracy diagnostic
                with torch.no_grad():
                    gate_preds = (auxiliaries["zero_gate_logits"] > 0).float()
                    self._last_gate_accuracy = float(
                        (gate_preds == gate_target).float().mean().item()
                    )

        return total if found else None

    def _training_loop(
        self,
        forward_fn: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        lr_scheduler,
        epochs: int,
        gradient_accumulation_steps: int,
        clip_grad_norm: float,
        patience: int,
        checkpoint_path: Path,
        phase_prefix: str,
        all_params: list[torch.nn.Parameter],
        metrics_mode: str = "full",
        gc_frequency_steps: int = 0,
        empty_cache_frequency_steps: int = 0,
        forward_fn_eval: Callable | None = None,
        trunk_adapter: TrunkAdapter | None = None,
        trunk_checkpoint_path: Path | None = None,
        aux_gene_lambda: float = 0.0,
    ) -> dict[str, list]:
        """Shared training loop.

        Parameters
        ----------
        forward_fn
            Forward pass function (training).
        train_loader
            DataLoader for training.
        val_loader
            DataLoader for validation.
        optimizer
            Optimizer.
        criterion
            Loss function.
        lr_scheduler
            LR scheduler.
        epochs
            Max epochs.
        gradient_accumulation_steps
            Gradient accumulation steps.
        clip_grad_norm
            Max gradient norm for clipping.
        patience
            Early stopping patience.
        checkpoint_path
            Path to save best head weights.
        phase_prefix
            Prefix for metric logging (``'HO'`` or ``'FT'``).
        all_params
            All trainable parameters for gradient clipping.
        metrics_mode
            ``'full'`` or ``'fast'`` metrics computation.
        gc_frequency_steps
            Run ``gc.collect()`` every N steps. (0 to disable)
        empty_cache_frequency_steps
            Run ``torch.cuda.empty_cache()`` every N steps. (0 to disable)
        forward_fn_eval
            Eval-mode forward pass. (default: ``forward_fn``)
        trunk_adapter
            Adapter around the Keras trunk for fine-tuning.
        trunk_checkpoint_path
            Path to save best trunk weights.
        aux_gene_lambda
            Weight for auxiliary per-gene mean loss.
        """
        if forward_fn_eval is None:
            forward_fn_eval = forward_fn

        history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_pearson_r": [],
            "val_pearson_r": [],
            "train_spearman_r": [],
            "val_spearman_r": [],
            "train_per_gene_pearson_r": [],
            "val_per_gene_pearson_r": [],
            "train_per_gene_spearman_r": [],
            "val_per_gene_spearman_r": [],
        }
        # Add LR keys based on training mode
        is_finetune = phase_prefix == "FT"
        if is_finetune:
            if len(optimizer.param_groups) > 1:
                history["trunk_lr"] = []
            history["head_lr"] = []
        else:
            history["lr"] = []

        best_val_loss = float("inf")
        patience_counter = 0
        epochs_trained = 0

        phase_desc = "Fine-tuning" if is_finetune else "Head-only"
        pbar = tqdm(
            range(epochs), 
            desc=phase_desc, 
            unit="epoch", 
            dynamic_ncols=True
        )
        for epoch in pbar:
            epochs_trained = epoch + 1
            self._global_epoch += 1
            epoch_start = time.monotonic()

            train_results = self._train_epoch(
                forward_fn=forward_fn,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                gradient_accumulation_steps=gradient_accumulation_steps,
                clip_grad_norm=clip_grad_norm,
                epoch=epoch,
                epochs=epochs,
                all_params=all_params,
                trunk_adapter=trunk_adapter,
                metrics_mode=metrics_mode,
                gc_frequency_steps=gc_frequency_steps,
                empty_cache_frequency_steps=empty_cache_frequency_steps,
                aux_gene_lambda=aux_gene_lambda,
            )

            val_results = self._eval_epoch(
                forward_fn=forward_fn_eval,
                loader=val_loader,
                criterion=criterion,
                epoch=epoch,
                epochs=epochs,
                trunk_adapter=trunk_adapter,
                metrics_mode=metrics_mode,
                gc_frequency_steps=gc_frequency_steps,
                empty_cache_frequency_steps=empty_cache_frequency_steps,
                aux_gene_lambda=aux_gene_lambda,
            )

            if lr_scheduler is not None:
                if isinstance(
                    lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    lr_scheduler.step(val_results["loss"])
                else:
                    lr_scheduler.step()

            epoch_time = time.monotonic() - epoch_start
            history["train_loss"].append(train_results["loss"])
            history["val_loss"].append(val_results["loss"])
            history["train_pearson_r"].append(train_results["pearson_r"])
            history["val_pearson_r"].append(val_results["pearson_r"])
            history["train_spearman_r"].append(train_results["spearman_r"])
            history["val_spearman_r"].append(val_results["spearman_r"])
            history["train_per_gene_pearson_r"].append(train_results["per_gene_pearson_r"])
            history["val_per_gene_pearson_r"].append(val_results["per_gene_pearson_r"])
            history["train_per_gene_spearman_r"].append(train_results["per_gene_spearman_r"])
            history["val_per_gene_spearman_r"].append(val_results["per_gene_spearman_r"])

            self._record_lr(
                history, optimizer, is_finetune, pbar,
                train_results, val_results,
            )

            self._log_metrics(
                train_results=train_results,
                val_results=val_results,
                optimizer=optimizer,
                epoch_time=epoch_time,
                phase_prefix=phase_prefix,
                is_finetune=is_finetune,
            )

            if val_results["loss"] < best_val_loss:
                best_val_loss = val_results["loss"]
                patience_counter = 0
                self.model.save(checkpoint_path)
                if trunk_adapter is not None and trunk_checkpoint_path is not None:
                    trunk_adapter.save_weights(str(trunk_checkpoint_path))
                pbar.write(
                    f"  Epoch {epoch + 1}: New best model "
                    f"(val_loss={best_val_loss:.4f})"
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                pbar.write(
                    f"  Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)"
                )
                break

        pbar.close()

        logger.info(
            f"{phase_desc} complete. Best val_loss={best_val_loss:.4f} "
            f"(trained {epochs_trained} epochs)"
        )
        return history

    def _train_epoch(
        self,
        forward_fn: Callable,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        gradient_accumulation_steps: int,
        clip_grad_norm: float,
        epoch: int,
        epochs: int,
        all_params: list[torch.nn.Parameter],
        metrics_mode: str,
        gc_frequency_steps: int,
        empty_cache_frequency_steps: int,
        trunk_adapter: TrunkAdapter | None = None,
        aux_gene_lambda: float = 0.0,
    ) -> dict[str, float]:
        """Run a training epoch."""
        self.model.train()
        fast_metrics = metrics_mode == "fast"
        has_gate = hasattr(self.model.head, "zero_gate_lambda")
        metrics = MetricsTracker(
            track_per_gene=not fast_metrics,
            track_spearman=not fast_metrics,
            track_nonzero_pearson=has_gate and not fast_metrics,
        )
        optimizer.zero_grad()
        total_aux_loss = 0.0
        aux_count = 0
        total_gate_accuracy = 0.0
        gate_acc_count = 0

        train_pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_idx, batch in enumerate(train_pbar):
            preds, targets = forward_fn(batch)
            main_loss = criterion(preds, targets)

            aux_loss = self._compute_aux_loss(batch, aux_gene_lambda)
            if aux_loss is not None:
                loss = main_loss + aux_loss
                total_aux_loss += aux_loss.item()
                aux_count += 1
                if self._last_gate_accuracy is not None:
                    total_gate_accuracy += self._last_gate_accuracy
                    gate_acc_count += 1
                    self._last_gate_accuracy = None
            else:
                loss = main_loss

            loss = loss / gradient_accumulation_steps

            loss.backward()

            with torch.no_grad():
                batch_loss = main_loss.item()
                batch_y_true = targets.detach().cpu().numpy()
                batch_y_pred = preds.detach().cpu().numpy()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(loader):
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            batch_gene_names = batch.get("gene_name")

            del preds, targets, loss, batch
            if trunk_adapter is not None:
                trunk_adapter.clear_losses()
            if hasattr(self.model.head, "clear_auxiliaries"):
                self.model.head.clear_auxiliaries()
            metrics.update(
                loss=batch_loss,
                y_true=batch_y_true,
                y_pred=batch_y_pred,
                gene_names=batch_gene_names,
            )
            del batch_y_true, batch_y_pred

            self._maybe_memory_cleanup(
                step=batch_idx + 1,
                gc_frequency_steps=gc_frequency_steps,
                empty_cache_frequency_steps=empty_cache_frequency_steps,
            )

        train_pbar.close()
        results = metrics.compute()
        if aux_count > 0:
            results["aux_loss"] = total_aux_loss / aux_count
        if gate_acc_count > 0:
            results["gate_accuracy"] = total_gate_accuracy / gate_acc_count
        return results

    def _eval_epoch(
        self,
        forward_fn: Callable,
        loader: DataLoader,
        criterion: nn.Module,
        epoch: int,
        epochs: int,
        metrics_mode: str,
        gc_frequency_steps: int,
        empty_cache_frequency_steps: int,
        trunk_adapter: TrunkAdapter | None = None,
        aux_gene_lambda: float = 0.0,
    ) -> dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        fast_metrics = metrics_mode == "fast"
        has_gate = hasattr(self.model.head, "zero_gate_lambda")
        metrics = MetricsTracker(
            track_per_gene=not fast_metrics,
            track_spearman=not fast_metrics,
            track_nonzero_pearson=has_gate and not fast_metrics,
        )
        total_aux_loss = 0.0
        aux_count = 0
        total_gate_accuracy = 0.0
        gate_acc_count = 0

        val_pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Val]",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                preds, targets = forward_fn(batch)
                main_loss = criterion(preds, targets)

                aux_loss = self._compute_aux_loss(batch, aux_gene_lambda)
                if aux_loss is not None:
                    total_aux_loss += aux_loss.item()
                    aux_count += 1
                    if self._last_gate_accuracy is not None:
                        total_gate_accuracy += self._last_gate_accuracy
                        gate_acc_count += 1
                        self._last_gate_accuracy = None

                batch_loss = main_loss.item()
                batch_y_true = targets.cpu().numpy()
                batch_y_pred = preds.cpu().numpy()

                batch_gene_names = batch.get("gene_name")

                del preds, targets, main_loss, batch
                if trunk_adapter is not None:
                    trunk_adapter.clear_losses()
                if hasattr(self.model.head, "clear_auxiliaries"):
                    self.model.head.clear_auxiliaries()

                metrics.update(
                    loss=batch_loss,
                    y_true=batch_y_true,
                    y_pred=batch_y_pred,
                    gene_names=batch_gene_names,
                )
                del batch_y_true, batch_y_pred

                self._maybe_memory_cleanup(
                    step=batch_idx + 1,
                    gc_frequency_steps=gc_frequency_steps,
                    empty_cache_frequency_steps=empty_cache_frequency_steps,
                )

        val_pbar.close()
        results = metrics.compute()
        if aux_count > 0:
            results["aux_loss"] = total_aux_loss / aux_count
        if gate_acc_count > 0:
            results["gate_accuracy"] = total_gate_accuracy / gate_acc_count
        return results

    def _maybe_memory_cleanup(
        self,
        *,
        step: int,
        gc_frequency_steps: int,
        empty_cache_frequency_steps: int,
    ) -> None:
        if gc_frequency_steps > 0 and step % gc_frequency_steps == 0:
            gc.collect()
        if (
            self.device.type == "cuda"
            and empty_cache_frequency_steps > 0
            and step % empty_cache_frequency_steps == 0
        ):
            torch.cuda.empty_cache()

    def _log_metrics(
        self,
        train_results: dict[str, float],
        val_results: dict[str, float],
        optimizer: torch.optim.Optimizer,
        epoch_time: float,
        phase_prefix: str,
        is_finetune: bool,
    ) -> None:
        """Log metrics to wandb if enabled."""
        if self._run is None:
            return

        log_dict = {
            "epoch": self._global_epoch,
            "train/phase": "FT" if is_finetune else "HO",
            "epoch_time_s": epoch_time,
        }

        # Log all result keys under train/ val/ and phase-prefixed keys
        all_keys = [
            "loss", "pearson_r", "spearman_r",
            "per_gene_pearson_r", "per_gene_spearman_r",
            "aux_loss", "gate_accuracy", "nz_per_gene_pearson_r",
        ]
        for key in all_keys:
            for split, results in [("train", train_results), ("val", val_results)]:
                if key in results:
                    log_dict[f"{split}/{key}"] = results[key]
                    log_dict[f"{phase_prefix}/{split}_{key}"] = results[key]

        # Derived total_loss (base + aux)
        for split, results in [("train", train_results), ("val", val_results)]:
            if "aux_loss" in results:
                total = results["loss"] + results["aux_loss"]
                log_dict[f"{split}/total_loss"] = total
                log_dict[f"{phase_prefix}/{split}_total_loss"] = total

        # Learning rates
        if is_finetune and len(optimizer.param_groups) >= 2:
            log_dict[f"{phase_prefix}/trunk_lr"] = optimizer.param_groups[0]["lr"]
            log_dict[f"{phase_prefix}/head_lr"] = optimizer.param_groups[1]["lr"]
        else:
            log_dict[f"{phase_prefix}/lr"] = optimizer.param_groups[0]["lr"]

        wandb.log(log_dict, step=self._global_epoch)

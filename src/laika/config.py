"""Typed configuration dataclasses for Laika training."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn


def serialize_config_value(value: Any) -> Any:
    """Convert config values to W&B-safe primitives."""
    if dataclasses.is_dataclass(value):
        return {
            f.name: serialize_config_value(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, nn.Module):
        return type(value).__name__
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serialize_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_config_value(v) for v in value]
    return value


class ConfigBase:
    """Mixin providing serialisation for all config dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        return serialize_config_value(self)


@dataclass
class DataModuleConfig(ConfigBase):
    """Configuration for BaseDataModule initialisation.

    Parameters
    ----------
    gtf_path
        Path to GTF annotation file.
    val_frac
        Fraction of genes held out for validation.
    val_genes
        Explicit validation gene list. (overrides ``val_frac``)
    spatial_emb_key
        Key in ``adata.obsm`` for spatial embeddings.
    seq_length
        DNA window length (bp).
    seed
        Random seed for train/val split.
    max_stochastic_shift
        Extra bases fetched on each side for stochastic shifting.
    sequence_cache_in_memory
        Preload all sequences into RAM.
    sequence_cache_onehot
        Cache one-hot arrays in RAM (only if ``sequence_cache_in_memory``
        is also True).
    encoder_genes
        Genes whose expression values are fed to the cell encoder.
        ``None`` means "use all AnnData genes" when a cell encoder is
        active; ignored when no cell encoder is configured.
    """

    gtf_path: str | Path
    val_frac: float = 0.1
    val_genes: list[str] | None = None
    spatial_emb_key: str = "spatial"
    seq_length: int = 524_288
    seed: int = 42
    max_stochastic_shift: int = 0
    sequence_cache_in_memory: bool = True
    sequence_cache_onehot: bool = True
    encoder_genes: list[str] | None = None



@dataclass
class TrainerInitConfig(ConfigBase):
    """Configuration for Trainer initialisation.

    Parameters
    ----------
    save_dir
        Directory to save checkpoints and logs.
    device
        PyTorch device string. (e.g. ``"cuda"`` or ``"cpu"``; ``None`` auto-detects)
    wandb_project
        W&B project name. (``None`` disables W&B logging)
    wandb_run_name
        W&B run name.
    wandb_group
        W&B group.
    wandb_job_type
        W&B job type.
    wandb_tags
        W&B tags.
    wandb_notes
        W&B notes.
    wandb_config
        Extra user config logged to W&B.
    """

    save_dir: str | Path = "laika_runs"
    device: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str | None = None
    wandb_config: dict[str, Any] | None = None


@dataclass
class TrunkConfig(ConfigBase):
    """Configuration for loading the sequence trunk model.

    Parameters
    ----------
    model_path
        Path or CREsted model name.
    extraction_layer
        Name of the intermediate layer to use as trunk output.
    """

    model_path: str
    extraction_layer: str = "add_15"


@dataclass
class CellEncoderConfig(ConfigBase):
    """Configuration for the learnable cell encoder.

    Parameters
    ----------
    encoder
        Registry name of the encoder architecture (e.g. ``"transformer"``).
    encoder_kwargs
        Extra keyword arguments forwarded to the encoder constructor.
    output_dim
        Dimension of the produced cell embedding.  Must match the head's
        ``spatial_emb_dim`` — when using :class:`CellEncoderConfig` the
        Laika model sets ``spatial_emb_dim`` to this value automatically.
    """

    encoder: str = "transformer"
    encoder_kwargs: dict[str, Any] = field(default_factory=dict)
    output_dim: int = 64


@dataclass
class ModelConfig(ConfigBase):
    """Configuration for constructing the Laika model.

    Parameters
    ----------
    head
        Head name from the registry. (e.g. ``"mlp"``, ``"film"``,
        ``"cross_attention"``, ``"decomposed"``, ``"hurdle"``,
        ``"hybrid"``, ``"hyperconv"``)
    head_kwargs
        Extra keyword arguments forwarded to the head constructor.
    spatial_emb_dim
        Spatial embedding dimension. (``None`` infers from the data module;
        overridden by ``cell_encoder.output_dim`` when a cell encoder is set)
    pool_factor
        Average-pooling factor applied to trunk output before the head.
    cell_encoder
        Cell encoder configuration.  ``None`` uses precomputed spatial
        embeddings (default behaviour).
    cell_encoder_chunk_size
        When set, the cell encoder processes cells in chunks of this size
        during eval/inference. Has no effect during training (where ``cells_per_gene`` already
        limits the batch size).
    """

    head: str = "mlp"
    head_kwargs: dict[str, Any] = field(default_factory=dict)
    spatial_emb_dim: int | None = None
    pool_factor: int = 8
    cell_encoder: CellEncoderConfig | None = None
    cell_encoder_chunk_size: int | None = None


@dataclass
class BasePhaseConfig(ConfigBase):
    """Shared configuration fields for both training phases.

    Parameters
    ----------
    weight_decay
        Weight decay.
    epochs
        Max epochs.
    loss_fn
        Loss function name or ``nn.Module``.
        Built-in names: ``"mse"``, ``"poisson"``, ``"huber"``, ``"pearson"``,
        ``"weighted_mse"``, ``"nz_poisson"``, ``"list_mle"``, ``"hybrid_ranking"``.
    gradient_accumulation_steps
        Batches to accumulate gradients.
    clip_grad_norm
        Max gradient norm.
    scheduler
        LR scheduler name. (``"cosine"``, ``"plateau"``, ``"none"``)
    scheduler_patience
        Scheduler patience.
    scheduler_factor
        Scheduler reduction factor.
    scheduler_min_lr
        Scheduler minimum LR.
    genes_per_batch
        Number of genes per batch.
    cells_per_gene
        Cells subsampled per gene (training).
    val_cells_per_gene
        Cells subsampled per gene for validation. ``None`` uses all cells.
    num_workers
        DataLoader workers.
    patience
        Early stopping patience.
    normalize_targets
        Normalize expression targets per gene.
    correlation_loss_lambda
        Weight for Pearson correlation loss.
    base_loss_lambda
        Weight for the base loss term.
    warmup_epochs
        Linear warmup epochs.
    aux_gene_loss_lambda
        Weight for auxiliary per-gene mean expression loss.
    metrics_mode
        ``'full'`` (Pearson + Spearman + per-gene) or ``'fast'`` (loss only).
    gene_repeat_factor
        Repeat each gene this many times per epoch (each with fresh random
        cells). Increases cell coverage without increasing per-step memory.
    gc_frequency_steps
        Run ``gc.collect()`` every N steps. (0 to disable)
    empty_cache_frequency_steps
        Run ``torch.cuda.empty_cache()`` every N steps. (0 to disable)
    checkpoint_metric
        Metric to determine best checkpoint. Either ``"loss"`` or ``"per_gene_pearson"``.
    """
    weight_decay: float = 0.0
    epochs: int = 50
    loss_fn: str | nn.Module = "mse"
    base_loss_lambda: float = 1.0
    gradient_accumulation_steps: int = 4
    clip_grad_norm: float = 1.0
    scheduler: str = "cosine"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    genes_per_batch: int = 1
    cells_per_gene: int = 4096
    num_workers: int = 4
    patience: int = 10
    normalize_targets: bool = False
    correlation_loss_lambda: float = 0.0
    warmup_epochs: int = 0
    aux_gene_loss_lambda: float = 0.0
    val_cells_per_gene: int | None = None
    metrics_mode: str = "full"
    gene_repeat_factor: int = 1
    gc_frequency_steps: int = 0
    empty_cache_frequency_steps: int = 0
    checkpoint_metric: str = "loss"

    def __post_init__(self):
        if self.metrics_mode not in {"full", "fast"}:
            raise ValueError(
                f"metrics_mode must be 'full' or 'fast', got {self.metrics_mode!r}"
            )
        if self.base_loss_lambda < 0:
            raise ValueError("base_loss_lambda must be >= 0")
        if self.correlation_loss_lambda < 0:
            raise ValueError("correlation_loss_lambda must be >= 0")
        if self.base_loss_lambda == 0 and self.correlation_loss_lambda == 0:
            raise ValueError(
                "At least one of base_loss_lambda or correlation_loss_lambda must be > 0."
            )
        if self.gene_repeat_factor < 1:
            raise ValueError("gene_repeat_factor must be >= 1")
        if self.checkpoint_metric not in {"loss", "per_gene_pearson"}:
            raise ValueError(
                f"checkpoint_metric must be 'loss' or 'per_gene_pearson', got {self.checkpoint_metric!r}"
            )
        if self.gc_frequency_steps < 0:
            raise ValueError("gc_frequency_steps must be >= 0")
        if self.empty_cache_frequency_steps < 0:
            raise ValueError("empty_cache_frequency_steps must be >= 0")



@dataclass
class HeadOnlyConfig(BasePhaseConfig):
    """Configuration for head-only training.

    Parameters
    ----------
    learning_rate
        Learning rate.
    precompute_batch_size
        Batch size.
    precomputed_save_path
        Save path.
    """
    learning_rate: float = 1e-3
    precompute_batch_size: int = 4
    precomputed_save_path: str | None = None


@dataclass
class FinetuneConfig(BasePhaseConfig):
    """Configuration for fine-tuning.

    Parameters
    ----------
    trunk_lr
        Trunk learning rate.
    head_lr
        Head learning rate.
    load_head_only_checkpoint
        Load head weights before fine-tuning.
    freeze_trunk
        Freeze the entire trunk during fine-tuning.
    use_lora
        Enable LoRA (Low-Rank Adaptation) on trunk attention layers.
    lora_rank
        Rank of the low-rank decomposition for LoRA.
    trunk_forward_training
        Whether to run the trunk in ``training=True`` mode during fine-tuning.
        ``None`` chooses a safe default: ``False`` for LoRA, otherwise ``True``
        when the trunk is not frozen.
    """
    trunk_lr: float = 1e-5
    head_lr: float = 1e-4
    load_head_only_checkpoint: bool = True
    freeze_trunk: bool = False
    use_lora: bool = False
    lora_rank: int = 8
    trunk_forward_training: bool | None = None
    epochs: int = 20
    cells_per_gene: int = 1024
    num_workers: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.use_lora and self.freeze_trunk:
            raise ValueError(
                "use_lora=True and freeze_trunk=True are mutually exclusive."
            )


@dataclass
class TrainingPlanConfig(ConfigBase):
    """Training plan describing which phases to run.

    Parameters
    ----------
    run_head_only
        Enable head-only training phase.
    run_finetune
        Enable fine-tuning phase.
    head_only
        Head-only phase config.
    finetune
        Fine-tuning phase config.
    """

    run_head_only: bool = False
    run_finetune: bool = True
    head_only: HeadOnlyConfig = field(default_factory=HeadOnlyConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)

    def __post_init__(self):
        if not self.run_head_only and not self.run_finetune:
            raise ValueError("At least one phase must be enabled in TrainingPlanConfig.")



@dataclass
class ExperimentConfig(ConfigBase):
    """Top-level configuration for a full Laika experiment run."""

    trunk: TrunkConfig
    data: DataModuleConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerInitConfig = field(default_factory=TrainerInitConfig)
    training: TrainingPlanConfig = field(default_factory=TrainingPlanConfig)


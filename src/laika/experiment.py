"""High-level experiment orchestration for end-to-end Laika runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anndata import AnnData

from .cell_encoders import get_cell_encoder
from .config import ExperimentConfig
from .data import SequenceDataModule, SpatialDataModule
from .model import Laika
from .trainer import Trainer
from .trunk import get_trunk_dims, load_borzoi_trunk


@dataclass
class ExperimentResult:
    """Artifacts returned by :func:`run_experiment`.

    Attributes
    ----------
    config
        Full experiment config.
    model
        Trained Laika model.
    trunk
        Keras trunk model.
    trainer
        Trainer instance.
    histories
        Training histories keyed by phase (``"head_only"``, ``"finetune"``).
    data_modules
        Data modules keyed by phase.
    """
    config: ExperimentConfig
    model: Laika
    trunk: Any
    trainer: Trainer
    histories: dict[str, dict[str, list]]
    data_modules: dict[str, Any]


def run_experiment(
    *,
    adata: AnnData,
    genome,
    genes: list[str],
    config: ExperimentConfig,
    precomputed_embeddings: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Run a configured Laika experiment end-to-end.

    Parameters
    ----------
    adata
        AnnData with gene expression and spatial embeddings.
    genome
        CREsted Genome instance.
    genes
        Genes to train on.
    config
        Full experiment config.
    precomputed_embeddings
        Pre-computed trunk embeddings (skips precomputation if provided).

    Returns
    -------
    ExperimentResult
    """
    sequence_dm = None
    spatial_dm = None

    # Determine encoder genes when a cell encoder is requested
    encoder_genes_for_dm: list[str] | None = None
    if config.model.cell_encoder is not None:
        encoder_genes_for_dm = (
            config.data.encoder_genes
            if config.data.encoder_genes is not None
            else list(adata.var_names)
        )

    if config.training.run_finetune:
        sequence_dm = SequenceDataModule.from_config(
            adata=adata,
            genome=genome,
            genes=genes,
            config=config.data,
            encoder_genes=encoder_genes_for_dm,
        )
        sequence_dm.setup()

    if config.training.run_head_only:
        spatial_dm = SpatialDataModule.from_config(
            adata=adata,
            genome=genome,
            genes=genes,
            config=config.data,
            encoder_genes=encoder_genes_for_dm,
        )
        spatial_dm.setup()

    # Resolve spatial_emb_dim: cell encoder output_dim takes priority
    if config.model.cell_encoder is not None:
        spatial_emb_dim = config.model.cell_encoder.output_dim
    elif config.model.spatial_emb_dim is not None:
        spatial_emb_dim = config.model.spatial_emb_dim
    elif sequence_dm is not None:
        spatial_emb_dim = sequence_dm.spatial_emb_dim
    elif spatial_dm is not None:
        spatial_emb_dim = spatial_dm.spatial_emb_dim
    else:
        raise RuntimeError("Could not infer spatial_emb_dim from data modules.")

    trunk = load_borzoi_trunk(
        config.trunk.model_path,
        extraction_layer=config.trunk.extraction_layer,
    )
    trunk_seq_len, trunk_channels = get_trunk_dims(trunk)

    # Build cell encoder when configured
    cell_encoder = None
    if config.model.cell_encoder is not None:
        # Determine n_input_genes from whichever data module was set up
        ref_dm = sequence_dm if sequence_dm is not None else spatial_dm
        if ref_dm is None or ref_dm.n_encoder_genes is None:
            raise RuntimeError(
                "Could not determine n_encoder_genes for the cell encoder. "
                "Ensure at least one data module is set up and encoder_genes is configured."
            )
        n_input_genes = ref_dm.n_encoder_genes
        cell_encoder = get_cell_encoder(
            config.model.cell_encoder.encoder,
            **config.model.cell_encoder.encoder_kwargs,
        )
        cell_encoder.setup(n_input_genes, spatial_emb_dim)

    model = Laika(
        head=config.model.head,
        trunk_seq_len=trunk_seq_len,
        trunk_channels=trunk_channels,
        spatial_emb_dim=spatial_emb_dim,
        pool_factor=config.model.pool_factor,
        cell_encoder=cell_encoder,
        **config.model.head_kwargs,
    )

    trainer = Trainer.from_config(model=model, config=config.trainer)
    trainer.update_run_config({"experiment": config.to_dict()})

    histories: dict[str, dict[str, list]] = {}
    data_modules: dict[str, Any] = {}

    if spatial_dm is not None:
        histories["head_only"] = trainer.run_head_only(
            data_module=spatial_dm,
            config=config.training.head_only,
            trunk=trunk,
            precomputed_embeddings=precomputed_embeddings,
        )
        data_modules["head_only"] = spatial_dm

    if sequence_dm is not None:
        histories["finetune"] = trainer.run_finetune(
            data_module=sequence_dm,
            trunk=trunk,
            config=config.training.finetune,
        )
        data_modules["finetune"] = sequence_dm

    return ExperimentResult(
        config=config,
        model=model,
        trunk=trunk,
        trainer=trainer,
        histories=histories,
        data_modules=data_modules,
    )

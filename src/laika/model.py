"""Laika PyTorch model: pooled trunk + spatial head."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

from .cell_encoders.base import BaseCellEncoder
from .heads import get_head
from .heads.base import SpatialHead


class Laika(nn.Module):
    """Trainable PyTorch model combining trunk pooling and spatial head.

    Parameters
    ----------
    head
        SpatialHead instance or registry name.
    trunk_seq_len
        Sequence length of trunk output.
    trunk_channels
        Channel dimension of trunk output.
    spatial_emb_dim
        Dimensionality of spatial cell embedding.
    pool_factor
        Average pooling factor applied to trunk embeddings.
    cell_encoder
        Optional learnable cell encoder.  When provided, raw expression
        vectors are encoded to cell embeddings on-the-fly, replacing
        precomputed spatial embeddings.  The encoder's ``output_dim``
        must equal ``spatial_emb_dim``.
    **head_kwargs
        Extra keyword arguments for get_head.
    """

    def __init__(
        self,
        head: SpatialHead | str,
        trunk_seq_len: int,
        trunk_channels: int = 1536,
        spatial_emb_dim: int = 64,
        pool_factor: int = 8,
        cell_encoder: BaseCellEncoder | None = None,
        **head_kwargs,
    ):
        super().__init__()

        if pool_factor < 1:
            raise ValueError("pool_factor must be >= 1")
        if pool_factor > 1 and trunk_seq_len % pool_factor != 0:
            logger.warning(
                f"trunk_seq_len ({trunk_seq_len}) is not divisible by "
                f"pool_factor ({pool_factor}). The last "
                f"{trunk_seq_len % pool_factor} positions will be discarded."
            )

        self._pool_factor = pool_factor
        self._trunk_seq_len = trunk_seq_len
        self._trunk_channels = trunk_channels
        self._spatial_emb_dim = spatial_emb_dim
        self._pooled_seq_len = trunk_seq_len // pool_factor

        if pool_factor > 1:
            self.pool = nn.AvgPool1d(kernel_size=pool_factor)
            logger.info(
                f"Trunk pooling: {trunk_seq_len} -> {self._pooled_seq_len} "
                f"(pool_factor={pool_factor})"
            )
        else:
            self.pool = nn.Identity()

        if isinstance(head, str):
            head_instance = get_head(head, **head_kwargs)
        else:
            head_instance = head

        self._head_name = head_instance.name
        head_instance.setup(
            input_seq_len=self._pooled_seq_len,
            trunk_channels=trunk_channels,
            spatial_emb_dim=spatial_emb_dim,
        )
        self.head = head_instance

        # Optional learnable cell encoder
        if cell_encoder is not None:
            if cell_encoder.output_dim is not None and cell_encoder.output_dim != spatial_emb_dim:
                raise ValueError(
                    f"cell_encoder.output_dim ({cell_encoder.output_dim}) must match "
                    f"spatial_emb_dim ({spatial_emb_dim}). Call encoder.setup() with the "
                    "correct output_dim before passing it to Laika."
                )
        self.cell_encoder: BaseCellEncoder | None = cell_encoder

        n_params = sum(p.numel() for p in self.parameters())
        encoder_info = f", encoder={cell_encoder.name}" if cell_encoder is not None else ""
        logger.info(f"Laika model built: {n_params:,} parameters (head={self._head_name}{encoder_info})")

    @property
    def pool_factor(self) -> int:
        """Pooling factor applied to trunk sequence."""
        return self._pool_factor

    @property
    def pooled_seq_len(self) -> int:
        """Sequence length after pooling."""
        return self._pooled_seq_len

    @property
    def trunk_channels(self) -> int:
        """Trunk output channel dimension."""
        return self._trunk_channels

    @property
    def trunk_seq_len(self) -> int:
        """Raw trunk output sequence length (before pooling)."""
        return self._trunk_seq_len

    @property
    def spatial_emb_dim(self) -> int:
        """Expected spatial cell embedding dimension."""
        return self._spatial_emb_dim

    def forward(
        self,
        trunk_emb: torch.Tensor,
        cell_embs: torch.Tensor,
        expression_vectors: torch.Tensor | None = None,
        gene_encoder_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        trunk_emb
            ``(B, seq_len, channels)`` trunk embeddings.
        cell_embs
            ``(B, N, emb_dim)`` spatial cell embeddings.  Used directly
            when no cell encoder is present; otherwise serves as a
            fallback if ``expression_vectors`` is ``None``.
        expression_vectors
            ``(B, N, n_encoder_genes)`` raw expression vectors.  Required
            when a cell encoder is attached and should be used.
        gene_encoder_idx
            ``(B,)`` index of the target gene in the encoder vocabulary,
            used for masking.  ``-1`` means no masking.

        Returns
        -------
        torch.Tensor
            ``(B, N)`` predictions.
        """
        if trunk_emb.shape[2] != self._trunk_channels:
            raise ValueError(
                f"trunk_emb has {trunk_emb.shape[2]} channels, "
                f"expected {self._trunk_channels}. "
                "Check that trunk and model were built with the same trunk_channels."
            )

        # Encode expression vectors when a cell encoder is present
        if self.cell_encoder is not None and expression_vectors is not None:
            cell_embs = self.cell_encoder(expression_vectors, gene_encoder_idx)

        if cell_embs.shape[2] != self._spatial_emb_dim:
            raise ValueError(
                f"cell_embs has {cell_embs.shape[2]} dimensions, "
                f"expected {self._spatial_emb_dim}. "
                "Check that spatial_emb_dim matches your spatial embeddings."
            )

        # Pool trunk only if not already pre-pooled (e.g. precomputed embeddings)
        seq_len = trunk_emb.shape[1]
        if seq_len == self._trunk_seq_len:
            # Raw trunk output — apply pooling
            x = trunk_emb.transpose(1, 2)  # (B, C, L)
            x = self.pool(x)               # (B, C, L_pooled)
            trunk_emb = x.transpose(1, 2)  # (B, L_pooled, C)
        elif seq_len != self._pooled_seq_len:
            raise ValueError(
                f"trunk_emb seq_len {seq_len} doesn't match expected "
                f"{self._trunk_seq_len} (raw) or "
                f"{self._pooled_seq_len} (pre-pooled)."
            )

        return self.head(trunk_emb, cell_embs)

    def save(self, path: str | Path) -> None:
        """Save model state dict with embedded config.

        Parameters
        ----------
        path
            Output path (parent directories created automatically).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cell_encoder_config = None
        if self.cell_encoder is not None:
            cell_encoder_config = {
                "name": self.cell_encoder.name,
                "kwargs": self.cell_encoder.config,
                "output_dim": self.cell_encoder.output_dim,
                "n_input_genes": getattr(self.cell_encoder, "_n_input_genes", None),
            }
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "head": self._head_name,
                "head_config": self.head.config,
                "trunk_seq_len": self._trunk_seq_len,
                "trunk_channels": self._trunk_channels,
                "spatial_emb_dim": self._spatial_emb_dim,
                "pool_factor": self._pool_factor,
                "cell_encoder_config": cell_encoder_config,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path, map_location: str = "cpu") -> None:
        """Load model state dict from a checkpoint.

        Parameters
        ----------
        path
            Checkpoint path.
        map_location
            Device string passed to ``torch.load``.
        """
        data = torch.load(path, map_location=map_location, weights_only=True)
        if isinstance(data, dict) and "state_dict" in data:
            state_dict = data["state_dict"]
        else:            
            raise ValueError(f"{path} does not contain a state_dict")
        self.load_state_dict(state_dict)
        logger.info(f"Model loaded from {path}")

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location: str = "cpu") -> "Laika":
        """Reconstruct a Laika model from a checkpoint.

        Parameters
        ----------
        path
            Checkpoint path saved by :meth:`save`.
        map_location
            Device string passed to ``torch.load``.
        """
        data = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(data, dict) or "config" not in data:
            raise ValueError(f"{path} has no config")
        cfg = data["config"]

        # Reconstruct cell encoder if present in checkpoint (graceful for old checkpoints)
        cell_encoder = None
        enc_cfg = cfg.get("cell_encoder_config")
        if enc_cfg is not None:
            from .cell_encoders import get_cell_encoder
            cell_encoder = get_cell_encoder(enc_cfg["name"], **enc_cfg["kwargs"])
            if enc_cfg.get("n_input_genes") is not None:
                cell_encoder.setup(enc_cfg["n_input_genes"], enc_cfg["output_dim"])

        model = cls(
            head=cfg["head"],
            trunk_seq_len=cfg["trunk_seq_len"],
            trunk_channels=cfg["trunk_channels"],
            spatial_emb_dim=cfg["spatial_emb_dim"],
            pool_factor=cfg["pool_factor"],
            cell_encoder=cell_encoder,
            **cfg["head_config"],
        )
        model.load_state_dict(data["state_dict"])
        logger.info(f"Model reconstructed from checkpoint {path}")
        return model

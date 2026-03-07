"""Gene DNA sequence cache with optional in-memory and one-hot encoding."""

from __future__ import annotations

import numpy as np
from loguru import logger

from crested._genome import Genome
from crested.utils import one_hot_encode_sequence


class GeneSequenceCache:
    """Efficiently stores one DNA sequence per gene.

    Parameters
    ----------
    genome
        Genome instance.
    gene_to_region
        Mapping from gene name to region string.
    in_memory
        Preload all sequences.
    max_stochastic_shift
        Extra bases for shifting.
    cache_onehot
        Cache one-hot array if ``in_memory`` is True.
    """

    def __init__(
        self,
        genome: Genome,
        gene_to_region: dict[str, str],
        in_memory: bool = True,
        max_stochastic_shift: int = 0,
        cache_onehot: bool = False,
    ):
        self.genome = genome
        self.gene_to_region = gene_to_region
        self.in_memory = in_memory
        self.max_stochastic_shift = max_stochastic_shift
        self.cache_onehot = cache_onehot and in_memory
        self._cache: dict[str, tuple[str, int]] = {}  # gene -> (seq, actual_left_ext)
        self._onehot_cache: dict[str, np.ndarray] = {}
        self._parsed_regions: dict[str, tuple[str, int, int, str, int]] = {
            gene: self._parse_region(region)
            for gene, region in self.gene_to_region.items()
        }

        if in_memory:
            self._preload()

    def _preload(self) -> None:
        """Load all (extended) sequences into memory."""
        logger.info(f"Preloading {len(self.gene_to_region)} gene sequences into memory...")
        for gene in self.gene_to_region:
            seq, left_ext = self._fetch_extended(self._parsed_regions[gene])
            self._cache[gene] = (seq, left_ext)
            if self.cache_onehot:
                self._onehot_cache[gene] = one_hot_encode_sequence(
                    seq, expand_dim=False
                )
        logger.info("Sequence preloading complete.")

    @staticmethod
    def _parse_region(region: str) -> tuple[str, int, int, str, int]:
        chrom, start_end, strand = region.split(":")
        start, end = map(int, start_end.split("-"))
        return chrom, start, end, strand, end - start

    def _fetch_extended(
        self, parsed_region: tuple[str, int, int, str, int]
    ) -> tuple[str, int]:
        """Fetch a sequence extended by ``max_stochastic_shift`` on each side.

        Returns
        -------
        tuple of (sequence_string, actual_left_extension)
        """
        chrom, start, end, _, _ = parsed_region
        ext_start = max(0, start - self.max_stochastic_shift)
        ext_end = end + self.max_stochastic_shift

        # Clamp to chromosome size
        chrom_size = self.genome.fasta.get_reference_length(chrom)
        ext_end = min(ext_end, chrom_size)

        actual_left_ext = start - ext_start
        seq = self.genome.fasta.fetch(reference=chrom, start=ext_start, end=ext_end).upper()
        return seq, actual_left_ext

    def _get_seq_len(self, gene: str) -> int:
        """Return sequence length."""
        return self._parsed_regions[gene][4]

    @staticmethod
    def _clamp_shift(actual_left_ext: int, extended_len: int, seq_len: int, shift: int) -> int:
        """Return clamped start index for a shifted window."""
        start_idx = actual_left_ext + shift
        return max(0, min(start_idx, extended_len - seq_len))

    def get_sequence(self, gene: str, shift: int = 0) -> str:
        """Return the DNA string.

        Parameters
        ----------
        gene
            Gene name.
        shift
            Base-pair shift

        Returns
        -------
        str
            DNA string.
        """
        if self.in_memory:
            extended, actual_left_ext = self._cache[gene]
        else:
            extended, actual_left_ext = self._fetch_extended(self._parsed_regions[gene])

        seq_len = self._get_seq_len(gene)
        start_idx = self._clamp_shift(actual_left_ext, len(extended), seq_len, shift)
        sub = extended[start_idx: start_idx + seq_len]

        # Pad with 'N' (unknown base, one-hot encodes to [0,0,0,0])
        if len(sub) < seq_len:
            logger.warning(
                f"Gene '{gene}' sequence too short after shift={shift} "
                f"({len(sub)} < {seq_len}). Padding with 'N'."
            )
            sub = sub.ljust(seq_len, "N")

        return sub

    def get_onehot(self, gene: str, shift: int = 0) -> np.ndarray:
        """Return one-hot encoded DNA.

        Parameters
        ----------
        gene
            Gene name.
        shift
            Base-pair shift.

        Returns
        -------
        np.ndarray
            One-hot DNA ``(seq_len, 4)``.
        """
        if self.cache_onehot and gene in self._onehot_cache:
            seq_len = self._get_seq_len(gene)
            actual_left_ext = self._cache[gene][1]
            extended_oh = self._onehot_cache[gene]
            start_idx = self._clamp_shift(actual_left_ext, len(extended_oh), seq_len, shift)
            oh = extended_oh[start_idx: start_idx + seq_len]
            if oh.shape[0] < seq_len:
                pad = np.zeros((seq_len - oh.shape[0], 4), dtype=np.float32)
                oh = np.concatenate([oh, pad], axis=0)
            return oh
        else:
            dna_str = self.get_sequence(gene, shift=shift)
            return one_hot_encode_sequence(dna_str, expand_dim=False)

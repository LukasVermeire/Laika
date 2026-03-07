"""GTF parsing and gene-centred sequence window computation."""

from __future__ import annotations

from pathlib import Path

from crested._genome import Genome
from loguru import logger


def _parse_gtf_gene_coords(gtf_path: str | Path) -> dict[str, tuple[str, int, int, str]]:
    """Parse a GTF file and return gene_name -> (chrom, start, end, strand).

    Parameters
    ----------
    gtf_path
        Path to GTF file.

    Returns
    -------
    dict
        Gene coordinates mapping.
    """
    gene_coords: dict[str, tuple[str, int, int, str]] = {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            # Extract gene_name from the attributes column
            attrs = fields[8]
            gene_name_parts = [x for x in attrs.split(";") if "gene_name" in x]
            if not gene_name_parts:
                continue
            gene_name = gene_name_parts[0].split('"')[1]
            if gene_name not in gene_coords:
                gene_coords[gene_name] = (chrom, start, end, strand)
    return gene_coords


def _compute_gene_regions(
    genes: list[str],
    gene_coords: dict[str, tuple[str, int, int, str]],
    genome: Genome,
    seq_length: int = 524_288,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Compute centred sequence windows for a list of genes.

    Parameters
    ----------
    genes
        Gene names.
    gene_coords
        Mapping from gene name to coordinates.
    genome
        Genome instance.
    seq_length
        Total sequence length.

    Returns
    -------
    valid_genes
        List of genes that could be mapped to valid windows.
    skipped_genes
        List of genes that were skipped (not in GTF or near chrom boundary).
    gene_to_region
        Mapping from gene name to region string ``chr:start-end:strand``.
    """
    half = seq_length // 2
    valid_genes: list[str] = []
    skipped_genes: list[str] = []
    gene_to_region: dict[str, str] = {}

    for gene in genes:
        if gene not in gene_coords:
            logger.warning(f"Gene '{gene}' not found in GTF annotation – skipping.")
            skipped_genes.append(gene)
            continue

        chrom, g_start, g_end, strand = gene_coords[gene]
        gene_center = (g_start + g_end) // 2
        seq_start = gene_center - half
        seq_end = seq_start + seq_length

        # Boundary checks
        if seq_start < 0:
            logger.warning(
                f"Gene '{gene}' too close to chromosome start "
                f"(seq_start={seq_start}) – skipping."
            )
            skipped_genes.append(gene)
            continue

        try:
            chrom_length = genome.fasta.get_reference_length(chrom)
            if seq_end > chrom_length:
                logger.warning(
                    f"Gene '{gene}' too close to chromosome end "
                    f"(seq_end={seq_end}, chrom_length={chrom_length}) – skipping."
                )
                skipped_genes.append(gene)
                continue
        except Exception as exc:
            logger.warning(
                f"Could not validate chromosome length for gene '{gene}': {exc} – skipping."
            )
            skipped_genes.append(gene)
            continue

        region = f"{chrom}:{seq_start}-{seq_end}:{strand}"
        valid_genes.append(gene)
        gene_to_region[gene] = region

    return valid_genes, skipped_genes, gene_to_region

#!/usr/bin/env python
"""Plot empirical codon frequencies separately for each gene.

The input CSV must have the wide format produced by TOMBOMBADIL's
``--codon-frequencies`` option. Each output plot contains one gene's bars and
the per-codon median +/- one standard deviation calculated from the other
genes.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from tombombadil.__main__ import CODON_LIST


def load_codon_frequency_csv(file_name):
    """Load named codon frequencies from a wide codon-frequency CSV."""
    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Codon-frequency CSV is empty: {file_name}") from exc

        expected_header = ["gene", *CODON_LIST]
        if header != expected_header:
            raise ValueError(
                "Codon-frequency CSV must have header: "
                + ",".join(expected_header)
            )

        gene_names = []
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(expected_header):
                raise ValueError(
                    f"{file_name} line {line_number} has {len(row)} fields; "
                    f"expected {len(expected_header)}"
                )
            gene_names.append(row[0])
            try:
                rows.append([float(value) for value in row[1:]])
            except ValueError as exc:
                raise ValueError(
                    f"{file_name} line {line_number} contains a non-numeric frequency"
                ) from exc

    if not rows:
        raise ValueError(f"Codon-frequency CSV contains no gene rows: {file_name}")

    frequencies = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Codon frequencies must be finite")
    if np.any(frequencies < 0):
        raise ValueError("Codon frequencies must be non-negative")
    return gene_names, frequencies


def plot_codon_frequencies(gene_names, frequencies, figsize=None):
    """Create one codon-frequency bar plot per gene.

    The median and standard deviation are calculated independently for every
    codon using the other genes as the comparison set. If only one gene is
    supplied, that gene is used as its own comparison set and the SD is zero.

    Returns a list of Matplotlib figures in the same order as ``gene_names``.
    """
    gene_names = list(gene_names)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.ndim != 2 or frequencies.shape[1] != len(CODON_LIST):
        raise ValueError(
            f"Expected frequencies with shape (n_genes, {len(CODON_LIST)}), "
            f"got {frequencies.shape}"
        )
    if len(gene_names) != frequencies.shape[0]:
        raise ValueError("Number of gene names must match frequency rows")
    if frequencies.shape[0] == 0:
        raise ValueError("At least one gene is required")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
        raise ValueError("Codon frequencies must be finite and non-negative")

    x = np.arange(len(CODON_LIST))
    figure_size = figsize or (max(14, 0.35 * len(CODON_LIST)), 7)
    figures = []

    for index, gene_name in enumerate(gene_names):
        comparison = (
            np.delete(frequencies, index, axis=0)
            if frequencies.shape[0] > 1
            else frequencies
        )
        median = np.median(comparison, axis=0)
        standard_deviation = np.std(comparison, axis=0)

        fig, ax = plt.subplots(figsize=figure_size)
        ax.bar(
            x,
            frequencies[index],
            width=0.8,
            color="steelblue",
            alpha=0.8,
            label=gene_name,
        )
        ax.errorbar(
            x,
            median,
            yerr=standard_deviation,
            fmt="D",
            color="black",
            markersize=4,
            capsize=2,
            linewidth=1,
            label="Other genes: median ± SD" if len(gene_names) > 1 else "Median ± SD",
            zorder=5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(CODON_LIST, rotation=90, fontsize=7)
        ax.set_ylabel("Empirical codon frequency")
        ax.set_xlabel("Codon")
        ax.set_title(f"Empirical codon frequencies: {gene_name}")
        ax.set_xlim(-0.6, len(CODON_LIST) - 0.4)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        figures.append(fig)

    return figures


def save_codon_frequency_plots(gene_names, frequencies, output_dir, figsize=None):
    """Save one codon-frequency plot per gene, closing each figure immediately."""
    gene_names = list(gene_names)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.ndim != 2 or frequencies.shape[1] != len(CODON_LIST):
        raise ValueError(
            f"Expected frequencies with shape (n_genes, {len(CODON_LIST)}), "
            f"got {frequencies.shape}"
        )
    if len(gene_names) != frequencies.shape[0]:
        raise ValueError("Number of gene names must match frequency rows")
    if frequencies.shape[0] == 0:
        raise ValueError("At least one gene is required")

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    for index, (gene_name, frequency_row) in enumerate(
        zip(gene_names, frequencies), start=1
    ):
        # Create only this gene's figure, rather than keeping all figures open.
        fig = plot_codon_frequencies(
            [gene_name],
            frequency_row[np.newaxis, :],
            figsize=figsize,
        )[0]
        # Replace the single-gene summary with the comparison summary when
        # there are other genes available.
        if len(gene_names) > 1:
            plt.close(fig)
            comparison = np.delete(frequencies, index - 1, axis=0)
            median = np.median(comparison, axis=0)
            standard_deviation = np.std(comparison, axis=0)
            x = np.arange(len(CODON_LIST))
            fig, ax = plt.subplots(figsize=figsize or (max(14, 0.35 * len(CODON_LIST)), 7))
            ax.bar(x, frequency_row, width=0.8, color="steelblue", alpha=0.8,
                   label=gene_name)
            ax.errorbar(x, median, yerr=standard_deviation, fmt="D", color="black",
                        markersize=4, capsize=2, linewidth=1,
                        label="Other genes: median ± SD", zorder=5)
            ax.set_xticks(x)
            ax.set_xticklabels(CODON_LIST, rotation=90, fontsize=7)
            ax.set_ylabel("Empirical codon frequency")
            ax.set_xlabel("Codon")
            ax.set_title(f"Empirical codon frequencies: {gene_name}")
            ax.set_xlim(-0.6, len(CODON_LIST) - 0.4)
            ax.legend(fontsize=8, loc="upper right")
            fig.tight_layout()

        safe_gene_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in gene_name
        ) or "gene"
        output_path = os.path.join(
            output_dir,
            f"{index:03d}_{safe_gene_name}_codon_frequencies.png",
        )
        try:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        finally:
            plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot empirical codon frequencies separately for each gene."
    )
    parser.add_argument("--input", required=True,
                        help="Codon-frequency CSV produced by TOMBOMBADIL")
    parser.add_argument("--output", default="codon_frequency_plots",
                        help="Output directory for per-gene PNGs "
                             "(default: codon_frequency_plots)")
    args = parser.parse_args()

    gene_names, frequencies = load_codon_frequency_csv(args.input)
    os.makedirs(args.output, exist_ok=True)
    output_paths = save_codon_frequency_plots(gene_names, frequencies, args.output)
    for output_path in output_paths:
        print(f"Wrote codon-frequency plot: {output_path}")


if __name__ == "__main__":
    main()

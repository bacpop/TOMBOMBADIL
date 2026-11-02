"""FASTA input and alignment-derived codon data helpers."""

import gzip
import numpy as np


# Map the A/C/G/T byte encoding used while reading sequences into model order.
CODON_INDEX_ORDER = np.array([
    63, 61, 60, 62, 55, 53, 52, 54, 51, 49, 59, 57, 58, 31, 29, 28, 30,
    23, 21, 20, 22, 19, 17, 16, 18, 27, 25, 24, 26, 15, 13, 12, 14, 7,
    5, 4, 6, 3, 1, 0, 2, 11, 9, 8, 10, 47, 45, 44, 46, 39, 37, 36, 38,
    35, 33, 32, 34, 43, 41, 40, 42, 48, 50, 56, 64,
])
BASE_ORDER = ("T", "C", "A", "G")
STOP_CODONS = {"TAA", "TAG", "TGA"}
CODON_LIST = tuple(
    codon
    for codon in (
        first + second + third
        for first in BASE_ORDER
        for second in BASE_ORDER
        for third in BASE_ORDER
    )
    if codon not in STOP_CODONS
)
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASE_ORDER)}


def _open_fasta(path):
    """Open plain or gzip-compressed FASTA text based on its file signature."""
    with open(path, "rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if is_gzip else open(path, "rt")


def read_fasta(handle):
    """Yield ``(header, sequence)`` records from an open FASTA text handle."""
    name, sequence = None, []
    for line in handle:
        line = line.strip()
        if line.startswith(">"):
            if name:
                yield name, "".join(sequence)
            name, sequence = line[1:], []
        else:
            sequence.append(line)
    if name:
        yield name, "".join(sequence)


def read_alignment(path):
    """Read all named FASTA records from a plain or gzip alignment file."""
    with _open_fasta(path) as handle:
        return list(read_fasta(handle))


def count_codons(file_name):
    """Count unambiguous sense codons at each alignment site."""
    n_samples = 0
    with _open_fasta(file_name) as fasta:
        counts = None
        for _header, sequence in read_fasta(fasta):
            n_samples += 1
            bases = np.frombuffer(sequence.lower().encode(), dtype=np.int8)
            if counts is None:
                counts = np.zeros((65, bases.shape[0] // 3), dtype=np.int32)

            ambiguous = (bases != 97) & (bases != 99) & (bases != 103) & (bases != 116)
            bases = np.copy(bases)
            bases[ambiguous] = 64
            codon_bases = bases.reshape(-1, 3).copy()
            codon_bases[codon_bases == 97] = 0  # A
            codon_bases[codon_bases == 99] = 1  # C
            codon_bases[codon_bases == 103] = 2  # G
            codon_bases[codon_bases == 116] = 3  # T
            codon_bases[:, 1] = np.left_shift(codon_bases[:, 1], 2)
            codon_bases[:, 0] = np.left_shift(codon_bases[:, 0], 4)
            codon_map = np.fmin(np.sum(codon_bases, axis=1), 64)
            for site, codon in enumerate(codon_map):
                counts[codon, site] += 1

    # Reorder to the model's codon order and remove stops and ambiguous bases.
    counts = counts[CODON_INDEX_ORDER]
    counts = counts[:61, :]
    return counts, n_samples


def estimate_pi_from_counts(counts, pseudocount):
    counts = np.asarray(counts)
    if counts.shape[0] != 61:
        raise ValueError(f"Expected codon count matrix with 61 rows, got {counts.shape[0]}")
    if pseudocount < 0:
        raise ValueError("--pi-pseudocount must be non-negative")

    observed_counts = counts.sum(axis=1, dtype=np.float64)
    if observed_counts.sum() <= 0:
        raise ValueError("Cannot estimate pi: no non-stop codons were observed in the alignment")

    smoothed_counts = observed_counts + pseudocount
    total = smoothed_counts.sum()
    if total <= 0:
        raise ValueError("Cannot estimate pi: smoothed codon counts sum to zero")

    pi = smoothed_counts / total
    if pi.shape != (61,) or not np.all(np.isfinite(pi)) or np.any(pi <= 0):
        raise ValueError(
            "Estimated pi must contain 61 finite, strictly positive non-stop codon frequencies"
        )
    return pi


def estimate_f3x4_frequencies_from_counts(counts, pseudocount):
    counts = np.asarray(counts)
    if counts.shape[0] != 61:
        raise ValueError(f"Expected codon count matrix with 61 rows, got {counts.shape[0]}")
    if pseudocount < 0:
        raise ValueError("--pi-pseudocount must be non-negative")

    observed_counts = counts.sum(axis=1, dtype=np.float64)
    if observed_counts.sum() <= 0:
        raise ValueError("Cannot estimate F3x4 pi: no non-stop codons were observed in the alignment")

    nucleotide_counts = np.full((3, 4), pseudocount, dtype=np.float64)
    for codon, count in zip(CODON_LIST, observed_counts):
        for position, base in enumerate(codon):
            nucleotide_counts[position, BASE_TO_INDEX[base]] += count

    row_totals = nucleotide_counts.sum(axis=1, keepdims=True)
    if np.any(row_totals <= 0):
        raise ValueError("Cannot estimate F3x4 pi: smoothed nucleotide counts sum to zero")

    frequencies = nucleotide_counts / row_totals
    if frequencies.shape != (3, 4) or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError(
            "Estimated F3x4 nucleotide frequencies must be a finite, strictly positive 3x4 matrix"
        )
    return frequencies


def estimate_f3x4_pi_from_counts(counts, pseudocount):
    frequencies = estimate_f3x4_frequencies_from_counts(counts, pseudocount)
    pi = np.array(
        [
            frequencies[0, BASE_TO_INDEX[codon[0]]]
            * frequencies[1, BASE_TO_INDEX[codon[1]]]
            * frequencies[2, BASE_TO_INDEX[codon[2]]]
            for codon in CODON_LIST
        ],
        dtype=np.float64,
    )
    total = pi.sum()
    if total <= 0:
        raise ValueError("Cannot estimate F3x4 pi: codon frequencies sum to zero")

    pi = pi / total
    if pi.shape != (61,) or not np.all(np.isfinite(pi)) or np.any(pi <= 0):
        raise ValueError("Estimated F3x4 pi must contain 61 finite, strictly positive codon frequencies")
    return pi

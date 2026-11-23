import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tombombadil.alignment import count_codons, read_alignment, read_fasta
from tombombadil.domains import parse_domain_labels


class TestFastaReading(unittest.TestCase):
    def test_read_fasta_handles_wrapped_sequences_and_blank_lines(self):
        records = list(read_fasta(io.StringIO(
            ">sample one\nAAA\nCCC\n\n>sample two\nGGG\nTTT\n"
        )))

        self.assertEqual(records, [("sample one", "AAACCC"), ("sample two", "GGGTTT")])

    def test_read_alignment_reads_plain_and_gzip_records(self):
        contents = ">one\nTTT\nCCC\n>two\nATG\nGGG\n"
        with tempfile.TemporaryDirectory() as tmp:
            plain_path = Path(tmp) / "alignment.fasta"
            gzip_path = Path(tmp) / "alignment.fasta.gz"
            plain_path.write_text(contents)
            with gzip.open(gzip_path, "wt") as handle:
                handle.write(contents)

            expected = [("one", "TTTCCC"), ("two", "ATGGGG")]
            self.assertEqual(read_alignment(plain_path), expected)
            self.assertEqual(read_alignment(gzip_path), expected)


class TestCodonCounting(unittest.TestCase):
    def test_count_codons_matches_across_plain_and_gzip_with_ambiguous_sites(self):
        contents = ">one\nNNNTTTTAA\n>two\nNNNTTCTGA\n"
        with tempfile.TemporaryDirectory() as tmp:
            plain_path = Path(tmp) / "counts.fasta"
            gzip_path = Path(tmp) / "counts.fasta.gz"
            plain_path.write_text(contents)
            with gzip.open(gzip_path, "wt") as handle:
                handle.write(contents)

            observed, samples = count_codons(plain_path)
            compressed, compressed_samples = count_codons(gzip_path)

        np.testing.assert_array_equal(observed, compressed)
        self.assertEqual((samples, compressed_samples), (2, 2))
        self.assertEqual(observed.shape, (61, 3))
        self.assertEqual(observed[:, 0].sum(), 0)  # ambiguous codons are excluded
        self.assertEqual(observed[:, 1].sum(), 2)
        self.assertEqual(observed[:, 2].sum(), 0)  # stop codons are excluded


class TestDomainLabels(unittest.TestCase):
    def test_domain_parser_uses_shared_reader_for_alignment_and_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            alignment_path = Path(tmp) / "alignment.fasta.gz"
            reference_path = Path(tmp) / "reference.fasta"
            domains_path = Path(tmp) / "domains.json"
            with gzip.open(alignment_path, "wt") as handle:
                handle.write(">proxy\nAAA---\nCCC\n>other\nAAGTTTCCA\n")
            reference_path.write_text(">part one\nA\n>part two\nA\n")
            domains_path.write_text(json.dumps({
                "features": [{
                    "description": "extracellular",
                    "location": {"start": {"value": 2}, "end": {"value": 2}},
                }]
            }))

            labels = parse_domain_labels(
                domains_path, alignment_path, reference_path, n_sites=3
            )

        np.testing.assert_array_equal(labels, ["other", "other", "extracellular"])


if __name__ == "__main__":
    unittest.main()

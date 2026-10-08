import csv
import os
import tempfile
import unittest

import numpy as np

from plot_codon_frequencies import (
    CODON_LIST,
    load_codon_frequency_csv,
    plot_codon_frequencies,
    save_codon_frequency_plots,
)


class TestCodonFrequencyPlot(unittest.TestCase):
    def test_loads_frequency_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "frequencies.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["gene", *CODON_LIST])
                writer.writerow(["gene_a", *([1 / 61] * 61)])
                writer.writerow(["gene_b", *([2 / 61] * 61)])

            names, frequencies = load_codon_frequency_csv(path)

        self.assertEqual(["gene_a", "gene_b"], names)
        self.assertEqual((2, 61), frequencies.shape)
        np.testing.assert_allclose(frequencies[0], 1 / 61)

    def test_plot_creates_separate_gene_figures_with_summary_errorbars(self):
        frequencies = np.vstack([
            np.full(61, 0.01),
            np.full(61, 0.02),
        ])
        figures = plot_codon_frequencies(["gene_a", "gene_b"], frequencies)
        try:
            self.assertEqual(2, len(figures))
            for fig in figures:
                ax = fig.axes[0]
                self.assertEqual(61, len(ax.patches))
                self.assertGreaterEqual(len(ax.lines), 1)
            self.assertIn("gene_a", figures[0].axes[0].get_title())
            self.assertIn("gene_b", figures[1].axes[0].get_title())
        finally:
            import matplotlib.pyplot as plt
            for fig in figures:
                plt.close(fig)

    def test_save_closes_each_figure(self):
        import matplotlib.pyplot as plt

        frequencies = np.vstack([
            np.full(61, 0.01),
            np.full(61, 0.02),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_codon_frequency_plots(
                ["gene_a", "gene_b"], frequencies, tmp
            )
            self.assertEqual(2, len(paths))
            self.assertTrue(all(os.path.exists(path) for path in paths))
        self.assertEqual([], plt.get_fignums())


if __name__ == "__main__":
    unittest.main()

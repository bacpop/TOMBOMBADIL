import os
import tempfile
import unittest

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from tombombadil.sample import _optimize_params
from tombombadil.sample import likelihood_plot_path
from tombombadil.sample import plot_likelihood_history


class TestMapProgressHistory(unittest.TestCase):
    def test_fixed_step_run_records_checkpoints_and_reports_each_iteration(self):
        params = {"x": jnp.array(1.0, dtype=jnp.float64)}
        solver = optax.sgd(0.0)
        fn = lambda values: -jnp.square(values["x"])
        progress = []

        result = _optimize_params(
            fn,
            params,
            solver,
            n_iter=25,
            verbose=False,
            progress_callback=lambda step, total, objective: progress.append(
                (step, total, objective)
            ),
        )

        self.assertEqual(
            [step for step, _ in result["objective_history"]], [0, 10, 20, 25]
        )
        self.assertEqual([step for step, _, _ in progress], list(range(26)))
        objective_updates = [
            step for step, _, objective in progress if objective is not None
        ]
        self.assertEqual(objective_updates, [0, 10, 20, 25])
        self.assertAlmostEqual(result["objective"], -1.0)

    def test_likelihood_plot_labels_axis_and_highlights_best_replicate(self):
        metadata = [
            {"objective_history": [(0, -4.0), (10, -2.0)]},
            {"objective_history": [(0, -3.0), (10, -1.0)]},
        ]
        fig, ax = plot_likelihood_history(metadata, best_idx=1)
        try:
            self.assertEqual(ax.get_ylabel(), "log-likelihood")
            self.assertEqual(ax.get_xlabel(), "iteration")
            self.assertEqual(ax.lines[1].get_label(), "Replicate 2 (best)")
            self.assertGreater(ax.lines[1].get_linewidth(), ax.lines[0].get_linewidth())
        finally:
            plt.close(fig)

    def test_default_and_stem_based_plot_paths(self):
        self.assertEqual(
            likelihood_plot_path(None, "scalar"), "scalar_likelihood_plot.pdf"
        )
        self.assertEqual(
            likelihood_plot_path(None, "per-site"), "per_site_likelihood_plot.pdf"
        )
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "fit")
            self.assertEqual(
                likelihood_plot_path(stem, "per-site"),
                os.path.join(tmp, "per_site_fit_likelihood_plot.pdf"),
            )


if __name__ == "__main__":
    unittest.main()

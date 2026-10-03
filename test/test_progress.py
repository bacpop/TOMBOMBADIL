import os
import tempfile
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from tombombadil import sample
from tombombadil.__main__ import main
from tombombadil.sample import _optimize_params
from tombombadil.sample import _run_replicates
from tombombadil.sample import likelihood_plot_path
from tombombadil.sample import plot_likelihood_history
from tombombadil.sample import plot_per_site_omega
from tombombadil.sample import softplus_inverse


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
            [step for step, _ in result["objective_history"]], list(range(26))
        )
        self.assertEqual([step for step, _, _ in progress], list(range(26)))
        objective_updates = [
            step for step, _, objective in progress if objective is not None
        ]
        self.assertEqual(objective_updates, list(range(26)))
        self.assertAlmostEqual(result["objective"], -1.0)

    def test_redirected_progress_logs_every_objective_and_startup_timing(self):
        params = {"x": jnp.array(1.0, dtype=jnp.float64)}
        fn = lambda values: -jnp.square(values["x"])

        with patch("tombombadil.sample.sys.stderr") as stderr:
            stderr.isatty.return_value = False
            with self.assertLogs(level="INFO") as captured:
                _run_replicates(
                    fn,
                    params,
                    {"x": "scalar"},
                    n_reps=1,
                    n_iter=3,
                    progress=True,
                )

        objective_logs = [
            line for line in captured.output
            if "MAP replicate 1/1, step " in line and "log-likelihood =" in line
        ]
        self.assertEqual(len(objective_logs), 4)
        self.assertIn("step 0/3", objective_logs[0])
        self.assertIn("step 3/3", objective_logs[-1])
        self.assertTrue(any("MAP startup" in line for line in captured.output))

    def test_likelihood_plot_labels_axis_and_highlights_best_replicate(self):
        metadata = [
            {"objective_history": [(0, -4.0), (1, -3.0), (2, -2.0)]},
            {"objective_history": [(0, -3.0), (1, -2.0), (2, -1.0)]},
        ]
        fig, ax = plot_likelihood_history(metadata, best_idx=1)
        try:
            self.assertEqual(ax.get_ylabel(), "log-likelihood")
            self.assertEqual(ax.get_xlabel(), "iteration")
            self.assertEqual(ax.lines[1].get_label(), "Replicate 2 (best)")
            self.assertGreater(ax.lines[1].get_linewidth(), ax.lines[0].get_linewidth())
            self.assertEqual(list(ax.lines[0].get_xdata()), [0, 1, 2])
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

    def test_omega_plot_starts_at_zero_and_keeps_one_guide_visible(self):
        params = {
            "omega": jnp.array(
                [softplus_inverse(0.1), softplus_inverse(0.5)], dtype=jnp.float64
            )
        }
        fig, ax = plot_per_site_omega(params)
        try:
            self.assertEqual(ax.get_yscale(), "linear")
            self.assertEqual(ax.get_ylim(), (0.0, 1.0))
            self.assertEqual(ax.lines[0].get_ydata()[0], 1.0)
            formatter = ax.yaxis.get_major_formatter()
            self.assertFalse(formatter._scientific)
        finally:
            plt.close(fig)

    def test_omega_plot_upper_limit_tracks_observed_values_above_one(self):
        params = {
            "omega": jnp.array(
                [softplus_inverse(0.1), softplus_inverse(2.0)], dtype=jnp.float64
            )
        }
        fig, ax = plot_per_site_omega(params)
        try:
            self.assertEqual(ax.get_ylim(), (0.0, 2.0))
        finally:
            plt.close(fig)

    def test_map_writes_default_per_site_result_files_without_output_stem(self):
        X = np.zeros((61, 1))
        X[15, 0] = 4
        X[47, 0] = 19
        pi = np.full(61, 1 / 61)

        def return_initial_params(fn, start_params, labels, n_reps, **kwargs):
            metadata = {
                "objective": -1.0,
                "converged": False,
                "n_steps": 0,
                "objective_history": [(0, -1.0)],
            }
            return [start_params], 0, [metadata]

        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with patch(
                    "tombombadil.sample.make_fn", return_value=lambda params: jnp.array(-1.0)
                ), patch(
                    "tombombadil.sample._run_replicates",
                    side_effect=return_initial_params,
                ):
                    fn, start_params, mask = sample._prepare_model(
                        X,
                        pi,
                        include_invariant=True,
                        aggregate="sum",
                        prior_mode="stan_unconstrained",
                        estimate_eta=True,
                        eigen_jitter=True,
                        omega_floor=True,
                        omega_mode="per-site",
                    )
                    sample.run_map_optimizer(
                        fn,
                        start_params,
                        mask,
                        samples=1,
                        omega_mode="per-site",
                    )
                expected_files = (
                    "per_site_output_GTRparams.csv",
                    "per_site_output_omega.csv",
                    "per_site_output_omega_plot.pdf",
                    "per_site_likelihood_plot.pdf",
                )
                for filename in expected_files:
                    with self.subTest(filename=filename):
                        self.assertGreater(os.path.getsize(filename), 0)
            finally:
                os.chdir(original_cwd)

    def test_main_uses_default_nuts_result_stem_without_output_stem(self):
        X = np.zeros((61, 1))
        with patch(
            "sys.argv",
            ["tombombadil", "--alignment", "alignment.fasta", "--fit-method", "nuts"],
        ), patch(
            "tombombadil.__main__.configure_jax_for_options"
        ), patch(
            "tombombadil.__main__.count_codons", return_value=(X, 1)
        ), patch(
            "tombombadil.sample._prepare_model",
            return_value=(lambda params: jnp.array(-1.0), {}, np.ones(1)),
        ), patch(
            "tombombadil.sample.run_nuts_sampler"
        ) as run_nuts:
            main()
        self.assertEqual(run_nuts.call_args.kwargs["output"], "output")


if __name__ == "__main__":
    unittest.main()

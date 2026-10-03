import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from tombombadil.sample import make_fn, softplus_inverse, transforms


REFERENCE_DIR = Path(__file__).parent / "fixtures"
MAP_REFERENCE_PATH = REFERENCE_DIR / "porB3_per_site_map.json"
MAP_REFERENCE = json.loads(MAP_REFERENCE_PATH.read_text())
RTOL = MAP_REFERENCE["tolerance"]["rtol"]
ATOL = MAP_REFERENCE["tolerance"]["atol"]


class TestLikelihoodGradientReference(unittest.TestCase):
    def test_likelihood_and_gradient_match_fixed_reference(self):
        X = np.zeros((61, 1))
        X[15, 0] = 4
        X[47, 0] = 19
        pi = np.full(61, 1 / 61)
        log_pi, pimat, pimatinv, pimult = transforms(X, pi)
        fn = make_fn(pi, log_pi, pimat, pimatinv, pimult, X, jnp.ones(1))
        params = {
            "alpha": softplus_inverse(1.0),
            "beta": softplus_inverse(1.0),
            "gamma": softplus_inverse(1.0),
            "delta": softplus_inverse(1.0),
            "epsilon": softplus_inverse(1.0),
            "eta": softplus_inverse(1.0),
            "theta": softplus_inverse(0.5),
            "omega": jnp.array(softplus_inverse(0.5), dtype=jnp.float64),
        }

        likelihood, gradient = jax.value_and_grad(fn)(params)
        expected_gradient = np.array([
            -1.229974550937346,
            -1.1830645041714325,
            -1.2878611181358206,
            -0.09967741966327304,
            -1.1079022028842331,
            0.0,
            -0.4781776509112939,
            0.41828300010419445,
        ])
        observed_gradient = np.array([float(gradient[key]) for key in params])

        np.testing.assert_allclose(
            float(likelihood), -19.270575644321788, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            observed_gradient, expected_gradient, rtol=RTOL, atol=ATOL
        )


class TestPerSiteMapIntegration(unittest.TestCase):
    def test_cli_outputs_match_fixed_porB3_reference(self):
        project_root = Path(__file__).resolve().parents[1]
        alignment = project_root / "porB3_aligned.fasta"
        reference = MAP_REFERENCE

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_stem = tmp_path / "porB3_map"
            mpl_config = tmp_path / "mplconfig"
            mpl_config.mkdir()
            env = os.environ.copy()
            env["MPLCONFIGDIR"] = str(mpl_config)
            command = [
                sys.executable,
                "-m",
                "tombombadil",
                "--alignment",
                str(alignment),
                "--omega-mode",
                "per-site",
                "--fit-method",
                "map",
                "--sample-it",
                "500",
                "--output-jax",
                str(output_stem),
                "--fit-until-convergence",
                "--convergence-patience",
                "5",
                "--convergence-check-every",
                "10",
                "--convergence-min-steps",
                "50",
                "--exclude-invariant",
                "--platform",
                "cpu",
            ]
            result = subprocess.run(
                command,
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"CLI failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertRegex(
                result.stderr,
                r"MAP replicate 1/1, step 10/500: log-likelihood =",
            )
            likelihood_match = re.search(
                r"Final log-likelihood:\s*([-+0-9.eE]+)", result.stderr
            )
            self.assertIsNotNone(likelihood_match, result.stderr)
            np.testing.assert_allclose(
                float(likelihood_match.group(1)),
                reference["log_likelihood"],
                rtol=RTOL,
                atol=ATOL,
            )

            gtr_path = output_stem.parent / f"per_site_{output_stem.name}_GTRparams.csv"
            omega_path = output_stem.parent / f"per_site_{output_stem.name}_omega.csv"
            likelihood_plot = output_stem.parent / (
                f"per_site_{output_stem.name}_likelihood_plot.pdf"
            )
            omega_plot = output_stem.parent / (
                f"per_site_{output_stem.name}_omega_plot.pdf"
            )
            self.assertGreater(likelihood_plot.stat().st_size, 0)
            self.assertGreater(omega_plot.stat().st_size, 0)
            with gtr_path.open(newline="") as handle:
                observed_gtr = {
                    row["variable"]: float(row["value"])
                    for row in csv.DictReader(handle)
                }
            with omega_path.open(newline="") as handle:
                observed_omega = [
                    float(row["omega_map"]) for row in csv.DictReader(handle)
                ]

        self.assertEqual(observed_gtr.keys(), reference["gtr"].keys())
        np.testing.assert_allclose(
            [observed_gtr[key] for key in reference["gtr"]],
            list(reference["gtr"].values()),
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            observed_omega, reference["omega"], rtol=RTOL, atol=ATOL
        )

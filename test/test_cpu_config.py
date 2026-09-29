import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from tombombadil.__main__ import configure_jax_for_options, get_options


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestCpuWorkerConfiguration(unittest.TestCase):
    def parse_options(self, *arguments):
        argv = ["tombombadil", "--alignment", "unused.fasta", *arguments]
        with mock.patch.object(sys, "argv", argv):
            return get_options()

    def test_default_cpu_worker_count_is_four(self):
        options = self.parse_options()
        self.assertEqual(options.cpus, 4)

    def test_non_positive_cpu_count_is_rejected(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit):
                    self.parse_options("--cpus", value)

    def test_explicit_one_sets_nproc_for_map_and_overrides_environment(self):
        options = self.parse_options("--cpus", "1")
        with mock.patch.dict(os.environ, {"NPROC": "9"}, clear=True):
            configure_jax_for_options(options)
            self.assertEqual(os.environ["NPROC"], "1")
            self.assertNotIn("xla_force_host_platform_device_count", os.environ.get("XLA_FLAGS", ""))

    def test_sequential_nuts_uses_requested_nproc(self):
        options = self.parse_options(
            "--fit-method", "nuts", "--nuts-chain-mode", "sequential", "--cpus", "2"
        )
        with mock.patch.dict(
            os.environ,
            {"NPROC": "9", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false"},
            clear=True,
        ):
            configure_jax_for_options(options)
            self.assertEqual(os.environ["NPROC"], "2")
            self.assertEqual(os.environ["XLA_FLAGS"], "--xla_cpu_multi_thread_eigen=false")

    def test_nuts_pmap_uses_requested_devices_and_preserves_other_xla_flags(self):
        options = self.parse_options(
            "--fit-method", "nuts", "--nuts-chain-mode", "pmap", "--cpus", "3"
        )
        with mock.patch.dict(
            os.environ,
            {
                "NPROC": "9",
                "XLA_FLAGS": (
                    "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=8"
                ),
            },
            clear=True,
        ):
            configure_jax_for_options(options)
            flags = os.environ["XLA_FLAGS"].split()
            device_flags = [
                token for token in flags
                if token.startswith("--xla_force_host_platform_device_count=")
            ]
            self.assertEqual(os.environ["NPROC"], "3")
            self.assertEqual(device_flags, ["--xla_force_host_platform_device_count=3"])
            self.assertIn("--xla_cpu_multi_thread_eigen=false", flags)

    def test_non_cpu_platform_does_not_change_nproc(self):
        for platform in ("gpu", "tpu"):
            with self.subTest(platform=platform):
                options = self.parse_options("--platform", platform, "--cpus", "1")
                with mock.patch.dict(
                    os.environ,
                    {"NPROC": "9", "XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
                    clear=True,
                ):
                    configure_jax_for_options(options)
                    self.assertEqual(os.environ["NPROC"], "9")
                    self.assertEqual(
                        os.environ["XLA_FLAGS"],
                        "--xla_force_host_platform_device_count=8",
                    )

    def test_default_pmap_startup_configures_four_devices_before_jax(self):
        child_script = textwrap.dedent(
            """
            import sys
            from tombombadil.__main__ import configure_jax_for_options, get_options

            sys.argv = [
                "tombombadil", "--alignment", "unused.fasta",
                "--fit-method", "nuts", "--nuts-chain-mode", "pmap",
            ]
            options = get_options()
            configure_jax_for_options(options)
            assert "jax" not in sys.modules
            import jax
            import jax.numpy as jnp
            from tombombadil.sample import run_nuts_sampler

            assert jax.local_device_count() == 4, jax.local_device_count()
            assert __import__("os").environ["NPROC"] == "4"
            fn = lambda params: -0.5 * jnp.square(params["alpha"])
            result = run_nuts_sampler(
                fn,
                {"alpha": jnp.array(0.1)},
                num_warmup=5,
                num_samples=6,
                num_chains=2,
                print_summary=False,
                chain_mode="pmap",
            )
            assert result["samples"]["alpha"].shape == (2, 6)
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["NPROC"] = "1"
            env["XLA_FLAGS"] = ""
            env["MPLCONFIGDIR"] = tmp
            result = subprocess.run(
                [sys.executable, "-c", child_script],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Subprocess failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

"""Reproducible PERF-04 screening and MAP comparisons (see perf04_results.md)."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter


from test.run_gtr_benchmark import measure
from test.jit_benchmark_variants import VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--mode", choices=("check", "diagnose", "inspect", "update", "step", "objective", "map", "memory"), default="objective")
    parser.add_argument("--alignment", default="porB3_aligned.fasta")
    parser.add_argument("--baseline-source", help="Exported original sample.py, if git history is unavailable")
    parser.add_argument("--site-repeat", type=int, default=1)
    parser.add_argument("--omega-mode", choices=("scalar", "per-site"), default="per-site")
    parser.add_argument("--batches", type=int, default=5, help="Warm batches/evaluations")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--params", choices=("initial", "fitted"), default="initial")
    parser.add_argument("--log-compiles", action="store_true", help="Diagnostic only; not for timing comparisons")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--rss-limit-gib", type=float, default=12)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.site_repeat, args.batches, args.timeout, args.rss_limit_gib) <= 0:
        parser.error("site-repeat, batches, timeout and RSS limit must be positive")
    if args.mode == "map" and (args.omega_mode != "per-site" or args.site_repeat != 1 or args.params != "initial"):
        parser.error("MAP comparisons use the pinned 294-site per-site workflow")

    if not args.worker:
        from test.jit_benchmark_limits import supervise
        supervise(args)
        return

    # Configure before any import that can initialize JAX.
    os.environ["NPROC"] = "4"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    import jax
    jax.config.update("jax_log_compiles", args.log_compiles)
    import numpy as np
    import jaxlib
    from test.jit_benchmark_variants import (
        BASELINE_REVISION, baseline_source, install_variant, make_variant,
    )
    from tombombadil import sample
    from tombombadil.alignment import count_codons

    source = baseline_source(args.baseline_source)
    variant = make_variant(args.variant, source)
    report = {
        "variant": args.variant, "mode": args.mode,
        "variants_source_sha256": hashlib.sha256(Path("test/jit_benchmark_variants.py").read_bytes()).hexdigest(),
        "baseline_revision": BASELINE_REVISION,
        "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "production_source_sha256": hashlib.sha256(Path("tombombadil/sample.py").read_bytes()).hexdigest(),
        "likelihood_source_sha256": hashlib.sha256(Path("tombombadil/likelihood.py").read_bytes()).hexdigest(),
        "gtr_source_sha256": hashlib.sha256(Path("tombombadil/gtr.py").read_bytes()).hexdigest(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(), "processor": platform.processor(),
        "python": sys.version, "executable": sys.executable,
        "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "devices": [str(d) for d in jax.devices()], "x64": jax.config.x64_enabled,
        "environment": {k: os.environ.get(k) for k in (
            "NPROC", "XLA_FLAGS", "JAX_ENABLE_COMPILATION_CACHE", "CONDA_PREFIX")},
        "alignment_sha256": hashlib.sha256(Path(args.alignment).read_bytes()).hexdigest(),
        "arguments": vars(args),
    }
    if args.mode in ("check", "diagnose"):
        from test.jit_benchmark_checks import check_variant, diagnose
        report["correctness" if args.mode == "check" else "diagnosis"] = (
            check_variant(args.variant, variant, make_variant("baseline", source))
            if args.mode == "check" else diagnose(make_variant("baseline", source)))
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
        install_variant(variant)
        counts, n_samples = count_codons(args.alignment)
        counts = np.tile(counts, args.site_repeat)
        fn, params, _ = sample._prepare_model(
            counts, np.full(61, 1 / 61), include_invariant=False,
            aggregate="sum", prior_mode="stan_unconstrained", estimate_eta=True,
            eigen_jitter=True, omega_floor=True, omega_mode=args.omega_mode,
        )
        if args.params == "fitted":
            ref = json.loads(Path("test/fixtures/porB3_per_site_map.json").read_text())
            natural = dict(ref["gtr"], omega=np.tile(ref["omega"], args.site_repeat)
                           if args.omega_mode == "per-site" else 0.5)
            params = sample.natural_to_raw_params(natural, omega_mode=args.omega_mode)
        report.update(n_samples=n_samples, n_sites=counts.shape[1])
        from test.jit_benchmark_checks import make_solver
        from test.jit_benchmark_variants import make_kernels
        solver = make_solver(params)
        loss, value_grad, update, advance = make_kernels(lambda p: -fn(p), solver, args.variant)
        if args.mode in ("objective", "memory", "update", "step"):
            device_params = jax.device_put(params)
            if args.mode in ("objective", "memory"):
                report["timing"], result = measure(value_grad, (device_params,), args.batches, 1)
                report["objective"] = -float(result[0])
                report["gradient"] = {k: (-np.asarray(v)).tolist() for k,v in result[1].items()}
            else:
                # Input gradient preparation is outside update/step timing.
                # Fused step timing includes gradient evaluation at new params.
                _, grad = jax.value_and_grad(lambda p: -fn(p))(device_params)
                state = solver.init(device_params)
                jax.block_until_ready((device_params, state, grad))
                timings = []
                for _ in range(args.batches + 1):
                    started = perf_counter()
                    if args.mode == "step":
                        device_params, state, current_loss, grad = jax.block_until_ready(advance(device_params, state, grad))
                    else:
                        device_params, state = jax.block_until_ready(update(device_params, state, grad))
                    timings.append(perf_counter() - started)
                report["timing"] = {"first_seconds": timings[0], "seconds": timings[1:]}
                report["final_raw_params"] = {k: np.asarray(v).tolist() for k,v in device_params.items()}
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["peak_rss_bytes"] = peak if sys.platform == "darwin" else peak*1024
        elif args.mode == "inspect":
            from test.site_benchmark_checks import inspect_objective
            report["inspection"] = inspect_objective(fn, params)
            if args.variant in ("value-grad", "separate", "fused"):
                executable = value_grad.lower(params).compile()
                report["compiled_memory"] = str(executable.memory_analysis())
        else:
            class Timings(logging.Handler):
                def __init__(self):
                    super().__init__()
                    self.entries = []
                    self.startups = []
                    self.optimizations = []

                def emit(self, record):
                    if record.msg.startswith(("MAP startup:", "MAP optimization:")):
                        self.entries.append(record.getMessage())
                        destination = self.startups if record.msg.startswith("MAP startup:") else self.optimizations
                        destination.append(float(record.args[0]))

            handler = Timings()
            logging.getLogger().addHandler(handler)
            runs = []
            for progress in (False, True):
                started = perf_counter()
                result = sample._run_map_replicates(
                    fn, params, sample.make_param_labels(params), 1, n_iter=500,
                    convergence={"enabled": True, "tol": 1e-6, "patience": 5,
                                 "check_every": 10, "min_steps": 50}, progress=progress,
                )
                jax.block_until_ready(result)
                elapsed = perf_counter() - started
                metadata = result[2][0]
                runs.append({"seconds": elapsed, "n_steps": metadata["n_steps"],
                             "objective": float(metadata["objective"]),
                             "timing_log": handler.entries[-2:],
                             "startup_seconds": handler.startups[-1],
                             "optimization_seconds": handler.optimizations[-1]})
            report["warmup"], report["timed"] = runs
            report["natural_params"] = {
                k: np.asarray(sample.positive_transform(v)).tolist() for k, v in result[0][0].items()
            }
            reference = json.loads(Path("test/fixtures/porB3_per_site_map.json").read_text())
            tolerance = reference["tolerance"]
            for run in runs:
                np.testing.assert_allclose(run["objective"], reference["log_likelihood"], **tolerance)
            for key, value in reference["gtr"].items():
                np.testing.assert_allclose(report["natural_params"][key], value, **tolerance)
            np.testing.assert_allclose(report["natural_params"]["omega"], reference["omega"], **tolerance)
            report["map_reference"] = "passed"
    from test.jit_benchmark_variants import TRACES
    report["new_kernel_traces"] = dict(TRACES)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {output}", flush=True)
    if report.get("correctness", {}).get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

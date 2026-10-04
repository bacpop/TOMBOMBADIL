"""Reproducible PERF-01 screening and MAP comparisons (see perf01_results.md)."""

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


def measure(fn, args, batches, calls):
    import jax
    # Inputs have already been placed on device; compile and synchronize once.
    started = perf_counter()
    result = jax.block_until_ready(fn(*args))
    first = perf_counter() - started
    timings = []
    for _ in range(batches):
        started = perf_counter()
        for _ in range(calls):
            result = jax.block_until_ready(fn(*args))
        timings.append((perf_counter() - started) / calls)
    return {"first_seconds": first, "seconds": timings}, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--mode", choices=("kernels", "objective", "map", "memory"), default="kernels")
    parser.add_argument("--alignment", default="porB3_aligned.fasta")
    parser.add_argument("--baseline-source", help="Exported original gtr.py, if git history is unavailable")
    parser.add_argument("--site-repeat", type=int, default=1)
    parser.add_argument("--omega-mode", choices=("scalar", "per-site"), default="per-site")
    parser.add_argument("--objective-jit", action="store_true",
                        help="Diagnostic only: add an outer JIT absent from the MAP optimizer")
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--output", required=True, help="JSON report path")
    args = parser.parse_args()

    # Configure before any import that can initialize JAX.
    os.environ["NPROC"] = "4"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jaxlib
    from test.gtr_benchmark_variants import (
        BASELINE_REVISION, baseline_source, install_variant, make_variant,
    )
    from tombombadil import sample
    from tombombadil.alignment import count_codons

    source = baseline_source(args.baseline_source)
    variant = make_variant(args.variant, source)
    report = {
        "variant": args.variant, "mode": args.mode,
        "baseline_revision": BASELINE_REVISION,
        "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
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
    if args.mode == "kernels":
        results = {}
        for frequency in ("uniform", "nonuniform"):
            pi = np.ones(61) if frequency == "uniform" else np.arange(1, 62, dtype=float)
            pi /= pi.sum()
            pimat, pimult = jax.device_put((np.diag(np.sqrt(pi)), np.sqrt(pi[None, :] / pi[:, None])))
            rates = jax.device_put(np.array([0.7, 1.3, 0.9, 1.7, 0.4, 1.1]))
            weight = jax.device_put(np.sin(np.arange(61 * 61).reshape(61, 61)))
            neutral = variant.build_GTR(*rates, 1.0, pimat, pimult)
            jax.block_until_ready(neutral)
            for omega in (0.1, 1.0, 3.0):
                params = jnp.concatenate((rates, jnp.array([omega])))
                build = lambda p: variant.build_GTR(*p[:6], p[6], pimat, pimult)
                update = lambda p: variant.update_GTR(neutral, p[6], pimult)
                composed = lambda p: variant.update_GTR(
                    variant.build_GTR(*p[:6], 1.0, pimat, pimult), p[6], pimult)
                for label, fn in (("build", build), ("update", update), ("composed", composed)):
                    for derivative in (False, True):
                        measured = jax.value_and_grad(lambda p: jnp.sum(fn(p) * weight)) if derivative else fn
                        key = f"{frequency}/{omega}/{label}/{'gradient' if derivative else 'value'}"
                        results[key], _ = measure(jax.jit(measured), (params,), args.batches, args.calls)
        report["results"] = results
    else:
        install_variant(variant)
        counts, n_samples = count_codons(args.alignment)
        counts = np.tile(counts, args.site_repeat)
        fn, params, _ = sample._prepare_model(
            counts, np.full(61, 1 / 61), include_invariant=False,
            aggregate="sum", prior_mode="stan_unconstrained", estimate_eta=True,
            eigen_jitter=True, omega_floor=True, omega_mode=args.omega_mode,
        )
        report.update(n_samples=n_samples, n_sites=counts.shape[1])
        if args.mode in ("objective", "memory"):
            measured = jax.value_and_grad(fn)
            if args.objective_jit:
                measured = jax.jit(measured)
            report["timing"], result = measure(
                measured, (jax.device_put(params),), args.batches if args.mode == "objective" else 1, 1)
            report["objective"] = float(result[0])
            report["gradient"] = {k: np.asarray(v).tolist() for k, v in result[1].items()}
            # macOS reports bytes; Linux reports KiB. Process high-water mark,
            # including imports and compilation, not device allocation size.
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
        else:
            logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
            class Timings(logging.Handler):
                def __init__(self):
                    super().__init__()
                    self.entries = []

                def emit(self, record):
                    if record.msg.startswith(("MAP startup:", "MAP optimization:")):
                        self.entries.append(record.getMessage())

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
                             "timing_log": handler.entries[-2:]})
            report["warmup"], report["timed"] = runs
            report["natural_params"] = {
                k: np.asarray(sample.positive_transform(v)).tolist() for k, v in result[0][0].items()
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()

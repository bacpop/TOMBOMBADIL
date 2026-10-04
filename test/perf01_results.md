# PERF-01: GTR benchmark campaign

This campaign compares GTR operations only. The original implementation is
anchored to commit `d22f582c9ecf1c438254824e6c6f227db98b231d`; the benchmark loader
uses its exact source, rather than a reconstruction. Production does not import
the benchmark alternatives.

## Reproduction

Use the active mamba Python environment and run from the repository root. Set
`MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable temporary directories. All runs
configure CPU, `NPROC=4`, float64, and disable persistent JAX compilation caching
before importing JAX. Run one benchmark process at a time.

```sh
python -m test.run_gtr_benchmark --variant baseline --mode kernels \
  --output /tmp/perf01/kernels-baseline.json
python -m test.run_gtr_benchmark --variant frequency --mode objective \
  --omega-mode per-site --site-repeat 10 \
  --output /tmp/perf01/objective-per-site-10-frequency.json
python -m test.run_gtr_benchmark --variant combined-lookup --mode map \
  --output /tmp/perf01/map-combined-lookup.json
python -m test.run_gtr_benchmark --variant production --mode memory \
  --site-repeat 10 --output /tmp/perf01/memory-production.json
```

For a checkout without the baseline's git history, export the original `gtr.py`
from that revision and pass `--baseline-source /path/to/original-gtr.py`.

The runner records versions, hardware/platform, environment, input/source hashes,
individual timing samples, and numerical results. `test.gtr_benchmark_variants`
lists the available candidates. Each variant must run in a fresh process so that
previously traced model functions cannot retain another implementation.

## Workloads and timing boundaries

- Kernels: GTR construction, omega update, and their composition, both values and
  reverse derivatives of a weighted matrix reduction. Two frequency distributions
  and three omega values; five batches of 100 synchronized calls per case. Inputs
  are already on device. First-call time includes compilation, reported separately.
- Objectives: the actual optimizer's `jax.value_and_grad(fn)` boundary, scalar and
  per-site omega, at 294 and 2,940 sites. The larger input repeats porB3 columns
  ten times. Five synchronized warm evaluations after the first call. Other model
  settings match the MAP workflow below.
- MAP: exactly the `_run_map_replicates` timing boundary used by
  `test.run_map_benchmark`, with one full warm-up pass followed by a timed pass.
  Per-site omega, uniform frequencies, estimated eta, summed objective,
  `stan_unconstrained` prior, invariant sites excluded, one replicate, 500 maximum
  iterations, tolerance `1e-6`, patience 5, check interval 10, minimum 50 steps.
  Alignment loading, model preparation, and output/plot writing are excluded.
  The timed pass includes normal per-step logging; warm-up suppresses progress.
  Startup is compilation-inclusive, not a measurement of pure compiler time.
- Memory: separate fresh processes, 2,940-site per-site objective and gradient.
  Report process peak RSS including imports, compilation, and execution. This is
  not a device allocation measurement. Memory-run timings are not used for ranking.

Two initial MAP pilots were excluded because logging had been configured after
model preparation, leaving startup records empty and suppressing progress logs.
The runner now forces its logging configuration, and replacement runs use the
same configuration as the other accepted measurements.

Full MAP order rotates across rounds: baseline/frequency/combined-lookup,
combined-lookup/baseline/frequency, then frequency/combined-lookup/baseline.

Acceptance: require protected numerical tests to pass, then either a repeatable
5% improvement in median warm MAP runtime without a >5% startup/larger-objective
regression, or a 10% startup improvement with warm runtime within 5% of baseline.
Extend borderline full-MAP comparisons to five rounds. Prefer the simpler
implementation when measurements do not distinguish candidates.

## Numerical screening

The original sparse indices, value expressions, and dense omega masks agree on
526 allowed substitutions and 392 nonsynonymous entries. All twelve candidates
passed independent genetic-code checks for matrices, structural zeros, symmetry,
stationarity, reverse derivatives with respect to the six rates and omega, and
replacement of arbitrary incoming diagonals. The tests also cover arbitrary
off-diagonal input to `update_GTR`, not only structural zeros.

An initial diagnostic added an outer `jax.jit` around the entire objective and
gradient. At symmetric starting parameters this produced gradient differences
despite matching likelihoods. A one-site reproduction showed that adding this
JIT also changes the baseline's own gradients beyond the protected tolerance.
For example, its delta gradient changed from the protected
`-0.09967741966327304` to `-0.0996670027805008`. The combined candidate passed the
existing protected test. On the actual optimizer path the baseline and candidates
had identical neutral/omega matrices and matching objective gradients.

The benchmark was corrected to use the actual optimizer boundary, and all
objective measurements were repeated. `--objective-jit` retains the explicit
diagnostic, but those measurements are excluded from selection. Investigate this
JIT-sensitive numerical behavior before PERF-04; this task changes neither JIT
coverage nor eigensolver behavior, reference values, or tolerances. Include
other fully jitted consumers, such as the NUTS step, in that investigation;
sampler-specific numerical comparisons were outside this MAP-focused campaign.

## Results

Kernel screening (median of the six case medians):

| Candidate | Composed value (µs) | Composed gradient (µs) |
|---|---:|---:|
| baseline | 49.87 | 93.47 |
| diag-mask | 52.02 | 87.43 |
| diag-einsum | 54.28 | 91.26 |
| omega-where | 50.97 | 88.67 |
| omega-multiply | 52.28 | 89.66 |
| frequency | 31.63 | 47.97 |
| rates-lookup | 41.63 | 88.44 |
| rates-sum | 43.65 | 100.14 |
| rates-einsum | 42.60 | 96.42 |
| combined-lookup | 18.78 | 45.15 |
| combined-sum | 19.85 | 56.66 |
| combined-einsum | 19.16 | 56.89 |

Full MAP comparisons:

| Candidate | Startup median, range (s) | Warm MAP median, range (s) |
|---|---:|---:|
| baseline | 3.772 (3.752–3.863) | 88.843 (87.921–89.582) |
| frequency | 3.815 (3.766–3.853) | 88.863 (88.548–89.205) |
| combined-lookup | 3.288 (3.273–3.331) | 90.883 (89.591–92.295) |

Three valid fresh-process runs per candidate. Every warm-up and timed run used
210 steps and matched the fixed objective; all final GTR and 294 omega estimates
passed the existing `rtol=1e-6`, `atol=1e-8` comparisons.

The combined candidate reduced median compilation-inclusive startup by 12.83%
(0.484 seconds). Its median warmed optimizer runtime increased by 2.30%.
Frequency broadcasting alone did not improve either median meaningfully.

| Omega mode | Sites | Baseline warm objective/gradient (s) | Frequency (s) | Combined lookup (s) |
|---|---:|---:|---:|---:|
| scalar | 294 | 0.027400 | 0.027740 | 0.025747 |
| scalar | 2940 | 0.113885 | 0.122474 | 0.113281 |
| per-site | 294 | 0.385085 | 0.395807 | 0.406252 |
| per-site | 2940 | 7.703150 | 8.073322 | 8.069049 |

All actual-boundary objective and gradient comparisons passed. The 2,940-site
per-site warm objective probe was 4.75% slower for the combined candidate; its
first call was 19.68% faster. These are screening samples, not full MAP runs.

| Candidate | Memory probe peak RSS median, range (GB) |
|---|---:|
| baseline | 11.020 (9.975–11.175) |
| combined-lookup | 11.480 (11.260–11.514) |

Memory uses decimal GB. Three independent processes per row, alternating
baseline and candidate order. Median peak RSS was 4.17% higher for the
combined candidate. The separate, longer six-call objective probes peaked
at 11.780 GB (baseline) and 11.722 GB (combined); no memory improvement is
established. Process RSS varies and is not a pure JAX allocation measurement.

## Selection

Retain **combined-lookup** under the planned startup criterion: at least 10%
lower startup time with warm MAP runtime within 5% of baseline. This improves
startup, while the measured 210-step reference workflow is slightly slower
overall. It does not establish better sustained throughput or lower memory.

Production decodes one static substitution table, gathers the rates with a
zero sentinel, applies omega with a mask, replaces the diagonal using masks,
and broadcasts the diagonal frequency factors. The original helper signatures
and profiler annotation remain. No production benchmark-selection flags,
optimizer changes, eigensolver changes, or additional JIT boundaries are added.

Do not retain frequency-only, einsum, or six-mask alternatives separately:
they did not beat the selected combination under the campaign criteria.
PERF-02 remains a separate task.

Individual timing samples, environment/source/input metadata, and numerical
comparison errors are in [the machine-readable results](benchmarks/perf01_results.json).
The report preserves the excluded outer-JIT diagnostic separately.

## Production validation

The installed implementation was checked in an additional fresh-process MAP run:
startup 3.238 s, warmed optimizer 90.732 s, 210 steps, and the fixed objective,
GTR estimates, and all omega estimates within their original tolerances. This is
a confirmation run, not an additional three-run performance estimate.

`python -m pytest -q`: **72 passed, 10 subtests passed**, with the same two
upstream dependency warnings, in 120.13 seconds. This includes the unchanged
protected likelihood/gradient and MAP tests. CLI help, `compileall`, and
`git diff --check` passed. No reference fixture or tolerance changed.

Measurements used macOS 26.6.2 arm64, Python 3.14.7, JAX/jaxlib 0.8.2,
CPU with four workers, float64, and no persistent compilation cache. These
results do not establish performance on other hardware or on GPUs.

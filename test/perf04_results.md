# PERF-04: JIT coverage

Baseline: `0962b3e2e9f5d30af083165e797189f3bb731cf0`, including PERF-01/02 and
PERF-03's unchanged production vmap. Production retains this baseline: none of
the compiled alternatives meets the numerical and performance acceptance gates.

## Protocol

Qualify with >=5% faster complete MAP, >=10% faster the largest common
objective/gradient workload, or >=20% less peak RSS. Allow no runtime regressions,
including compilation-inclusive startup, and <=5% other RSS regressions. Use
three fresh processes, extending uncertain comparisons to five; use all valid
trials and unrounded medians. Keep protected references and tolerances fixed.

Workers use CPU, float64, NPROC=4 and no persistent compilation cache, in the
active mamba environment. Probes run serially, with a 12 GiB process RSS guard
and 600-second timeout. First calls are separate from five synchronized warm
calls. Update and step probes carry optimizer state and parameters; isolated
updates use a fixed input gradient, whereas complete steps recompute it.
Recorded platform: macOS 26.6.2 arm64, Python 3.14.7, JAX/jaxlib 0.8.2,
one reported CPU device. Full environment/source metadata is in each report.

Full MAP probes perform a complete warmup and then a timed complete invocation,
using 500 iterations maximum and convergence tolerance 1e-6, patience 5,
check_every 10, min_steps 50. Each invocation constructs its own fit kernels:
any recompilation in the timed invocation remains included. Within a fit,
replicates reuse kernels but initialize independent optimizer states. Startup
retains the original initial-evaluation/first-update boundary. The final
iteration still uses value-only evaluation, and history/convergence/logging
remain in Python.

## Candidates and numerical screening

- Baseline: exact source from the pinned revision.
- Update: JIT only Optax update and application; original likelihood AD.
- Value-grad: JIT the complete value-and-gradient call.
- Objective: differentiate the JIT-compiled objective.
- Separate: compiled value-and-gradient and update kernels.
- Fused: compile update followed by evaluation at the updated parameters;
  retain the original final value-only evaluation.

The candidate kernel is called directly against the protected one-site oracle,
read from the unchanged reference test. Eight further cases cover both omega
modes, initial/fitted/asymmetric parameters, frequency/jitter/eta choices,
omega-floor boundaries, masking/reductions and all prior modes. Compare all
objective/gradient leaves, updates, optimizer state, and three successive steps
at rtol=1e-6, atol=1e-8; reject non-finite results.

The objective candidate fails the one-site oracle by 1.04e-5. Value-grad passes
that oracle but fails the symmetric per-site case by 1.41e-5; separate and fused
likelihood compilation also fail. These are numerical rejections, not timing
results. Update-only passes the complete screen and proceeds to measurement.

## Numerical investigation

Symmetric rates leave an eigenvalue gap around 2.22e-16 after the existing
identity jitter. The asymmetric control's minimum gap is about 1.67e-4. Adding
an identity multiple shifts eigenvalues without mathematically separating equal
eigenvalues; it cannot guarantee a well-conditioned eigenvector derivative.

The eager and compiled substitution matrices match exactly in the diagnostic.
Alpha-matrix outputs agree within 1.67e-16, and downstream cotangents within
8.14e-13. Applying the *same* cotangent through eager versus compiled alpha
pullbacks differs by up to 4.40e-5 at symmetric rates, versus 2.01e-14 for the
asymmetric control. This localises the sensitivity to the alpha reverse path,
consistent with ill-conditioned differentiation around repeated eigenvalues;
it does not identify every compiler transformation responsible.

Scaling the same alpha cotangent by ten also exposes sensitivity within the
eager pullback: deviation from ten times the original result reaches 7.94e-4 at
symmetric rates, versus 4.72e-13 for the asymmetric control. This supports the
connection to the previously observed scalar repeated-input gradient issue;
scalar controls use direct same-size baseline comparisons, not that invalid
scaling identity. The diagnostic retains a hash/length of lowered compiler IR.

A finite-difference sweep of the protected delta parameter approaches roughly
-0.09967547, while the protected eager derivative is -0.09967742. The asymmetric
control agrees closely across eager, compiled and finite-difference evaluations.
These diagnostics are not new reference values or relaxed tests. No eigensolver,
jitter, derivative rule or reference change is made. A numerical fix needs a
separate proposal and independent model-level validation.

## Reproduction

```sh
export MPLCONFIGDIR=/tmp/perf04-mpl
export XDG_CACHE_HOME=/tmp/perf04-cache
python -m test.run_jit_benchmark --mode diagnose --output /tmp/perf04/diagnose.json
python -m test.run_jit_benchmark --variant update --mode check --output /tmp/perf04/update-check.json
python -m test.run_jit_benchmark --variant baseline --site-repeat 10 --output /tmp/perf04/baseline-objective-10-initial-1.json
python -m test.run_jit_benchmark --variant update --mode map --output /tmp/perf04/update-map-1.json
python -m test.summarize_jit_benchmark /tmp/perf04 --output test/benchmarks/perf04_results.json
```

Use distinct output filenames for fresh-process trials; alternate baseline and
candidate order. `--params fitted` and `--omega-mode scalar` select controls.
`--mode update/step` isolates update or full numerical-step timing. The optional
`--log-compiles` flag is diagnostic only and must not be used for acceptance
measurements. New-kernel trace counts are retained separately from timing.

## Results and limitations

Five fresh-process MAP trials give a median complete timed invocation of
106.480551 s for baseline and 107.703812 s for update-only JIT, a 1.15% regression.
Every trial converges at 210 steps and passes the fixed MAP objective, GTR and
294-omega checks. The first isolated update pilot measured about 1.65 ms baseline
versus 0.10 ms compiled, against roughly 0.49 s for a full likelihood step. That
kernel improvement does not translate to a qualifying end-to-end improvement.

| MAP trial | Baseline (s) | Update JIT (s) |
| --- | ---: | ---: |
| 1 | 177.958403 | 110.476344 |
| 2 | 108.791608 | 109.954834 |
| 3 | 105.228417 | 107.703812 |
| 4 | 106.480551 | 105.486032 |
| 5 | 104.153303 | 107.481561 |

Early trials show substantial timing variability. All valid measurements,
including the slow first baseline, are retained. Acceptance calculations use
unrounded values; displayed tables are rounded. No specific cause is assigned
to the variability.

The 2,940-site initial per-site objective uses five independent processes per
variant and five synchronized warm calls per process:

| Median metric | Baseline | Update JIT |
| --- | ---: | ---: |
| Warm objective/gradient (s) | 6.045230 | 6.217870 |
| First objective/gradient (s) | 7.862557 | 7.706924 |
| Peak process RSS (GiB) | 6.583221 | 6.624222 |

The warm objective is 2.86% slower and RSS is 0.62% higher. This candidate leaves
the objective graph unchanged; these measurements do not establish a speed or
memory benefit. Neither the >=5% MAP, >=10% objective nor >=20% RSS threshold is
met, and runtime regressions independently prevent adoption.

Early startup records used the existing millisecond-formatted log. Later probes
capture raw timing arguments without changing the measured work. The summary
bounds those early startup values by +/-0.0005 s and uses conservative median
bounds for the no-regression check. Complete invocation timings are unrounded
in every trial. Update MAP probes report two new update-kernel traces, one per
fit (warmup and timed); these are tracing observations, not backend compilation
counts. A behavioural test separately checks reuse across replicates in one fit.
The compilation-inclusive cold-startup median falls from about 2.035 s to
1.815 s (at least 10.80% by the rounding bound); startup improvement alone is
not one of this task's qualifying benefits.

The archive contains 46 reports: 38 completed probes, four numerical rejections,
and four allocation-preflight skips. All 20 objective reports pass direct
same-size baseline checks; per-site cases also pass the repeated-input identity.
Initial/fitted per-site and initial scalar controls cover 294 and 2,940 sites.
The 8,820/29,400-site per-site probes are skipped for both variants: estimated
process RSS is 24.80/80.48 GiB, over the 12 GiB guard. This estimate reuses the
PERF-03 saved-residual inspection. Update-only JIT leaves that objective graph
unchanged, so no larger-capacity benefit is inferred.

Production code and protected references are unchanged. The benchmark harness
retains the candidate implementations for reproducibility and future work.

Validation: `python -m pytest -q` passes **82 tests and 32 subtests** in 147.62 s,
including the fixed likelihood/gradient and CLI MAP oracles. Behavioural tests
cover optimizer history/convergence/callbacks, zero/one/multiple steps, replicate
state isolation and kernel reuse, and new objectives with unchanged shapes.
CLI help, Python compilation, archive source hashes and diff checks also pass.
The two dependency deprecation warnings remain. Since no production candidate
is installed, the full suite's normal production MAP test is the final
integration confirmation.

No inference extends to GPUs, sharding, NUTS, other JAX versions or independent
biological datasets. The repeated PorB3 ladder measures computational scaling.

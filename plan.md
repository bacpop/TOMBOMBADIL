Read `AGENTS.md` before starting work.

This file is the authoritative record of project state, priorities, decisions, and handoff context.

Do not begin a backlog item merely because it appears in this file. Work only on the Current Task unless the requested work requires updating it.

# Current Status

**Phase:** Maintainable MAP optimisation
**Status:** Complete

MAINT-01 and MAINT-02 are complete. The CPU worker count now configures MAP and
NUTS on the CPU backend; the full test suite and numerical references pass.
The MAINT-03, MAINT-04, and MAINT-05 batch is complete. MAP objective values
are reported at every step, compilation-inclusive startup time is separated
from later optimization time, result output uses logging and default files,
and the per-site omega plot uses a linear scale. Numerical references pass.

# Current Task

## MAINT-03, MAINT-04, and MAINT-05 — Compilation logging, consistent output, and omega plot

Complete the original MAINT-03, MAINT-05, and MAINT-06 together. At the start
of this batch, renumber the backlog so consistent output becomes MAINT-04,
omega plotting becomes MAINT-05, and the existing MAP/NUTS path separation
task becomes MAINT-06.

### Checklist

- [x] Renumber MAINT-05 to MAINT-04, MAINT-06 to MAINT-05, and the existing MAINT-04 to MAINT-06.
- [x] Report and record MAP objective values at step 0 and after every optimizer update in the progress display, logs, and likelihood plot; preserve configured convergence checks and best-replicate selection.
- [x] Log a compilation-inclusive startup duration covering the initial objective evaluation and first optimizer update, separately from the remaining optimization duration.
- [x] Replace final-result prints with logging for MAP, NUTS, Laplace summaries, and fixed-parameter diagnostics.
- [x] Always write MAP estimate and NUTS posterior CSVs. Without `--output-jax`, use the `output` stem and existing mode-prefixed filename conventions; preserve current CSV schemas.
- [x] Plot per-site omega on a linear axis from zero, use plain numeric y ticks where practical, and keep the omega=1 guide visible (the upper limit may therefore be 1 when the observed maximum is lower).
- [x] Save the per-site omega plot with a default mode-prefixed filename when no output stem is supplied; document default result files.
- [x] Add or update focused tests for every-step objective history/reporting, logging, default and explicit output paths, and omega plot scale and tick formatting.
- [x] Run the full test suite and help check; preserve protected numerical references and tolerances.
- [x] Record a warmed MAP benchmark with the existing timing boundary.
- [x] Update Current Status, Design Notes, Session Log, blockers, and Next Task.

### Guardrails

The likelihood/gradient unit test is a correctness oracle for subsequent work.

- It must not be removed, skipped, or weakened to accommodate implementation changes.
- Numerical differences are acceptable only within the documented floating-point tolerance.
- Changes outside tolerance must be investigated rather than accepted by changing the reference values.
- Reference values or tolerances may only be changed when there is an explicitly justified change to the intended model behaviour.

The MAP integration test is the primary end-to-end oracle for optimisation work.

Performance comparisons must use comparable inputs, configuration, hardware, and timing boundaries. Record enough information to make before/after measurements interpretable.

# Next Task

Continue with MAINT-06 — Separate MAP and NUTS execution paths. Do not begin it automatically.

---

# Project

## Goal

Improve the coding style, maintainability, and computational efficiency of TOMBOMBADIL.

TOMBOMBADIL uses JAX to calculate a likelihood for selection from genetic data. The primary target path for this work is MAP optimisation using Optax.

## Scope

Primary targets:

- `tombombadil/`
- `test/`

Out of scope unless explicitly requested:

- NUTS optimisation
- BlackJAX-specific implementation
- "Later tidy up" work described below

# Engineering Priorities

In descending order:

1. Computational efficiency
2. Memory efficiency
3. Readability and maintainability
4. Useful error reporting

Avoid excessive input/output validation that adds complexity without protecting realistic failure modes.

Correctness takes precedence over optimisation: performance improvements must preserve the numerical behaviour protected by the reference tests.

---

# Backlog

Backlog order does not authorize work. Move an item into Current Task before implementing it.

## Testing and baselines

Run tests in the active mamba-managed environment with `python -m pytest`.
The project-level pytest installation workflow is deferred.

### TEST-01 — Likelihood and gradient unit test

Add a unit test covering likelihood and gradient calculations.

The test must use fixed reference values and explicit floating-point tolerances.

### TEST-02 — MAP integration test

Add an integration test equivalent to:

`python -m tombombadil --alignment porB3_aligned.fasta --omega-mode per-site --fit-method map --sample-it 500 --output-jax porB3_map --fit-until-convergence --exclude-invariant`

Check that the following remain within documented tolerances of reference values:

- likelihood
- omega estimates
- GTR parameter estimates

### TEST-03 — MAP performance baseline

Time the optimisation portion of the MAP integration workflow.

Record:

- benchmark command/input
- relevant runtime configuration
- timing boundary
- baseline runtime

Use this benchmark when evaluating optimisation changes.

---

## Maintainability

### MAINT-01 — Optimisation progress reporting

Add default progress reporting for MAP optimisation using tqdm already installed
in the active mamba environment.

Include:

- iteration/progress
- current likelihood
- likelihood-versus-iteration plot

### MAINT-02 — CPU/thread configuration

MAP mode appears to use approximately 500% CPU regardless of whether the thread count is specified.

Investigate and ensure the requested CPU/thread configuration is respected.

### MAINT-03 — Compilation logging

Make it clear in logging that JAX model compilation is a one-off cost rather than part of steady-state optimisation performance.

Change the progress bar and plot to report the log-likelihood every step,
accepting the slight hit to performance for this evaluation.

### MAINT-04 — Consistent output/logging

Remove the mixture of `print` and logging for final output.

Use logging consistently.

Write final parameter estimates and dN/dS results to output files rather than printing them.

### MAINT-05 — Omega plot

Scale the omega plot from zero to the observed maximum.

Avoid scientific-notation labels on the y-axis where practical.

### MAINT-06 — Separate MAP and NUTS execution paths

The current `run_sampler` control flow between ML/MAP and NUTS is unclear and relies on an early return.

Separate these into explicit execution paths/functions.

Do not alter BlackJAX behaviour unless necessary for this separation.

### MAINT-07 — CLI organisation

Reorganise CLI options so the primary options appear first:

1. input/output
2. model fitting method
3. CPU/runtime configuration

Clearly separate options specific to:

- MAP / Optax
- NUTS / BlackJAX

Replace or deprecate `--sample-it` for MAP mode with terminology appropriate to optimisation iterations.

Consider making early convergence the default for MAP optimisation.

Setting defaults both in the CLI parser, and in function arguments is
confusing. Remove defaults in functions where these are set at input.

### MAINT-08 — Move I/O functions out of `__main__`

Move I/O-related functions from `__main__` into an appropriate module.

Consolidate duplicated `read_fasta()` / `read_alignment()` functionality currently present in `domains.py`.

### MAINT-09 — Improve function names

Review and rename unclear functions, including:

- `my_dirichlet_multinomial_logpmf`
- `make_fn`
- `_run_replicates`

Names should describe their role rather than implementation history.

### MAINT-10 – remove JAXopt

JAXopt is deprecated. We don't rely on it explicitly, but is giving
a testing warning. Fix this warning / remove the dependency.

---

# Optimisation Backlog

For every optimisation below, the MAP integration test is the correctness oracle and TEST-03 is the performance baseline.

Do not retain an optimisation solely because it appears theoretically faster. Measure it against the baseline.

### PERF-01 — GTR operations

Investigate whether the GTR functions can be improved using `einsum` or direct multiply/add operations instead of indexed `.at[...]` / `.set(...)` updates. See hints in the commented code especially in gtr.py for more of my thoughts, but other suggestions are welcome.

### PERF-02 — `_gen_alpha_impl`

Investigate alternatives to loops and repeated `.at[...]` / `.set(...)` operations in `_gen_alpha_impl`.

Prefer JAX operations that compile efficiently and avoid unnecessary intermediate arrays.

### PERF-03 — Site vectorisation

Verify whether `batched_loss` and its `jax.vmap()` usage efficiently vectorise over per-site omega values.

Test behaviour at alignment sizes representative of the intended large-scale workload.

### PERF-04 — JIT coverage

Review the MAP execution path for missed JIT-compilation opportunities.

Reference:

https://docs.jax.dev/en/latest/201/jit.html#jax-201-jit

Consider compilation boundaries as well as steady-state execution performance.

### PERF-05 — MAP parallelism and scaling

Investigate explicit parallelism for MAP/Adam optimisation. Consider adding sharding.

The design should consider eventual workloads with approximately one million sites and potential GPU execution.

---

# Later Tidy Up

**Do not commence this section without explicit approval.**

- Review and reduce obsolete or excessive comments.
- Remove profiler decorators.

---

# Definition of Done

## Current task

A Current Task is complete when:

- [x] Its checklist is complete.
- [x] Relevant existing tests pass.
- [x] New tests required by the task pass.
- [x] Protected numerical tests remain within their documented tolerances.
- [x] Any relevant performance comparison has been recorded.
- [x] Important implementation decisions are recorded in Design Notes.
- [x] Blockers or unresolved issues are recorded.
- [x] Current Status reflects the repository state.
- [x] Session Log records the work performed.
- [x] Next Task provides a clear handoff.

## Project

The project is complete when:

- [x] `python -m tombombadil --help` runs successfully.
- [x] The required unit and integration tests exist.
- [x] All required tests pass.
- [x] Numerical correctness guardrails are satisfied.
- [ ] Approved maintainability work is complete.
- [ ] Approved optimisation work has been benchmarked and accepted or rejected based on evidence.
- [x] `plan.md` accurately reflects the final repository state.

---

# Design Notes

Record decisions that future sessions need to understand.

Use entries of the form:

### YYYY-MM-DD — Short decision title

**Context:** Why a decision was needed.

**Decision:** What was chosen.

**Rationale:** Why this option was chosen.

**Consequences:** Important implications or follow-up work.

### 2026-09-29 — Fixed pytest correctness and MAP baselines

**Context:** The baseline phase had a likelihood-only unit oracle, no per-site
MAP regression test, and a codon-count test that referenced an absent alignment.

**Decision:** Preserve the existing `Testdiv` likelihood reference unchanged and
add a fixed likelihood/gradient oracle. Record full per-site MAP outputs in
`test/fixtures/porB3_per_site_map.json`. Use `rtol=1e-6` and `atol=1e-8` for the
new numerical comparisons. Run the suite as `python -m pytest` in the active
mamba-managed environment; do not add packaging-level pytest installation
metadata in this phase.

**Rationale:** Fixed references protect model and optimiser behavior, while the
active mamba environment already supplies pytest. The environment installation
workflow can be handled separately.

**Consequences:** The likelihood/gradient reference is `-19.270575644321788`
with gradients ordered as alpha, beta, gamma, delta, epsilon, eta, theta,
omega: `[-1.229974550937346, -1.1830645041714325, -1.2878611181358206,
-0.09967741966327304, -1.1079022028842331, 0.0, -0.4781776509112939,
0.41828300010419445]`. The per-site MAP objective reference is
`-1071.4185160607003`; GTR estimates and all 294 omega estimates are in the
JSON fixture. The optimizer-only baseline was `142.556440` seconds for 210
steps on macOS 26.6.2 arm64 with Python 3.14.7 and CPU backend. Timing starts
at `_run_replicates()` and includes first-step JAX compilation; it excludes
alignment loading, transform setup, and output writing.

### 2026-09-29 — Default MAP progress and likelihood history

**Context:** MAINT-01 requested MAP iteration progress, current likelihood
updates, and a likelihood-versus-iteration plot. The active mamba environment
already provides tqdm.

**Decision:** Use a tqdm bar for each MAP replicate in an interactive terminal.
Record the starting objective and objective values at the existing convergence
checks, or every 10 iterations in fixed-step mode, plus the final iterate when
needed. In non-TTY runs, log those checkpoints. Save the plot on every MAP run
with the y-axis label `log-likelihood`; highlight the best replicate. Use
`scalar_likelihood_plot.pdf` or `per_site_likelihood_plot.pdf` without an output
stem, and the existing mode-prefixed output stem when one is supplied. Add no
CLI flag or package dependency.

**Rationale:** Reusing convergence-check evaluations avoids changing the
convergence rule. Fixed-step objective samples make progress visible without
evaluating the model at every iteration. The optimizer updates, replicate
selection, and reference values remain unchanged.

**Consequences:** The benchmark helper now runs one full untimed optimizer pass
in-process, blocks until its JAX results are ready, then times a fresh run. On
the 23-sample, 294-site porB3 input with CPU backend, one replicate, 500 maximum
steps, convergence enabled, and invariant sites excluded, the pre-change warm
pass took 144.394336 s and the timed pass 130.472883 s. After MAINT-01, the warm
pass took 138.051521 s and the timed pass 133.801273 s (210 optimizer steps in
both cases; final objective `-1071.4185160607003`). The post-change timed run is
2.55% above the pre-change warm measurement and 6.14% below the earlier cold
baseline of 142.556440 s; these single runs do not establish a performance
improvement. Timing starts at `_run_replicates()` after the warm-up and includes
progress reporting and result synchronization; it excludes alignment loading,
transform setup, plot writing, and parameter output. No unresolved blockers.

---

### 2026-09-29 — CPU worker configuration

**Context:** `--cpus` previously affected only CPU NUTS `pmap` device creation,
defaulted to 1, and did not constrain CPU work during MAP or sequential NUTS.

**Decision:** Default to 4 and apply `NPROC=<cpus>` on the CPU backend before
JAX import for MAP and both NUTS modes. For CPU NUTS `pmap`, normalize the
`--xla_force_host_platform_device_count` entry in `XLA_FLAGS` to match the
requested count while preserving unrelated flags. Explicit positive counts,
including 1, override inherited `NPROC`. Do not alter GPU/TPU worker settings.
Describe `--cpus` as a JAX worker setting, not a strict process CPU cap.

**Rationale:** Environment probes showed that `NPROC` affects CPU worker
utilization in this JAX/mamba environment, while common Eigen thread flags did
not. Four CPU pmap devices are configured before JAX import; the fresh-process
two-chain smoke check confirms they are usable.

**Consequences:** A 30-iteration per-site MAP probe on `porB3_aligned.fasta`
completed in 29.17 s with `--cpus 1` (median 168.8% CPU after the first 10 s)
and 19.61 s with `--cpus 4` (median 358.6%). CPU usage can exceed 100% per
worker due to helper threads and compilation. The warmed benchmark used 4
workers and the existing optimizer-only timing boundary: an untimed warm pass
took 99.071470 s, followed by a timed pass of 101.753761 s for 210 steps. It
used the same 23-sample, 294-site input and produced the fixed objective
`-1071.4185160607003`. The prior MAINT-01 measurement (then `--cpus 1`, before
this option configured CPU workers) was 138.051521 s warm and 133.801273 s timed
for 210 steps. The current timed run is about 24% shorter, but one run per
configuration does not establish a repeatable performance comparison or
attribute the difference to worker count. The benchmark used:

`env NPROC=4 MPLCONFIGDIR=/private/tmp/maint02-benchmark/mplconfig python -m test.run_map_benchmark --alignment porB3_aligned.fasta --omega-mode per-site --fit-method map --sample-it 500 --output-jax /private/tmp/maint02-benchmark/default4_reference_settings --fit-until-convergence --convergence-patience 5 --convergence-check-every 10 --convergence-min-steps 50 --exclude-invariant --platform cpu`

The fixed MAP integration fixture predates the CLI convergence-default change
already in `HEAD` (patience 3, check every step,
minimum 10), whose shorter stopping point is 16 steps at likelihood
`-1074.174409853388`. Pin the fixture test and warmed benchmark to the previous
settings (patience 5, check every 10 steps, minimum 50) so they continue to
check the established reference without changing its values or tolerances. The
CLI default assertions also now reflect the 3/1/10 defaults already selected in
`HEAD`. The fixed likelihood/gradient and MAP references pass unchanged.

**Full-suite verification:** `python -m pytest` completed with 51 passed and 2
existing dependency deprecation warnings in 131.57 s. `python -m tombombadil
--help` and `git diff --check` also succeeded.

### 2026-09-30 — Per-step MAP reporting and result output

**Context:** MAINT-03, MAINT-05, and MAINT-06 were combined into one
implementation loop, with the user-requested renumbering of output logging to
MAINT-04, omega plotting to MAINT-05, and MAP/NUTS path separation to MAINT-06.

**Decision:** Record the objective at step 0 and after each optimizer update,
while keeping convergence decisions at their configured check intervals and
selecting the best replicate as before. Measure startup from the initial
objective evaluation through the first update and report later optimization
time separately; describe startup as including JAX compilation when it occurs.
Use `jax.value_and_grad` to share intermediate objective and gradient
evaluations, with a separate objective evaluation for the final state. Replace
final result printing with logging and always write existing MAP/NUTS result
formats using the `output` stem when no `--output-jax` stem is provided. Plot
per-site omega linearly from zero, with the upper limit at least 1 so the
omega=1 guide remains visible.

**Rationale:** Per-step values support the requested progress display and plot.
Sharing objective and gradient work limits the cost of this additional
reporting. A default output stem ensures MAP and NUTS estimates are retained
even when callers omit `--output-jax`.

**Consequences:** Focused tests passed (8 tests). The full suite passed (56
tests, 2 existing dependency deprecation warnings) in 171.30 s; the protected
likelihood/gradient and MAP numerical references remain unchanged. CLI help
and `git diff --check` passed. The warmed benchmark used `porB3_aligned.fasta`
(23 samples, 294 sites), CPU with 4 workers, one per-site MAP replicate,
500-step maximum, convergence patience 5, check every 10 steps, minimum 50,
and invariant sites excluded. The benchmark timing starts at `_run_replicates`
and includes progress reporting in the timed pass; it excludes alignment
loading, transform setup, plots, and output writing. Warm and timed durations
were 126.371255 s and 122.072659 s respectively for 210 steps, with final
objective `-1071.4185160607`. The timed result is about 20% above the prior
MAINT-02 timed result of 101.753761 s, while the warm result is about 28% above
the prior warm result of 99.071470 s. These are single runs and do not
establish a repeatable performance difference. No numerical references or
tolerances were changed.

# Blockers

None.

For each blocker, record:

- affected task
- problem
- information or action required to unblock it

Remove resolved blockers from this section once their resolution has been captured in the Session Log or Design Notes.

---

# Session Log

Keep entries concise. Record outcomes rather than a transcript of the work.

## 2026-09-29

**Task:** TEST-01–03 — Correctness and performance baselines

**Completed:**
- Added fixed likelihood/gradient and per-site MAP regression references.
- Replaced the missing codon-count input with a self-contained temporary FASTA.
- Documented `python -m pytest` in `README.md`.

**Changed:**
- Added `test/test_baselines.py`, `test/run_map_benchmark.py`, and the full MAP
  reference fixture; preserved the original `Testdiv` reference.

**Tests/benchmarks:**
- `python -m tombombadil --help` succeeded.
- `python -m pytest`: 40 passed, 2 dependency deprecation warnings, 194.69 s.
- MAP optimizer benchmark: 142.556440 s, 210 steps, including first-step JAX
  compilation. Input: `porB3_aligned.fasta` (23 samples, 294 codon sites), CPU,
  one replicate, per-site omega, 500 maximum steps, convergence enabled,
  invariant sites excluded.
- Benchmark command:
  `MPLCONFIGDIR=/private/tmp/tombombadil-phase1-map/mplconfig python -m test.run_map_benchmark --alignment porB3_aligned.fasta --omega-mode per-site --fit-method map --sample-it 500 --output-jax /private/tmp/tombombadil-phase1-map/porB3_map --fit-until-convergence --exclude-invariant --platform cpu`

**Decisions:**
- Numerical comparisons use `rtol=1e-6`, `atol=1e-8`; pytest dependency
  installation is deferred because mamba currently manages the environment.

**Remaining:**
- None for this task.

**Next:** Review the MAP execution path and select the first optimisation target;
wait for the user to request that task before starting.

## 2026-09-29 — MAINT-01 progress reporting

**Task:** MAINT-01 — Optimisation progress reporting

**Completed:**
- Added per-replicate tqdm progress for interactive terminals and checkpoint
  likelihood logs when stderr is redirected.
- Recorded objective history and saved a best-replicate-highlighted PDF by
  default, using the exact y-axis label `log-likelihood`.
- Documented default output paths and behavior in CLI help and `README.md`.

**Tests/benchmarks:**
- `python -m tombombadil --help` succeeded; `git diff --check` passed.
- Focused progress and convergence tests: 5 passed.
- `python -m pytest`: 44 passed, 2 existing dependency deprecation warnings,
  187.55 s. The likelihood/gradient and per-site MAP references remain within
  their existing tolerances.
- Warmed benchmark results and timing boundaries are recorded in Design Notes.

**Blockers:** None.

**Next:** Continue with MAINT-02 — CPU/thread configuration investigation; do
not begin until requested.

## 2026-09-29 — MAINT-02 CPU/thread configuration

**Task:** MAINT-02 — CPU/thread configuration

**Completed:**
- Set the CPU `--cpus` default to 4, reject non-positive values, and apply the
  requested `NPROC` before JAX backend initialization for MAP and both NUTS
  chain modes. CPU NUTS `pmap` also gets a matching host-device count while
  unrelated XLA flags are retained; GPU and TPU settings are unchanged.
- Documented the worker-setting semantics in CLI help and `README.md`.
- Added coverage for default/explicit values, invalid inputs, inherited
  environment precedence, MAP and sequential NUTS worker settings, pmap device
  flags, GPU/TPU behavior, and a fresh-process two-chain/four-device smoke run.
- Pinned the MAP regression test to its established convergence settings and
  updated stale CLI default assertions. Numerical references and tolerances
  were not changed.

**Tests/benchmarks:**
- `python -m pytest`: 51 passed, 2 existing dependency deprecation warnings,
  131.57 s. The likelihood/gradient and MAP reference tests passed.
- `python -m tombombadil --help` succeeded; `git diff --check` passed.
- A 30-step CPU probe completed in 29.17 s at `--cpus 1` (168.8% median CPU
  after 10 s) and 19.61 s at `--cpus 4` (358.6%).
- Warmed per-site MAP at 4 workers: 99.071470 s warm-up, 101.753761 s timed,
  210 steps, objective `-1071.4185160607003`.

**Blockers:** None.

**Next:** MAINT-03 — Compilation logging; wait for the user to request it.

## 2026-09-30 — MAINT-03/04/05 batched implementation

**Task:** MAINT-03, MAINT-04, and MAINT-05 — Compilation logging, consistent
output, and omega plot

**Completed:**
- Renumbered consistent output to MAINT-04, omega plotting to MAINT-05, and
  MAP/NUTS path separation to MAINT-06.
- Added per-step MAP likelihood history and reporting, compilation-inclusive
  startup timing, consistent final-result logging, default MAP/NUTS output
  files, and linear per-site omega plots.
- Fused intermediate objective and gradient calculations after the first
  per-step implementation increased the timed benchmark to 211.717 s.

**Tests/benchmarks:**
- Focused progress/output/plot tests: 8 passed.
- `python -m pytest`: 56 passed, 2 existing dependency deprecation warnings,
  171.30 s. Protected numerical references passed unchanged.
- `python -m tombombadil --help` succeeded.
- Final warmed benchmark: 126.371255 s warm and 122.072659 s timed, 210 steps,
  objective `-1071.4185160607`; configuration and timing boundary are in
  Design Notes.

**Blockers:** None.

**Next:** MAINT-06 — Separate MAP and NUTS execution paths; do not begin until
requested.

---

# Deviations

Record significant deviations from the planned approach or scope.

Defer project-level pytest installation metadata and lockfile changes; the active
environment is mamba-managed and already has pytest, per user direction.

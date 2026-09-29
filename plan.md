Read `AGENTS.md` before starting work.

This file is the authoritative record of project state, priorities, decisions, and handoff context.

Do not begin a backlog item merely because it appears in this file. Work only on the Current Task unless the requested work requires updating it.

# Current Status

**Phase:** Maintainable MAP optimisation
**Status:** Complete

MAINT-01 is complete. MAP progress and likelihood plots are enabled by default;
the full pytest suite passes and the protected numerical references are unchanged.

# Current Task

## MAINT-01 — Optimisation progress reporting

Add default progress reporting and likelihood history plotting to the MAP optimisation path. Use tqdm already installed in the active mamba environment; dependency installation or packaging changes are out of scope.

### Checklist

- [x] Warm the MAP benchmark in-process, then record a pre-change steady-state optimizer runtime: 130.472883 s after a 144.394336 s untimed warm-up.
- [x] Show MAP iteration progress by default, with current objective updates at recorded checkpoints and a useful non-TTY logging fallback.
- [x] Record objective history without changing optimizer updates, convergence decisions, replicate selection, or reference values.
- [x] Save a likelihood-versus-iteration PDF by default; label its y-axis exactly `log-likelihood` and highlight the best replicate when several are fitted.
- [x] When `--output-jax` is provided, save the plot alongside the existing mode-prefixed outputs; otherwise use `scalar_likelihood_plot.pdf` or `per_site_likelihood_plot.pdf` in the current directory.
- [x] Document the default progress and plot behavior in CLI help and `README.md`; add no CLI flag.
- [x] Add focused tests for progress/history and plot labeling/output naming, plus the MAP integration plot artifact assertion.
- [x] Record a warmed post-change benchmark using the same timing boundary and configuration: 133.801273 s after a 138.051521 s untimed warm-up.
- [x] Run the complete test suite and preserve the likelihood/gradient and MAP numerical references.
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

Continue with MAINT-02 — CPU/thread configuration investigation. Do not begin it automatically.

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

### MAINT-04 — Separate MAP and NUTS execution paths

The current `run_sampler` control flow between ML/MAP and NUTS is unclear and relies on an early return.

Separate these into explicit execution paths/functions.

Do not alter BlackJAX behaviour unless necessary for this separation.

### MAINT-05 — Consistent output/logging

Remove the mixture of `print` and logging for final output.

Use logging consistently.

Write final parameter estimates and dN/dS results to output files rather than printing them.

### MAINT-06 — Omega plot

Scale the omega plot from zero to the observed maximum.

Avoid scientific-notation labels on the y-axis where practical.

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

Investigate explicit parallelism for MAP/Adam optimisation.

The design should consider eventual workloads with approximately one million sites and potential GPU execution.

Avoid introducing parallelism until the existing CPU/thread behaviour is understood.

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

# Blockers

None currently recorded.

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

---

# Deviations

Record significant deviations from the planned approach or scope.

Defer project-level pytest installation metadata and lockfile changes; the active
environment is mamba-managed and already has pytest, per user direction.

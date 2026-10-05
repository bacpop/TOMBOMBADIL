Read `AGENTS.md` before starting work.

This file is the authoritative record of project state, priorities, decisions, and handoff context.

Do not begin a backlog item merely because it appears in this file. Work only on the Current Task unless the requested work requires updating it.

# Current Status

**Phase:** Likelihood optimisation
**Status:** Complete — PERF-02

PERF-02 is complete. `_gen_alpha_impl` now uses two static JAX loops while
preserving its column arithmetic. Five fresh-process trials reduced median warm
MAP runtime by 5.30% and startup by 50.52%; three larger-workload probes reduced
warm objective time by 46.61% and peak RSS by 21.31%. All fixed numerical
references and tolerances remain unchanged. The installed MAP confirmation and
full suite pass (77 tests, 10 subtests). Reproduction, rejected alternatives,
measurements, and limitations are in `test/perf02_results.md` and
`test/benchmarks/perf02_results.json`. PERF-03 remains queued and unstarted.

PERF-01 is complete. GTR construction now uses a compact static rate lookup,
masked omega/diagonal updates, and broadcast frequency scaling. The selected
implementation reduced median compilation-inclusive MAP startup by 12.83%,
with median warm MAP runtime 2.30% higher (within the agreed 5% limit). No
sustained-throughput or memory improvement is claimed. All protected references
remain unchanged. Benchmark commands, alternatives, raw measurements, and
limitations are in `test/perf01_results.md` and
`test/benchmarks/perf01_results.json`.

MAINT-01 and MAINT-02 are complete. The CPU worker count now configures MAP and
NUTS on the CPU backend; the full test suite and numerical references pass.
The MAINT-03, MAINT-04, and MAINT-05 batch is complete. MAP objective values
are reported at every step, compilation-inclusive startup time is separated
from later optimization time, result output uses logging and default files,
and the per-site omega plot uses a linear scale. Numerical references pass.
MAINT-06 is complete. Shared model preparation and MAP execution have dedicated
functions; `main()` prepares the model and dispatches directly to MAP or NUTS.
MAINT-07 is complete. CLI help is grouped by workflow, MAP uses `--max-it` with
early convergence enabled by default, and former CLI-owned function defaults
are explicit at call sites. MAINT-08, MAINT-09, and MAINT-10 are complete.
Alignment parsing, codon counting, and frequency estimation live in
`tombombadil/alignment.py`; approved helper names now describe their roles.
The JAXopt pytest warning is documented as coming from BlackJAX's eager
imports; the project has no direct JAXopt dependency. Numerical references
remain unchanged.

# Current Task

## PERF-02 — `_gen_alpha_impl`

Benchmark spectral reconstruction, diagonal frequency scaling, and column
normalisation against revision `4273181`, retaining the PERF-01 GTR code.
Require at least 5% faster median warm MAP runtime with no more than 5%
regression in compilation-inclusive startup, larger-workload runtime, or peak
RSS. Preserve all numerical references and existing JIT boundaries.

### Checklist

- [x] Freeze baseline and implement reproducible benchmark alternatives.
- [x] Gate alternatives on matrix, derivative, and protected reference checks.
- [x] Screen kernels and scalar/per-site objectives at 294 and 2,940 sites.
- [x] Compare finalists with five fresh-process MAP and three memory trials.
- [x] Select qualifying production changes and confirm installed performance.
- [x] Run protected tests, full suite, and diff checks.
- [x] Record results, decisions, blockers, Session Log, and Next Task.

### Guardrails

The likelihood/gradient unit test is a correctness oracle for subsequent work.

- It must not be removed, skipped, or weakened to accommodate implementation changes.
- Numerical differences are acceptable only within the documented floating-point tolerance.
- Changes outside tolerance must be investigated rather than accepted by changing the reference values.
- Reference values or tolerances may only be changed when there is an explicitly justified change to the intended model behaviour.

The MAP integration test is the primary end-to-end oracle for optimisation work.

Performance comparisons must use comparable inputs, configuration, hardware, and timing boundaries. Record enough information to make before/after measurements interpretable.

# Next Task

After PERF-02, continue with PERF-03 — Site vectorisation; do not start it automatically.

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

`python -m tombombadil --alignment porB3_aligned.fasta --omega-mode per-site --fit-method map --max-it 500 --output-jax porB3_map --exclude-invariant --convergence-patience 5 --convergence-check-every 10 --convergence-min-steps 50`

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

For MAP mode, replace `--sample-it` with `--max-it` and make early convergence
the default. `--fixed-iterations` opts out. `--sample-it` is removed; NUTS
continues to use `--num-samples`.

Remove function defaults for settings owned by the CLI parser. Keep defaults
for optional data and internal controls that have no CLI counterpart.

### MAINT-08 — Move I/O functions out of `__main__`

Move FASTA reading, codon counting, and alignment-derived codon frequency
estimation from `__main__` into `tombombadil/alignment.py`.

Consolidate the duplicate FASTA readers in `domains.py` on the shared parser.

### MAINT-09 — Improve function names

Review and rename unclear functions, including:

- `my_dirichlet_multinomial_logpmf` and its alternate implementation
- the log-density factory and MAP replicate runner
- likelihood transforms, codon-site likelihoods, the variable-site mask, and
  the positive parameter transform and its inverse

Names should describe their role rather than implementation history.

### MAINT-10 — Investigate the JAXopt pytest warning

JAXopt is deprecated. Confirm whether the warning comes from a direct project
dependency or an upstream import; remove a direct dependency if present, or
record the upstream cause when no project change is needed.

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
- [x] Approved maintainability work is complete.
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

### 2026-10-03 — Separate MAP and NUTS execution paths

**Context:** MAINT-06 asked to clarify the MAP/NUTS control flow without
changing BlackJAX sampling behavior or the MAP benchmark boundary. The wrapper
based dispatch was subsequently removed at the user's direction because the
CLI is the only production caller.

**Decision:** Extract shared transforms, mask creation, initial parameters, and
objective construction into `_prepare_model`. Keep MAP execution, result
selection, plots, uncertainty, and output in `run_map_optimizer`. Have
`main()` call `_prepare_model()` and dispatch directly to `run_map_optimizer`
or the unchanged `run_nuts_sampler`; remove `run_sampler()`. Preserve output
naming and the location and signature of `_run_replicates`. Clarify the
post-startup timing log label without changing its calculation.

**Rationale:** The CLI is the sole production entry point, so a general wrapper
with a duplicate call signature adds no value. Preparation is common to both
fitters, while their execution and outputs differ. The timing label describes
the already separate post-startup duration.

**Consequences:** Updated the existing MAP output test to call preparation and
MAP execution directly, and updated the NUTS default-stem test to exercise
`main()`. The full suite passed (56 tests, 2 dependency deprecation warnings)
in 187.61 s; protected numerical references remained unchanged. CLI help and
`git diff --check` passed. No new dispatch test was added and no performance
benchmark was needed because optimizer execution and the `_run_replicates`
timing boundary are unchanged.

### 2026-10-03 — Organize CLI options and remove duplicated defaults

**Context:** MAINT-07 requested clearer help sections, explicit MAP iteration
semantics, and a single source of truth for settings already owned by the CLI.

**Decision:** Order CLI groups from input/output and fitting/runtime through
shared model settings, MAP, NUTS, diagnostics, and other options. Use
`ArgumentDefaultsHelpFormatter` to display parser values. Make `--max-it` the
positive MAP iteration limit (default 500); enable existing early-convergence
settings by default and let `--fixed-iterations` opt out. Keep NUTS draws on
`--num-samples`, and remove `--sample-it` and `--fit-until-convergence` from
the CLI. Remove matching function defaults and require callers to pass values
explicitly, preserving each call site's former effective configuration.

**Rationale:** The CLI parser remains authoritative for user-facing settings,
and grouped help makes MAP and NUTS controls easier to distinguish. Required
function arguments expose omitted configuration at call sites without changing
their behavior.

**Consequences:** The protected likelihood/gradient test now spells out its
former effective `make_fn` configuration (`include_invariant=True`,
`aggregate="mean"`, `prior_mode="current"`, `estimate_eta=False`, jitter and
omega floor enabled, scalar omega). The MAP fixture continues to pin its
established 5/10/50 convergence settings. No reference value or tolerance is
changed.

### 2026-10-03 — Shared alignment input, clearer helper names, and JAXopt warning

**Context:** MAINT-08 and MAINT-09 were batched to extract alignment I/O and
clarify several ambiguous helper names. Pytest also reported that JAXopt is no
longer maintained.

**Decision:** Move FASTA parsing, codon counting/order, and empirical/F3x4
frequency estimation into `tombombadil.alignment`. Use its shared reader in
domain labeling and support gzip input through the same reader. Rename the
likelihood helpers to `dirichlet_multinomial_logpmf` and
`dirichlet_multinomial_logpmf_scipy_form`, `make_log_density_fn`,
`_run_map_replicates`, `prepare_likelihood_transforms`,
`make_variable_site_mask`, `positive_transform` and
`positive_transform_inverse`, and `codon_site_log_likelihood` and
`codon_site_log_likelihood_no_jitter`. Preserve their calculations and update
all callers, tests, and the benchmark wrapper. Leave the JAXopt warning visible
and make no environment or dependency changes.

**Rationale:** BlackJAX 1.3 eagerly imports its L-BFGS and Pathfinder utilities,
which import JAXopt while pytest collects the package. The active environment
has JAXopt 0.8.4 although BlackJAX metadata declares `jaxopt<=0.8.3`;
`pyproject.toml` has no direct JAXopt requirement. This warning therefore does
not identify a project dependency to remove.

**Consequences:** Added alignment tests for wrapped and multi-record FASTA,
plain/gzip parity, codon counts, and domain labels using the shared reader. The
focused run passed 55 tests and 6 subtests; the full suite passed 64 tests and
10 subtests in 174.18 seconds. The protected likelihood/gradient and pinned
MAP references passed unchanged. Both existing dependency warnings remain:
the JAXopt warning and a `fastcore` asyncio deprecation warning. CLI help and
`git diff --check` passed. No performance benchmark was needed because MAP
execution and its timing boundary did not change.

### 2026-10-04 — PERF-01 benchmark boundaries and numerical screening

**Context:** GTR candidates can have substantially different kernel timings while
whole-model execution is dominated by other operations. The optimizer currently
uses `jax.value_and_grad(fn)` with existing inner JIT boundaries.

**Decision:** Preserve the original GTR source at revision
`d22f582c9ecf1c438254824e6c6f227db98b231d` as the benchmark baseline, loaded from
git or an explicitly exported source file. Screen all candidates against an
independent genetic-code matrix/derivative oracle. Run whole-objective benchmarks
with the production JIT boundaries; expose an additional outer JIT only as an
explicit diagnostic option.

**Rationale:** Initial diagnostic runs with an extra whole-objective JIT showed
gradient differences between algebraically equivalent candidates at symmetric
parameters, even though likelihoods matched. A one-site reproduction also showed
that adding this JIT changes the baseline's own gradients beyond the protected
tolerance. The actual optimizer path produced matching baseline/candidate matrices,
likelihoods, and gradients. Introducing an outer JIT would mix PERF-04 into this
task and obscure the intended GTR comparison.

**Consequences:** Retain the diagnostic evidence separately and repeat screening
using the actual optimizer boundary. Do not change reference values, tolerances,
eigensolver behavior, or JIT coverage. Treat the outer-JIT gradient sensitivity as
a correctness concern to investigate before any future PERF-04 change, including
other fully jitted consumers such as the NUTS step. This MAP-focused campaign did
not add sampler-specific numerical comparisons.

### 2026-10-04 — PERF-01 selected GTR representation

**Context:** Twelve kernel variants and two full-MAP finalists were compared
against the original implementation, with the same CPU/four-worker float64 setup.

**Decision:** Retain the combined lookup implementation: decode one compact static
codon substitution table into rate indices and a nonsynonymous mask; gather the
six rates plus a zero sentinel; use masked omega scaling and diagonal replacement;
replace diagonal-frequency matrix products with broadcasting. Preserve GTR helper
signatures and profiler annotation. No likelihood, optimizer, or JIT boundaries
change. The static table matches all 526 original entries, including 392 omega
entries, and is checked against an independent genetic-code oracle.

**Rationale:** Across three valid fresh-process MAP comparisons per variant,
median compilation-inclusive startup fell from 3.772 s to 3.288 s (12.83%).
Median warm MAP time increased from 88.843 s to 90.883 s (2.30%), within the
planned 5% limit for accepting a startup improvement of at least 10%. The
2,940-site warm per-site objective/gradient probe increased 4.75%; first-call
time decreased 19.68%. Frequency broadcasting alone did not establish a benefit.

**Consequences:** This is a startup optimization, not a demonstrated improvement
to sustained MAP throughput. Separate three-process large-input memory probes had
median peak RSS 11.020 GB (baseline) and 11.480 GB (selected), about 4.17% higher;
the longer six-call objective probes peaked at 11.780 and 11.722 GB respectively.
Do not claim a memory improvement. Raw timings, numerical comparisons, rejected
variants, and reproduction commands are retained with the benchmark report.
Two initial pilots were excluded and replaced because the runner initially
configured logging too late to capture startup timing and enable progress logs.

### 2026-10-05 — PERF-02 measurement protocol

**Decision:** Compare broadcast spectral scaling, matrix multiplication/einsum
reconstruction, static-loop fallback, frequency broadcasting, and vectorised
column normalisation individually and in promising combinations. Use exact
baseline likelihood source from `4273181` and current GTR for every variant.

CPU/four workers/float64, no persistent compilation cache; synchronised kernel
samples (five batches of 100), actual optimizer value-and-gradient boundary
(no extra outer JIT), scalar/per-site 294/2,940-site probes. Full MAP uses one
warm-up plus one timed pass with 500 maximum steps and convergence 1e-6/5/10/50.
Use three rotated fresh-process MAP trials and three larger per-site memory
trials (first plus five warm evaluations); extend borderline comparisons to
five. Throughput must improve at least 5%, with startup/larger-runtime/RSS
regressions capped at 5%. Startup-only wins do not qualify. Preserve protected
references, clipping semantics, eigensolver, jitter, and JIT boundaries.

**Screening:** Initial matmul/einsum reconstruction and broadcast normalisation
fail the unchanged protected gradient reference at symmetric initial parameters.
They are excluded, not accommodated by changing references. Scale-only,
frequency-only, and static reconstruction loops pass the one-site check.
The real per-site objective rejects spectral broadcasting and all combinations
containing it (gradient differences up to 0.82); explicit-reduction vmap also
fails the one-site check. Static normalization loops pass. Both static loops
together, with and without frequency broadcasting, pass the real per-site
gradient screen. The initial both-loops probe improves warm
objective time by 8%. Five full MAP trials select both static loops: median
warm runtime 86.538 to 81.955 s (5.30% improvement), startup 3.157 to 1.562 s
(50.52%), larger-probe runtime 7.547 to 4.030 s (46.61%), peak RSS 11.642 to
9.161 GB (21.31%). Production now uses these two loops with original spectral
and frequency arithmetic. Installed numerical checks and MAP confirmation pass: startup 1.690 s,
warm MAP 83.049 s, 210 steps, unchanged fixed outputs. Full suite: 77 tests and
10 subtests pass, with the same two dependency warnings. CLI/benchmark help and
diff checks pass.

**Decision:** Retain the two static loops. The measured warm gain (5.30%) passes
the agreed 5% threshold after extending from three to five trials; individual
ranges overlap, and no valid trial was excluded. Both startup and memory improve
substantially. A partial-unrolling experiment passed numerical screens but its
90.950 s MAP pilot offered no warmed-throughput benefit and was rejected.

**Consequences:** Public interfaces, profiler annotations, JIT boundaries,
eigensolver, jitter, priors, clipping semantics and optimizer controls are
unchanged. Preserve the benchmark alternatives and numerical failures for
future investigation of gradient sensitivity during PERF-04. PERF-03 onward
and NUTS optimisation remain out of scope. No blockers remain.

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

## 2026-10-03 — MAINT-06 MAP/NUTS execution separation

**Task:** MAINT-06 — Separate MAP and NUTS execution paths

**Completed:**
- Extracted shared model preparation into `_prepare_model` and MAP execution,
  result handling, plotting, uncertainty, and outputs into `run_map_optimizer`.
- Moved model preparation and the MAP/NUTS branch into `main()` and removed
  the redundant `run_sampler()` wrapper. Preserved default filenames, NUTS
  sampling, and the `_run_replicates` timing boundary.
- Clarified the existing post-startup optimization timing log label; timing
  behavior is unchanged.

**Tests:**
- `python -m pytest`: 56 passed, 2 existing dependency deprecation warnings,
  187.61 s. Protected likelihood/gradient and MAP references passed unchanged.
- `python -m tombombadil --help` succeeded; `git diff --check` passed.
- Existing test callers now use the direct MAP functions or `main()`; no new
  dispatch test was added.

**Blockers:** None.

**Next:** MAINT-07 — CLI organisation; do not begin until requested.

## 2026-10-03 — MAINT-07 CLI organisation

**Task:** MAINT-07 — CLI organisation

**Completed:**
- Grouped CLI help by input/output, fitting method, runtime, shared model,
  MAP/Optax, NUTS/BlackJAX, fixed-parameter diagnostics, and other options.
- Replaced MAP `--sample-it` with positive `--max-it` (default 500), enabled
  early convergence by default with the existing 3/1/10 settings, and added
  `--fixed-iterations`. NUTS continues to use `--num-samples`; removed flags
  are rejected.
- Removed CLI-owned function defaults and updated repository callers and
  examples to pass the prior effective values explicitly. Kept optional data
  and internal-control defaults.
- Recorded the protected likelihood/gradient function's former settings
  explicitly and kept MAP fixture convergence pinned to 5/10/50.

**Tests:**
- Focused parser, helper, progress, and CPU configuration tests: 56 passed,
  10 subtests passed.
- `python -m pytest test/test_baselines.py -q`: 2 passed, including the fixed
  likelihood/gradient oracle and pinned MAP integration test.
- `python -m pytest -q`: 60 passed, 10 subtests passed, with 2 existing
  dependency deprecation warnings; completed in 180.26 s.
- `python -m tombombadil --help` and `git diff --check` succeeded.

**Blockers:** None.

**Next:** MAINT-08 — Move I/O functions out of `__main__`; do not begin until
requested.

## 2026-10-03 — MAINT-08/09/10 alignment and naming batch

**Task:** MAINT-08, MAINT-09, and MAINT-10 — Alignment input extraction,
function naming, and JAXopt warning investigation

**Completed:**
- Moved FASTA readers, codon counting/order, and alignment-derived frequency
  estimation to `tombombadil/alignment.py`. Updated the CLI, diversity plot,
  domain labeling, tests, and benchmark caller.
- Renamed the reviewed likelihood, MAP replicate, likelihood transform,
  variable-site mask, and positive parameter transformation functions and
  updated their callers.
- Traced the pytest JAXopt warning to BlackJAX 1.3 importing its optimizer and
  Pathfinder utilities. The active environment has JAXopt 0.8.4 while
  BlackJAX declares `jaxopt<=0.8.3`; `pyproject.toml` has no direct JAXopt
  dependency. No dependency or warning-filter changes were made.

**Tests:**
- Focused input, existing helper, progress/output, and baseline tests:
  55 passed, 6 subtests passed.
- `python -m pytest -q`: 64 passed, 10 subtests passed, with the same 2
  dependency deprecation warnings; completed in 174.18 s. Protected numerical
  references remained unchanged.
- `python -m tombombadil --help`, `python -m compileall -q tombombadil test
  plot_codon_diversity.py`, and `git diff --check` succeeded.

**Blockers:** None.

**Next:** PERF-01 — GTR operations; wait for the user to request it.

## 2026-10-04 — PERF-01 GTR benchmarks and startup optimisation

**Task:** PERF-01 — GTR operations.

**Completed:**
- Compared twelve kernel variants, screened scalar/per-site objective gradients
  at 294 and 2,940 sites, and ran three valid fresh-process MAP comparisons for
  baseline and each of two finalists. Excluded/replaced two logging-affected
  pilots; repeated baseline/selected large-input memory probes three times.
- Replaced GTR scatter assembly and redundant static tables with one compact
  substitution table, rate lookup, masked updates, and frequency broadcasting.
  Kept helper signatures, numerical semantics, and profiler annotation.
- Added independent genetic-code matrix, structural-zero, stationarity,
  diagonal replacement, and reverse-derivative checks. Added reusable benchmark
  variants/runner and retained reproducible measurements with the report.

**Tests/benchmarks:**
- `python -m pytest -q`: 72 passed, 10 subtests passed, 2 existing dependency
  warnings, 120.13 seconds. Protected likelihood/gradient and MAP references,
  including all 294 omega estimates, passed unchanged.
- CLI help, `python -m compileall -q tombombadil test`, and `git diff --check`
  passed. Existing user-generated CSV/PDF outputs were left untouched.
- Median MAP startup: 3.772 -> 3.288 s (12.83% shorter). Median warm MAP:
  88.843 -> 90.883 s (2.30% longer), satisfying the agreed startup criterion.
  All accepted runs converged in 210 steps to the fixed objective.
- A separate installed-production confirmation measured 3.238 s startup and
  90.732 s warm MAP, with all reference estimates passing.
- Separate large-input memory medians were 11.020 -> 11.480 GB peak RSS;
  longer objective probes were both approximately 11.7 GB. No memory reduction
  or sustained MAP speedup is claimed. Details and individual samples are in
  `test/perf01_results.md` and `test/benchmarks/perf01_results.json`.

**Decisions/follow-up:**
- Retain combined lookup under the pre-agreed startup improvement criterion;
  reject the isolated frequency, einsum, and six-mask alternatives.
- Adding an outer JIT exposed gradient sensitivity even in the original
  baseline. Corrected the benchmark to match the actual optimizer boundary;
  recorded the diagnostic separately for future JIT/numerical investigation.
  No JIT coverage, eigensolver, reference values, or tolerances were changed.

**Blockers:** None for PERF-01.

**Next:** PERF-02 — `_gen_alpha_impl`; do not start automatically.


## 2026-10-05 — PERF-02 complete

**Task:** Benchmark and optimise `_gen_alpha_impl` using the approved plan.

**Completed:**
- Updated Current Task, checklist, and protocol before implementation.
- Added exact-baseline benchmark loading, nineteen alternatives, numerical
  screening, kernel/objective/MAP/memory modes, metadata, and recorded results.
- Added five independent alpha-matrix/derivative/normalization tests.
- Rejected numerically incompatible contractions, broadcasting, and vmap;
  retained the two static loops after five MAP trials and three memory trials.
- Warm MAP median 86.538 to 81.955 s (5.30% faster); startup 3.157 to 1.562 s
  (50.52% lower); large warm objective 7.547 to 4.030 s (46.61% lower); peak RSS
  11.642 to 9.161 GB (21.31% lower).
- Installed MAP confirmation: 1.690 s startup, 83.049 s warmed optimizer,
  210 steps; objective, GTR and all omega references pass unchanged.
- `python -m pytest -q`: 77 passed, 10 subtests passed, two existing dependency
  warnings, 115.14 s. CLI/benchmark help, compilation and diff checks pass.
- Updated `test/perf02_results.md`, `test/benchmarks/perf02_results.json`,
  Current Status, Design Notes, blockers, and this handoff.

**Decisions:** Extend borderline/variable MAP comparisons to five trials; use
all valid trials and unrounded medians. Reject partial unrolling after its slow
MAP pilot. Keep references, tolerances, model behaviour and JIT scope unchanged.

**Blockers:** None.

**Next:** PERF-03 — Site vectorisation; do not start automatically.

---

# Deviations

Record significant deviations from the planned approach or scope.

Defer project-level pytest installation metadata and lockfile changes; the active
environment is mamba-managed and already has pytest, per user direction.

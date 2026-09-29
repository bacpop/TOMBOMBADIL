Read `AGENTS.md` before starting work.

This file is the authoritative record of project state, priorities, decisions, and handoff context.

Do not begin a backlog item merely because it appears in this file. Work only on the Current Task unless the requested work requires updating it.

# Current Status

**Phase:** Establish baseline and regression tests  
**Status:** Not started

No implementation changes have yet been recorded in this plan.

# Current Task

## Establish correctness and performance baselines

Create the tests and reference measurements needed to protect correctness during subsequent refactoring and optimisation.

### Checklist

- [ ] Confirm `python -m tombombadil --help` runs successfully.
- [ ] Add a unit test covering likelihood and gradient calculations.
- [ ] Record the expected likelihood and gradient reference values.
- [ ] Add a MAP integration test equivalent to:

  `python -m tombombadil --alignment porB3_aligned.fasta --omega-mode per-site --fit-method map --sample-it 500 --output-jax porB3_map --fit-until-convergence --exclude-invariant`

- [ ] Record reference likelihood, omega estimates, and GTR parameter estimates.
- [ ] Define and document numerical tolerances for the reference values.
- [ ] Measure and record the baseline runtime of the MAP optimisation step.
- [ ] Confirm unit and integration tests pass.
- [ ] Update Current Status, Design Notes, Session Log, and Next Task.

### Guardrails

The likelihood/gradient unit test is a correctness oracle for subsequent work.

- It must not be removed, skipped, or weakened to accommodate implementation changes.
- Numerical differences are acceptable only within the documented floating-point tolerance.
- Changes outside tolerance must be investigated rather than accepted by changing the reference values.
- Reference values or tolerances may only be changed when there is an explicitly justified change to the intended model behaviour.

The MAP integration test is the primary end-to-end oracle for optimisation work.

Performance comparisons must use comparable inputs, configuration, hardware, and timing boundaries. Record enough information to make before/after measurements interpretable.

# Next Task

Review the MAP execution path and identify the first optimisation/refactoring target using the correctness tests and performance baseline established by the Current Task.

Do not begin this task until the Current Task is complete.

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

Add progress reporting for MAP optimisation.

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

- [ ] Its checklist is complete.
- [ ] Relevant existing tests pass.
- [ ] New tests required by the task pass.
- [ ] Protected numerical tests remain within their documented tolerances.
- [ ] Any relevant performance comparison has been recorded.
- [ ] Important implementation decisions are recorded in Design Notes.
- [ ] Blockers or unresolved issues are recorded.
- [ ] Current Status reflects the repository state.
- [ ] Session Log records the work performed.
- [ ] Next Task provides a clear handoff.

## Project

The project is complete when:

- [ ] `python -m tombombadil --help` runs successfully.
- [ ] The required unit and integration tests exist.
- [ ] All required tests pass.
- [ ] Numerical correctness guardrails are satisfied.
- [ ] Approved maintainability work is complete.
- [ ] Approved optimisation work has been benchmarked and accepted or rejected based on evidence.
- [ ] `plan.md` accurately reflects the final repository state.

---

# Design Notes

Record decisions that future sessions need to understand.

Use entries of the form:

### YYYY-MM-DD — Short decision title

**Context:** Why a decision was needed.

**Decision:** What was chosen.

**Rationale:** Why this option was chosen.

**Consequences:** Important implications or follow-up work.

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

## YYYY-MM-DD

**Task:** `<task ID or Current Task>`

**Completed:**
- ...

**Changed:**
- ...

**Tests/benchmarks:**
- ...

**Decisions:**
- ...

**Remaining:**
- ...

**Next:** ...

---

# Deviations

Record significant deviations from the planned approach or scope.

None currently recorded.

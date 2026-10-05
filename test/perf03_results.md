# PERF-03: site vectorisation and reverse-mode memory

Baseline: `6de8aa61c7b605701cb97680385cddc3d2f6a8d5`, including PERF-01/02.
The exact baseline objective is loaded from `sample.py` at that revision.
Benchmark alternatives live entirely under `test/`.

## Decision and acceptance rule

**Decision: retain the production `vmap`.** No candidate meets the agreed gate.
Adoption requires that an alternative improve median warmed objective/gradient runtime by at least 10% or peak process
RSS by at least 20%, with **no runtime regression** in startup, warmed objectives,
or small MAP, and no other RSS regression above 5%. The primary comparison is
the largest workload completed by both implementations within the resource budget.
Use three fresh processes, extending uncertain comparisons to five; retain all
valid trials and use unrounded medians. Capacity improvements alone do not qualify.

## What the graph shows

The current `vmap` already shares rates, frequency transforms and neutral GTR
construction across sites. Its mapped arguments are omega and count columns.
The expensive storage is in reverse mode: graph inspection reports saved
`float64[61, N, 61, 61]` arrays, among other intermediates. Ordinary `lax.map`
chunking retains those intermediates across *all* chunks. Checkpointing a whole
chunk removes that storage but recomputes the chunk during the backward pass.

Alternatives preserve site-model arithmetic, eigen jitter, the loss order,
masking, reduction and one full-input prior. Whole-site `vmap`, sequential map,
and batches 16/64/256/1024 are included. Checkpointed chunks of the best two
screened sizes (64/256) are investigated separately. The candidate policy uses
the original path for scalar omega and for N <= max(512, batch size). `--force`
and the numerical-check mode exercise chunks even on small inputs. There is no
outer objective JIT, new application CLI option, GPU or NUTS change.

## Reproduction

Use the active mamba environment from the repository root. The supervisor uses
`psutil` (already available in the benchmark environment) to monitor its fresh
worker process. Each probe has a 12 GiB RSS limit and 600 second timeout.
Uncheckpointed oversized inputs are conservatively skipped before execution.
All commands below create scratch reports; no application output files are touched.

```sh
export MPLCONFIGDIR=/tmp/perf03-mpl
export XDG_CACHE_HOME=/tmp/perf03-cache
python -m test.run_site_benchmark --variant baseline --mode inspect --output /tmp/perf03/baseline-inspect.json
python -m test.run_site_benchmark --variant batch256 --mode check --output /tmp/perf03/batch256-check.json
python -m test.run_site_benchmark --variant baseline --site-repeat 10 --output /tmp/perf03/baseline-10-initial-1.json
python -m test.run_site_benchmark --variant checkpoint64 --site-repeat 10 --output /tmp/perf03/checkpoint64-10-initial-1.json
```

Repeat timing commands in fresh processes with distinct output paths; run them
serially and alternate order. The ladder uses `--site-repeat 1/10/30/100`, giving
294/2,940/8,820/29,400 sites. `--params fitted` uses the fixed MAP fixture's global
parameters and tiled omega vector. `--omega-mode scalar` is an unchanged control.
`--mode kernels` measures forced site-loss forward and reduced-gradient calls;
`--mode map` supports a reproduction comparison with a complete warmup MAP
followed by a timed full MAP using 500/5/10/50 convergence settings, checking the
fixed reference.

Before JAX initialization, workers select CPU, float64, NPROC=4 and disable the
compilation cache. Objective and memory modes use the actual optimizer boundary,
`jax.value_and_grad(fn)` without outer JIT, one first call and five synchronized
warm evaluations. RSS includes imports, compilation and those same calls;
macOS `ru_maxrss` is bytes. Reports retain software/platform/source/input hashes,
raw timings, all gradient leaves, monitored memory and process exit status.

## Numerical validation

The protected likelihood/gradient and MAP fixture remain unchanged. Checks compare
ordered losses, objectives and every gradient leaf with rtol=1e-6, atol=1e-8,
rejecting non-finite values. Cases cover nonuniform frequencies, asymmetric rates,
omega floor boundaries, jitter choices, fixed/estimated eta, sum/mean, all prior
modes and masks including all-masked data. Sizes B-1/B/B+1 and multiple chunks with
a remainder exercise the candidate bodies directly.

`check_repeated_report` validates large repeated per-site data using a small baseline
reference: subtract the small-input prior, multiply the data objective and shared
parameter gradients, tile per-site omega gradients, then add the full-input prior
once. Thus validation need not allocate the large baseline AD graph. Lightweight
analytic-model tests independently check loss ordering, scalar policy, cutoff,
shared gradients and remainder handling.

All 35 per-site objective probes pass the repeated-input identity (maximum
absolute difference across values/gradients 8.19e-12); all eight scalar probes
match their same-sized frozen baseline exactly. Thirty forced model cases also
pass ordered-loss and objective/gradient comparisons. `python -m pytest -q`:
**79 passed, 28 subtests passed, two existing dependency warnings**, 112.86 s.
This includes the unchanged fixed likelihood/gradient and full MAP references.
Benchmark help, Python compilation and `git diff --check` also pass.

An additional scalar diagnostic exposes a **pre-existing numerical sensitivity**:
repeated-input gradient scaling fails in the frozen scalar baseline at the
symmetric initial parameters for 2,940/8,820/29,400 sites. For example, at 2,940
sites the baseline alpha gradient is 128.253868, while the small-input identity
predicts 129.248044 after accounting for priors. Scalar data-value scaling passes.
The archive retains all six failing diagnostic records (baseline and unchanged
candidate paths). Scalar regression comparisons use the full-size frozen baseline
at the original tolerance; the per-site identity remains an enforced gate.
No reference or tolerance was changed, and no scalar fix is claimed. Investigate
this sensitivity before accepting compilation changes in PERF-04; this experiment
does not establish its exact numerical cause.

## Measurements and limitations

Five fresh-process trials at 2,940 sites, standard initial parameters:

| Implementation | First call (s) | Warm objective + gradient (s) | Peak RSS (GiB) | Decision |
| --- | ---: | ---: | ---: | --- |
| Existing vmap | 6.352 | 5.272 | 6.514 | Retain |
| Batch 256 | 6.703 | 5.232 | 6.377 | Only 0.76% warm gain; startup +5.52% |
| Checkpoint 64 | 6.599 | 5.998 | 1.070 | RSS -83.57%; warm runtime +13.76% |
| Checkpoint 256 | 6.750 | 6.202 | 1.654 | RSS -74.61%; warm runtime +17.63% |

Values are medians across fresh processes; each warm sample is itself the median
of five synchronized calls. Ratios use unrounded values. Trial ranges and every
individual timing are in `benchmarks/perf03_results.json`. Earlier one-process
screens of ordinary batches 16/64/1024 were slower than the baseline; batch 256
was the strongest ordinary candidate. The best two screened sizes also determined
the checkpoint experiments. All initial 2,940-site gradients match within
2.85e-13 absolute error.

Fitted-parameter control probes at 2,940 sites preserve the objective/gradients:
one-process warm medians are 5.063 s (baseline), 4.575 s (batch 256), and 6.404 s
(checkpoint 64). These are diagnostic controls, not repeated acceptance trials;
the primary initial-parameter gate determines selection. The unchanged 294-site
policy also passes fitted controls. Forced 294-site forward/gradient kernel
medians are 0.253/0.388 s for vmap, 0.412/0.650 s for sequential map,
0.393/0.624 s for batch 256, and 0.374/0.746 s for checkpoint 64.

Capacity probes (one process each, still five warmed calls):

| Sites | Checkpoint-64 warm median (s) | Peak RSS (GiB) | Baseline per-site |
| ---: | ---: | ---: | --- |
| 8,820 | 17.017 | 1.052 | Preflight skipped |
| 29,400 | 56.551 | 0.888 | Preflight skipped |

The baseline's saved `[61, N, 61, 61]` float64 array alone exceeds 12 GiB at
8,820 sites, so these larger baselines were deliberately not launched. Their
preflight estimates/skips are retained in the archive. These are capacity results,
not relative speedups. RSS is a process high-water mark; remainder size,
compilation and allocator behaviour mean these single-process values need not
increase monotonically. Scalar controls complete the entire ladder; at 29,400
sites their peak RSS is about 8.49 GiB with both variants.

All 58 reports, including graph inspections, numerical checks, kernel diagnostics,
controls and skipped probes, are retained in the archive. No timeouts or RSS-limit
terminations occurred. Since all candidates fail the primary gate, further full
MAP candidate comparisons and production installation were unnecessary. The full
protected suite passed. No production model, optimizer, API or reference changed.

Repeated PorB3 inputs test scaling of this model;
they do not represent a new independent biological dataset. No performance claim
extends to other hardware, JAX versions, accelerator backends or NUTS.

To rebuild the archive and validate objective/gradient results against repeated
small-input identities after running the probes:

```sh
python -m test.summarize_site_benchmark /tmp/perf03 --output test/benchmarks/perf03_results.json
```

The archive command expects the standard PorB3 294-site baseline controls for
every parameter/omega configuration represented in the directory. It validates
all completed objective reports before writing the archive, using full-size
baselines for scalar regression and retaining the separate scalar identity
diagnostics. The archive contains 43 objective checks and their reference files.

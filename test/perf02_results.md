# PERF-02: alpha-matrix benchmark campaign

Selected **two static `jax.lax.fori_loop` loops** in `_gen_alpha_impl`.
Five fresh-process comparisons reduced median warm MAP time by **5.30%**,
startup by **50.52%**, larger-workload warm objective time by **46.61%**, and
peak RSS by **21.31%**. Protected references and tolerances are unchanged.

The exact baseline likelihood source is pinned to `4273181fae048963e9ad60fde6d2a86eeb925f75`; every candidate
uses the same post-PERF-01 GTR implementation. Production does not import any
benchmark alternative.

## Reproduction

Use the active mamba Python from the repository root. Set `MPLCONFIGDIR` and
`XDG_CACHE_HOME` to writable temporary directories. Run one process at a time.
The runner configures CPU, four workers, float64, and disables persistent JAX
compilation caching before JAX imports.

```sh
python -m test.run_alpha_benchmark --variant baseline --mode check \
  --output /tmp/perf02/check-baseline.json
python -m test.run_alpha_benchmark --variant loops --mode kernels \
  --output /tmp/perf02/kernels-loops.json
python -m test.run_alpha_benchmark --variant baseline --mode objective \
  --omega-mode per-site --site-repeat 10 \
  --output /tmp/perf02/objective-per-site-10-baseline.json
python -m test.run_alpha_benchmark --variant baseline --mode map \
  --output /tmp/perf02/map-baseline-1.json
python -m test.run_alpha_benchmark --variant baseline --mode memory \
  --site-repeat 10 --output /tmp/perf02/memory-baseline-1.json
```

Export the original `likelihood.py` and pass `--baseline-source` if baseline git
history is unavailable. The exported baseline must be from the pinned revision.
Candidates import the
current checkout's GTR implementation; reproduce the historical measurements
with the unchanged GTR from that revision (its hash is in the JSON report).
`test.alpha_benchmark_variants.VARIANTS` lists all alternatives. `scale` broadcasts
the spectral diagonal; `loop` changes reconstruction to a static loop;
`normalize-loop` changes normalization to a static loop; `loops` changes both.
`loops-unroll4` partially unrolls both static loops by four.
`frequency` broadcasts diagonal frequency factors. Hyphenated combinations
apply the named operations, while `combined-*` also includes broadcast
normalization. Matmul/einsum/vmap are alternative reconstruction operations.

## Controls and acceptance

Numerical screening runs separately from timings: screening must not warm
compilation caches in timed processes. Compare full matrices and weighted-output
reverse derivatives for uniform/nonuniform frequencies, omega 0.1/1/3, and both
jitter settings. The unchanged protected likelihood/gradient test gates each
candidate before expensive screening. The full MAP runner also checks the fixed
objective and all final parameters before writing a successful report. Failed candidates remain reproducible in
the benchmark module, but cannot be selected for production.

Kernel timings use five batches of 100 synchronized evaluations, with input
arrays on device. The first invocation is recorded separately; subsequent omega
values reuse the compiled shape. Kernel reverse derivatives include GTR
construction. The objective boundary is exactly `jax.value_and_grad(fn)`, without
an added outer JIT. Scalar/per-site omega probes use 294 and 2,940 sites. Each
objective and memory probe uses one initial plus five warm evaluations.

Full MAP uses `_run_map_replicates`, with a complete warm-up followed by a timed
pass from the same parameters. Keep progress logging enabled for the timed pass.
The workload is per-site omega, uniform frequencies, estimated eta, summed
objective, `stan_unconstrained` prior, invariant sites excluded, one replicate,
500 maximum iterations, tolerance 1e-6, patience 5, check interval 10, minimum
50 steps. Loading, preparation and output plotting are outside the timing
boundary. Startup is compilation-inclusive, not pure compilation time.

Compare three fresh-process trials per finalist and baseline with rotated order.
Memory trials use separate processes and the 2,940-site per-site objective;
process peak RSS includes imports, compilation and execution, not just JAX
allocations. Report medians and ranges with macOS bytes/Linux KiB normalized.

Require at least 5% faster median warm full MAP, no more than 5% regression in
startup, larger-workload warm runtime, or peak RSS, and unchanged numerical
references. Extend borderline or inconsistent comparisons to five trials.
Startup-only improvements do not qualify. Retain baseline if nothing qualifies.
JIT scope, eigensolver, jitter, optimizer settings and model behaviour are fixed.

## Numerical findings and candidate selection

Nineteen alternatives were evaluated. Matmul/einsum reconstruction, broadcast
normalisation, explicit-reduction vmap reconstruction, and their combinations
fail the protected symmetric-start gradient reference. In the initial
combined-matmul probe, asymmetric-screen values and derivatives matched, but the
protected gradient differed by up to 1.18e-5 at unchanged rtol=1e-6/atol=1e-8.

The real per-site objective additionally rejects spectral column broadcasting,
including combinations that passed the one-site reference. Its largest initial
gradient discrepancy was about 0.82. The real optimizer boundary is therefore an
essential numerical gate. No reference, tolerance, eigensolver, jitter, or JIT
boundary was changed to accommodate any candidate. The previously documented
outer-JIT sensitivity remains a separate PERF-04 investigation.

Static reconstruction and normalisation loops preserve the protected and
real-objective gradients. Both loops together improve the initial 294-site warm
objective probe by 8% and the 2,940-site probe by 46%. Adding frequency
broadcasting regresses the large scalar probe by 6% and does not improve the
small per-site probe. The simpler two-loop candidate proceeds to repeated MAP
and memory comparisons.

Partial unrolling (`unroll=4`) was screened after the first MAP pairs were
inconsistent. It passed numerical checks and improved kernel gradient timing,
but its MAP pilot took 90.950 s, offering no warmed-throughput benefit. Its
larger objective also used more time and memory than default static loops
(6.075 s, 11.094 GB versus 4.134 s, 9.064 GB in screening). This exploratory
variant was rejected after the pilot; that single run is not a repeated
performance estimate.

Five new tests independently check full matrices against NumPy, weighted-output
derivatives against finite differences for all six rates/omega/scale, both
jitter settings, nonuniform frequency orientation, and diagonal/negative/zero/
small-positive behaviour. Existing fixed likelihood/gradient and MAP tests
remain unchanged.

## Kernel screening

All timings are microseconds: the median of the twelve frequency/omega/jitter case medians.

| Candidate | Forward | Reverse derivative |
|---|---:|---:|
| baseline | 1144.26 | 1750.72 |
| frequency | 1201.38 | 1872.72 |
| loop-frequency | 923.45 | 1741.00 |
| loop | 991.65 | 1887.90 |
| loops-frequency | 794.68 | 1470.62 |
| loops-unroll4 | 845.03 | 1397.74 |
| loops | 882.16 | 1596.53 |
| normalize-loop | 1030.61 | 1666.42 |
| scale-frequency | 1181.40 | 1718.64 |
| scale-loop-frequency | 932.15 | 1738.99 |
| scale-loops-frequency | 789.18 | 1420.18 |
| scale | 1140.97 | 1714.56 |

Spectral-scaling candidates in this table subsequently failed the real-objective gradient gate. Kernel speed alone does not qualify a candidate.

## Objective screening

Warm value-and-gradient seconds, median of five evaluations after the first call.

| Omega mode | Sites | Baseline | Two loops | Two loops + frequency |
|---|---:|---:|---:|---:|
| scalar | 294 | 0.025276 | 0.022263 | 0.023272 |
| scalar | 2940 | 0.110056 | 0.107141 | 0.116737 |
| per-site | 294 | 0.417164 | 0.383420 | 0.392722 |
| per-site | 2940 | 7.641886 | 4.133563 | 4.097931 |

## Repeated MAP comparisons

Five fresh processes per candidate, each with a complete warm-up and a timed pass. Order: baseline/loops, loops/baseline, baseline/loops, baseline/loops, loops/baseline. Extended from three because the first paired gains varied (9.2% and 2.0%).

| Candidate | Startup median, range (s) | Warm MAP median, range (s) |
|---|---:|---:|
| baseline | 3.157 (3.079–3.281) | 86.538 (85.223–88.495) |
| loops | 1.562 (1.460–1.687) | 81.955 (80.322–86.867) |

Every warm-up and timed pass used 210 steps and matched the fixed objective, GTR parameters and all 294 omega estimates at their original tolerances. Warm MAP is the full timed pass, including its first step and progress logging.

## Repeated memory and larger-workload probes

Three separate fresh processes per candidate; one initial plus five warm objective/gradient calls at 2,940 sites. Order: baseline/loops, loops/baseline, baseline/loops. GB denotes decimal gigabytes.

| Candidate | Peak RSS median, range (GB) | Warm objective median, range across processes (s) |
|---|---:|---:|
| baseline | 11.642 (11.470–11.775) | 7.547 (7.477–7.921) |
| loops | 9.161 (8.790–9.344) | 4.030 (3.887–4.288) |

RSS includes imports and compilation and varies between processes. It is not pure device allocation. Memory-run timings confirm the larger-workload result; the separate objective screen remains reported above.

## Production decision

Retain the two static loops under the agreed throughput criterion: warm MAP
median 86.538 to 81.955 s (5.30% faster), startup 3.157 to 1.562 s (50.52% lower),
large-probe warm runtime 7.547 to 4.030 s (46.61% lower), and peak RSS 11.642 to
9.161 GB (21.31% lower). The larger scalar probe is also within its regression
limit. Selection uses unrounded numbers; no valid trial was discarded.

The warm MAP margin is modest and individual trial ranges overlap. Five trials
were used because the initial paired gains varied (9.2%, 2.0%, -0.4%). The
reported median satisfies the chosen 5% threshold on this machine; timing
variability remains visible in the ranges and raw samples.

Production preserves the original column arithmetic, spectral diagonal
multiplication, frequency matrix multiplications, normalization, strict-negative
clipping, transpose and identity addition. Static loops reduce the unrolled JIT
program without adding JIT boundaries or public configuration. Obsolete comments
and an unused index within the rewritten function were removed. No other
production module changed.

Individual timings, source/input/environment metadata, numerical comparison
errors, rejected checks, and the selection calculation are in
[the machine-readable results](benchmarks/perf02_results.json).

## Installed validation

Installed numerical checks passed (six targeted tests plus the benchmark matrix/
derivative screen). A fresh-process production MAP confirmation took 1.690 s
startup and 83.049 s for the warmed full optimizer, used 210 steps, and matched
the fixed objective, GTR parameters, and all 294 omega estimates. This separate
confirmation is not included in the five-trial selection medians.

`python -m pytest -q`: **77 passed, 10 subtests passed**, with the same two
upstream dependency warnings, in 115.14 s. This includes the unchanged protected
likelihood/gradient and CLI MAP integration tests. CLI/benchmark help,
compilation, and `git diff --check` pass. No reference fixture or tolerance was
changed.

Measurements use macOS 26.6.2 arm64, Python 3.14.7, JAX/jaxlib 0.8.2, CPU with
`NPROC=4`, float64, and disabled persistent compilation caching. Large inputs
repeat the 294-site alignment ten times. These results cover this CPU workload;
GPU, NUTS, and substantially larger real alignments remain outside this campaign.

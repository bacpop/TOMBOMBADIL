"""Run the MAP CLI and report time spent in the optimizer loop only.

Example:
    python -m test.run_map_benchmark --alignment porB3_aligned.fasta \
      --omega-mode per-site --fit-method map --sample-it 500 \
      --output-jax /tmp/porB3_map --fit-until-convergence \
      --exclude-invariant --platform cpu
"""

from time import perf_counter

from tombombadil import sample
from tombombadil.__main__ import main


_run_replicates = sample._run_replicates


def _timed_run_replicates(*args, **kwargs):
    started = perf_counter()
    result = _run_replicates(*args, **kwargs)
    elapsed = perf_counter() - started
    print(f"MAP_OPTIMIZATION_SECONDS={elapsed:.6f}")
    return result


sample._run_replicates = _timed_run_replicates


if __name__ == "__main__":
    main()

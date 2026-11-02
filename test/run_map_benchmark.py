"""Run the MAP CLI and report time spent in the optimizer loop only.

Example:
    python -m test.run_map_benchmark --alignment porB3_aligned.fasta \
      --omega-mode per-site --fit-method map --max-it 500 \
      --output-jax /tmp/porB3_map \
      --exclude-invariant --platform cpu
"""

from inspect import signature
from time import perf_counter

import jax
from tombombadil import sample
from tombombadil.__main__ import main


_run_map_replicates = sample._run_map_replicates


def _timed_run_map_replicates(*args, **kwargs):
    warmup_kwargs = dict(kwargs)
    if "progress" in signature(_run_map_replicates).parameters:
        warmup_kwargs["progress"] = False

    warmup_started = perf_counter()
    warmup_result = _run_map_replicates(*args, **warmup_kwargs)
    _block_until_ready(warmup_result)
    warmup_elapsed = perf_counter() - warmup_started
    print(f"MAP_WARMUP_SECONDS={warmup_elapsed:.6f}")

    started = perf_counter()
    result = _run_map_replicates(*args, **kwargs)
    _block_until_ready(result)
    elapsed = perf_counter() - started
    print(f"MAP_OPTIMIZATION_SECONDS={elapsed:.6f}")
    return result


def _block_until_ready(result):
    for leaf in jax.tree.leaves(result):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


sample._run_map_replicates = _timed_run_map_replicates


if __name__ == "__main__":
    main()

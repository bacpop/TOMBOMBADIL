"""Serial fresh-process budget enforcement for PERF-04 benchmark probes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def supervise(args):
    import psutil
    from tombombadil.alignment import count_codons

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Graph inspection finds an f64[61, N, 61, 61] saved residual alone.
    # Add headroom for other arrays/imports; plain maps keep all chunks.
    # This is a conservative estimate, not a claim of measured peak memory.
    n_sites = count_codons(args.alignment)[0].shape[1] * args.site_repeat
    predicted = 1_000_000_000 + int(1.6 * 8 * 61**3 * n_sites)
    if (args.mode in ("objective", "memory", "step", "update") and args.omega_mode == "per-site"
            and predicted > args.rss_limit_gib * 1024**3):
        output.write_text(json.dumps({"status": "preflight_skipped", "arguments": vars(args),
            "predicted_rss_bytes": predicted, "reason": "Conservative uncheckpointed AD residual estimate exceeds budget"}, indent=2) + "\n")
        return
    command = [sys.executable, "-m", "test.run_jit_benchmark", *sys.argv[1:], "--worker"]
    output.unlink(missing_ok=True)
    start = time.monotonic()
    peak = 0
    status = "completed"
    with subprocess.Popen(command, env=os.environ.copy()) as child:
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                peak = max(peak, process.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            if peak > args.rss_limit_gib * 1024**3:
                status = "rss_limit"
            elif time.monotonic() - start > args.timeout:
                status = "timeout"
            if status != "completed":
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            time.sleep(0.1)
        returncode = child.wait()
    if status == "completed" and returncode:
        status = "failed"
    result = json.loads(output.read_text()) if output.exists() else {"arguments": vars(args)}
    result.update(status=status, monitored_peak_rss_bytes=peak,
                  process_seconds=time.monotonic() - start, returncode=returncode)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if status == "failed":
        raise SystemExit(returncode)

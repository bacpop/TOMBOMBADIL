"""Validate and archive a directory of PERF-03 probe reports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median


def summarize(directory):
    from test.site_benchmark_checks import check_repeated_report, compare_trees
    import numpy as np

    trials = []
    for path in sorted(Path(directory).glob("*.json")):
        report = json.loads(path.read_text())
        if "arguments" in report:
            trials.append(dict(report, report_file=path.name))
    references = {}
    for report in trials:
        if (report.get("variant") == "baseline" and report.get("n_sites") == 294
                and "gradient" in report and report.get("status") == "completed"):
            args = report["arguments"]
            references[(args["params"], args["omega_mode"], report["alignment_sha256"])] = report
    scalar_references = {
        (r["n_sites"], r["arguments"]["params"], r["alignment_sha256"]): r
        for r in trials if r.get("variant") == "baseline" and "gradient" in r
        and r["arguments"]["omega_mode"] == "scalar"
    }
    checks = []
    scalar_diagnostics = []
    for report in trials:
        if "gradient" not in report:
            continue
        args = report["arguments"]
        reference = references[(args["params"], args["omega_mode"], report["alignment_sha256"])]
        if args["omega_mode"] == "per-site":
            error = check_repeated_report(report, reference)
            method = "repeated-input identity"
        else:
            # At symmetric rates even the frozen scalar baseline violates the
            # mathematical gradient identity. Preserve that diagnostic; use the
            # full-size frozen baseline for the actual scalar regression gate.
            try:
                identity_error = check_repeated_report(report, reference)
                diagnostic = {"status": "passed", "max_abs_error": identity_error}
            except AssertionError as error:
                diagnostic = {"status": "failed", "failure": str(error)}
            scalar_diagnostics.append(dict(diagnostic, report_file=report["report_file"],
                                           reference=reference["report_file"]))
            reference = scalar_references[(report["n_sites"], args["params"], report["alignment_sha256"])]
            def result(r):
                return r["objective"], {k: np.asarray(v) for k,v in r["gradient"].items()}
            error = compare_trees(result(report), result(reference))
            method = "same-size frozen scalar baseline"
        checks.append({"report_file": report["report_file"],
                       "reference": reference["report_file"], "method": method,
                       "status": "passed", "max_abs_error": error})
    groups = {}
    for report in trials:
        args = report["arguments"]
        if (report.get("n_sites") == 2940 and args["params"] == "initial"
                and args["omega_mode"] == "per-site" and "timing" in report):
            groups.setdefault(report["variant"], []).append(report)
    statistics = {}
    for name, reports in groups.items():
        first = [r["timing"]["first_seconds"] for r in reports]
        warm = [median(r["timing"]["seconds"]) for r in reports]
        rss = [r["peak_rss_bytes"] for r in reports]
        statistics[name] = {"trials": len(reports), "first_seconds": median(first),
                            "warm_seconds": median(warm), "peak_rss_bytes": median(rss),
                            "first_range": [min(first), max(first)],
                            "warm_range": [min(warm), max(warm)],
                            "rss_range": [min(rss), max(rss)]}
    baseline = statistics["baseline"]
    for name, stats in statistics.items():
        ratios = {k: stats[k]/baseline[k] for k in ("first_seconds", "warm_seconds", "peak_rss_bytes")}
        stats["change_percent"] = {k: 100*(r-1) for k,r in ratios.items()}
        stats["passes_primary_gate"] = (
            (ratios["warm_seconds"] <= 0.9 or ratios["peak_rss_bytes"] <= 0.8)
            and ratios["first_seconds"] <= 1 and ratios["warm_seconds"] <= 1
            and ratios["peak_rss_bytes"] <= 1.05)
    sources = ("tombombadil/sample.py", "tombombadil/likelihood.py", "tombombadil/gtr.py",
               "test/fixtures/porB3_per_site_map.json", "test/site_benchmark_variants.py",
               "test/site_benchmark_checks.py", "test/site_benchmark_limits.py",
               "test/run_site_benchmark.py", "test/summarize_site_benchmark.py")
    qualifiers = [name for name, stats in statistics.items() if stats["passes_primary_gate"]]
    return {"task": "PERF-03", "primary_n_sites": 2940,
            "primary_qualifiers": qualifiers,
            "decision": "further controls required" if qualifiers else "retain baseline",
            "final_source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
            "primary_statistics": statistics, "objective_checks": checks,
            "scalar_repeated_input_diagnostics": scalar_diagnostics,
            "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory of individual probe JSON files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["NPROC"] = "4"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    result = summarize(args.directory)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["primary_statistics"], indent=2))
    print(f"Validated {len(result['objective_checks'])} objectives; archive: {output}")


if __name__ == "__main__":
    main()

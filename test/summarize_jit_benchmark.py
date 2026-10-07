"""Validate and archive PERF-04 reports, retaining rejected numerical probes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median


def samples(values):
    return {"median":median(values),"range":[min(values),max(values)],"samples":values}


def startup(run):
    if "startup_seconds" in run:
        return run["startup_seconds"], 0.0
    # Early pilots retained the original formatted log. Bound the rounding
    # uncertainty; never treat the printed millisecond precision as exact.
    value = float(next(x for x in run["timing_log"] if x.startswith("MAP startup:")).split()[2])
    return value, .0005


def summarize(directory):
    import numpy as np
    from test.site_benchmark_checks import compare_trees, check_repeated_report
    trials=[]
    for path in sorted(Path(directory).glob("*.json")):
        report=json.loads(path.read_text())
        if "arguments" in report:
            trials.append(dict(report,report_file=path.name))
    references={}
    for r in trials:
        if r.get("variant")=="baseline" and "gradient" in r:
            a=r["arguments"]
            references.setdefault((r["n_sites"],a["params"],a["omega_mode"]),r)
    checks=[]
    for r in trials:
        if "gradient" not in r:
            continue
        a=r["arguments"]
        ref=references[(r["n_sites"],a["params"],a["omega_mode"])]
        def result(r):
            return r["objective"],{k:np.asarray(v) for k,v in r["gradient"].items()}
        err=compare_trees(result(r),result(ref))
        item={"report_file":r["report_file"],"reference":ref["report_file"],"max_abs_error":err}
        if a["omega_mode"]=="per-site":
            small=references[(294,a["params"],a["omega_mode"])]
            item["repeated_input_error"]=check_repeated_report(r,small)
        checks.append(item)
    statistics={}
    for variant in ("baseline","update"):
        maps=[r for r in trials if r.get("variant")==variant and r.get("mode")=="map"
              and r.get("status")=="completed" and r.get("map_reference")=="passed"]
        objectives=[r for r in trials if r.get("variant")==variant and r.get("n_sites")==2940
              and r.get("mode")=="objective" and r.get("status")=="completed"
              and r["arguments"]["params"]=="initial" and r["arguments"]["omega_mode"]=="per-site"]
        cold=[startup(r["warmup"]) for r in maps]
        statistics[variant]={
            "map_trials":len(maps),"objective_trials":len(objectives),
            "map_seconds":samples([r["timed"]["seconds"] for r in maps]),
            "cold_startup_seconds":samples([v for v,e in cold]),
            "cold_startup_median_bounds":[median(v-e for v,e in cold),median(v+e for v,e in cold)],
            "objective_seconds":samples([median(r["timing"]["seconds"]) for r in objectives]),
            "objective_first_seconds":samples([r["timing"]["first_seconds"] for r in objectives]),
            "objective_peak_rss_bytes":samples([r["peak_rss_bytes"] for r in objectives]),
        }
    base,candidate=statistics["baseline"],statistics["update"]
    fields=("map_seconds","objective_seconds","objective_first_seconds","objective_peak_rss_bytes")
    ratios={k:candidate[k]["median"]/base[k]["median"] for k in fields}
    # Conservative startup bound also covers pilots recorded at millisecond precision.
    startup_ratio=candidate["cold_startup_median_bounds"][1]/base["cold_startup_median_bounds"][0]
    enough_trials=min(base["map_trials"],candidate["map_trials"],base["objective_trials"],candidate["objective_trials"])>=3
    primary_gate=(enough_trials and (ratios["map_seconds"]<=.95 or ratios["objective_seconds"]<=.9
                   or ratios["objective_peak_rss_bytes"]<=.8)
                  and ratios["map_seconds"]<=1 and ratios["objective_seconds"]<=1
                  and ratios["objective_first_seconds"]<=1 and startup_ratio<=1
                  and ratios["objective_peak_rss_bytes"]<=1.05)
    source_files=("tombombadil/sample.py","tombombadil/likelihood.py","tombombadil/gtr.py",
        "test/jit_benchmark_variants.py","test/jit_benchmark_checks.py","test/run_jit_benchmark.py",
        "test/jit_benchmark_limits.py","test/summarize_jit_benchmark.py","test/fixtures/porB3_per_site_map.json",
        "test/test_baselines.py","test/test_jit_benchmark.py")
    return {"task":"PERF-04","statistics":statistics,
        "change_percent":{k:100*(v-1) for k,v in ratios.items()},
        "cold_startup_change_percent_upper_bound":100*(startup_ratio-1),
        "passes_primary_gate":primary_gate,"objective_checks":checks,"trials":trials,
        "final_source_sha256":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    os.environ.update(NPROC="4",JAX_ENABLE_X64="true",JAX_PLATFORMS="cpu",JAX_ENABLE_COMPILATION_CACHE="false")
    report=summarize(args.directory)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("trials","objective_checks","final_source_sha256")},indent=2))
    print(f"Validated {len(report['objective_checks'])} objective reports; archive: {output}")


if __name__=="__main__":
    main()

"""Validate and archive PERF-05 measurements and experimental solution quality."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median


def stats(values):
    return {"median":median(values),"range":[min(values),max(values)],"samples":values}


def summarize(directory):
    import numpy as np
    from test.site_benchmark_checks import compare_trees,check_repeated_report
    reports=[]
    for p in sorted(Path(directory).glob("*.json")):
        r=json.loads(p.read_text())
        if "arguments" in r:
            reports.append(dict(r,report_file=p.name))
    screens={f"{r['arguments']['variant']}-d{r['arguments']['devices']}":r["correctness"]
        for r in reports if r["arguments"]["mode"]=="check" and "correctness" in r}
    changed=[(k,v) for k,v in screens.items() if not k.startswith("baseline")]
    if len(changed)!=14:
        raise ValueError("Expected all fourteen changed candidate/device screens")
    objectives=[r for r in reports if "gradient" in r and r["status"]=="completed"]
    references={(r["n_sites"],r["arguments"]["params"],r["arguments"]["omega_mode"]):r
        for r in objectives if r["arguments"]["variant"]=="baseline" and r["arguments"]["devices"]==1}
    checks=[]
    for r in objectives:
        a=r["arguments"];key=(r["n_sites"],a["params"],a["omega_mode"])
        ref=references[key]
        value=lambda x:(x["objective"],{k:np.asarray(v) for k,v in x["gradient"].items()})
        item={"report_file":r["report_file"],"reference":ref["report_file"],
              "max_abs_error":compare_trees(value(r),value(ref))}
        if a["omega_mode"]=="per-site":
            item["repeated_input_error"]=check_repeated_report(r,references[(294,a["params"],a["omega_mode"])])
        checks.append(item)
    primary={}
    for r in objectives:
        a=r["arguments"]
        if r["n_sites"]==2940 and a["params"]=="initial" and a["omega_mode"]=="per-site":
            primary.setdefault(f"{a['variant']}-d{a['devices']}",[]).append(r)
    measurements={k:{"trials":len(rs),
        "warm_seconds":stats([median(r["timing"]["seconds"]) for r in rs]),
        "first_seconds":stats([r["timing"]["first_seconds"] for r in rs]),
        "peak_rss_bytes":stats([r["peak_rss_bytes"] for r in rs])} for k,rs in primary.items()}
    base=measurements["baseline-d1"]
    decisions={}
    for key,screen in changed:
        if screen["status"]!="passed":
            decisions[key]={"decision":"numerically_rejected"};continue
        if key not in measurements:
            attempted=[r for r in reports if r["arguments"]["mode"]=="objective" and
                f"{r['arguments']['variant']}-d{r['arguments']['devices']}"==key]
            if not attempted: raise ValueError(f"Missing screen timing for {key}")
            decisions[key]={"decision":"performance_probe_incomplete",
                            "statuses":[r["status"] for r in attempted]}
            continue
        candidate=measurements[key]
        ratios={k:candidate[k]["median"]/base[k]["median"] for k in ("warm_seconds","first_seconds","peak_rss_bytes")}
        nonregressing=ratios["warm_seconds"]<=1 and ratios["first_seconds"]<=1 and ratios["peak_rss_bytes"]<=1.05
        if candidate["trials"]>=3 and nonregressing:
            raise ValueError(f"{key} needs complete MAP/control confirmation before selection")
        decisions[key]={"decision":"runtime_regression" if candidate["trials"]>=3 else "screen_only_not_finalist",
                        "trials":candidate["trials"],"change_percent":{k:100*(v-1) for k,v in ratios.items()}}
    maps=[r for r in reports if "timed" in r and r["status"]=="completed" and r["arguments"]["variant"]=="baseline"]
    map_by_start={start:[r for r in maps if r["arguments"]["start"]==start] for start in (0,1,2)}
    baseline_stats={str(s):{
        "cold_seconds":stats([r["warmup"]["seconds"] for r in rs]),
        "timed_seconds":stats([r["timed"]["seconds"] for r in rs]),
        "startup_seconds":stats([r["warmup"]["startup_seconds"] for r in rs]),
        "steps":[r["timed"]["n_steps"] for r in rs]}
        for s,rs in map_by_start.items() if rs}
    alternating=[]
    for r in reports:
        a=r["arguments"]
        if a["mode"]!="alternate": continue
        item={"report_file":r["report_file"],"status":r["status"],"rounds":a["rounds"],
              "variant":a["variant"],"devices":a["devices"],
              "start":a["start"],"n_sites":r.get("n_sites"),"update_budget":a["updates"]}
        if "alternating" in r and r["status"]=="completed":
            item.update(seconds=r["alternating"]["seconds"],objective=r["endpoint_objective"],
                gradient_norm=r["gradient_norm"],block_updates=r["alternating"]["block_updates"])
            if r["n_sites"]==294:
                base=map_by_start[a["start"]][0]
                target=base["timed"]["objective"]
                tolerance=1e-8+1e-6*abs(target)
                hits=[h["seconds"] for h in r["alternating"]["history"] if h["objective"]>=target-tolerance]
                item.update(baseline_report=base["report_file"],baseline_target=target,
                    baseline_gradient_norm=base.get("endpoint_gradient_norm"),
                    objective_difference=r["endpoint_objective"]-target,
                    seconds_to_baseline_quality=min(hits) if hits else None,
                    reaches_baseline_quality=r["endpoint_objective"]>=target-tolerance,
                    max_parameter_abs_difference=max(float(np.max(np.abs(np.asarray(v)-np.asarray(base["timed"]["natural_params"][k])))) for k,v in r["natural_params"].items()))
        else:
            item["partial_block_updates"]=r.get("partial_block_updates")
            history=r.get("partial_history",[])
            if history:
                item["last_completed_cycle"]=history[-1]
            item["time_budget_seconds"]=a["timeout"]
        alternating.append(item)
    alternating_stats={}
    for item in alternating:
        if item["status"]!="completed" or item["n_sites"]!=294:
            continue
        key=f"{item['variant']}-d{item['devices']}-{item['rounds']}-start{item['start']}"
        alternating_stats.setdefault(key,[]).append(item)
    alternating_stats={key:{
        "trials":len(items),
        "seconds":stats([x["seconds"] for x in items]),
        "gradient_norm":stats([x["gradient_norm"] for x in items]),
        "objective_difference":stats([x["objective_difference"] for x in items]),
        "reaches_baseline_quality":sum(x["reaches_baseline_quality"] for x in items),
        "seconds_to_baseline_quality":[x["seconds_to_baseline_quality"] for x in items],
    } for key,items in alternating_stats.items()}
    source_files=[*Path("test").glob("*parallel_benchmark*.py"),Path("tombombadil/sample.py"),
        Path("tombombadil/likelihood.py"),Path("tombombadil/gtr.py"),Path("tombombadil/device.py"),
        Path("test/test_baselines.py"),Path("test/fixtures/porB3_per_site_map.json")]
    return {"task":"PERF-05","selection":{"production":"baseline",
        "reason":"Chunk-local and compiled candidates fail the numerical gate; passing plain-sharding finalists regress runtime; alternating remains experimental"},
        "joint_decisions":decisions,"objective_statistics":measurements,
        "screens":screens,"objective_checks":checks,"baseline_statistics":baseline_stats,
        "alternating_comparisons":alternating,
        "alternating_statistics":alternating_stats,
        "validation_logs":{p.name:p.read_text() for name in ("pytest.log","unit-d4-final.log",
                           "unit-d4-placement-failure.log","pytest-pre-placement-fix.log")
                           if (p:=Path(directory)/name).exists()},
        "million_site_projection":{"sites":1_000_000,"counts_float64_bytes":61*8*1_000_000,
            "omega_and_adam_moments_bytes":3*8*1_000_000,
            "omega_gradient_bytes":8*1_000_000,"float64_mask_bytes":8*1_000_000,
            "one_unbounded_saved_residual_bytes":8*61**3*1_000_000,
            "one_64_site_residual_four_devices_bytes":8*61**3*64*4,
            "linear_baseline_seconds_per_gradient":measurements["baseline-d1"]["warm_seconds"]["median"]*1_000_000/2940,
            "note":"Arithmetic storage estimates only, not validated capacity or measured million-site execution"},
        "final_source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "reports":reports}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory");parser.add_argument("--output",required=True)
    args=parser.parse_args()
    os.environ.update(NPROC="4",JAX_PLATFORMS="cpu",JAX_ENABLE_X64="true",JAX_ENABLE_COMPILATION_CACHE="false")
    report=summarize(args.directory)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("reports","screens","final_source_sha256")},indent=2))


if __name__=="__main__":
    main()

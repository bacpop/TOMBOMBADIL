"""Bounded fresh-process CPU benchmarks for PERF-05. See perf05_results.md."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from test.parallel_benchmark_variants import VARIANTS, BASELINE_REVISION


def supervise(args):
    import psutil
    from tombombadil.alignment import count_codons
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    n=count_codons("porB3_aligned.fasta")[0].shape[1]*args.site_repeat
    chunk=int(args.variant.removeprefix("local").removeprefix("shard")) if args.variant not in ("baseline","shard","shard-jit") else n
    active=min(n,chunk*(args.devices if args.variant.startswith("shard") else 1))
    predicted=1_000_000_000+int(1.6*8*61**3*active)+n*61*8*3
    if args.omega_mode=="per-site" and args.mode not in ("check","blocks") and predicted>args.rss_limit_gib*1024**3:
        output.write_text(json.dumps({"status":"preflight_skipped","arguments":vars(args),
            "predicted_rss_bytes":predicted,"reason":"Conservative active-residual estimate exceeds RSS budget"},indent=2)+"\n")
        return
    output.unlink(missing_ok=True)
    started=time.monotonic(); peak=0; status="completed"
    with subprocess.Popen([sys.executable,"-m","test.run_parallel_benchmark",*sys.argv[1:],"--worker"]) as child:
        process=psutil.Process(child.pid)
        while child.poll() is None:
            try: peak=max(peak,process.memory_info().rss)
            except psutil.NoSuchProcess: break
            if peak>args.rss_limit_gib*1024**3: status="rss_limit"
            elif time.monotonic()-started>args.timeout: status="timeout"
            if status!="completed":
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill()
                break
            time.sleep(.1)
        code=child.wait()
    result=json.loads(output.read_text()) if output.exists() else {"arguments":vars(args)}
    if status=="completed" and code: status="failed"
    result.update(status=status,returncode=code,monitored_peak_rss_bytes=peak,
                  process_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    if status=="failed": raise SystemExit(code)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=VARIANTS,default="baseline")
    parser.add_argument("--devices",type=int,choices=(1,2,4),default=1)
    parser.add_argument("--mode",choices=("check","blocks","block-time","inspect","objective","step","map","alternate"),default="objective")
    parser.add_argument("--site-repeat",type=int,default=1)
    parser.add_argument("--params",choices=("initial","fitted"),default="initial")
    parser.add_argument("--omega-mode",choices=("scalar","per-site"),default="per-site")
    parser.add_argument("--rounds",choices=("1:5","5:20"),default="1:5")
    parser.add_argument("--start",type=int,choices=(0,1,2),default=0)
    parser.add_argument("--policy",action="store_true",help="Keep baseline for scalar omega and fewer than 2940 sites; checks always force candidates")
    parser.add_argument("--diagnostic-cases",action="store_true",help="Retain initial/fitted diagnostics after a failed protected gate; does not permit adoption")
    parser.add_argument("--updates",type=int,default=500)
    parser.add_argument("--batches",type=int,default=5)
    parser.add_argument("--timeout",type=float,default=600)
    parser.add_argument("--rss-limit-gib",type=float,default=12)
    parser.add_argument("--output",required=True)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if min(args.site_repeat,args.updates,args.batches,args.timeout,args.rss_limit_gib)<=0:
        parser.error("sizes, updates, batches and budgets must be positive")
    if args.updates>500: parser.error("maximum 500 updates")
    if args.mode=="map" and (args.params!="initial" or args.omega_mode!="per-site"):
        parser.error("MAP comparisons require initial per-site parameters")
    if not args.worker:
        supervise(args); return
    os.environ.update(JAX_ENABLE_X64="true",JAX_ENABLE_COMPILATION_CACHE="false")
    from tombombadil.device import configure_platform, _host_device_count_flags
    configure_platform("cpu",cpus=4)
    os.environ["XLA_FLAGS"]=_host_device_count_flags(os.environ.get("XLA_FLAGS",""),args.devices)
    import jax
    import numpy as np
    import jaxlib
    import logging
    import psutil
    from test.parallel_benchmark_variants import baseline_module,Evaluation,optimizer_module,alternating
    from test.parallel_benchmark_checks import model_case,check,check_map,check_blocks
    from test.run_gtr_benchmark import measure
    from test.jit_benchmark_checks import make_solver
    if len(jax.devices())!=args.devices:
        raise RuntimeError(f"Requested {args.devices} CPU devices, got {jax.devices()}")
    module=baseline_module()
    output=Path(args.output)
    report={"arguments":vars(args),"baseline_revision":BASELINE_REVISION,
        "baseline_source_sha256":hashlib.sha256(module.source.encode()).hexdigest(),
        "platform":platform.platform(),"python":sys.version,"executable":sys.executable,
        "jax":jax.__version__,"jaxlib":jaxlib.__version__,"devices":[str(d) for d in jax.devices()],
        "x64":jax.config.x64_enabled,"backend":jax.default_backend(),
        "host":{"logical_cpus":os.cpu_count(),"physical_cpus":psutil.cpu_count(logical=False),
                "ram_bytes":psutil.virtual_memory().total},
        "environment":{k:os.environ.get(k) for k in ("NPROC","XLA_FLAGS","JAX_ENABLE_COMPILATION_CACHE","CONDA_PREFIX")},
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [*Path("test").glob("*parallel_benchmark*.py"),Path("tombombadil/sample.py"),
             Path("tombombadil/likelihood.py"),Path("tombombadil/gtr.py"),Path("porB3_aligned.fasta")]}}
    def save():
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if args.mode in ("check","blocks"):
        report["correctness"]=check(args.variant,args.devices,args.diagnostic_cases) if args.mode=="check" else check_blocks(args.variant,args.devices,tuple(map(int,args.rounds.split(":"))))
        save()
        if report["correctness"]["status"]=="failed": raise SystemExit(1)
        return
    started=time.perf_counter()
    X,pi,mask,options,params=model_case(module,args.site_repeat,args.params=="fitted",args.omega_mode,args.start)
    effective="baseline" if args.policy and (X.shape[1]<2940 or args.omega_mode=="scalar") else args.variant
    evaluation=Evaluation(module,X,pi,mask,options,effective,args.devices)
    params=jax.device_put(params)
    jax.block_until_ready(params)
    report.update(n_sites=X.shape[1],effective_variant=effective,preparation_seconds=time.perf_counter()-started)
    if args.mode in ("objective","inspect"):
        if args.mode=="inspect":
            from test.site_benchmark_checks import inspect_objective
            report["inspection"]=inspect_objective(evaluation.value,params)
            graph=str(jax.make_jaxpr(evaluation.value_grad)(params))
            report["explicit_gradient_graph"]={"sha256":hashlib.sha256(graph.encode()).hexdigest(),"characters":len(graph)}
        else:
            report["timing"],(v,g)=measure(evaluation.value_grad,(params,),args.batches,1)
            report["objective"]=float(v)
            report["gradient"]={k:np.asarray(v).tolist() for k,v in g.items()}
            report["forward_timing"],_=measure(evaluation.value,(params,),args.batches,1)
    elif args.mode=="block-time":
        from test.jit_benchmark_checks import compare
        report["blocks"]={}
        results={}
        for label,ks in (("shared",[k for k in params if k!="omega"]),("omega",["omega"])):
            active={k:params[k] for k in ks}
            partial=jax.value_and_grad(lambda p:evaluation.value(dict(params,**p)))
            timing,result=measure(partial,(active,),args.batches,1)
            report["blocks"][label]={"timing":timing}
            results[label]=(ks,result)
        expected=evaluation.value_grad(params)
        for label,(ks,result) in results.items():
            report["blocks"][label]["correctness"]=compare(result,(expected[0],{k:expected[1][k] for k in ks}))
        report["block_order"]="shared then omega; omega first call may reuse common kernels; correctness evaluated after timing"
    elif args.mode=="step":
        import optax
        solver=make_solver(params); state=solver.init(params)
        timings=[]
        for _ in range(args.batches+1):
            started=time.perf_counter()
            value,g=evaluation.value_grad(params)
            u,state=solver.update(jax.tree.map(lambda x:-x,g),state,params)
            params=optax.apply_updates(params,u)
            jax.block_until_ready((params,state,value))
            timings.append(time.perf_counter()-started)
        report["timing"]={"first_seconds":timings[0],"seconds":timings[1:]}
    elif args.mode=="alternate":
        def checkpoint(index,p,states,counts):
            # A timeout still leaves completed cycle evidence via this callback.
            if sum(counts)%25==0:
                report["partial_block_updates"]=counts
                save()
        def cycle(history, counts):
            report["partial_history"]=history
            report["partial_block_updates"]=counts
            save()
            print(f"Alternating updates {sum(counts)}: objective={history[-1]['objective']:.9f}",flush=True)
        result=alternating(evaluation,params,tuple(map(int,args.rounds.split(":"))),
                           args.updates,checkpoint,cycle)
        params=result.pop("params"); result.pop("states")
        report["alternating"]=result
        v,g=jax.value_and_grad(evaluation.reference)(params)
        report["endpoint_objective"]=float(v)
        report["gradient_norm"]=float(np.sqrt(sum(np.sum(np.asarray(x)**2) for x in g.values())))
        report["natural_params"]={k:np.asarray(module.positive_transform(v)).tolist() for k,v in params.items()}
    else:
        logging.basicConfig(level=logging.INFO,format="%(message)s",force=True)
        class Timers(logging.Handler):
            def emit(self,record):
                if record.msg.startswith("MAP startup:"): self.startup=float(record.args[0])
        handler=Timers(); logging.getLogger().addHandler(handler)
        optimizer=module if effective=="baseline" else optimizer_module(evaluation)
        original_optimize=optimizer._optimize_params
        history=[]
        fit_start=0.
        def observed(*a,**kw):
            callback=kw.get("progress_callback")
            def record(step,total,objective):
                history.append({"updates":step,"seconds":time.perf_counter()-fit_start,"objective":objective})
                if callback: callback(step,total,objective)
            kw["progress_callback"]=record
            return original_optimize(*a,**kw)
        optimizer._optimize_params=observed
        runs=[]
        for progress in (False,True):
            started=time.perf_counter()
            fit_start=started; history=[]
            if hasattr(evaluation,"_kernels"):
                evaluation._kernels.clear()
            result=optimizer._run_map_replicates(evaluation.value,params,module.make_param_labels(params),1,
                n_iter=args.updates,convergence={"enabled":True,"tol":1e-6,"patience":5,"check_every":10,"min_steps":50},progress=progress)
            jax.block_until_ready(result)
            seconds=time.perf_counter()-started
            metadata=result[2][0]
            record={"seconds":seconds,"startup_seconds":handler.startup,"n_steps":metadata["n_steps"],
                "objective":float(metadata["objective"]),"natural_params":{k:np.asarray(module.positive_transform(v)).tolist() for k,v in result[0][0].items()},
                "objective_history":metadata["objective_history"],"history":history}
            if args.site_repeat==1 and args.start==0 and args.updates==500:
                check_map(record); record["map_reference"]="passed"
            runs.append(record)
        report["warmup"],report["timed"]=runs
        report["placement_inclusive_cold_seconds"]=report["preparation_seconds"]+runs[0]["seconds"]
        _,gradient=jax.value_and_grad(evaluation.reference)(result[0][0])
        report["endpoint_gradient_norm"]=float(np.sqrt(sum(np.sum(np.asarray(x)**2) for x in gradient.values())))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_bytes"]=peak if sys.platform=="darwin" else peak*1024
    save()


if __name__=="__main__":
    main()

"""Direct numerical screens for PERF-05; fixed references are never rewritten."""
import json
from pathlib import Path


def model_case(module, repeat=1, fitted=False, omega_mode="per-site", start=0):
    import numpy as np
    from tombombadil.alignment import count_codons
    counts, _ = count_codons("porB3_aligned.fasta")
    counts = np.tile(counts, repeat)
    pi = np.full(61, 1/61)
    options = dict(include_invariant=False, aggregate="sum", prior_mode="stan_unconstrained",
        estimate_eta=True, eigen_jitter=True, omega_floor=True, omega_mode=omega_mode)
    p = module.make_base_params(estimate_eta=True, n_sites=counts.shape[1], omega_mode=omega_mode)
    if fitted:
        ref = json.loads(Path("test/fixtures/porB3_per_site_map.json").read_text())
        p = module.natural_to_raw_params(dict(ref["gtr"], omega=np.tile(ref["omega"], repeat)
            if omega_mode == "per-site" else .5), omega_mode=omega_mode)
    if start:
        rng = np.random.default_rng(0)
        for _ in range(start):
            perturbed = {k: v+rng.normal(0, .5, np.shape(v)) for k,v in p.items()}
        p = perturbed
    return counts, pi, module.make_variable_site_mask(counts), options, p


def check(variant, devices, diagnostic_cases=False):
    import jax
    import numpy as np
    import optax
    from test.parallel_benchmark_variants import Evaluation, baseline_module
    from test.jit_benchmark_checks import protected_case, compare, make_solver

    m = baseline_module()
    _, p, expected = protected_case(m)
    X = np.zeros((61,1)); X[15,0]=4; X[47,0]=19
    options = dict(include_invariant=True, aggregate="mean", prior_mode="current",
                   estimate_eta=False, eigen_jitter=True, omega_floor=True, omega_mode="scalar")
    evaluator = Evaluation(m, X, np.full(61,1/61), np.ones(1), options, variant, devices)
    oracle = compare(evaluator.value_grad(p), expected)
    print(f"Protected reference: {oracle['status']}",flush=True)
    report = {"protected_reference":oracle,"cases":[],"status":"passed"}
    if oracle["status"] != "passed":
        report["status"] = "failed"
        if not diagnostic_cases:
            return report

    counts, pi, mask, options, params = model_case(m)
    cases = [(counts, pi, mask, options, params, "initial"),
             (*model_case(m, fitted=True), "fitted")]
    batch = 64 if variant in ("baseline", "shard", "shard-jit") else int(variant.removeprefix("local").removeprefix("shard"))
    rng = np.random.default_rng(0)
    for index, n in enumerate(sorted({1, max(devices-1,1), batch-1, batch, batch+1, 2*batch+3})):
        X = rng.multinomial(23, np.full(61,1/61), size=n).T
        pi = np.arange(1,62)/1891 if index%2 else np.full(61,1/61)
        mask = np.zeros(n) if index==3 else (np.arange(n)%3!=0).astype(float)
        opts = dict(options, include_invariant=index%2==0, aggregate="mean" if index%2 else "sum",
            prior_mode=("none","current","stan_constrained","stan_unconstrained")[index%4],
            estimate_eta=index%2==0, eigen_jitter=index%3!=2, omega_floor=index%3!=1)
        p = m.make_base_params(estimate_eta=True,n_sites=n,omega_mode="per-site")
        for j,k in enumerate(p):
            p[k] = m.positive_transform_inverse(np.resize([.009,.01,.011,.3,.8,2.1],n)
                if k=="omega" else .31+.19*j)
        cases.append((X,pi,mask,opts,p,f"boundary-{n}"))
    for X,pi,mask,opts,p,label in cases:
        print(f"Checking {label} ({X.shape[1]} sites)",flush=True)
        base = Evaluation(m,X,pi,mask,opts)
        candidate = Evaluation(m,X,pi,mask,opts,variant,devices)
        result = {"label":label,"n_sites":X.shape[1],"options":opts,
                  "objective_gradient":compare(candidate.value_grad(p),base.value_grad(p)),
                  "ordered_losses":compare(candidate.losses(p),base.losses(p))}
        cv,cg=candidate.value_grad(p)
        result["loss_boundary"]=compare((-cv,jax.tree.map(lambda x:-x,cg)),
            jax.value_and_grad(lambda p:-base.value(p))(p))
        # Stop rejected initial cases before expensive timing; retain failure.
        report["cases"].append(result)
        if any(result[k]["status"]=="failed" for k in ("objective_gradient","ordered_losses","loss_boundary")):
            report["status"]="failed"
        if report["status"]=="failed":
            # Retain initial and fitted per-site diagnostics even when the
            # one-site gate fails; do not benchmark this kernel for adoption.
            if label=="fitted": return report
            continue
        solver = make_solver(p)
        bp,cp = p,p
        bs,cs = solver.init(p),solver.init(p)
        result["trajectory"]=[]
        for _ in range(3):
            for which,e in ((0,base),(1,candidate)):
                pp,ss = (bp,bs) if which==0 else (cp,cs)
                if which==0:
                    _,g=jax.value_and_grad(lambda p:-e.value(p))(pp)
                else:
                    _,g=e.value_grad(pp)
                    g=jax.tree.map(lambda x:-x,g)
                u,ss = solver.update(g,ss,pp)
                pp = optax.apply_updates(pp,u)
                if which==0: bp,bs = pp,ss
                else: cp,cs = pp,ss
            result["trajectory"].append(compare((cp,cs,candidate.value_grad(cp)),
                                                (bp,bs,base.value_grad(bp))))
        if any(x["status"]=="failed" for x in result["trajectory"]):
            report["status"]="failed"
            return report
        jax.clear_caches()
    return report


def check_map(result):
    import numpy as np
    ref = json.loads(Path("test/fixtures/porB3_per_site_map.json").read_text())
    np.testing.assert_allclose(result["objective"],ref["log_likelihood"],**ref["tolerance"])
    for k,v in dict(ref["gtr"],omega=ref["omega"]).items():
        np.testing.assert_allclose(result["natural_params"][k],v,**ref["tolerance"])


def check_blocks(variant="baseline", devices=1, rounds=(1,5)):
    """Partial AD used by serial alternating rounds must match joint AD."""
    import jax
    from test.parallel_benchmark_variants import baseline_module, Evaluation, alternating, parallel_block_value
    from test.jit_benchmark_checks import compare
    m=baseline_module(); cases=[]
    for fitted,start in ((False,0),(True,0),(False,1),(False,2)):
        X,pi,mask,opts,p=model_case(m,fitted=fitted,start=start)
        e=Evaluation(m,X,pi,mask,opts)
        candidate=Evaluation(m,X,pi,mask,opts,variant,devices)
        value,full=jax.value_and_grad(lambda p:-e.value(p))(p)
        case={"fitted":fitted,"start":start,"blocks":[]}
        for ks in ([k for k in p if k!="omega"],["omega"]):
            active={k:p[k] for k in ks}
            fn=e.value if ks!=["omega"] or variant=="baseline" else lambda p:parallel_block_value(candidate,p)
            actual=jax.value_and_grad(lambda a:-fn(dict(p,**a)))(active)
            case["blocks"].append(compare(actual,(value,{k:full[k] for k in ks})))
        cases.append(case)
    checks=[b for c in cases for b in c["blocks"]]
    trajectory=None
    if variant!="baseline" and all(b["status"]=="passed" for b in checks):
        X,pi,mask,opts,p=model_case(m)
        a=alternating(Evaluation(m,X,pi,mask,opts),p,rounds=rounds,max_updates=2*sum(rounds))
        b=alternating(Evaluation(m,X,pi,mask,opts,variant,devices),p,rounds=rounds,max_updates=2*sum(rounds))
        trajectory=compare((a["params"],a["states"],a["history"][-1]["objective"]),
                           (b["params"],b["states"],b["history"][-1]["objective"]))
        checks.append(trajectory)
    return {"cases":cases,"trajectory":trajectory,"rounds":list(rounds),
            "status":"passed" if all(b["status"]=="passed" for b in checks) else "failed"}

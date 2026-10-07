"""PERF-04 numerical gates; no timing assertions or reference changes."""
import ast
import inspect
import textwrap


def make_solver(params):
    import optax
    from tombombadil.sample import make_param_labels
    schedule = optax.cosine_decay_schedule(0.2, 500, alpha=1e-3/0.2)
    return optax.multi_transform({"vec": optax.adam(schedule), "scalar": optax.adam(schedule)},
                                make_param_labels(params))


def protected_case(module):
    import numpy as np
    import jax.numpy as jnp
    X = np.zeros((61, 1)); X[15, 0] = 4; X[47, 0] = 19
    pi = np.full(61, 1/61)
    params = module.make_base_params(estimate_eta=True, n_sites=1, omega_mode="scalar")
    fn = module.make_log_density_fn(pi, *module.prepare_likelihood_transforms(X, pi), X, jnp.ones(1),
        include_invariant=True, aggregate="mean", prior_mode="current", estimate_eta=False,
        eigen_jitter=True, omega_floor=True, omega_mode="scalar")
    # Read the oracle verbatim from the protected test, rather than maintaining
    # a second reference or accidentally leaving the new kernel unexercised.
    from test.test_baselines import TestLikelihoodGradientReference
    tree = ast.parse(textwrap.dedent(inspect.getsource(
        TestLikelihoodGradientReference.test_likelihood_and_gradient_match_fixed_reference)))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "expected_gradient")
    gradients = ast.literal_eval(assignment.value.args[0])
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "assert_allclose")
    value = ast.literal_eval(call.args[1])
    keys = ("alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "omega")
    return fn, params, (value, dict(zip(keys, gradients)))


def compare(actual, expected):
    import jax
    import numpy as np
    from test.site_benchmark_checks import compare_trees
    a = jax.tree.map(lambda x: np.asarray(x), actual)
    b = jax.tree.map(lambda x: np.asarray(x), expected)
    maximum = max(float(np.max(np.abs(x-y))) for x,y in
                  zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))
    try:
        compare_trees(a, b)
        return {"status": "passed", "max_abs_error": maximum}
    except AssertionError as error:
        return {"status": "failed", "max_abs_error": maximum if np.isfinite(maximum) else None, "failure": str(error)}


def check_variant(name, candidate, baseline):
    import jax
    import numpy as np
    from pathlib import Path
    import json
    from tombombadil.alignment import count_codons
    from test.jit_benchmark_variants import make_kernels
    fn, params, expected = protected_case(baseline)
    loss, vg, update, advance = make_kernels(lambda p: -fn(p), make_solver(params), name)
    value, grad = vg(params)
    result = compare((-value, jax.tree.map(lambda x: -x, grad)), expected)
    report = {"protected_reference": result, "cases": []}
    # Reject widened likelihood compilation immediately if it misses the oracle.
    # Update-only candidates still exercise their update and state below.
    if result["status"] == "failed":
        report["status"] = "failed"
        return report
    counts, _ = count_codons("porB3_aligned.fasta")
    fitted = json.loads(Path("test/fixtures/porB3_per_site_map.json").read_text())
    for index in range(8):
        mode = "scalar" if index < 4 else "per-site"
        n = 294 if index % 4 in (0, 1) else 7
        X = counts[:, :n]
        pi = np.full(61, 1/61) if index % 2 == 0 else np.arange(1,62)/1891
        options = dict(include_invariant=index%3==0, aggregate="mean" if index%2 else "sum",
            prior_mode=("none", "current", "stan_constrained", "stan_unconstrained")[index%4],
            estimate_eta=index%2==0, eigen_jitter=index%3!=2, omega_floor=index%3!=1, omega_mode=mode)
        mask = baseline.make_variable_site_mask(X) if index%4 != 3 else np.zeros(n)
        f = baseline.make_log_density_fn(pi, *baseline.prepare_likelihood_transforms(X,pi), X,mask,**options)
        p = baseline.make_base_params(estimate_eta=True, n_sites=n, omega_mode=mode)
        if index%4 == 1:
            natural = dict(fitted["gtr"],omega=np.asarray(fitted["omega"]) if mode=="per-site" else .5)
            p = baseline.natural_to_raw_params(natural,omega_mode=mode)
        elif index%4 >= 2:
            for j,k in enumerate(("alpha","beta","gamma","delta","epsilon","eta","theta")):
                p[k] = baseline.positive_transform_inverse(.31 + .19*j)
            p["omega"] = baseline.positive_transform_inverse(
                np.resize([.009,.01,.011,.3,.8,1.3,2.1],n) if mode=="per-site" else .3)
        solver = make_solver(p)
        base = make_kernels(lambda x: -f(x),solver,"baseline")
        selected = make_kernels(lambda x: -f(x),solver,name)
        expected_result = base[1](p)
        actual = selected[1](p)
        case = {"index":index, "omega_mode":mode, "n_sites":n, "options":options,
                "objective_gradient":compare(actual, expected_result)}
        state = solver.init(p)
        case["update_state"] = compare(selected[2](p,state,expected_result[1]),base[2](p,state,expected_result[1]))
        # Carry both parameters and state; a tiny update difference can amplify
        # through the eigensolver even when each update itself is close.
        bp,bs,bg = p,state,expected_result[1]
        cp,cs,cg = p,state,actual[1]
        case["trajectory"] = []
        for step in range(3):
            bp,bs,bv,bg = base[3](bp,bs,bg)
            cp,cs,cv,cg = selected[3](cp,cs,cg)
            case["trajectory"].append(compare((cp,cs,cv,cg),(bp,bs,bv,bg)))
        report["cases"].append(case)
        jax.clear_caches()
    entries = [c[k] for c in report["cases"] for k in ("objective_gradient","update_state")]
    entries += [t for c in report["cases"] for t in c["trajectory"]]
    report["status"] = "passed" if all(r["status"]=="passed" for r in entries) else "failed"
    return report


def diagnose(baseline):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from test.jit_benchmark_variants import make_kernels
    from tombombadil.gtr import build_GTR, update_GTR
    from tombombadil.alignment import count_codons

    fn,p,expected = protected_case(baseline)
    output = {"protected": {}, "spectra": {}, "finite_difference": {}, "repeated_scalar": {}, "alpha_vjp": {}}
    for name in ("baseline","update","value-grad","objective","separate","fused"):
        vg = make_kernels(lambda x:-fn(x),make_solver(p),name)[1]
        v,g = vg(p)
        output["protected"][name] = dict(compare((-v,jax.tree.map(lambda x:-x,g)),expected),
            value=float(-v), gradient={k:float(-v) for k,v in g.items()})
    counts,_ = count_codons("porB3_aligned.fasta")
    pi = np.full(61,1/61)
    logpi,pm,pmi,pmul = baseline.prepare_likelihood_transforms(counts,pi)
    def matrix(raw):
        x = jax.tree.map(baseline.positive_transform,raw)
        A = build_GTR(*(x[k] for k in ("alpha","beta","gamma","delta","epsilon","eta")),1,pm,pmul)
        return update_GTR(A,x["omega"],pmul)
    for label,raw in (("symmetric",p),("asymmetric",dict(p,alpha=p["alpha"]+.13,beta=p["beta"]-.07,gamma=p["gamma"]+.19,delta=p["delta"]-.11,epsilon=p["epsilon"]+.23))):
        m = matrix(raw)
        w = np.asarray(jnp.linalg.eigvalsh(m))
        wj = np.asarray(jnp.linalg.eigvalsh(m+1e-6*jnp.eye(61)))
        output["spectra"][label] = {"min_gap":float(np.min(np.diff(w))),
            "min_gap_with_jitter":float(np.min(np.diff(wj))),
            "gaps_below_1e_minus_12":int(np.sum(np.diff(wj)<1e-12)),
            "matrix_jit_max_error":float(jnp.max(jnp.abs(m-jax.jit(matrix)(raw))))}
        def alpha_matrix(raw):
            x = jax.tree.map(baseline.positive_transform,raw)
            neutral = build_GTR(*(x[k] for k in ("alpha","beta","gamma","delta","epsilon")),
                                jnp.array(1.,dtype=jnp.float64),1,pm,pmul)
            scale = (x["theta"]/2)/(-jnp.dot(jnp.diag(neutral),pi))
            return baseline.gen_alpha(x["omega"],neutral,pm,pmul,pmi,scale)
        eager_alpha, eager_pullback = jax.vjp(alpha_matrix,raw)
        jit_alpha, jit_pullback = jax.vjp(jax.jit(alpha_matrix),raw)
        obs = jnp.zeros(61).at[15].set(4).at[47].set(19)
        def tail(a):
            return jax.scipy.special.logsumexp(baseline.dirichlet_multinomial_logpmf(obs,a)+logpi)
        cotangent = jax.grad(tail)(eager_alpha)
        output["alpha_vjp"][label] = {
            "forward":compare(eager_alpha,jit_alpha),
            "same_cotangent_pullback":compare(eager_pullback(cotangent),jit_pullback(cotangent)),
            "cotangent":compare(jax.grad(tail)(eager_alpha),jax.grad(tail)(jit_alpha)),
            "scaled_cotangent_linearity":compare(eager_pullback(10*cotangent),jax.tree.map(lambda x:10*x,eager_pullback(cotangent))),
        }
        eager = jax.value_and_grad(fn)(raw)
        compiled = jax.jit(jax.value_and_grad(fn))(raw)
        samples = []
        for h in (1e-3,1e-4,1e-5,1e-6):
            plus,minus=dict(raw),dict(raw)
            plus["delta"]+=h;minus["delta"]-=h
            samples.append({"h":h,"delta_derivative":float((fn(plus)-fn(minus))/(2*h))})
        output["finite_difference"][label] = {"eager_delta":float(eager[1]["delta"]),
            "compiled_delta":float(compiled[1]["delta"]),"sweep":samples}
    for repeat in (1,10):
        X = np.tile(counts,repeat)
        f = baseline.make_log_density_fn(pi,*baseline.prepare_likelihood_transforms(X,pi),X,
            baseline.make_variable_site_mask(X),include_invariant=False,aggregate="sum",
            prior_mode="none",estimate_eta=True,eigen_jitter=True,omega_floor=True,omega_mode="scalar")
        v,g=jax.value_and_grad(f)(p)
        output["repeated_scalar"][str(repeat)]={"data_value":float(v),"gradient":{k:float(x) for k,x in g.items()}}
    import hashlib
    lowered = jax.jit(jax.value_and_grad(fn)).lower(p).as_text()
    output["lowered_ir"] = {"sha256":hashlib.sha256(lowered.encode()).hexdigest(),
                             "characters":len(lowered)}
    return output

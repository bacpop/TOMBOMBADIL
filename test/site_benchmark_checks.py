"""Numerical gates and graph inspection, kept outside timed processes."""
import contextlib
import io
from collections import Counter


def inspect_objective(fn, params):
    import jax
    from jax.ad_checkpoint import print_saved_residuals

    graph = jax.make_jaxpr(jax.value_and_grad(fn))(params)
    shapes = Counter()
    primitives = Counter()

    def visit(jaxpr):
        if hasattr(jaxpr, "jaxpr"):
            jaxpr = jaxpr.jaxpr
        if not hasattr(jaxpr, "eqns"):
            return
        for eqn in jaxpr.eqns:
            primitives[eqn.primitive.name] += 1
            for var in eqn.outvars:
                shape = getattr(getattr(var, "aval", None), "shape", ())
                if shape:
                    shapes[str(shape)] += 1
            for value in eqn.params.values():
                for sub in value if isinstance(value, (list, tuple)) else (value,):
                    visit(sub)
    visit(graph)
    saved = io.StringIO()
    with contextlib.redirect_stdout(saved):
        print_saved_residuals(fn, params)
    return {"primitives": dict(primitives), "array_shapes": dict(shapes),
            "saved_residuals": saved.getvalue()}


def compare_trees(actual, expected):
    import jax
    import numpy as np
    errors = []
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise AssertionError("Non-finite objective or gradient")
        errors.append(float(np.max(np.abs(a-b))))
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)
    return max(errors, default=0.0)


def check_variant(candidate, baseline):
    import jax
    import numpy as np
    from test.site_benchmark_variants import site_mapper
    from tombombadil import sample
    from test.test_baselines import TestLikelihoodGradientReference

    original = sample.make_log_density_fn
    try:
        sample.make_log_density_fn = candidate.make_log_density_fn
        TestLikelihoodGradientReference().test_likelihood_and_gradient_match_fixed_reference()
    finally:
        sample.make_log_density_fn = original
    report = {"protected_reference": "passed", "cases": []}
    # Pairwise coverage avoids an enormous Cartesian product of expensive AD
    # compilations while varying every objective switch and boundary condition.
    name = getattr(candidate, "_variant_name", "baseline")
    batch = 16 if name in ("baseline", "production", "sequential") else int(name.removeprefix("checkpoint").removeprefix("batch"))
    for i, n in enumerate(sorted({1, batch-1, batch, batch+1, batch*2+3})):
        pi = np.full(61, 1/61) if i % 2 == 0 else np.arange(1, 62) / 1891
        X = np.zeros((61, n), dtype=np.int64)
        for col in range(n):
            X[col % 61, col] = 4
            X[(col+17) % 61, col] = 19 if col % 3 else 0
        transforms = sample.prepare_likelihood_transforms(X, pi)
        params = sample.make_base_params(estimate_eta=True, n_sites=n, omega_mode="per-site")
        if i:
            for j, key in enumerate(("alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta")):
                params[key] = sample.positive_transform_inverse(0.3 + j*0.21)
        params["omega"] = sample.positive_transform_inverse(np.resize([0.009, 0.01, 0.011, 0.3, 0.8, 2.1], n))
        mask = sample.make_variable_site_mask(X) if i != 3 else np.zeros(n)
        options = dict(include_invariant=bool(i % 2), aggregate="sum" if i % 2 else "mean",
                       prior_mode=("none", "current", "stan_constrained", "stan_unconstrained", "none")[i],
                       estimate_eta=bool(i % 2), eigen_jitter=bool(i % 2), omega_floor=(i != 2),
                       omega_mode="per-site")
        expected = jax.value_and_grad(baseline.make_log_density_fn(pi, *transforms, X, mask, **options))(params)
        actual = jax.value_and_grad(candidate.make_log_density_fn(pi, *transforms, X, mask, **options))(params)
        try:
            error = compare_trees(actual, expected)
            report["cases"].append({"n_sites": n, "status": "passed", "max_abs_error": error})
        except AssertionError as error:
            report["cases"].append({"n_sites": n, "status": "failed", "failure": str(error)})
        # Ordered losses are checked separately from reductions and priors.
        natural = jax.tree.map(sample.positive_transform, params)
        model = sample.codon_site_log_likelihood if options["eigen_jitter"] else sample.codon_site_log_likelihood_no_jitter
        args = tuple(natural[k] for k in ("alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "omega")) + (pi, *transforms, X)
        expected_losses = site_mapper(model, "baseline", "per-site", n)(*args)
        actual_losses = site_mapper(model, name, "per-site", n, force=True)(*args)
        try:
            compare_trees(actual_losses, expected_losses)
            report["cases"][-1]["ordered_losses"] = "passed"
        except AssertionError as error:
            report["cases"][-1].update(ordered_losses="failed", loss_failure=str(error))
        jax.clear_caches()
    report["status"] = "passed" if all(c["status"] == "passed" and c["ordered_losses"] == "passed" for c in report["cases"]) else "failed"
    return report


def kernel_function(module, counts, omega_mode):
    import jax
    import numpy as np
    from test.site_benchmark_variants import site_mapper

    pi = np.full(61, 1/61)
    transforms = module.prepare_likelihood_transforms(counts, pi)
    name = getattr(module, "_variant_name", "baseline")
    mapped = site_mapper(module.codon_site_log_likelihood, name, omega_mode,
                         counts.shape[1], force=True)
    def losses(raw):
        x = jax.tree.map(module.positive_transform, raw)
        args = tuple(x[k] for k in ("alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "omega"))
        return mapped(*args, pi, *transforms, counts)
    return losses


def check_repeated_report(actual, reference):
    """Validate a large objective using only one repeat of baseline data AD.

    Undo the small-input prior, apply the repeated-data identity, then add the
    full-input prior exactly once. This never builds the large baseline graph.
    """
    import jax
    import numpy as np
    from pathlib import Path
    import json
    from tombombadil import sample

    n = reference["n_sites"]
    repeats = actual["n_sites"] // n
    assert repeats * n == actual["n_sites"]
    mode = actual["arguments"]["omega_mode"]
    assert mode == reference["arguments"]["omega_mode"]
    assert actual["arguments"]["params"] == reference["arguments"]["params"]
    small = sample.make_base_params(estimate_eta=True, n_sites=n, omega_mode=mode)
    if actual["arguments"]["params"] == "fitted":
        ref = json.loads(Path("test/fixtures/porB3_per_site_map.json").read_text())
        small = sample.natural_to_raw_params(dict(ref["gtr"], omega=np.asarray(ref["omega"]) if mode == "per-site" else 0.5), omega_mode=mode)
    large = dict(small)
    if mode == "per-site":
        large["omega"] = np.tile(small["omega"], repeats)
    def prior(params, size):
        return sample.prior_log_likelihood(params, size, prior_mode="stan_unconstrained",
            estimate_eta=True, omega_mode=mode, aggregate="sum")
    small_prior, small_grad = jax.value_and_grad(prior)(small, n)
    large_prior, large_grad = jax.value_and_grad(prior)(large, n*repeats)
    value = (reference["objective"] - float(small_prior)) * repeats + float(large_prior)
    gradient = {}
    for key, leaf in reference["gradient"].items():
        data_grad = np.asarray(leaf) - np.asarray(small_grad[key])
        data_grad = np.tile(data_grad, repeats) if key == "omega" and mode == "per-site" else repeats*data_grad
        gradient[key] = data_grad + np.asarray(large_grad[key])
    observed = {k: np.asarray(v) for k, v in actual["gradient"].items()}
    return compare_trees((actual["objective"], observed), (value, gradient))

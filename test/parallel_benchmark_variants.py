"""PERF-05 CPU execution alternatives; production never imports this module."""
import ast
from pathlib import Path
import subprocess
from types import ModuleType

BASELINE_REVISION = "c9762e18c101102e487b417e558f51ba09384aa8"
VARIANTS = ("baseline", "shard", "shard-jit", "local64", "local256", "shard64", "shard256")


def baseline_module():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_REVISION}:tombombadil/sample.py"],
        cwd=Path(__file__).resolve().parents[1], text=True)
    tree = ast.parse(source)
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "make_log_density_fn")
    factory.name = "make_ordered_loss_fn"
    body = next(n for n in factory.body if isinstance(n, ast.FunctionDef))
    index = next(i for i, n in enumerate(body.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "losses")
    body.body = body.body[:index+1] + [ast.Return(ast.Name("losses", ast.Load()))]
    module = ModuleType("tombombadil._perf05_baseline")
    module.__package__ = "tombombadil"
    exec(compile(source, "<PERF-05 frozen baseline>", "exec"), module.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module([factory], [])),
                 "<PERF-05 ordered baseline losses>", "exec"), module.__dict__)
    module.source = source
    return module


class Evaluation:
    """Positive log density and explicit gradient, including one full prior."""

    def __init__(self, module, counts, pi, mask, options, variant="baseline", devices=1):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        self.module, self.variant, self.options = module, variant, options
        self.n = counts.shape[1]
        self.per_site = options["omega_mode"] == "per-site"
        self.chunk = int(variant.removeprefix("local").removeprefix("shard")) if variant not in ("baseline", "shard", "shard-jit") else None
        transforms = module.prepare_likelihood_transforms(counts, pi)
        self.reference = module.make_log_density_fn(pi, *transforms, counts, mask, **options)
        self.reference_losses = module.make_ordered_loss_fn(pi, *transforms, counts, mask, **options)
        self.prior = lambda p: module.prior_log_likelihood(p, self.n,
            **{k: options[k] for k in ("prior_mode", "estimate_eta", "omega_mode", "aggregate")})
        self.denominator = (1. if options["aggregate"] == "sum" else
                            float(self.n if options["include_invariant"] else max(np.sum(mask), 1)))
        if variant == "baseline":
            self.value = self.reference
            self.value_grad = jax.value_and_grad(self.reference)
            self.losses = self.reference_losses
            return
        if variant not in VARIANTS:
            raise ValueError(variant)
        sharded = variant.startswith("shard")
        d = devices if sharded else 1
        multiple = d * (self.chunk or 1)
        self.padded = ((self.n + multiple - 1) // multiple) * multiple
        padded_counts = np.pad(counts.T, ((0, self.padded-self.n), (0, 0)), mode="edge")
        weights = np.ones(self.n) if options["include_invariant"] else np.asarray(mask, dtype=float)
        weights = np.pad(weights, (0, self.padded-self.n))
        self.mesh = Mesh(np.array(jax.devices()[:d]), ("site",))
        spec = P("site") if sharded else P()
        self.X = jax.device_put(padded_counts, NamedSharding(self.mesh, spec))
        self.weights = jax.device_put(weights, NamedSharding(self.mesh, spec))
        self.param_spec = lambda p: {k: P("site") if self.per_site and k == "omega" else P() for k in p}

        def losses(p, X):
            f = module.make_ordered_loss_fn(pi, *transforms, X.T, jnp.ones(X.shape[0]), **options)
            return f(p)

        def contribution(p, X, w):
            return jnp.sum(losses(p, X) * w) / self.denominator

        def chunked(p, X, w):
            shared = {k: v for k, v in p.items() if not (self.per_site and k == "omega")}
            def one(inputs):
                x, weight, omega = inputs
                def f(s, o):
                    return contribution(dict(s, omega=o) if self.per_site else s, x, weight)
                return jax.value_and_grad(f, argnums=(0, 1))(shared, omega)
            b = self.chunk
            omega = p["omega"].reshape(-1, b) if self.per_site else jnp.zeros((X.shape[0]//b, b))
            inputs = X.reshape(-1, b, 61), w.reshape(-1, b), omega
            if sharded:
                values, (shared_grads, omega_grads) = jax.lax.map(one, inputs)
                value = jnp.sum(values)
                gradient = jax.tree.map(lambda x: jnp.sum(x, axis=0), shared_grads)
                if self.per_site:
                    gradient["omega"] = omega_grads.reshape(-1)
            else:
                # Python accumulation avoids an enclosing AD/compilation boundary.
                value, gradient, omega_parts = jnp.array(0.), jax.tree.map(jnp.zeros_like, shared), []
                for i in range(X.shape[0]//b):
                    v, (g, o) = one(tuple(x[i] for x in inputs))
                    value += v
                    gradient = jax.tree.map(lambda a, b: a+b, gradient, g)
                    omega_parts.append(o)
                if self.per_site:
                    gradient["omega"] = jnp.concatenate(omega_parts)
            if sharded:
                value = jax.lax.psum(value, "site")
                gradient = {k: v if self.per_site and k == "omega" else jax.lax.psum(v, "site")
                            for k, v in gradient.items()}
            return value, gradient

        self._local_losses = losses
        self._contribution = contribution
        self._chunked = chunked
        self._sharded = sharded
        self._kernels = {}
        self.value = self._value
        self.value_grad = self._value_grad
        self.losses = self._losses

    def _prepare(self, params):
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding
        p = dict(params)
        if self.per_site:
            p["omega"] = jnp.pad(p["omega"], (0, self.padded-self.n), mode="edge")
        return jax.tree.map(lambda x, spec: jax.device_put(x, NamedSharding(self.mesh, spec)),
                            p, self.param_spec(p))

    def _kernel(self, params, kind):
        import jax
        from jax.sharding import PartitionSpec as P
        if kind not in self._kernels:
            ps = self.param_spec(params)
            if kind == "losses":
                fn = lambda p, X, w: self._local_losses(p, X)
                output = P("site")
            elif kind == "gradient" and self.chunk:
                fn, output = self._chunked, (P(), ps)
            else:
                def fn(p, X, w):
                    v = self._contribution(p, X, w)
                    return jax.lax.psum(v, "site") if self._sharded else v
                output = P()
            if self._sharded:
                fn = jax.shard_map(fn, mesh=self.mesh,
                    in_specs=(ps, P("site"), P("site")), out_specs=output,
                    # The unchanged alpha loops start with replicated zeros.
                    # Use explicit collectives and test their replication on
                    # analytic models instead of changing those loop carries.
                    check_vma=False)
            if self.variant=="shard-jit":
                fn=jax.jit(fn)
            if kind == "gradient" and not self.chunk:
                fn = jax.value_and_grad(fn)
            self._kernels[kind] = fn
        return self._kernels[kind]

    def _losses(self, params):
        p = self._prepare(params)
        return self._kernel(p, "losses")(p, self.X, self.weights)[:self.n]

    def _value(self, params):
        p = self._prepare(params)
        return self._kernel(p, "value")(p, self.X, self.weights) + self.prior(params)

    def _value_grad(self, params):
        import jax
        p = self._prepare(params)
        v, g = self._kernel(p, "gradient")(p, self.X, self.weights)
        if self.per_site:
            g["omega"] = g["omega"][:self.n]
        prior, pg = jax.value_and_grad(self.prior)(params)
        return v+prior, jax.tree.map(lambda a, b: a+b, g, pg)


def optimizer_module(evaluation):
    """Frozen Python optimiser with only its numerical evaluation seam replaced."""
    module = baseline_module()
    source = module.source.replace("loss_and_grad = jax.value_and_grad(loss_fn)",
        "loss_and_grad = _negative_value_grad")
    import jax
    def negative(p):
        value, gradient = evaluation.value_grad(p)
        return -value, jax.tree.map(lambda x: -x, gradient)
    module.__dict__["_negative_value_grad"] = negative
    exec(compile(source, "<PERF-05 optimizer seam>", "exec"), module.__dict__)
    return module


def parallel_block_value(evaluation, params):
    """Bridge single-device optimizer state to the parallel omega objective.

    Replicate the unpadded inputs so the full prior and mapped likelihood use
    the same device set. The mapped kernel partitions/pads omega internally;
    AD through device_put returns gradients to the optimizer's input placement.
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec
    replicated = NamedSharding(evaluation.mesh, PartitionSpec())
    params = jax.tree.map(lambda x: jax.device_put(x, replicated), params)
    return evaluation.value(params)


def alternating(evaluation, params, rounds=(1, 5), max_updates=500, callback=None, on_cycle=None):
    """Experimental block Adam: inactive blocks and their moments stay frozen."""
    import jax
    import optax
    from time import perf_counter

    p = dict(params)
    value = evaluation.value
    if evaluation.variant != "baseline":
        value = lambda p: parallel_block_value(evaluation, p)
    keys = [k for k in p if k != "omega" and (k != "eta" or evaluation.options["estimate_eta"])]
    blocks = (keys, ["omega"])
    solvers = [optax.adam(optax.cosine_decay_schedule(.2, 500, alpha=.001/.2)) for _ in blocks]
    states = [solver.init({k: p[k] for k in ks}) for solver, ks in zip(solvers, blocks)]
    counts = [0, 0]
    started = perf_counter()
    initial = float(value(p))
    history = [{"updates":0,"seconds":perf_counter()-started,"objective":initial}]
    cycles = 0
    while sum(counts) < max_updates:
        for index, (ks, length) in enumerate(zip(blocks, rounds)):
            # A new closure per phase captures the latest frozen block. It is
            # reused within that phase; no cache survives a shared update.
            frozen = dict(p)
            def loss(active):
                # Parallel experiment distributes only omega rounds. Shared
                # rounds use the frozen original objective in both methods.
                fn = evaluation.reference if index==0 and evaluation.variant!="baseline" else value
                return -fn(dict(frozen, **active))
            vg = jax.value_and_grad(loss)
            for _ in range(min(length, max_updates-sum(counts))):
                active = {k:p[k] for k in ks}
                _, gradient = vg(active)
                updates, states[index] = solvers[index].update(gradient, states[index], active)
                p.update(optax.apply_updates(active, updates))
                jax.block_until_ready(p)
                counts[index] += 1
                if callback:
                    callback(index, p, states, tuple(counts))
        cycles += 1
        objective = float(value(p))
        history.append({"updates":sum(counts),"seconds":perf_counter()-started,
                        "objective":objective})
        if on_cycle:
            on_cycle(history, tuple(counts))
    return {"params":p,"states":states,"block_updates":counts,"cycles":cycles,
            "history":history,"seconds":perf_counter()-started}

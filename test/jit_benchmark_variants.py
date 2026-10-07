"""PERF-04 alternatives with the exact post-PERF-03 source as baseline."""
import ast
from pathlib import Path
import subprocess
from types import ModuleType

BASELINE_REVISION = "0962b3e2e9f5d30af083165e797189f3bb731cf0"
VARIANTS = ("baseline", "update", "value-grad", "objective", "separate", "fused", "production")
TRACES = {}


def baseline_source(path=None):
    if path:
        return Path(path).read_text()
    return subprocess.check_output(
        ["git", "show", f"{BASELINE_REVISION}:tombombadil/sample.py"],
        cwd=Path(__file__).resolve().parents[1], text=True)


def make_kernels(loss_fn, solver, name):
    import jax
    import optax

    def compiled(fn, label):
        def traced(*args):
            TRACES[label] = TRACES.get(label, 0) + 1
            return fn(*args)
        return jax.jit(traced)

    def update(params, state, grad):
        updates, state = solver.update(grad, state, params)
        return optax.apply_updates(params, updates), state

    if name == "objective":
        loss_fn = compiled(loss_fn, "objective")
    value_grad = jax.value_and_grad(loss_fn)
    if name in ("value-grad", "separate", "fused"):
        value_grad = compiled(value_grad, "value_grad")
    if name in ("update", "separate", "fused"):
        update = compiled(update, "update")

    def advance(params, state, grad):
        params, state = update(params, state, grad)
        loss, grad = value_grad(params)
        return params, state, loss, grad

    if name == "fused":
        advance = compiled(advance, "advance")
    return loss_fn, value_grad, update, advance


def make_variant(name, source=None):
    if name == "production":
        from tombombadil import sample
        return sample
    if name not in VARIANTS:
        raise ValueError(name)
    source = baseline_source() if source is None else source
    if name != "baseline":
        tree = ast.parse(source)
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        fn = functions["_optimize_params"]
        fn.args.args.append(ast.arg(arg="_kernels"))
        fn.args.defaults.append(ast.Constant(value=None))
        for i, node in enumerate(fn.body):
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "loss_and_grad":
                fn.body[i:i+1] = ast.parse('''loss_fn, loss_and_grad, update, advance = (
    _make_kernels(loss_fn, solver, _variant_name) if _kernels is None else _kernels)
''').body
                break
        loop = next(n for n in fn.body if isinstance(n, ast.For))
        # Fuse only the non-final steps; preserve the final value-only call.
        loop.body[:2] = ast.parse('''if _variant_name == "fused" and step < n_iter:
    params, opt_state, fused_loss, fused_grad = advance(params, opt_state, grad)
else:
    params, opt_state = update(params, opt_state, grad)
''').body
        condition = next(n for n in loop.body if isinstance(n, ast.If)
                         and ast.unparse(n.test) == "step < n_iter")
        condition.body = ast.parse('''if _variant_name == "fused":
    current_loss, next_grad = fused_loss, fused_grad
else:
    current_loss, next_grad = loss_and_grad(params)
''').body
        fit = functions["_run_map_replicates"]
        rep = next(n for n in fit.body if isinstance(n, ast.For))
        # Schedule/solver are immutable; numerical callables live for this fit.
        # Optimizer state still initializes independently inside each replicate.
        setup = rep.body[1:3]
        del rep.body[1:3]
        index = fit.body.index(rep)
        fit.body[index:index] = setup + ast.parse(
            '_kernels = _make_kernels(lambda p: -fn(p), solver, _variant_name)').body
        for node in ast.walk(rep):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_optimize_params":
                node.keywords.append(ast.keyword(arg="_kernels", value=ast.Name(id="_kernels", ctx=ast.Load())))
        source = ast.unparse(ast.fix_missing_locations(tree))
    module = ModuleType("tombombadil._perf04_" + name.replace("-", "_"))
    module.__package__ = "tombombadil"
    module.__dict__.update(_make_kernels=make_kernels, _variant_name=name)
    exec(compile(source, f"<PERF-04 {name}>", "exec"), module.__dict__)
    return module


def install_variant(module):
    from tombombadil import sample
    for name in ("make_log_density_fn", "_optimize_params", "_run_map_replicates"):
        setattr(sample, name, getattr(module, name))

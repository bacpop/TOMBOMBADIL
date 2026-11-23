"""PERF-03 site batching alternatives; the baseline is exact git source."""

import ast
from pathlib import Path
import subprocess
from types import ModuleType

BASELINE_REVISION = "6de8aa61c7b605701cb97680385cddc3d2f6a8d5"
VARIANTS = ("baseline", "sequential", "batch16", "batch64", "batch256", "batch1024",
            "checkpoint16", "checkpoint64", "checkpoint256", "checkpoint1024", "production")


def baseline_source(path=None):
    if path:
        return Path(path).read_text()
    return subprocess.check_output(
        ["git", "show", f"{BASELINE_REVISION}:tombombadil/sample.py"],
        cwd=Path(__file__).resolve().parents[1], text=True)


def site_mapper(model_fn, name, omega_mode, n_sites, *, force=False):
    import jax
    import jax.numpy as jnp

    axes = (None,) * 7 + (None if omega_mode == "scalar" else 0,) + (None,) * 5 + (1,)
    whole = jax.vmap(model_fn, in_axes=axes)
    if name == "baseline" or omega_mode == "scalar":
        return whole
    checkpoint = name.startswith("checkpoint")
    batch_size = 1 if name == "sequential" else int(name.removeprefix("checkpoint").removeprefix("batch"))
    if not force and n_sites <= max(512, batch_size):
        return whole

    def mapped(*args):
        def one(pair):
            omega, counts = pair
            return model_fn(*args[:7], omega, *args[8:13], counts)

        omega, counts = args[7], args[13].T
        if name == "sequential":
            return jax.lax.map(one, (omega, counts))
        if not checkpoint:
            return jax.lax.map(one, (omega, counts), batch_size=batch_size)
        # Rematerialize a whole chunk in reverse mode, including the site JIT.
        # The leftover sites are evaluated separately, without synthetic padding.
        stop = n_sites // batch_size * batch_size
        chunks = (omega[:stop].reshape(-1, batch_size),
                  counts[:stop].reshape(-1, batch_size, counts.shape[-1]))
        losses = jax.lax.map(jax.checkpoint(jax.vmap(one)), chunks).reshape(-1)
        if stop < n_sites:
            losses = jnp.concatenate((losses, jax.vmap(one)((omega[stop:], counts[stop:]))))
        return losses

    return mapped


def make_variant(name, source=None, *, force=False):
    if name == "production":
        from tombombadil import sample
        return sample
    if name not in VARIANTS:
        raise ValueError(name)
    source = baseline_source() if source is None else source
    tree = ast.parse(source)
    if name != "baseline":
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "make_log_density_fn")
        for i, node in enumerate(fn.body):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "batched_loss" for t in node.targets):
                fn.body[i] = ast.parse("batched_loss = _site_mapper(model_fn, _variant_name, omega_mode, X.shape[1], force=_force)").body[0]
        source = ast.unparse(ast.fix_missing_locations(tree))
    module = ModuleType("tombombadil._perf03_" + name)
    module.__package__ = "tombombadil"
    module.__dict__.update(_site_mapper=site_mapper, _variant_name=name, _force=force)
    exec(compile(source, f"<PERF-03 {name}>", "exec"), module.__dict__)
    return module


def install_variant(module):
    from tombombadil import sample
    sample.make_log_density_fn = module.make_log_density_fn

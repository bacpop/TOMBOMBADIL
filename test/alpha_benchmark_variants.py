"""PERF-02 alternatives, isolated from production; load the exact git baseline."""

import ast
from pathlib import Path
import subprocess
from types import ModuleType


BASELINE_REVISION = "4273181fae048963e9ad60fde6d2a86eeb925f75"
VARIANTS = (
    "baseline", "scale", "matmul", "einsum", "loop", "frequency", "normalize",
    "combined-matmul", "combined-einsum", "combined-loop",
    "scale-frequency", "scale-loop-frequency", "normalize-loop",
    "scale-loops-frequency", "vmap", "scale-vmap-frequency",
    "loops", "loop-frequency", "loops-frequency", "loops-unroll4", "production",
)


def baseline_source(path=None):
    if path is not None:
        return Path(path).read_text()
    return subprocess.check_output(
        ["git", "show", f"{BASELINE_REVISION}:tombombadil/likelihood.py"],
        cwd=Path(__file__).resolve().parents[1], text=True,
    )


def make_variant(name, source=None):
    if name == "production":
        from tombombadil import likelihood
        return likelihood
    if name not in VARIANTS:
        raise ValueError(name)
    source = baseline_source() if source is None else source
    if name != "baseline":
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_gen_alpha_impl")
        body = []
        combined = name.startswith("combined-")
        reconstruction = name.removeprefix("combined-")
        if name in ("scale-loop-frequency", "scale-loops-frequency", "scale-vmap-frequency"):
            reconstruction = name.split("-")[1].replace("loops", "loop")
        if name in ("loops", "loop-frequency", "loops-frequency", "loops-unroll4"):
            reconstruction = "loop"
        scale = name == "scale" or combined or name.startswith("scale-")
        frequency = name == "frequency" or combined or name.endswith("-frequency")
        for node in fn.body:
            assigned = (node.targets[0].id if isinstance(node, ast.Assign)
                        and isinstance(node.targets[0], ast.Name) else None)
            replacement = None
            if assigned == "V_inv" and scale:
                replacement = "V_inv = v * E[None, :]"
            elif isinstance(node, ast.For) and "Va =" in ast.unparse(node):
                if reconstruction == "matmul":
                    replacement = "m_AB = V_inv @ v.T"
                elif reconstruction == "einsum":
                    replacement = 'm_AB = jnp.einsum("ik,jk->ij", V_inv, v)'
                elif reconstruction == "vmap":
                    replacement = "m_AB = jax.vmap(lambda row: jnp.sum(row[:, None] * V_inv.T, axis=0))(v).T"
                elif reconstruction == "loop":
                    replacement = '''def column(i, matrix):
    Va = jnp.repeat(v[i, :], 61).reshape(61, 61)
    return matrix.at[:, i].set(jnp.sum(Va * V_inv.T, axis=0))
m_AB = jax.lax.fori_loop(0, 61, column, m_AB)'''
            elif assigned == "m_AB" and "pimatinv" in ast.unparse(node):
                if frequency:
                    replacement = "m_AB = (jnp.diag(pimatinv)[:, None] * m_AB) * jnp.diag(pimat)[None, :]"
            elif isinstance(node, ast.For):
                if name in ("normalize-loop", "scale-loops-frequency", "loops", "loops-frequency", "loops-unroll4"):
                    replacement = """def normalize_column(i, matrix):
    matrix = matrix.at[:, i].set(matrix[:, i] / matrix[i, i])
    return matrix.at[i, i].set(1e-6)
m_AB = jax.lax.fori_loop(0, 61, normalize_column, m_AB)"""
                elif name == "normalize" or combined:
                    replacement = '''m_AB = m_AB / jnp.diag(m_AB)[None, :]
m_AB = m_AB.at[jnp.diag_indices(61)].set(1e-6)'''
            if replacement is not None and name == "loops-unroll4":
                replacement = replacement.replace(
                    "column, m_AB)", "column, m_AB, unroll=4)")
            body.extend(ast.parse(replacement).body if replacement is not None else [node])
        fn.body = body
        source = ast.unparse(ast.fix_missing_locations(tree))
    module = ModuleType("tombombadil._perf02_" + name.replace("-", "_"))
    module.__package__ = "tombombadil"
    exec(compile(source, f"<PERF-02 {name}>", "exec"), module.__dict__)
    return module


def install_variant(module):
    """Patch direct imports before model tracing in a fresh process."""
    from tombombadil import likelihood, sample
    for name in ("gen_alpha", "gen_alpha_no_jitter"):
        setattr(sample, name, getattr(module, name))
        setattr(likelihood, name, getattr(module, name))

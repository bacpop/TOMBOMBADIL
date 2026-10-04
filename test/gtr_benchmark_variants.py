"""PERF-01 alternatives; never imported by production code.

The baseline is the exact pre-PERF-01 source, loaded from git (or an exported
source file). This avoids approximating its tracing/compilation behavior.
"""

import ast
from pathlib import Path
import subprocess
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np


BASELINE_REVISION = "d22f582c9ecf1c438254824e6c6f227db98b231d"
VARIANTS = (
    "baseline", "diag-mask", "diag-einsum", "omega-where", "omega-multiply",
    "frequency", "rates-lookup", "rates-sum", "rates-einsum",
    "combined-lookup", "combined-sum", "combined-einsum", "production",
)


def baseline_source(path=None):
    if path is not None:
        return Path(path).read_text()
    return subprocess.check_output(
        ["git", "show", f"{BASELINE_REVISION}:tombombadil/gtr.py"],
        cwd=Path(__file__).resolve().parents[1], text=True,
    )


def topology(source):
    """Extract and cross-check all three original representations of omega."""
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def assigned(fn, name):
        return next(n.value.args[0] for n in functions[fn].body
                    if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == name)

    indices = np.array(ast.literal_eval(assigned("build_GTR", "IDX")))
    names = ("alpha", "beta", "gamma", "delta", "epsilon", "eta")
    rate_index = np.full((61, 61), 6, dtype=np.int32)
    nonsyn = np.zeros((61, 61), dtype=bool)
    for (i, j), value in zip(indices, assigned("build_GTR", "values").elts, strict=True):
        rate = value.left if isinstance(value, ast.BinOp) else value
        rate_index[i, j] = names.index(rate.id)
        nonsyn[i, j] = isinstance(value, ast.BinOp)
    update_indices = np.array(ast.literal_eval(assigned("update_GTR", "IDX")))
    update_mask = np.zeros((61, 61), dtype=bool)
    update_mask[tuple(update_indices.T)] = True
    np.testing.assert_array_equal(nonsyn, update_mask)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("omega_mat", "non_omega_mat"):
                mask = np.array(ast.literal_eval(node.value.args[0]), dtype=bool)
                expected = nonsyn if node.targets[0].id == "omega_mat" else ~nonsyn
                np.testing.assert_array_equal(mask, expected)
    assert len(indices) == len(set(map(tuple, indices))) == 526
    assert nonsyn.sum() == 392
    return rate_index, nonsyn


def make_variant(name, source=None):
    if name == "production":
        from tombombadil import gtr
        return gtr
    if name not in VARIANTS:
        raise ValueError(name)
    source = baseline_source() if source is None else source
    if name == "frequency" or name.startswith("combined-"):
        source = source.replace(
            "M = jnp.matmul(jnp.matmul(M, pimat).T, pimat).T",
            "M = (M * jnp.diag(pimat)[None, :]) * jnp.diag(pimat)[:, None]",
        )
    module = ModuleType("gtr_" + name.replace("-", "_"))
    exec(compile(source, "<PERF-01 baseline>", "exec"), module.__dict__)
    if name in ("baseline", "frequency"):
        return module

    rate_index, nonsyn = topology(source)
    eye = jnp.eye(61, dtype=bool)
    nonsyn = jnp.asarray(nonsyn)
    if name.startswith("diag-") or name.startswith("combined-"):
        @jax.jit
        def diag_update(matrix, pimult):
            offdiag = jnp.where(eye, 0.0, matrix)
            if name == "diag-einsum":
                diagonal = -jnp.einsum("ij,ij->i", offdiag, pimult)
            else:
                diagonal = -jnp.sum(offdiag * pimult, axis=1)
            return jnp.where(eye, diagonal[:, None], offdiag)

        module.diag_update = diag_update

    if name.startswith("omega-") or name.startswith("combined-"):
        @jax.jit
        def update_GTR(matrix, omega, pimult):
            if name == "omega-multiply":
                multiplier = nonsyn.astype(matrix.dtype) * omega + (~nonsyn).astype(matrix.dtype)
            else:
                multiplier = jnp.where(nonsyn, omega, 1.0)
            return module.diag_update(matrix * multiplier, pimult)

        module.update_GTR = update_GTR

    if name.startswith("rates-") or name.startswith("combined-"):
        masks = jnp.asarray(np.arange(6)[:, None, None] == rate_index[None, :, :])
        rate_index = jnp.asarray(rate_index)

        @jax.jit
        def build_GTR(alpha, beta, gamma, delta, epsilon, eta, omega, pimat, pimult):
            rates = jnp.stack((alpha, beta, gamma, delta, epsilon, eta))
            if name.endswith("lookup"):
                matrix = jnp.concatenate((rates, jnp.zeros(1, dtype=rates.dtype)))[rate_index]
            elif name.endswith("sum"):
                matrix = sum(rates[i] * masks[i] for i in range(6))
            else:
                matrix = jnp.einsum("k,kij->ij", rates, masks.astype(rates.dtype))
            matrix = matrix * jnp.where(nonsyn, omega, 1.0)
            if name.startswith("combined-"):
                matrix = (matrix * jnp.diag(pimat)[None, :]) * jnp.diag(pimat)[:, None]
            else:
                matrix = jnp.matmul(jnp.matmul(matrix, pimat).T, pimat).T
            return module.diag_update(matrix, pimult)

        module.build_GTR = build_GTR
    return module


def install_variant(module):
    """Install before tracing any model functions, in a fresh process."""
    from tombombadil import sample, likelihood
    sample.build_GTR = module.build_GTR
    likelihood.build_GTR = module.build_GTR
    likelihood.update_GTR = module.update_GTR

"""Numerical screens and shared inputs for PERF-02 (no extra objective JIT)."""

import jax
import jax.numpy as jnp
import numpy as np

from tombombadil.gtr import build_GTR


RTOL, ATOL = 1e-6, 1e-8


def cases(module):
    for frequency in ("uniform", "nonuniform"):
        pi = np.ones(61) if frequency == "uniform" else np.arange(1, 62, dtype=float)
        pi /= pi.sum()
        pimat, pimult, pimatinv = jax.device_put((
            np.diag(np.sqrt(pi)), np.sqrt(pi[None, :] / pi[:, None]),
            np.diag(1 / np.sqrt(pi)),
        ))
        for jitter in (False, True):
            alpha = module.gen_alpha if jitter else module.gen_alpha_no_jitter
            # Bind closure arguments: cases are also consumed after iteration.
            def fn(p, alpha=alpha, pimat=pimat, pimult=pimult, pimatinv=pimatinv):
                neutral = build_GTR(*p[:6], 1.0, pimat, pimult)
                return alpha(p[6], neutral, pimat, pimult, pimatinv, p[7])
            for omega in (0.1, 1.0, 3.0):
                params = jax.device_put(np.array([0.7, 1.3, 0.9, 1.7, 0.4, 1.1, omega, 0.5]))
                yield f"{frequency}/{omega}/jitter-{jitter}", fn, params


def kernel_cases(module):
    weight = jax.device_put(np.sin(np.arange(61 * 61).reshape(61, 61)))
    previous = None
    for key, fn, params in cases(module):
        if fn is not previous:
            forward = jax.jit(fn)
            derivative = jax.jit(jax.value_and_grad(
                lambda p, fn=fn: jnp.sum(fn(p) * weight)))
            previous = fn
        yield key + "/value", forward, params
        yield key + "/gradient", derivative, params


def check_variant(module, baseline):
    errors = {}
    for (key, fn, params), (_, reference, _) in zip(
        kernel_cases(module), kernel_cases(baseline), strict=True
    ):
        actual, expected = fn(params), reference(params)
        differences = []
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), key
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=key)
            differences.append(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
        errors[key] = max(differences)
    return errors

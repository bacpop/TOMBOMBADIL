"""Independent alpha-matrix and derivative checks, including frequency orientation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tombombadil import likelihood
from tombombadil.gtr import build_GTR
from test.test_gtr import reference_matrix


jax.config.update("jax_enable_x64", True)
RTOL, ATOL = 1e-6, 1e-8


def reference_alpha(params, pi, jitter):
    matrix = reference_matrix(params[:7], pi) + jitter * np.eye(61)
    values, vectors = np.linalg.eigh(matrix, UPLO="U")
    spectral = (vectors / (1 - 2 * params[7] * values)) @ vectors.T
    transformed = spectral * np.sqrt(pi[None, :] / pi[:, None])
    transformed /= np.diag(transformed).copy()[None, :]
    np.fill_diagonal(transformed, 1e-6)
    transformed[transformed < 0] = 1e-6
    return transformed.T + np.eye(61)


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("jitter", [0.0, 1e-6])
def test_alpha_values_and_derivatives(nonuniform, jitter):
    pi = np.arange(1, 62, dtype=float) if nonuniform else np.ones(61)
    pi /= pi.sum()
    pimat = jnp.array(np.diag(np.sqrt(pi)))
    pimatinv = jnp.array(np.diag(1 / np.sqrt(pi)))
    pimult = jnp.array(np.sqrt(pi[None, :] / pi[:, None]))
    alpha = likelihood.gen_alpha if jitter else likelihood.gen_alpha_no_jitter
    weight = np.sin(np.arange(61 * 61).reshape(61, 61))

    def fn(p):
        neutral = build_GTR(*p[:6], 1.0, pimat, pimult)
        return alpha(p[6], neutral, pimat, pimult, pimatinv, p[7])

    derivative = jax.jit(jax.grad(lambda p: jnp.sum(fn(p) * weight)))
    for omega in (0.1, 1.0, 3.0):
        params = np.array([0.7, 1.3, 0.9, 1.7, 0.4, 1.1, omega, 0.5])
        expected = reference_alpha(params, pi, jitter)
        actual = np.asarray(fn(jnp.array(params)))
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.diag(actual), 1 + 1e-6, rtol=0, atol=1e-15)
        assert np.all(np.isfinite(actual)) and np.all(actual >= 0)
        finite_difference = []
        for k in range(8):
            offset = np.eye(8)[k] * 1e-5
            difference = reference_alpha(params + offset, pi, jitter) - reference_alpha(params - offset, pi, jitter)
            finite_difference.append(np.sum(difference * weight) / 2e-5)
        np.testing.assert_allclose(derivative(jnp.array(params)), finite_difference, rtol=RTOL, atol=ATOL)


def test_alpha_normalization_preserves_zero_and_small_positive_entries(monkeypatch):
    # Inject sparse spectral factors to isolate normalization and clipping from
    # the eigensolver. Negative, zero, and tiny positive entries are intentional.
    vectors = np.eye(61)
    vectors[1, 0] = -0.2
    vectors[3, 2] = 1e-8
    eigenvalues = np.linspace(-2, -0.1, 61)
    monkeypatch.setattr(likelihood.jnp.linalg, "eigh",
                        lambda *args, **kwargs: (jnp.array(eigenvalues), jnp.array(vectors)))
    monkeypatch.setattr(likelihood, "update_GTR", lambda *args: jnp.zeros((61, 61)))
    pi = np.arange(1, 62, dtype=float)
    pi /= pi.sum()
    factors = np.sqrt(pi)
    spectral = (vectors / (1 - eigenvalues)) @ vectors.T
    expected = spectral * factors[None, :] / factors[:, None]
    expected /= np.diag(expected).copy()[None, :]
    np.fill_diagonal(expected, 1e-6)
    expected[expected < 0] = 1e-6
    expected = expected.T + np.eye(61)
    actual = np.asarray(likelihood._gen_alpha_impl(
        1.0, jnp.zeros((61, 61)), jnp.diag(jnp.array(factors)),
        jnp.ones((61, 61)), jnp.diag(1 / jnp.array(factors)), 0.5, 0.0,
    ))
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0)
    assert actual[0, 1] == 1e-6
    assert actual[0, 4] == 0
    assert 0 < actual[2, 3] < 1e-6

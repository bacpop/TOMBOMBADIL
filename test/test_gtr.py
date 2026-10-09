"""Independent codon-model checks for PERF-01, including reverse derivatives."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tombombadil import gtr
from tombombadil.alignment import CODON_LIST


jax.config.update("jax_enable_x64", True)
RTOL, ATOL = 1e-6, 1e-8
# Standard genetic code in T/C/A/G lexical order, independent of GTR tables.
AMINO_ACIDS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
TRANSLATION = dict(zip(map("".join, itertools.product("TCAG", repeat=3)), AMINO_ACIDS))
RATE_PAIRS = ("AG", "AC", "AT", "CG", "GT", "CT")


def reference_matrix(params, pi):
    matrix = np.zeros((61, 61))
    for i, source in enumerate(CODON_LIST):
        for j, target in enumerate(CODON_LIST):
            changes = ["".join(sorted((a, b))) for a, b in zip(source, target) if a != b]
            if len(changes) == 1:
                rate = params[RATE_PAIRS.index(changes[0])]
                if TRANSLATION[source] != TRANSLATION[target]:
                    rate *= params[6]
                matrix[i, j] = rate * np.sqrt(pi[i]) * np.sqrt(pi[j])
    np.fill_diagonal(matrix, -np.sum(matrix * np.sqrt(pi[None, :] / pi[:, None]), axis=1))
    return matrix


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("omega", [0.1, 1.0, 3.0])
def test_gtr_values_and_reverse_derivatives(nonuniform, omega):
    pi = np.arange(1, 62, dtype=float) if nonuniform else np.ones(61)
    pi /= pi.sum()
    pimat = jnp.array(np.diag(np.sqrt(pi)))
    pimult = jnp.array(np.sqrt(pi[None, :] / pi[:, None]))
    params = jnp.array([0.7, 1.3, 0.9, 1.7, 0.4, 1.1, omega])
    expected = reference_matrix(np.asarray(params), pi)
    weight = np.sin(np.arange(61 * 61).reshape(61, 61))

    def direct(p):
        return gtr.build_GTR(*p[:6], p[6], pimat, pimult)

    def updated(p):
        neutral = gtr.build_GTR(*p[:6], 1.0, pimat, pimult)
        return gtr.update_GTR(neutral, p[6], pimult)

    expected_gradient = []
    for k in range(7):
        offset = np.eye(7)[k] * 1e-5
        difference = reference_matrix(np.asarray(params) + offset, pi) - reference_matrix(np.asarray(params) - offset, pi)
        expected_gradient.append(np.sum(difference * weight) / 2e-5)

    for fn in (direct, updated):
        actual = np.asarray(fn(params))
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(actual, actual.T, rtol=RTOL, atol=ATOL)
        np.testing.assert_array_equal(actual[expected == 0], 0)
        gradient = jax.grad(lambda p: jnp.sum(fn(p) * weight))(params)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(actual @ np.sqrt(pi), 0, atol=ATOL)


def test_diagonal_replacement_discards_old_diagonal_and_its_derivative():
    rng = np.random.default_rng(41)
    matrix = rng.normal(size=(61, 61))
    pimult = rng.uniform(0.2, 2, size=(61, 61))
    weight = rng.normal(size=(61, 61))
    expected = matrix.copy()
    np.fill_diagonal(expected, 0)
    np.fill_diagonal(expected, -np.sum(expected * pimult, axis=1))
    np.testing.assert_allclose(gtr.diag_update(matrix, pimult), expected, rtol=RTOL, atol=ATOL)
    derivative = jax.grad(lambda m: jnp.sum(gtr.diag_update(m, pimult) * weight))(jnp.array(matrix))
    expected_derivative = weight - np.diag(weight)[:, None] * pimult
    np.fill_diagonal(expected_derivative, 0)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=RTOL, atol=ATOL)


def test_omega_update_preserves_unselected_offdiagonal_entries():
    # Exercise the helper on arbitrary input, not only structural GTR zeros.
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(61, 61))
    pi = np.arange(1, 62, dtype=float)
    pi /= pi.sum()
    pimult = np.sqrt(pi[None, :] / pi[:, None])
    expected = matrix.copy()
    for i, a in enumerate(CODON_LIST):
        for j, b in enumerate(CODON_LIST):
            if sum(x != y for x, y in zip(a, b)) == 1 and TRANSLATION[a] != TRANSLATION[b]:
                expected[i, j] *= 0.3
    np.fill_diagonal(expected, 0)
    np.fill_diagonal(expected, -np.sum(expected * pimult, axis=1))
    np.testing.assert_allclose(gtr.update_GTR(matrix, 0.3, pimult), expected, rtol=RTOL, atol=ATOL)

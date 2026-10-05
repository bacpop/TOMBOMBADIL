
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit

from .gtr import update_GTR
from .gtr import build_GTR

def _gen_alpha_impl(omega, A, pimat, pimult, pimatinv, scale, eigen_jitter):
    mutmat = update_GTR(A, omega, pimult)
    # add jitter to diagonal (avoids repeated eigenvalues --> eigenvectors are not uniquely defined --> gradient of eigenvectors is undefined / discontinuous --> nans in optimizer)
    mutmat = mutmat + eigen_jitter * jnp.eye(mutmat.shape[-1])
    # computes eigen vectors (v) and values (w)
    w, v = jnp.linalg.eigh(mutmat, UPLO='U')
    E = 1 / (1 - 2 * scale * jnp.reshape(w, (61)))
    V_inv = jnp.matmul(jnp.reshape(v, (61, 61)), jnp.diag(E))

    # Create m_AB for each ancestral codon
    # Static loops avoid unrolling 61 copies under JIT.)
    def reconstruct_column(i, matrix):
        Va = jnp.repeat(v[i, :], 61).reshape(61, 61)
        return matrix.at[:, i].set(jnp.sum(Va * V_inv.T, axis=0))

    m_AB = jax.lax.fori_loop(0, 61, reconstruct_column, jnp.zeros((61, 61)))
    m_AB = jnp.matmul(jnp.matmul(m_AB.T, pimatinv).T, pimat)

    def normalize_column(i, matrix):
        matrix = matrix.at[:, i].set(matrix[:, i] / matrix[i, i])
        return matrix.at[i, i].set(1e-6)

    m_AB = jax.lax.fori_loop(0, 61, normalize_column, m_AB)
    m_AB = jnp.where(m_AB < 0, 1e-6, m_AB)
    return m_AB.T + jnp.eye(61, 61)

@jax.profiler.annotate_function
@jit
def gen_alpha(omega, A, pimat, pimult, pimatinv, scale):
    return _gen_alpha_impl(omega, A, pimat, pimult, pimatinv, scale, 1e-6)


@jax.profiler.annotate_function
@jit
def gen_alpha_no_jitter(omega, A, pimat, pimult, pimatinv, scale):
    return _gen_alpha_impl(omega, A, pimat, pimult, pimatinv, scale, 0.0)


import numpy as np
from jax import jit
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.lax import lgamma, dynamic_update_slice_in_dim, dynamic_slice_in_dim

from .likelihood import update_GTR

# data at each site is
# obs_vec: vector of length 61 with counts
# obs_mat: 61x61 matrix with repeat(obs_vec, 61) [repeated along rows, so a column is invariant]

def partial_likelihood(A, obs_mat, N, l, omega, lp, pimat, pimult,
                       pimatinv, scale, phi):
    lik = 0
    for pos in range(l):
        mutmat = update_GTR(A, omega[l], pimult)

        V, Ve = jnp.linalg.eig(mutmat)
        E = 1 / (1 - 2 * scale * Ve)
        V_inv = jnp.matmul(V, jnp.diag(E)) # TODO probably can be made more efficient

        # Create m_AB for each ancestral codon
        m_AB = jnp.zeros((61, 61))
        index = jnp.arange(0, 61, 1, jnp.uint16)
        #sqp = jnp.sqrt(pi_eq)
        for i in range(61):
            # Va = rep_matrix(row(V, i), 61)
            # Va should be matrix where rows are repeats of row i of V
            Va = jnp.reshape(jnp.repeat(jnp.take(V, index), 61), (61, 61)) # Not sure if this is repeat or tile
            index += 61
            # m_AB[i, ] = to_row_vector(rows_dot_product(Va, V_inv))
            # Possible with jax.vmap?
            # row_sum_fn = jax.vmap(lambda x, y: jnp.vdot(x, y), (1, 1), 0)
            row = jnp.einsum('ij,ij->i', Va, V_inv)
            # If using option 1 below
            # row = jnp.divide(row, sqp.at[i].get())
            dynamic_update_slice_in_dim(m_AB, row, i, 0)

        # Add equilibrium frequencies (option 1)
        #for i in range(61):
        #    col = jnp.multiply(dynamic_slice_in_dim(m_AB, (0, i), 61), sqp.at[i].get())
        #    dynamic_update_slice_in_dim(m_AB, col, i, 1)

        # Add equilibrium frequencies (option 2)
        m_AB = jnp.matmul(jnp.matmul(m_AB.T, pimatinv).T, pimat)

        # Normalise by m_AA
        m_AA = jnp.reshape(jnp.repeat(jnp.diag(m_AB), 61), (61, 61)) # Creates matrix with diagonals copied along each row
        m_AB = jnp.maximum(jnp.divide(m_AB, m_AA) - jnp.eye(61, 61), 1.0e-06) # Makes min value 1e-6 (and sets diagonal, as -I makes this 0)

        muti = m_AB + jnp.eye(61, 61)
        lgmuti = lgamma(muti)
        # ttheta = m_AB * ones; # I think this is rowSum()?
        ttheta = jnp.sum(m_AB, 0)
        ltheta = jnp.log(ttheta)
        lgtheta = lgamma(ttheta)

        poslp = lgtheta - lgamma(N[pos] + ttheta) - jnp.log(N[pos] + ttheta) + ltheta

        gam_mat = lgamma(obs_mat[pos] + muti) - lgmuti

        likposanc = lp
        likposanc += jnp.sum(gam_mat, 0)
        likposanc += poslp + phi[pos]
        lik += logsumexp(likposanc)
    return lik









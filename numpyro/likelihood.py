import numpy as np
from jax import jit
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.lax import lgamma, dynamic_update_slice_in_dim, dynamic_slice_in_dim

from .likelihood import build_GTR, update_GTR

def transforms(X, pi_eq):
    N = np.sum(X, 0)

    # pi transforms
    lp = np.array(np.log(pi_eq))
    pimat = np.diag(np.sqrt(pi_eq))
    pimatinv = np.diag(np.divide(1, np.sqrt(pi_eq)))

    pimult = np.array((61, 61))
    for j in range(61):
        for i in range(61):
            pimult[i, j] = np.sqrt(pi_eq[j] / pi_eq[i])


    # These may need to be 3D tensors, or some other transform may be better here
    obs_mat = []
    obs_vec = []
    l = X.shape()[0]
    for i in range(l):
        obs_mat.append(np.broadcast_to(X[i, :], (61, 61)))
        obs_vec.append(X[i, :])

    return N, l, lp, pimat, pimatinv, pimult, obs_mat, obs_vec

# data at each site is
# obs_vec: vector of length 61 with counts
# obs_mat: 61x61 matrix with repeat(obs_vec, 61) [repeated along rows, so a column is invariant]

def likelihood(obs_vec, obs_mat, N, l, theta, omega, pi_eq, lp, pimat, pimult,
               pimatinv, alpha, beta, gamma, delta, epsilon, eta):

    # TODO This can be precomputed in transforms, will be of length N
    Np = N.at[pos].get()
    phi = lgamma(Np + 1) - jnp.sum(lgamma(obs_vec.at[pos].get() + 1))

    # Calculate substitution rate matrix under neutrality
    # TODO can this be moved out to model function?
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult)

    meanrate = -jnp.dot(jnp.diagonal(A), pi_eq)
    #meanrate = 0
    #for i in range(61):
    #    meanrate -= pi_eq[i] * A.at[i, i].get()

    # Calculate substitution rate matrix
    scale = (theta / 2.0) / meanrate
    # TODO fix this loop, all needs to be over codon ('pos' in stan code)
    for l in range(N):
        # TODO replace this with build_GTR/update_GTR
        mutmat = build_GTR(alpha, beta, gamma, delta, epsilon, eta, omega, pimat, pimult)

        V, Ve = jnp.linalg.eig(mutmat)
        E = 1 / (1 - 2 * scale * Ve)
        V_inv = jnp.matmul(V, jnp.diag(E))

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

        # Original C
        # for(i in 1:61){
        #     m_AA = m_AB[i, i]
        #     for(j in 1:61){
        #     if(j != i){
        #         m_AB[i, j] /= m_AA
        #     }
        #     if(m_AB[i, j] < 0){
        #         m_AB[i, j] = 1.0e-06
        #     }
        #     }
        #     m_AB[i, i] = 1.0e-06
        # }

        # Likelihood calculation

        # Parts shared over all positions (at least while omega is fixed)


        lik = 0
        muti = m_AB + jnp.eye(61, 61)
        lgmuti = lgamma(muti)
        # ttheta = m_AB * ones; # I think this is rowSum()?
        ttheta = jnp.sum(m_AB, 0)
        ltheta = jnp.log(ttheta)
        lgtheta = lgamma(ttheta)

        poslp = lgtheta - lgamma(Np + ttheta) - jnp.log(Np + ttheta) + ltheta

        gam_mat = lgamma(obs_mat[pos] + muti) - lgmuti

        likposanc = lp
        likposanc += jnp.sum(gam_mat, 0)
        likposanc += poslp + phi

        lik += logsumexp(likposanc)
    return lik










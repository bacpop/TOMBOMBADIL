#!/usr/bin/env python

import logging
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.lax
import jax.numpy as jnp
import jax.random as random

from .gtr import build_GTR
from .likelihood import gen_alpha

# data at each site is
# obs_mat: 61x61 matrix with repeat(obs_vec, 61) [repeated along rows, so a column is invariant]
# so obs_mat is [locus, repeat x 61, codon]
def model(pi_eq, N, l, pimat, pimatinv, pimult, obs_mat):
    # GTR params (shared)
    log_alpha = numpyro.sample("log_alpha", dist.Normal(0, 1))
    alpha = numpyro.deterministic("alpha", jax.lax.exp(log_alpha))
    log_beta = numpyro.sample("log_beta", dist.Normal(0, 1))
    beta = numpyro.deterministic("beta", jax.lax.exp(log_beta))
    log_gamma = numpyro.sample("log_gamma", dist.Normal(0, 1))
    gamma = numpyro.deterministic("gamma", jax.lax.exp(log_gamma))
    log_delta = numpyro.sample("log_delta", dist.Normal(0, 1))
    delta = numpyro.deterministic("delta", jax.lax.exp(log_delta))
    log_epsilon = numpyro.sample("log_epsilon", dist.Normal(0, 1))
    epsilon = numpyro.deterministic("epsilon", jax.lax.exp(log_epsilon))
    log_eta = numpyro.sample("log_eta", dist.Normal(0, 1))
    eta = numpyro.deterministic("eta", jax.lax.exp(log_eta))

    # Rate params
    theta_prior = dist.Gamma(1, 2)
    theta = numpyro.sample("theta", theta_prior)

    # Calculate substitution rate matrix under neutrality
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult)
    meanrate = -jnp.dot(jnp.diagonal(A), pi_eq)
    # Calculate substitution rate matrix
    scale = (theta / 2.0) / meanrate

    with numpyro.plate('locus', l, dim=1) as codons: # minibatch here?
        omega = numpyro.sample("omega", dist.Exponential(0.5))
        N_batch = N[codons]
        alpha = gen_alpha(A, omega, pimat, pimult, pimatinv, scale)
        with numpyro.plate('ancestor', 61, dim=1) as anc, numpyro.poutine.scale(scale=pi_eq):
            numpyro.sample('obs', dist.DirichletMultinomial(concentration=alpha[anc, :], total_count=N_batch), obs=obs_mat)

# TODO - not all of these may be needed
def transforms(X, pi_eq):
    import numpy as np
    from math import lgamma
    N = np.sum(X, 0)

    # pi transforms
    lp = np.array(np.log(pi_eq))
    pimat = np.diag(np.sqrt(pi_eq))
    pimatinv = np.diag(np.divide(1, np.sqrt(pi_eq)))

    pimult = np.zeros((61, 61))
    for j in range(61)  :
        for i in range(61):
            pimult[i, j] = np.sqrt(pi_eq[j] / pi_eq[i])

    phi = []
    obs_mat = np.empty((X.shape[1], 61, 61))
    for l in range(X.shape[1]):
        phi.append(lgamma(N[l] + 1) - sum([lgamma(x + 1) for x in X[:, l]]))
        obs_mat[l, :, :] = np.broadcast_to(X[:, l], (61, 61))

    return N, l, lp, pimat, pimatinv, pimult, obs_mat, phi

def run_sampler(X, pi_eq, warmup=500, samples=500):
    logging.info("Precomputing transforms...")
    N, l, lp, pimat, pimatinv, pimult, obs_mat, phi = transforms(X, pi_eq)

    logging.info("Running model...")
    numpyro.set_platform('cpu')
    numpyro.set_host_device_count(16)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=warmup, num_samples=samples+warmup)
    rng_key = random.PRNGKey(0)
    results = mcmc.run(rng_key, pi_eq, N, l, pimat, pimatinv, pimult, obs_mat,
                       extra_fields=('potential_energy',))

    return results



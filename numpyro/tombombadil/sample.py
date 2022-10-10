#!/usr/bin/env python

from likelihood import likelihood, transforms

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.lax
import jax.numpy as jnp
import jax.random as random

from .gtr import build_GTR
from .likelihood import partial_likelihood

def model(X, pi_eq, N, l, lp, pimat, pimatinv, pimult, obs_mat, phi):
    # GTR params (shared)
    alpha_prior = dist.Normal(0, 1)
    beta_prior = dist.Normal(0, 1)
    gamma_prior = dist.Normal(0, 1)
    delta_prior = dist.Normal(0, 1)
    epsilon_prior = dist.Normal(0, 1)
    eta_prior = dist.Normal(0, 1)
    log_alpha = numpyro.sample("alpha", alpha_prior)
    log_beta = numpyro.sample("beta", beta_prior)
    log_gamma = numpyro.sample("gamma", gamma_prior)
    log_delta = numpyro.sample("delta", delta_prior)
    log_epsilon = numpyro.sample("epsilon", epsilon_prior)
    log_eta = numpyro.sample("eta", eta_prior)
    alpha = jax.lax.exp(log_alpha)
    beta = jax.lax.exp(log_beta)
    gamma = jax.lax.exp(log_gamma)
    delta = jax.lax.exp(log_delta)
    epsilon = jax.lax.exp(log_epsilon)
    eta = jax.lax.exp(log_eta)

    # Rate params
    theta_prior = dist.Gamma(1, 2)
    theta = numpyro.sample("theta", theta_prior)

    # Calculate substitution rate matrix under neutrality
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult)
    meanrate = -jnp.dot(jnp.diagonal(A), pi_eq)
    # Calculate substitution rate matrix
    scale = (theta / 2.0) / meanrate

    with numpyro.plate('locus', l) as codons: # minibatch here?
        omega = numpyro.sample("omega", dist.Exponential(0.5))
        N_batch = N[codons]
        l = len(N_batch)
        obs_mat_batch = obs_mat[codons]
        numpyro.sample('obs', partial_likelihood(A, obs_mat_batch, N_batch, l,
                                                 omega, lp, pimat, pimult, pimatinv,
                                                 scale, phi))

def transforms(X, pi_eq):
    import numpy as np
    from math import lgamma
    N = np.sum(X, 0)

    # pi transforms
    lp = np.array(np.log(pi_eq))
    pimat = np.diag(np.sqrt(pi_eq))
    pimatinv = np.diag(np.divide(1, np.sqrt(pi_eq)))

    pimult = np.array((61, 61))
    for j in range(61)  :
        for i in range(61):
            pimult[i, j] = np.sqrt(pi_eq[j] / pi_eq[i])

    phi = []
    obs_mat = []
    for l in range(X.shape[1]):
        phi.append(lgamma(N[l] + 1) - sum([lgamma(x + 1) for x in X[l, :]]))
        obs_mat.append(np.broadcast_to(X[i, :], (61, 61)))

    return N, l, lp, pimat, pimatinv, pimult, obs_mat, phi

def run_sampler(X, pi_eq, warmup=500, samples=500):
    N, l, lp, pimat, pimatinv, pimult, obs_mat, phi = transforms(X, pi_eq)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=warmup, num_samples=samples+warmup)
    rng_key = random.PRNGKey(0)
    results = mcmc.run(rng_key, X, pi_eq, N, l, lp, pimat, pimatinv, pimult, obs_mat,
                       phi, extra_fields=('potential_energy',))

    return results



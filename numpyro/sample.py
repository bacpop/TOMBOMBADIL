#!/usr/bin/env python

from likelihood import likelihood, transforms

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.random as random

def model(pi_eq, N, l, lp, pimat, pimatinv, pimult, obs_mat, obs_vec):
    # GTR params (shared)
    alpha_prior = dist.LogNormal(1, 1.5)
    beta_prior = dist.LogNormal(1, 1.5)
    gamma_prior = dist.LogNormal(1, 1.5)
    delta_prior = dist.LogNormal(1, 1.5)
    epsilon_prior = dist.LogNormal(1, 1.5)
    eta_prior = dist.LogNormal(1, 1.5)
    alpha = numpyro.sample("alpha", alpha_prior)
    beta = numpyro.sample("beta", beta_prior)
    gamma = numpyro.sample("gamma", gamma_prior)
    delta = numpyro.sample("delta", delta_prior)
    epsilon = numpyro.sample("epsilon", epsilon_prior)
    eta = numpyro.sample("eta", eta_prior)

    # Rate params
    theta_prior = dist.Gamma(1, 2)
    theta = numpyro.sample("theta", theta_prior)

    with numpyro.plate('locus', l) as codons: # minibatch here?
        omega = numpyro.sample("omega", dist.Exponential(0.5))
        N_batch = N[codons]
        l = len(N_batch)
        obs_vec_batch = obs_vec[codons]
        obs_mat_batch = obs_mat[codons]
        numpyro.sample('obs', likelihood(obs_vec_batch, obs_mat_batch, N_batch,
                                         l, theta, omega, pi_eq, lp, pimat,
                                         pimult, pimatinv, alpha, beta, gamma,
                                         delta, epsilon, eta))

def run_sampler(X, pi_eq, warmup=500, samples=1000):
    N, l, lp, pimat, pimatinv, pimult, obs_mat, obs_vec = transforms(X, pi_eq)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=warmup, num_samples=samples)
    rng_key = random.PRNGKey(0)
    results = mcmc.run(rng_key, pi_eq, N, l, lp, pimat, pimatinv, pimult, obs_mat,
                       obs_vec, extra_fields=('potential_energy',))

    return results



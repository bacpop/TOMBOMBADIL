# TOMBOMBADIL likelihood
import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.special as special
from jax.scipy.special import gammaln
from jax import jit

from .gtr import update_GTR, build_GTR

SCALAR_PARAM_KEYS_WITH_ETA = ["alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "omega"]
GTR_PARAM_KEYS_WITH_ETA = ["alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta"]
OMEGA_MODES = ("scalar", "per-site")

# This is the main TOMBOMBDAIL likelihood
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

# Closures for calling the likelihood:
@jit
def gen_alpha(omega, A, pimat, pimult, pimatinv, scale):
    return _gen_alpha_impl(omega, A, pimat, pimult, pimatinv, scale, 1e-6)


@jit
def gen_alpha_no_jitter(omega, A, pimat, pimult, pimatinv, scale):
    return _gen_alpha_impl(omega, A, pimat, pimult, pimatinv, scale, 0.0)

@jit
def codon_site_log_likelihood(alpha, beta, gamma, delta, epsilon, eta, mu, omega, pi_eq, log_pi, pimat, pimatinv, pimult, obs_vec):
    # Calculate substitution rate matrix under neutrality
    #print(pimat)
    #print(pimult)
    #A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult) # 61x61 subst rate matrix
    #A = build_GTR(1, 1, 1, 1, 1, 1, 1, pimat, pimult) # same as NY98?
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult) # 61x61 subst rate matrix # for building the GTR matrix you want omega=1 (mean mutation rate under neutrality)
    #print(A) # is all zeros at the moment
    #print(pi_eq)
    #print(jnp.diagonal(A))
    #print(-jnp.dot(jnp.diagonal(A), pi_eq))
    meanrate = -jnp.dot(jnp.diagonal(A), pi_eq)
    # Calculate substitution rate matrix
    scale = (mu / 2.0) / meanrate

    A2 = gen_alpha(omega, A, pimat, pimult, pimatinv, scale)
    #alpha = gen_alpha(omega, A, pimat, pimult, pimatinv, scale, alpha, beta, gamma, delta, epsilon, eta) # just for comparing runtime between build_GTR and update_GTR
    #print('alpha: ',alpha)
    #print("obs_vec: ", obs_vec)
    #print("N: ", N)
    #jax.debug.print("alpha = {alpha}", alpha=alpha)
    #jax.debug.print("A = {A}", A=A)
    #jax.debug.print("A2 = {A2}", A2=A2) # these calculations are done twice in one step (jit?) and the second time some NaNs appear in A
    # it seems to come from the parameters but not sure? my analysis in test_fn suggests that the likelihood becomes zero with omega close to zero, no NaNs in parameters needed...?
    #print(np.sum(alpha,axis=1).tolist()) # alpha rows clearly do not sum to one but this is what the pmf is expecting -- a problem? no, for dirichlet not a problem
    #log_prob = scipy.stats.multinomial.pmf(obs_vec, N, alpha) # this is where it breaks but is it because the code is broken or because of lack of diversity? It is not because of the lack of diversity
    #log_prob = scipy.stats.multinomial.logpmf(obs_vec, N, alpha) # this is pmf in John's code but we think it might need to be pmf?
    # log_prob = scipy.stats.dirichlet_multinomial.logpmf(obs_vec, alpha, N) # gives -10.21301 (correct)
    log_prob = dirichlet_multinomial_logpmf(obs_vec, A2) # our custom, jnp based dirichlet_multinomial.logpmf but something is wrong in the implementation this function gives us an integer, we want a vector of length 61

    #print("Difference between scipy and custom jax dirichlet-multinomial logpmf:", scipy.stats.dirichlet_multinomial.logpmf(obs_vec, alpha, N) - dirichlet_multinomial_logpmf(obs_vec, alpha))
    #print("Difference between scipy and other custom jax dirichlet-multinomial logpmf:", scipy.stats.dirichlet_multinomial.logpmf(obs_vec, alpha, N) - dirichlet_multinomial_logpmf_scipy_form(obs_vec, alpha))

    #print("log_prob_shape",log_prob.shape)
    #print('log_prob: ',log_prob)
    #jax.debug.print("obs_vec = {obs_vec}", obs_vec=obs_vec)
    #jax.debug.print("alpha = {alpha}", alpha=alpha)
    #jax.debug.print("log_prob = {log_prob}", log_prob=log_prob)
    #print('log_prop_pi',log_prob + log_pi)
    #print('logsumexp_prop_pi',special.logsumexp(log_prob + log_pi, axis=0))
    return special.logsumexp(log_prob + log_pi, axis=0) # check that these go in as different arguments

@jit
def dirichlet_multinomial_logpmf(x, a):
    x = jnp.asarray(x, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)

    N = jnp.sum(x, axis=-1)
    a0 = jnp.sum(a, axis=-1)

    term1 = gammaln(N + 1) - jnp.sum(gammaln(x + 1), axis=-1)
    term2 = gammaln(a0) - gammaln(N + a0)
    term3 = jnp.sum(gammaln(x + a) - (gammaln(a)), axis=-1)

    #print("term3",term3)
    #print("x",x)
    #print("a",a)
    #jax.debug.print("x = {x}", x=x)
    #jax.debug.print("a = {a}", a=a)
    #jax.debug.print("a = {a}", a=a)
    #test = gammaln(a)
    #jax.debug.print("test = {test}", test=test)
    #jax.debug.print("term1 = {term1}", term1=term1)
    #jax.debug.print("term2 = {term2}", term2=term2)
    #jax.debug.print("term3 = {term3}", term3=term3)

    return term1 + term2 + term3 # gives 1407.2288

# This version is adapted from the scipy implementation
def dirichlet_multinomial_logpmf_scipy_form(x, a):
    x = jnp.asarray(x)
    a = jnp.asarray(a)

    N = jnp.sum(x, axis=-1)
    a0 = jnp.sum(a, axis=-1)

    out = jnp.asarray(gammaln(a0) + gammaln(N + 1) - gammaln(N + a0))
    out += (gammaln(x + a) - (gammaln(a) + gammaln(x + 1))).sum(axis=-1)

    # The scipy version sets the logpmf to -inf if N and sum(x) disagree, but
    # we're calculating N from x here so not really relevant
    # out = jnp.place(out, N != x.sum(axis=-1), -jnp.inf, inplace=False)

    return out



@jit
def codon_site_log_likelihood_no_jitter(alpha, beta, gamma, delta, epsilon, eta, mu, omega, pi_eq, log_pi, pimat, pimatinv, pimult, obs_vec):
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult)
    meanrate = -jnp.dot(jnp.diagonal(A), pi_eq)
    scale = (mu / 2.0) / meanrate
    A2 = gen_alpha_no_jitter(omega, A, pimat, pimult, pimatinv, scale)
    log_prob = dirichlet_multinomial_logpmf(obs_vec, A2)
    return special.logsumexp(log_prob + log_pi, axis=0)

def prepare_likelihood_transforms(X, pi_eq):
    #N = np.sum(X, 0)
    #n_loci = len(N)

    # pi transforms
    log_pi = np.log(pi_eq)
    pimat = np.diag(np.sqrt(pi_eq))
    pimatinv = np.diag(np.divide(1, np.sqrt(pi_eq)))

    pimult = np.zeros((61, 61))
    for j in range(61)  :
        for i in range(61):
            pimult[i, j] = np.sqrt(pi_eq[j] / pi_eq[i])
            #pimult = pimult.at[i,j].set(jnp.sqrt(pi_eq[j] / pi_eq[i]))

    return log_pi, pimat, pimatinv, pimult

def positive_transform(a):
    """Map raw values to positive natural scale with the existing 1e-6 floor."""
    return jnp.exp(a) + 1e-6


def positive_transform_inverse(y, eps=1e-6):
    """Convert positive natural-scale values to the corresponding raw log scale."""
    z = y - eps
    return jnp.log(z)


def _log_transform_jacobian(raw_value):
    return jnp.log(jnp.exp(raw_value))


def prior_log_likelihood(raw_x, n_sites, *, prior_mode, estimate_eta,
                         omega_mode, aggregate):
    """Log prior contributions for MAP regularisation.

    The data log-likelihood is mean-aggregated over sites (mean(losses)), which
    is (1/n_sites) × Σᵢ log P(Dᵢ | θ). For the MAP to coincide with the mode of
    the true Bayesian posterior Σᵢ log P(Dᵢ | θ) + log P(θ), every prior term
    must enter with the same 1/n_sites weighting.

    omega:     LogNormal(log(0.5), 1). This is now a global scalar, so it is
               weighted like the other global priors.
    GTR rates: Half-normal priors. These are global (one prior per parameter,
               not per site), so we explicitly divide the summed log-prior by
               n_sites to put it at the same effective weight as one per-site
               quantity. eta is fixed to 1.0 (not sampled) to remove the global
               GTR-scale ambiguity, so it is excluded from the prior.
    theta:     Half-normal, same form as the GTR rates but NOT divided by n_sites.
               theta is a global mutation rate scalar that scales the overall branch
               length; its prior is intentionally kept at full strength.
    """

    if prior_mode == "none":
        return jnp.array(0.0, dtype=jnp.float64)

    validate_omega_mode(omega_mode)
    omega = positive_transform(raw_x["omega"])

    if prior_mode == "current":
        omega_prior = jax.scipy.stats.norm.logpdf(jnp.log(omega), jnp.log(0.5), 1.0)
    else:
        omega_prior = (
            jax.scipy.stats.norm.logpdf(jnp.log(omega), jnp.log(0.5), 1.0)
            - jnp.log(omega)
        )
        if prior_mode == "stan_unconstrained":
            omega_prior += _log_transform_jacobian(raw_x["omega"])

    gtr_keys = ["alpha", "beta", "gamma", "delta", "epsilon"]
    if estimate_eta:
        gtr_keys.append("eta")

    gtr_terms = []
    for k in gtr_keys:
        term = jax.scipy.stats.norm.logpdf(positive_transform(raw_x[k]), 0.0, 1.0)
        if prior_mode == "stan_unconstrained":
            term += _log_transform_jacobian(raw_x[k])
        gtr_terms.append(term)
    gtr_prior = jnp.sum(jnp.array(gtr_terms))

    theta_prior = jax.scipy.stats.norm.logpdf(positive_transform(raw_x["theta"]), 0.0, 1.0)
    if prior_mode == "stan_unconstrained":
        theta_prior += _log_transform_jacobian(raw_x["theta"])

    if omega_mode == "per-site":
        omega_prior = jnp.sum(omega_prior) if aggregate == "sum" else jnp.mean(omega_prior)

    if prior_mode in ("stan_constrained", "stan_unconstrained"):
        return omega_prior + gtr_prior + theta_prior

    if omega_mode == "per-site":
        return omega_prior + gtr_prior / n_sites + theta_prior
    return (omega_prior + gtr_prior) / n_sites + theta_prior

def compute_laplace_se(fn, params):
    """Diagonal Laplace approximation: per-parameter standard errors at the MAP.

    Flattens the parameter PyTree to a 1-D vector, computes the full Hessian of
    -fn (the negative log-likelihood), and uses its diagonal to approximate the
    marginal variance of each parameter:

        Var(theta_i) ≈ 1 / H_ii,   H = -d²(log L)/dtheta²

    Standard errors in unconstrained (raw) space and on the natural scale are
    both returned. For parameters transformed by positive_transform(), the delta method
    gives se_natural = se_raw * exp(raw).

    Args:
        fn:     the log-likelihood function (higher = better)
        params: PyTree of optimised (raw/unconstrained) parameter values

    Returns:
        se_raw:     PyTree matching params, SEs in unconstrained space
        se_natural: PyTree matching params, SEs on natural scale
    """
    flat_params, unflatten = ravel_pytree(params)

    def neg_ll_flat(p):
        return -fn(unflatten(p))

    # Compute only the diagonal of the Hessian via forward-over-reverse AD.
    # For each basis vector eᵢ, jvp(grad, params, eᵢ) returns H @ eᵢ;
    # the i-th element of that product is H_ii. This uses O(n) memory
    # rather than the O(n²) required by the full jax.hessian approach.
    grad_fn = jax.grad(neg_ll_flat)
    n = len(flat_params)
    def hess_diag_i(i):
        e_i = jnp.zeros(n).at[i].set(1.0)
        _, hv = jax.jvp(grad_fn, (flat_params,), (e_i,))
        return hv[i]
    hess_diag = jax.vmap(hess_diag_i)(jnp.arange(n))

    # 1 / H_ii gives the marginal variance under the diagonal approximation.
    # H_ii <= 0 means the likelihood is flat there (masked site) → NaN.
    var_raw = jnp.where(hess_diag > 0, 1.0 / hess_diag, jnp.nan)
    se_raw = unflatten(jnp.sqrt(jnp.clip(var_raw, 0)))

    # Delta method onto natural scale for positive_transform()-transformed parameters.
    se_natural = {k: se_raw[k] * jnp.exp(params[k]) for k in params}

    return se_raw, se_natural


def _log_laplace_summary(params, se_natural, *, omega_mode):
    """Log a human-readable summary of MAP estimates ± 1 SE (natural scale)."""
    validate_omega_mode(omega_mode)
    gtr_keys = GTR_PARAM_KEYS_WITH_ETA
    logging.info("Laplace approximation (diagonal), estimates on natural scale:")
    for k in gtr_keys:
        if k in params:
            est = float(positive_transform(params[k]))
            se  = float(se_natural[k])
            logging.info("  %-8s: %.4f +/- %.4f", k, est, se)
    if "omega" in params:
        omega = np.asarray(positive_transform(params["omega"]))
        se = np.asarray(se_natural["omega"])
        if omega_mode == "scalar":
            logging.info("  %-8s: %.4f +/- %.4f", "omega", float(omega), float(se))
        else:
            logging.info("  omega: %d site-wise standard errors", len(omega))


def validate_omega_mode(omega_mode):
    if omega_mode not in OMEGA_MODES:
        raise ValueError(f"Unknown omega mode: {omega_mode!r}")
    return omega_mode


def mode_output_stem(output_stem, omega_mode):
    """Prefix an output stem so files identify their omega parameterisation."""
    validate_omega_mode(omega_mode)
    directory, basename = os.path.split(str(output_stem))
    prefixed = f"{omega_mode.replace('-', '_')}_{basename}"
    return os.path.join(directory, prefixed) if directory else prefixed


def likelihood_plot_path(output_stem, omega_mode):
    """Return the default or output-stem-based MAP likelihood plot path."""
    validate_omega_mode(omega_mode)
    if output_stem is None:
        mode_name = omega_mode.replace("-", "_")
        return f"{mode_name}_likelihood_plot.pdf"
    return mode_output_stem(output_stem, omega_mode) + "_likelihood_plot.pdf"
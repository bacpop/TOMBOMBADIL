# Samplers (NUTS/MAP)

import csv as _csv
import logging
import sys
from time import perf_counter
import numpy as np
import jax
import jax.numpy as jnp

import blackjax
import optax
from jax import jit
from jax.flatten_util import ravel_pytree
jax.config.update('jax_enable_x64', True)
import matplotlib.pyplot as plt
from tqdm import tqdm

from .likelihood import *

# NUTS/Bayesian sampling
def run_nuts_sampler(fn, start_params, *, num_warmup, num_samples,
                     num_chains, rng_seed, target_acceptance_rate, output,
                     print_summary=True, chain_mode, omega_mode):
    """Run BlackJAX NUTS from raw unconstrained starting parameters."""
    if chain_mode not in ("sequential", "pmap"):
        raise ValueError(f"Unknown NUTS chain mode: {chain_mode}")

    rng_key = jax.random.PRNGKey(rng_seed)
    chain_keys = jax.random.split(rng_key, num_chains)
    initial_positions = []
    for chain in range(num_chains):
        if chain == 0:
            initial_positions.append(start_params)
        else:
            initial_positions.append(_perturb_params(start_params))

    if chain_mode == "pmap":
        logging.info(
            "Running %s BlackJAX NUTS chain(s) with pmap across %s local JAX device(s)",
            num_chains, jax.local_device_count(),
        )
        stacked_initial_positions = _stack_chain_pytrees(initial_positions)
        raw_samples, infos, adapted_parameters = _sample_blackjax_chains_pmap(
            fn,
            stacked_initial_positions,
            chain_keys,
            num_warmup,
            num_samples,
            target_acceptance_rate,
        )
    else:
        chain_positions = []
        chain_infos = []
        adapted_parameters = []
        for chain in range(num_chains):
            logging.info("Running BlackJAX NUTS chain %s/%s", chain + 1, num_chains)
            positions, infos, parameters = _sample_blackjax_chain(
                fn,
                initial_positions[chain],
                chain_keys[chain],
                num_warmup,
                num_samples,
                target_acceptance_rate,
            )
            chain_positions.append(positions)
            chain_infos.append(infos)
            adapted_parameters.append(parameters)

        raw_samples = _stack_chain_pytrees(chain_positions)
        infos = _stack_chain_pytrees(chain_infos)

    natural_samples, summaries, diagnostics = summarize_posterior_samples(
        raw_samples, infos, omega_mode=omega_mode
    )
    if print_summary:
        _log_posterior_summary(summaries, diagnostics)
    if output is not None:
        save_posterior_outputs(output, raw_samples, summaries, omega_mode=omega_mode)

    return {
        "raw_samples": raw_samples,
        "samples": natural_samples,
        "summaries": summaries,
        "diagnostics": diagnostics,
        "infos": infos,
        "adapted_parameters": adapted_parameters,
    }

def _sample_blackjax_chain(logdensity_fn, initial_position, rng_key, num_warmup,
                           num_samples, target_acceptance_rate):
    """Warm up and sample one NUTS chain with BlackJAX."""
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        target_acceptance_rate=target_acceptance_rate,
    )
    warmup_key, sample_key = jax.random.split(rng_key)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=num_warmup)
    kernel = blackjax.nuts(logdensity_fn, **parameters).step

    @jax.jit
    def one_step(current_state, step_key):
        new_state, info = kernel(step_key, current_state)
        sample_info = {
            "acceptance_rate": info.acceptance_rate,
            "is_divergent": info.is_divergent,
        }
        return new_state, (new_state.position, sample_info)

    sample_keys = jax.random.split(sample_key, num_samples)
    _, (positions, infos) = jax.lax.scan(one_step, state, sample_keys)
    return positions, infos, parameters


def _sample_blackjax_chains_pmap(logdensity_fn, initial_positions, rng_keys,
                                 num_warmup, num_samples,
                                 target_acceptance_rate):
    """Warm up and sample NUTS chains in parallel across JAX devices."""
    num_chains = int(rng_keys.shape[0])
    n_devices = jax.local_device_count()
    if num_chains > n_devices:
        raise ValueError(
            f"Requested {num_chains} pmap NUTS chain(s), but JAX sees only "
            f"{n_devices} local device(s). On CPU, run through the CLI with "
            f"--nuts-chain-mode pmap --cpus {num_chains} before JAX is imported, "
            "or use --nuts-chain-mode sequential."
        )

    def run_chain(initial_position, rng_key):
        return _sample_blackjax_chain(
            logdensity_fn,
            initial_position,
            rng_key,
            num_warmup,
            num_samples,
            target_acceptance_rate,
        )

    return jax.pmap(run_chain)(initial_positions, rng_keys)


def _stack_chain_pytrees(chain_pytrees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *chain_pytrees)

# ADAM/ML(MAP) sampling
def run_map_optimizer(fn, start_params, mask, *, max_it,
                      estimate_uncertainty, fit_replicates, output,
                      fit_until_convergence, convergence_tol,
                      convergence_patience, convergence_check_every,
                      convergence_min_steps, omega_mode, domain_labels):
    """Run MAP optimization and write its result files and diagnostics."""
    base_labels = make_param_labels(start_params)
    result_stem = output if output is not None else "output"
    logging.info(f"Running optimization — {fit_replicates} replicate(s)...")
    convergence = None
    if fit_until_convergence:
        convergence = {
            "enabled": True,
            "tol": convergence_tol,
            "patience": convergence_patience,
            "check_every": convergence_check_every,
            "min_steps": convergence_min_steps,
        }
    all_params, best_idx, all_metadata = _run_map_replicates(
        fn, start_params, base_labels, fit_replicates, n_iter=max_it,
        convergence=convergence,
    )
    params = all_params[best_idx]
    best_metadata = all_metadata[best_idx]

    status = "converged" if best_metadata["converged"] else "reached max steps"
    likelihood_fig, _ = plot_likelihood_history(all_metadata, best_idx)
    likelihood_path = likelihood_plot_path(output, omega_mode)
    likelihood_fig.savefig(likelihood_path, format="pdf", bbox_inches="tight")
    plt.close(likelihood_fig)
    logging.info("Saved MAP likelihood plot to: %s", likelihood_path)

    if fit_replicates > 1:
        plot_replicates(all_params, best_idx)
        plt.show()
    if omega_mode == "per-site":
        fig, _ = plot_per_site_omega(params, mask=mask, domain_labels=domain_labels)
        omega_plot_path = mode_output_stem(result_stem, omega_mode) + "_omega_plot.pdf"
        fig.savefig(omega_plot_path, format="pdf", bbox_inches="tight")
        logging.info("Saved MAP per-site omega plot to: %s", omega_plot_path)
        plt.close(fig)
    if estimate_uncertainty:
        logging.info("Computing Laplace uncertainty...")
        _, se_nat = compute_laplace_se(fn, params)
        _log_laplace_summary(params, se_nat, omega_mode=omega_mode)

    save_params(result_stem, params, mask=mask, omega_mode=omega_mode)
    logging.info("Final log-likelihood: %.10f", best_metadata["objective"])
    scalar_estimates = {
        key: float(positive_transform(params[key]))
        for key in GTR_PARAM_KEYS_WITH_ETA
        if key in params
    }
    logging.info("Final scalar parameter estimates (natural scale): %s", scalar_estimates)
    omega_estimates = np.asarray(positive_transform(params["omega"]))
    if omega_mode == "scalar":
        logging.info("Final dN/dS estimate: %.10g", float(omega_estimates))
    else:
        logging.info(
            "Final dN/dS estimates cover %d sites (range %.6g to %.6g); saved to: %s",
            len(omega_estimates), float(omega_estimates.min()),
            float(omega_estimates.max()),
            mode_output_stem(result_stem, omega_mode) + "_omega.csv",
        )
    logging.info("Optimization status: %s after %d step(s)", status, best_metadata["n_steps"])

    return {
        "params": params,
        "objective": float(best_metadata["objective"]),
        "metadata": best_metadata,
    }

def _run_map_replicates(fn, start_params, param_labels, n_reps, *, n_iter,
                    convergence=None, progress=True):
    """Run the optimizer n_reps times and return all results plus the index of the best.

    Replicate 0 uses the unperturbed starting point; subsequent replicates add
    Normal(0, 0.5) noise in raw parameter space. All runs are silent (verbose=False).

    Args:
        fn:            log-likelihood function
        start_params:  unperturbed starting parameter dict
        param_labels:  optax multi_transform label dict
        n_reps:        number of restarts
        n_iter:        optimisation iterations per replicate

    Returns:
        all_params:   list of param dicts, one per replicate
        best_idx:     index of the replicate with the highest log-likelihood
        all_metadata: optimizer metadata dicts, one per replicate
    """
    all_params = []
    all_lls = []
    all_metadata = []
    for rep in range(n_reps):
        start = dict(start_params) if rep == 0 else _perturb_params(start_params)
        schedule = optax.cosine_decay_schedule(
            init_value=0.2, decay_steps=n_iter, alpha=1e-3 / 0.2
        )
        solver = optax.multi_transform(
            {"vec": optax.adam(schedule), "scalar": optax.adam(schedule)},
            param_labels=param_labels,
        )

        interactive_progress = progress and sys.stderr.isatty()
        progress_bar = None
        last_progress_step = 0
        if interactive_progress:
            progress_bar = tqdm(
                total=n_iter,
                desc=f"MAP replicate {rep + 1}/{n_reps}",
                unit="step",
                file=sys.stderr,
            )

        def report_progress(step, total, objective):
            nonlocal last_progress_step
            if progress_bar is not None:
                if step > last_progress_step:
                    progress_bar.update(step - last_progress_step)
                    last_progress_step = step
                if objective is not None:
                    progress_bar.set_postfix_str(f"log-likelihood={objective:.6f}")
            elif objective is not None:
                logging.info(
                    "MAP replicate %d/%d, step %d/%d: log-likelihood = %.6f",
                    rep + 1, n_reps, step, total, objective,
                )

        try:
            result = _optimize_params(
                fn,
                start,
                solver,
                n_iter,
                verbose=False,
                convergence=convergence,
                progress_callback=report_progress if progress else None,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        params_rep = result["params"]
        ll = result["objective"]
        all_params.append(params_rep)
        all_lls.append(ll)
        all_metadata.append(result)
        status = "converged" if result["converged"] else "max steps"
        logging.info(
            f"  Replicate {rep + 1}/{n_reps}: log-likelihood = {ll:.4f}; "
            f"steps = {result['n_steps']}; status = {status}"
        )
    best_idx = int(np.argmax(all_lls))
    logging.info(f"Best replicate: {best_idx + 1} (log-likelihood = {all_lls[best_idx]:.4f})")
    return all_params, best_idx, all_metadata

def make_log_density_fn(pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
            *, include_invariant, aggregate, prior_mode, estimate_eta,
            eigen_jitter, omega_floor, omega_mode): # closure for defining fn
    validate_omega_mode(omega_mode)
    model_fn = codon_site_log_likelihood if eigen_jitter else codon_site_log_likelihood_no_jitter
    omega_axis = None if omega_mode == "scalar" else 0
    batched_loss = jax.vmap(
        model_fn,
        in_axes=(None, None, None, None, None, None, None, omega_axis,
                 None, None, None, None, None, 1)
    )
    def f(raw_x):

        #x = jnp.exp(x)
        #print('x: ',x)
        #return codon_site_log_likelihood(...)
        x = jax.tree.map(positive_transform, raw_x)

        if omega_mode == "per-site" and x["omega"].shape != (X.shape[1],):
            raise ValueError(
                f"Per-site omega must have shape ({X.shape[1]},), got {x['omega'].shape}"
            )

        if omega_floor:
            x["omega"] = jnp.where( # stops gradients for omegas <= 0.01
                x["omega"] > 0.01,
                x["omega"],
                jax.lax.stop_gradient(x["omega"])
            )


        # eta is fixed to 1.0 (no longer sampled) to identify the global GTR rate scale.
        eta = x["eta"] if estimate_eta else jnp.array(1.0, dtype=jnp.float64)
        losses = batched_loss(x["alpha"], x["beta"], x["gamma"], x["delta"], x["epsilon"], eta, x["theta"], x["omega"], pi_eq, log_pi, pimat, pimatinv, pimult, X)
        #print('losses: ',losses)
        if include_invariant:
            selected_losses = losses
        else:
            mask_f = mask.astype(jnp.float64)
            selected_losses = losses * mask_f
        if aggregate == "sum":
            total = jnp.sum(selected_losses)
        elif include_invariant:
            total = jnp.mean(selected_losses)
        else:
            mask_f = mask.astype(jnp.float64)
            total = jnp.sum(selected_losses) / jnp.maximum(jnp.sum(mask_f), 1.0)
        total = total + prior_log_likelihood(
            raw_x, X.shape[1], prior_mode=prior_mode,
            estimate_eta=estimate_eta, omega_mode=omega_mode, aggregate=aggregate,
        )
        return total
    return f


def natural_to_raw_params(params, *, omega_mode):
    """Convert positive natural-scale parameters to this code's raw log scale."""
    validate_omega_mode(omega_mode)
    return {k: jnp.array(positive_transform_inverse(v), dtype=jnp.float64) for k, v in params.items()}


def make_variable_site_mask(X):
    col_max = np.max(X, axis=0)
    col_sum = np.sum(X, axis=0)
    return np.where(col_max == col_sum, 0, 1)


def make_base_params(*, estimate_eta, n_sites=None, omega_mode):
    """Build default raw initial parameters for scalar-GTR fitting."""
    validate_omega_mode(omega_mode)
    if omega_mode == "per-site" and n_sites is None:
        raise ValueError("n_sites is required for per-site omega")
    omega = jnp.array(positive_transform_inverse(0.5), dtype=jnp.float64)
    if omega_mode == "per-site":
        omega = jnp.repeat(omega, int(n_sites))
    params = {
        "alpha":   jnp.array(positive_transform_inverse(1),   dtype=jnp.float64),
        "beta":    jnp.array(positive_transform_inverse(1),   dtype=jnp.float64),
        "gamma":   jnp.array(positive_transform_inverse(1),   dtype=jnp.float64),
        "delta":   jnp.array(positive_transform_inverse(1),   dtype=jnp.float64),
        "epsilon": jnp.array(positive_transform_inverse(1),   dtype=jnp.float64),
        "theta":   jnp.array(positive_transform_inverse(0.5), dtype=jnp.float64),
        "omega":   omega,
    }
    if estimate_eta:
        params["eta"] = jnp.array(positive_transform_inverse(1), dtype=jnp.float64)
    return params


def make_param_labels(params):
    return {k: ("vec" if k == "omega" and jnp.ndim(v) else "scalar")
            for k, v in params.items()}


def evaluate_fixed_params(X, pi_eq, natural_params, *, include_invariant,
                          aggregate, prior_mode, estimate_eta, eigen_jitter,
                          omega_floor, omega_mode):
    """Evaluate the scalar-GTR objective at fixed natural-scale parameters."""
    log_pi, pimat, pimatinv, pimult = prepare_likelihood_transforms(X, pi_eq)
    mask = make_variable_site_mask(X)
    raw_params = natural_to_raw_params(natural_params, omega_mode=omega_mode)
    fn = make_log_density_fn(
        pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
        include_invariant=include_invariant,
        aggregate=aggregate,
        prior_mode=prior_mode,
        estimate_eta=estimate_eta,
        eigen_jitter=eigen_jitter,
        omega_floor=omega_floor,
        omega_mode=omega_mode,
    )
    return float(fn(raw_params))


def _optimize_params(fn, params, solver, n_iter, verbose=True, convergence=None,
                     progress_callback=None):
    """Run the optimization loop and return final params plus convergence metadata."""
    loss_fn = lambda p: -fn(p)
    loss_and_grad = jax.value_and_grad(loss_fn)
    opt_state = solver.init(params)
    use_convergence = convergence is not None and convergence.get("enabled", False)
    check_every = max(int(convergence.get("check_every", 1)), 1) if use_convergence else 1
    patience = max(int(convergence.get("patience", 1)), 1) if use_convergence else 1
    min_steps = max(int(convergence.get("min_steps", 0)), 0) if use_convergence else 0
    tol = float(convergence.get("tol", 0.0)) if use_convergence else 0.0

    best_params = params
    optimization_started = perf_counter()
    if n_iter:
        initial_loss, grad = loss_and_grad(params)
    else:
        initial_loss = loss_fn(params)
        grad = None
    initial_objective = -float(initial_loss)
    best_objective = initial_objective if use_convergence else None
    objective_history = [(0, initial_objective)]
    last_objective = initial_objective
    startup_seconds = None
    steady_state_started = None
    if progress_callback is not None:
        progress_callback(0, n_iter, initial_objective)
    stale_checks = 0
    converged = False
    steps_run = 0

    for step in range(1, n_iter + 1):
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        steps_run = step
        if verbose:
            logging.debug('parameters: %s', jax.tree.map(positive_transform, jnp.array([
                params["alpha"], params["beta"], params["gamma"],
                params["delta"], params["epsilon"], params["theta"]
            ])))
            logging.debug('omegas: %s', jax.tree.map(positive_transform, params["omega"]))

        if step < n_iter:
            current_loss, next_grad = loss_and_grad(params)
        else:
            current_loss = loss_fn(params)
            next_grad = None
        current_objective = -float(current_loss)
        objective_history.append((step, current_objective))
        last_objective = current_objective

        if step == 1:
            startup_seconds = perf_counter() - optimization_started
            steady_state_started = perf_counter()

        if progress_callback is not None:
            progress_callback(step, n_iter, current_objective)

        if use_convergence and step % check_every == 0:
            improvement = current_objective - best_objective
            if current_objective > best_objective:
                best_objective = current_objective
                best_params = params

            if step >= min_steps:
                if improvement <= tol:
                    stale_checks += 1
                else:
                    stale_checks = 0
                if stale_checks >= patience:
                    converged = True
                    break
        if next_grad is not None:
            grad = next_grad

    if startup_seconds is None:
        startup_seconds = perf_counter() - optimization_started
        logging.info(
            "MAP startup: %.3f seconds",
            startup_seconds,
        )
        steady_state_seconds = 0.0
    else:
        steady_state_seconds = perf_counter() - steady_state_started
        logging.info(
            "MAP startup: %.3f seconds",
            startup_seconds,
        )
    logging.info(
        "MAP optimization: %.3f seconds across %d step(s)",
        steady_state_seconds,
        max(steps_run - 1, 0),
    )

    if not use_convergence:
        best_params = params
        best_objective = last_objective

    return {
        "params": best_params,
        "n_steps": steps_run,
        "objective": best_objective,
        "converged": converged,
        "objective_history": objective_history,
    }


def save_params(output_stem: str, params: dict, mask: np.ndarray = None,
                *, omega_mode) -> None:
    """Save MAP scalar parameter estimates to CSV.

    scalar_<stem>_Allparams.csv — scalar-mode parameters
    per_site_<stem>_GTRparams.csv — shared GTR/theta parameters
    """
    validate_omega_mode(omega_mode)
    output_stem = mode_output_stem(output_stem, omega_mode)
    scalar_keys = SCALAR_PARAM_KEYS_WITH_ETA if omega_mode == "scalar" else GTR_PARAM_KEYS_WITH_ETA
    parameter_name = "Allparams" if omega_mode == "scalar" else "GTRparams"
    scalar_path = output_stem + f"_{parameter_name}.csv"
    rows = [(k, float(positive_transform(params[k]))) for k in scalar_keys if k in params]
    with open(scalar_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["variable", "value"])
        w.writerows(rows)
    logging.info("Saved scalar parameters to: %s", scalar_path)
    if omega_mode == "per-site":
        if mask is None:
            mask = np.ones(len(np.asarray(params["omega"])), dtype=int)
        omega_path = output_stem + "_omega.csv"
        with open(omega_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["site", "omega_map", "variant"])
            for i, (value, variant) in enumerate(zip(np.asarray(positive_transform(params["omega"])), mask), start=1):
                w.writerow([i, float(value), int(variant)])
        logging.info("Saved per-site omega estimates to: %s", omega_path)





def _posterior_draws_natural(raw_samples):
    return {k: positive_transform(v) for k, v in raw_samples.items()}


def summarize_posterior_samples(raw_samples, infos, *, omega_mode):
    """Summarize posterior samples and BlackJAX diagnostics on natural scale."""
    samples = _posterior_draws_natural(raw_samples)
    validate_omega_mode(omega_mode)
    summaries = {}
    for k in SCALAR_PARAM_KEYS_WITH_ETA:
        if k not in samples:
            continue
        vals = np.asarray(samples[k])
        flat = vals.reshape(-1) if omega_mode == "scalar" or k != "omega" else vals.reshape(-1, vals.shape[-1])
        if omega_mode == "per-site" and k == "omega":
            summaries[k] = {
                "mean": np.mean(flat, axis=0), "sd": np.std(flat, axis=0, ddof=1),
                "median": np.quantile(flat, 0.5, axis=0),
                "q2.5": np.quantile(flat, 0.025, axis=0), "q25": np.quantile(flat, 0.25, axis=0),
                "q75": np.quantile(flat, 0.75, axis=0), "q97.5": np.quantile(flat, 0.975, axis=0),
                "ess": np.asarray(blackjax.ess(samples[k])),
                "rhat": np.asarray(blackjax.rhat(samples[k])) if vals.shape[0] > 1 else np.full(vals.shape[-1], np.nan),
            }
            continue
        summaries[k] = {
            "mean": float(np.mean(flat)),
            "sd": float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0,
            "median": float(np.quantile(flat, 0.5)),
            "q2.5": float(np.quantile(flat, 0.025)),
            "q25": float(np.quantile(flat, 0.25)),
            "q75": float(np.quantile(flat, 0.75)),
            "q97.5": float(np.quantile(flat, 0.975)),
            "ess": float(blackjax.ess(samples[k])),
            "rhat": float(blackjax.rhat(samples[k])) if vals.shape[0] > 1 else np.nan,
        }

    diagnostics = {
        "mean_acceptance_rate": float(jnp.mean(infos["acceptance_rate"])),
        "n_divergent": int(jnp.sum(infos["is_divergent"])),
    }
    return samples, summaries, diagnostics


def save_posterior_outputs(output_stem, raw_samples, summaries, *, omega_mode):
    """Save posterior draws and scalar summaries to CSV files."""
    samples = _posterior_draws_natural(raw_samples)
    validate_omega_mode(omega_mode)
    output_stem = mode_output_stem(output_stem, omega_mode)
    keys = [k for k in SCALAR_PARAM_KEYS_WITH_ETA if k in samples and (omega_mode == "scalar" or k != "omega")]
    samples_path = output_stem + "_posterior_samples.csv"
    with open(samples_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["chain", "draw"] + keys)
        n_chains, n_draws = np.asarray(samples[keys[0]]).shape[:2]
        for chain in range(n_chains):
            for draw in range(n_draws):
                w.writerow([chain, draw] + [float(samples[k][chain, draw]) for k in keys])

    if omega_mode == "per-site":
        omega_summary_path = output_stem + "_omega_posterior_summary.csv"
        omega = np.asarray(samples["omega"])
        summary = summaries["omega"]
        with open(omega_summary_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["site", "mean", "sd", "median", "q2.5", "q25", "q75", "q97.5", "ess", "rhat"])
            for site in range(omega.shape[-1]):
                w.writerow([site + 1] + [float(summary[field][site]) for field in
                                         ("mean", "sd", "median", "q2.5", "q25", "q75", "q97.5", "ess", "rhat")])

    summary_path = output_stem + "_posterior_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = _csv.writer(f)
        fields = ["variable", "mean", "sd", "median", "q2.5", "q25", "q75", "q97.5", "ess", "rhat"]
        w.writerow(fields)
        for k in keys:
            row = summaries[k]
            w.writerow([k] + [row[field] for field in fields[1:]])

    logging.info("Saved posterior samples to: %s", samples_path)
    logging.info("Saved posterior summary to: %s", summary_path)


def _log_posterior_summary(summaries, diagnostics):
    logging.info("BlackJAX NUTS posterior summary (natural scale):")
    for k in SCALAR_PARAM_KEYS_WITH_ETA:
        if k not in summaries:
            continue
        s = summaries[k]
        if np.ndim(s["mean"]) != 0:
            logging.info("  %-8s: %d site-wise posterior summaries", k, len(np.asarray(s["mean"])))
            continue
        logging.info(
            "  %-8s: mean=%.4f, median=%.4f, 95%% CI=(%.4f, %.4f), ESS=%.1f, R-hat=%.4f",
            k, s["mean"], s["median"], s["q2.5"], s["q97.5"], s["ess"], s["rhat"],
        )
    logging.info(
        "Diagnostics: mean acceptance=%.4f, divergences=%d",
        diagnostics["mean_acceptance_rate"], diagnostics["n_divergent"],
    )


def _perturb_params(params, scale=0.5):
    """Add Normal(0, scale) noise to all parameters in raw (unconstrained) space."""
    key = jax.random.PRNGKey(int(np.random.randint(0, 2**31)))
    flat, unflatten = ravel_pytree(params)
    noise = jax.random.normal(key, flat.shape) * scale
    return unflatten(flat + noise)


def plot_replicates(all_params_list, best_idx):
    """Plot scalar parameter estimates across replicate runs."""
    n_reps = len(all_params_list)
    scalar_keys = [k for k in GTR_PARAM_KEYS_WITH_ETA if k in all_params_list[0]]
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, params in enumerate(all_params_list):
        is_best = (i == best_idx)
        vals = [float(positive_transform(params[k])) for k in scalar_keys if k in params]
        ax.scatter(
            np.arange(len(vals)),
            vals,
            color=cmap(i % 10),
            alpha=0.9 if is_best else 0.35,
            s=60 if is_best else 25,
            zorder=4 if is_best else 2,
            label=f'Rep {i + 1} (best)' if is_best else f'Rep {i + 1}',
        )
    ax.set_xticks(np.arange(len(scalar_keys)))
    ax.set_xticklabels(scalar_keys)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.set_ylabel('Estimate (natural scale)')
    ax.set_title(f'Scalar parameter estimates across {n_reps} replicates')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)

    plt.tight_layout()
    return fig, ax


def plot_likelihood_history(all_metadata, best_idx):
    """Plot sampled MAP objective values, emphasizing the best replicate."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.tab10

    for rep, metadata in enumerate(all_metadata):
        history = metadata["objective_history"]
        iterations, objectives = zip(*history)
        is_best = rep == best_idx
        ax.plot(
            iterations,
            objectives,
            color=cmap(rep % 10),
            linewidth=2.5 if is_best else 1.2,
            alpha=1.0 if is_best else 0.55,
            zorder=3 if is_best else 2,
            label=f"Replicate {rep + 1}" + (" (best)" if is_best else ""),
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title("MAP log-likelihood by iteration")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_per_site_omega(params, mask=None, *, domain_labels):
    """Plot per-site omega estimates, optionally coloured by domain labels."""
    omega = np.asarray(positive_transform(params["omega"]))
    sites = np.arange(1, len(omega) + 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    colours = np.full(len(omega), "black", dtype=object)
    if mask is not None:
        colours[np.asarray(mask) == 0] = "lightgrey"
    if domain_labels is not None:
        labels = np.asarray(domain_labels, dtype=object)
        if labels.shape != omega.shape:
            raise ValueError("Domain labels must have one value per alignment site")
        colours = np.where(labels == "extracellular", "tomato", colours)
        colours = np.where(labels == "other", "steelblue", colours)
    ax.scatter(sites, omega, c=colours, s=15, alpha=0.75)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, max(float(np.max(omega)), 1.0))
    y_formatter = plt.ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.set_xlabel("Alignment codon site")
    ax.set_ylabel("omega")
    ax.set_title("Per-site omega estimates")
    fig.tight_layout()
    return fig, ax


def _prepare_model(X, pi_eq, include_invariant, aggregate, prior_mode,
                   estimate_eta, eigen_jitter, omega_floor, omega_mode):
    """Build the objective, initial parameters, and site mask shared by fitters."""
    validate_omega_mode(omega_mode)
    logging.info("Precomputing transforms...")
    #col = 30 # site in the alignment
    col = 7 # site in the alignment # this is a column with a bit of diversity (unlike 31)
    # add for loop later
    #X[:,7] = jnp.zeros(61)
    #X[:,7] = np.zeros(61)
    #X[15,7] = 4
    #X[47,7] = 19
    #X[15,7] = 4
    #X[47,7] = 19
    #X = np.zeros((61,1))
    #X[15,:] = 4
    #X[47,:] = 19
    #col = 0
    #X = np.zeros((61,10))
    #X[15,:] = 4
    #X[47,:] = 19
    #col = 0
    #X = np.array(X[:,11:16]) # found some diversity in these columns
    #X = np.array(X[:,11:18]) # found some diversity in these columns, and last column has no diversity
    #X = np.array(X[:,7:18]) # found some diversity in these columns, and last column has no diversity # position 9 is problematic has 5x one codon, 18x another, which corresponds to nonsyn mutation I think (so dS = 0)
    #X = np.array(X[:,9:10])
    #X = np.array(X[:,7:8])
    # probably need exceptions for these cases?
    #print("X shape",X.shape)
    # I think there's a problem with the function reading in the data (the order of the codons)
    """ X = np.zeros((61,5))
    X[9,0] = 5
    X[22,0] = 18
    X[24,1] = 5
    X[37,1] = 17
    X[55,1] = 1
    X[38,2] = 1 # 39  55  60
    X[54,2] = 5
    X[59,2] = 17
    X[49,3] = 8 # 50  58  59
    X[57,3] = 13
    X[58,3] = 2
    X[23,4] = 5 # 24  25  40
    X[24,4] = 17
    X[39,4] = 1
    X = np.array(X[:,1:4]) """
    #X = np.array(X[:,10:11]) # this one for example behaves like it has found stop codons, where actually there should be 17x of AAT
    # it should be (based on stan code)
    #X = np.zeros((61,1))
    #X[24,0] = 5
    #X[37,0] = 17
    #X[55,0] = 1
    #X = np.zeros((61,1))
    #X[9,0] = 5
    #X[22,0] = 18
    #X = np.array(X[:,10:14])
    #X = np.array(X[:,0:15])
    log_pi, pimat, pimatinv, pimult = prepare_likelihood_transforms(X, pi_eq)
    # l is length of alignment
    #print("X",X)
    #print("sum X", np.sum(X))
    #print("larger zero", np.where(X > 0))

    # calculate mask for masking parts of the alignment where there is no diversity
    # this will allow using these position for calculating gradient for constant parameters but excludes omega calculation for these positions
    mask = make_variable_site_mask(X) # create mask for positions without diversity
    #print("col_max",col_max)
    #print("col_sum",col_sum)
    #print("mask",mask)
    #mask = mask.at[0].set(0.0)
    #mask2 = mask ==1
    #print("mask where",mask2)
    #X = X[:,mask2] # this could be an alternative, where I filter X by positions that show diversity
    #print("X",X)
    #log_pi, pimat, pimatinv, pimult = prepare_likelihood_transforms(X, pi_eq)
    logging.info("Compiling model...")

    base_params = make_base_params(
        n_sites=X.shape[1], estimate_eta=estimate_eta, omega_mode=omega_mode
    )
    fn = make_log_density_fn(
        pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
        include_invariant=include_invariant,
        aggregate=aggregate,
        prior_mode=prior_mode,
        estimate_eta=estimate_eta,
        eigen_jitter=eigen_jitter,
        omega_floor=omega_floor,
        omega_mode=omega_mode,
    )
    return fn, base_params, mask




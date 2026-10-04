#!/usr/bin/env python

import logging
import argparse
import numpy as np

from .__init__ import __version__
from .alignment import (
    count_codons,
    estimate_f3x4_pi_from_counts,
    estimate_pi_from_counts,
)


def _positive_int(value):
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed

def get_options():
    parser = argparse.ArgumentParser(
        description=(
            'TOMBOMBADIL (Tree-free Omega Mapping By Observing Mutations of '
            'Bases and Amino acids Distributed Inside Loci)'
        ),
        prog='tombombadil',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group('Input/output')
    io_group.add_argument('--alignment', type=str, required=True,
                          help='Alignment file to fit model to')
    io_group.add_argument('--output-jax', type=str, default=None, metavar='STEM',
                          help='Result file stem for generated outputs')
    io_group.add_argument('--domains', type=str, default=None,
                          help='Optional domain JSON for colouring per-site omega plots')
    io_group.add_argument('--reference', type=str, default=None,
                          help='Reference protein FASTA required with --domains')

    fit_group = parser.add_argument_group('Fitting method')
    fit_group.add_argument('--fit-method', choices=['map', 'nuts'], default='map',
                           help='Fit with MAP optimisation or BlackJAX NUTS sampling')

    runtime_group = parser.add_argument_group('CPU/runtime')
    runtime_group.add_argument('--platform', choices=['cpu', 'gpu', 'tpu'], default='cpu',
                               help='Which hardware/device to run on')
    runtime_group.add_argument('--cpus', type=_positive_int, default=4,
                               help='JAX worker setting on CPU; also sets local devices for NUTS pmap')

    model_group = parser.add_argument_group('Shared model options')
    model_group.add_argument('--pi', choices=['uniform', 'empirical', 'F3x4'], default='uniform',
                             help='Codon equilibrium frequencies')
    model_group.add_argument('--pi-pseudocount', type=float, default=0.5,
                             help='Pseudocount for empirical or F3x4 codon frequencies')
    model_group.add_argument('--omega-mode', choices=['scalar', 'per-site'], default='scalar',
                             help='Estimate one omega for the alignment or one per codon site')
    model_group.add_argument('--objective-aggregate', choices=['mean', 'sum'], default='sum',
                             help='Aggregate site log-likelihoods by mean or sum')
    model_group.add_argument('--prior-mode',
                             choices=['current', 'none', 'stan_constrained', 'stan_unconstrained'],
                             default='stan_unconstrained',
                             help='Prior/Jacobian convention for model fitting')
    model_group.add_argument('--fix-eta', action='store_true', default=False,
                             help='Fix eta to 1.0 instead of estimating it')
    model_group.add_argument('--exclude-invariant', action='store_true', default=False,
                             help='Exclude invariant sites from the data likelihood')

    map_group = parser.add_argument_group('MAP / Optax')
    map_group.add_argument('--max-it', type=_positive_int, default=500, metavar='N',
                           help='Maximum MAP optimisation iterations')
    map_group.add_argument('--fit-replicates', type=int, default=1, metavar='N',
                           help='Run N perturbed starts and select the best MAP fit')
    map_group.add_argument('--estimate-uncertainty', action='store_true', default=False,
                           help='Estimate parameter standard errors with a diagonal Laplace approximation')
    map_group.add_argument('--fixed-iterations', action='store_true', default=False,
                           help='Run all --max-it steps without early convergence')
    map_group.add_argument('--convergence-tol', type=float, default=1e-6,
                           help='Minimum objective improvement counted as progress')
    map_group.add_argument('--convergence-patience', type=int, default=3,
                           help='Convergence checks without progress before stopping')
    map_group.add_argument('--convergence-check-every', type=int, default=1,
                           help='Check convergence every N optimizer steps')
    map_group.add_argument('--convergence-min-steps', type=int, default=10,
                           help='Minimum steps before convergence can stop fitting')

    nuts_group = parser.add_argument_group('NUTS / BlackJAX')
    nuts_group.add_argument('--num-warmup', type=int, default=1000,
                             help='NUTS warmup steps per chain')
    nuts_group.add_argument('--num-samples', type=int, default=1000,
                             help='NUTS posterior draws per chain')
    nuts_group.add_argument('--num-chains', type=int, default=4,
                             help='Number of NUTS chains')
    nuts_group.add_argument('--rng-seed', type=int, default=0,
                             help='Random seed for NUTS')
    nuts_group.add_argument('--target-acceptance-rate', type=float, default=0.8,
                             help='Target acceptance rate for NUTS adaptation')
    nuts_group.add_argument('--nuts-chain-mode', choices=['sequential', 'pmap'], default='sequential',
                             help='Run NUTS chains sequentially or in parallel across JAX devices')

    diagnostic_group = parser.add_argument_group('Fixed-parameter diagnostics')
    diagnostic_group.add_argument('--diagnostic-fixed-params', action='store_true', default=False,
                                  help='Evaluate the scalar-GTR objective at fixed natural-scale parameters and exit')
    diagnostic_group.add_argument('--diagnostic-alpha', type=float, default=1.0)
    diagnostic_group.add_argument('--diagnostic-beta', type=float, default=1.0)
    diagnostic_group.add_argument('--diagnostic-gamma', type=float, default=1.0)
    diagnostic_group.add_argument('--diagnostic-delta', type=float, default=1.0)
    diagnostic_group.add_argument('--diagnostic-epsilon', type=float, default=1.0)
    diagnostic_group.add_argument('--diagnostic-eta', type=float, default=1.0)
    diagnostic_group.add_argument('--diagnostic-theta', type=float, default=0.5)
    diagnostic_group.add_argument('--diagnostic-omega', type=float, default=0.5)
    diagnostic_group.add_argument('--diagnostic-prior-mode',
                                  choices=['none', 'current', 'stan_constrained', 'stan_unconstrained'],
                                  default='none',
                                  help='Prior/Jacobian convention for fixed scoring')
    diagnostic_group.add_argument('--diagnostic-aggregate', choices=['sum', 'mean'], default='sum',
                                  help='Aggregate site log-likelihoods for fixed scoring')
    diagnostic_group.add_argument('--diagnostic-fix-eta', action='store_true', default=False,
                                  help='Score with eta fixed to 1.0 instead of using --diagnostic-eta')
    diagnostic_group.add_argument('--diagnostic-enable-jitter', action='store_true', default=False,
                                  help='Enable eigen jitter while fixed scoring')
    diagnostic_group.add_argument('--diagnostic-enable-omega-floor', action='store_true', default=False,
                                  help='Enable the omega gradient floor while fixed scoring')

    other_group = parser.add_argument_group('Other options')
    other_group.add_argument('--version', action='version',
                             version='%(prog)s '+__version__)

    args = parser.parse_args()
    return args

def configure_jax_for_options(options):
    """Set JAX process flags that must exist before JAX is imported."""
    from .device import configure_platform
    configure_platform(
        options.platform,
        cpus=options.cpus,
        force_cpu_devices=(
            options.fit_method == "nuts" and options.nuts_chain_mode == "pmap"
        ),
    )

def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True)

    options = get_options()
    configure_jax_for_options(options)
    if options.platform == "cpu":
        logging.info("JAX CPU worker setting: %d", options.cpus)
    logging.info("Reading alignment...")
    X, n_samples = count_codons(options.alignment)
    logging.info(f"Read {n_samples} samples and {X.shape[1]} codons")

    #print("X",X.max())
    if options.pi == 'uniform':
        pi = np.full(61, 1 / 61)
        logging.info("Using uniform codon equilibrium frequencies")
    elif options.pi == 'empirical':
        pi = estimate_pi_from_counts(X, options.pi_pseudocount)
        logging.info(
            "Using empirical codon equilibrium frequencies estimated from alignment "
            f"with pseudocount {options.pi_pseudocount}"
        )
    elif options.pi == 'F3x4':
        pi = estimate_f3x4_pi_from_counts(X, options.pi_pseudocount)
        logging.info(
            "Using F3x4 codon equilibrium frequencies estimated from alignment "
            f"with pseudocount {options.pi_pseudocount}"
        )
    else:
        raise ValueError(f"Unsupported --pi mode: {options.pi}")

    if options.diagnostic_fixed_params:
        if options.omega_mode != 'scalar':
            raise ValueError('--diagnostic-fixed-params currently supports only --omega-mode scalar')
        from .sample import evaluate_fixed_params

        diagnostic_params = {
            "alpha": options.diagnostic_alpha,
            "beta": options.diagnostic_beta,
            "gamma": options.diagnostic_gamma,
            "delta": options.diagnostic_delta,
            "epsilon": options.diagnostic_epsilon,
            "eta": options.diagnostic_eta,
            "theta": options.diagnostic_theta,
            "omega": options.diagnostic_omega,
        }
        value = evaluate_fixed_params(
            X, pi, diagnostic_params,
            include_invariant=not options.exclude_invariant,
            aggregate=options.diagnostic_aggregate,
            prior_mode=options.diagnostic_prior_mode,
            estimate_eta=not options.diagnostic_fix_eta,
            eigen_jitter=options.diagnostic_enable_jitter,
            omega_floor=options.diagnostic_enable_omega_floor,
            omega_mode=options.omega_mode,
        )
        logging.info("Diagnostic scalar-GTR objective: %.10f", value)
        return

    from .sample import _prepare_model, run_map_optimizer, run_nuts_sampler

    domain_labels = None
    if options.domains is not None:
        if options.omega_mode != 'per-site':
            raise ValueError('--domains requires --omega-mode per-site')
        if options.reference is None:
            raise ValueError('--reference is required when --domains is specified')
        from .domains import parse_domain_labels
        domain_labels = parse_domain_labels(options.domains, options.alignment,
                                            options.reference, X.shape[1])

    fn, start_params, mask = _prepare_model(
        X, pi,
        include_invariant=not options.exclude_invariant,
        aggregate=options.objective_aggregate,
        prior_mode=options.prior_mode,
        estimate_eta=not options.fix_eta,
        eigen_jitter=True,
        omega_floor=True,
        omega_mode=options.omega_mode,
    )

    if options.fit_method == "map":
        run_map_optimizer(
            fn, start_params, mask,
            max_it=options.max_it,
            estimate_uncertainty=options.estimate_uncertainty,
            fit_replicates=options.fit_replicates,
            output=options.output_jax,
            fit_until_convergence=not options.fixed_iterations,
            convergence_tol=options.convergence_tol,
            convergence_patience=options.convergence_patience,
            convergence_check_every=options.convergence_check_every,
            convergence_min_steps=options.convergence_min_steps,
            omega_mode=options.omega_mode,
            domain_labels=domain_labels,
        )
    elif options.fit_method == "nuts":
        logging.info(
            "Running BlackJAX NUTS — %s chain(s), %s warmup step(s), %s draw(s)",
            options.num_chains, options.num_warmup, options.num_samples,
        )
        result_stem = options.output_jax if options.output_jax is not None else "output"
        run_nuts_sampler(
            fn,
            start_params,
            num_warmup=options.num_warmup,
            num_samples=options.num_samples,
            num_chains=options.num_chains,
            rng_seed=options.rng_seed,
            target_acceptance_rate=options.target_acceptance_rate,
            output=result_stem,
            chain_mode=options.nuts_chain_mode,
            omega_mode=options.omega_mode,
        )
    else:
        raise ValueError(f"Unsupported fit method: {options.fit_method!r}")

if __name__ == "__main__":
    main()

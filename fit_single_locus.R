library(cmdstanr)
library(bayesplot)
mod <- cmdstan_model("models/single_locus.stan", 
                              cpp_options = list(stan_threads = TRUE))

data_list <- list(n_genomes = 23,
                  n_observed = c(c(11, 12), rep(0, 59)),
                  pi_eq = rep(1/61, 61))

model_fit <- mod$sample(data = data_list, threads_per_chain = 1, parallel_chains = 4)
results <- model_fit$draws(variables = c("omega", "kappa", "theta"))
mcmc_areas(results)
mcmc_trace(results)
mcmc_hist(results)
model_fit$summary()


var_inf <- mod$variational(data = data_list, threads = 8)
var_inf$draws(variables = c("omega", "kappa", "theta"))
var_inf$summary(variables = c("omega", "kappa", "theta"))

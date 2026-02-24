library(cmdstanr)
library(bayesplot)
library(seqinr)
library(data.table)

source("R_funcs/generate_data.R")

# Read alignment
data <- seqinr::read.alignment(file = "data/porB3.carriage.noindels.txt", format = "fasta")
cores <- 8
nchains <- 1
thr_per_chain <- floor(cores / nchains)
Sys.setenv(STAN_NUM_THREADS = cores)

#data_list_singlelocus_example <- list(n_genomes = 100,
                  #n_observed = c(c(11, 12, 100 - 11 - 12), rep(0, 58)),
                  #pi_eq = rep(1/61, 61))

data_list_all <- generate_data(data, cores = cores)
data_list <- list(n_genomes = data_list_all$n,
                  n_observed = unname(data_list_all$X[8,]),
                  n_observed_array = unname(data_list_all$X[8,]),
                  pi_eq = rep(1/61, 61))

mod <- cmdstan_model("models/single_locus.stan",
                     cpp_options = list(stan_threads = TRUE))

model_fit <- mod$sample(
  iter_sampling = 1,
  chains = 1,
  fixed_param = TRUE,
  init = function() list(l_omega = log(0.003), l_kappa = log(1.0), l_theta = log(0.5)),
  #data = data_list_singlelocus_example,
  data = data_list,
  threads_per_chain = 1,
  parallel_chains = 1
)

#results <- model_fit$draws(variables = c("omega", "kappa", "theta"))


#model_fit$summary()

library(cmdstanr)
library(seqinr)
library(data.table)
source("R_funcs/generate_data.R")

# Read alignment
# data <- seqinr::read.alignment(file = "data/porB3.carriage.noindels.txt", format = "fasta")
data <- seqinr::read.alignment(file = "~/Downloads/recO.aln.fas", format = "fasta")

cores <- 8
nchains <- 1
thr_per_chain <- floor(cores / nchains)
Sys.setenv(STAN_NUM_THREADS = cores)

data_list <- generate_data(data, cores = cores)

# data list
data_list$K <- 2
data_list$om_priors <- c(1, 2.5)


# Compile model
mod_fit_test <- cmdstan_model("models/fit_omega.stan", 
                         cpp_options = list(stan_threads = TRUE))

fit_new <- mod_fit_test$sample(
  data = data_list,
  iter_warmup = 1000, 
  iter_sampling = 500,
  threads_per_chain = 4,
  fixed_param = FALSE,
  chains = 1, 
  refresh = 1
)
fit <- readRDS("~/Downloads/hierarchical_fit(1).RDS")

library(bayesplot)
mcmc_intervals(fit$draws("omega"))

fit$summary()
mcmc_intervals(fit$draws("omega"))
fit$summary("omega_mean")

library(cmdstanr)
library(seqinr)
library(data.table)
source("R_funcs/generate_data.R")

# Read alignment
data <- seqinr::read.alignment(file = "data/porB3.carriage.noindels.txt", format = "fasta")

cores <- 8
nchains <- 1
thr_per_chain <- floor(cores / nchains)
Sys.setenv(STAN_NUM_THREADS = cores)

data_list <- generate_data(data, cores = cores)

# data list
data_list$theta <- 0.17
data_list$omega <- 1
data_list$kappa <- 1

# Compile model
mod_fit_test <- cmdstan_model("models/fit_omega.stan", 
                         cpp_options = list(stan_threads = TRUE))

fit_new <- mod_fit_test$sample(
  data = data_list,
  iter_warmup = 1000, 
  iter_sampling = 1000,
  threads_per_chain = 8,
  fixed_param = FALSE,
  chains = 1, 
  refresh = 50
)


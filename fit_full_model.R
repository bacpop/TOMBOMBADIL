library(cmdstanr)
library(seqinr)
library(data.table)
library(bayesplot)
library(magrittr)
library(ggplot2)
source("R_funcs/generate_data.R")

# Read alignment
data <- seqinr::read.alignment(file = "data/porB3.carriage.noindels.txt", format = "fasta")
# data <- seqinr::read.alignment(file = "recO.aln-1.fas", format = "fasta")

cores <- 8
nchains <- 1
thr_per_chain <- floor(cores / nchains)
Sys.setenv(STAN_NUM_THREADS = cores)

data_list <- generate_data(data, cores = cores)

# Flags between NY98 and GTR substitution models
data_list$GTR <- 0
# Flags between omega being constant over loci or varying
data_list$omega_varies <- 0
# If omega varies over loci, flags between omega being i.i.d or hierarchical 
data_list$omega_hierarchical <- 0
# Mixture model for omega
data_list$omega_mixture <- 1
data_list$K <- 3
data_list$mixprop <- c(0.01, 0.1, 0.89)
data_list$mix_mean <- c(1.5, 0.8, 0.1)
data_list$mix_var <- c(1, 1, 1)

# Compile model
mod_fit_test <- cmdstan_model("tombombadil.stan", 
                         cpp_options = list(stan_threads = TRUE))

fit_new <- mod_fit_test$sample(
  data = data_list,
  iter_warmup = 1, 
  iter_sampling = 1,
  threads_per_chain = thr_per_chain,
  fixed_param = FALSE,
  chains = 1, 
  refresh = 1
)

mcmc_intervals(laplace_fit$draws("omega"))






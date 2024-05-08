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

## THIS IS FROM DANNY'S MCMC OUTPUT FOR CONSTANT OMEGA. 
## WE NEED OUR LIKELIHOOD OUTPUT TO MATCH loglik.seqs

# iteration loglikelihood loglik.seqs.     theta   kappa   omega0
# 1           5      -1957.71     -1958.48 0.1700000 1.00000 1.000000
# 3          15      -1939.37     -1940.16 0.1700000 1.63177 0.492084
# 151       755      -1934.01     -1934.22 0.1140420 2.83147 0.920608

# data list
data_list$theta <- 0.17
data_list$omega <- 1
data_list$kappa <- 1

# Compile model
mod_test <- cmdstan_model("models/exact.stan", 
                         cpp_options = list(stan_threads = TRUE))

fit <- mod_test$sample(
  data = data_list,
  iter_warmup = 1, 
  iter_sampling = 1,
  threads_per_chain = 8,
  chains = 1, 
  fixed_param = TRUE,
  refresh = 50
)

# We want the value -1958.48
as.vector(fit$draws("lik"))

library(cmdstanr)
library(seqinr)
library(data.table)
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
data_list$omega_varies <- 1
# If omega varies over loci, flags between omega being i.i.d or hierarchical 
data_list$omega_hierarchical <- 1

# Compile model
mod_fit_test <- cmdstan_model("full model/tombombadil.stan", 
                         cpp_options = list(stan_threads = TRUE))

fit_new <- mod_fit_test$sample(
  data = data_list,
  iter_warmup = 1000, 
  iter_sampling = 1000,
  threads_per_chain = thr_per_chain,
  fixed_param = FALSE,
  chains = 1, 
  refresh = 1
)
fit_new$save_object("laplace_fit_full.RDS")
library(bayesplot)
library(magrittr)
library(ggplot2)

mcmc_intervals(laplace_fit$draws("omega"))
fit_new$summary("omega_mean")
hierarchical_fit$summary("omega[208]")

mcmc_intervals(laplace_fit$draws("omega"))

mcmc_intervals(fit_new$draws("pr"))

summarise_draws(fit_new$draws("pr"),pr =function(x){sum(x == 2)/length(x)}) %>%
  ggplot(aes(x = variable, y = pr)) + 
  geom_point()

hierarchical_fit$draws()
exp(-0.88)

library(posterior)
library(data.table)
hf <- as.data.table(summarise_draws(hierarchical_fit$draws("omega"), ~quantile(.x, probs = c(0.975, 0.5, 0.025))))

hf[, codon_position := 1:.N]

dt <- merge.data.table(hf, recO, by = "codon_position")


dt[omega_mean == 0, order(`50%`, decreasing = TRUE)]
dt[omega_mean == 0][order(`50%`, decreasing = TRUE)]
dt[208, ]

data_list$X[208,]

data_list$n_samples[208]




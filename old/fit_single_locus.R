library(cmdstanr)
library(bayesplot)
mod <- cmdstan_model("models/single_locus.stan", 
                              cpp_options = list(stan_threads = TRUE))

geneticCode <- list(
  "TTT"="Phe","TTC"="Phe","TTA"="Leu","TTG"="Leu",
  "TCT"="Ser","TCC"="Ser","TCA"="Ser","TCG"="Ser",
  "TAT"="Tyr","TAC"="Tyr","TAA"="STO","TAG"="STO",
  "TGT"="Cys","TGC"="Cys","TGA"="STO","TGG"="Trp",
  "CTT"="Leu","CTC"="Leu","CTA"="Leu","CTG"="Leu",
  "CCT"="Pro","CCC"="Pro","CCA"="Pro","CCG"="Pro",
  "CAT"="His","CAC"="His","CAA"="Gln","CAG"="Gln",
  "CGT"="Arg","CGC"="Arg","CGA"="Arg","CGG"="Arg",
  "ATT"="Ile","ATC"="Ile","ATA"="Ile","ATG"="Met",
  "ACT"="Thr","ACC"="Thr","ACA"="Thr","ACG"="Thr",
  "AAT"="Asn","AAC"="Asn","AAA"="Lys","AAG"="Lys",
  "AGT"="Ser","AGC"="Ser","AGA"="Arg","AGG"="Arg",
  "GTT"="Val","GTC"="Val","GTA"="Val","GTG"="Val",
  "GCT"="Ala","GCC"="Ala","GCA"="Ala","GCG"="Ala",
  "GAT"="Asp","GAC"="Asp","GAA"="Glu","GAG"="Glu",
  "GGT"="Gly","GGC"="Gly","GGA"="Gly","GGG"="Gly")
tripletNames = names(geneticCode)
tripletNames_noSTO <- tripletNames[-c(11, 12, 15)]
triprev <- rev(tripletNames_noSTO)

data_list <- list(n_genomes = 100,
                  n_observed = c(c(11, 12, 100 - 11 - 12), rep(0, 58)),
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

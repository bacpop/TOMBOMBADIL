#install.packages("cmdstanr", repos = c('https://stan-dev.r-universe.dev', getOption("repos")))
#library(cmdstanr)
#install.packages("seqinr", repos='https://cran.ma.imperial.ac.uk/')
library(seqinr)
#install.packages("bayesplot", repos='https://cran.ma.imperial.ac.uk/')
library(bayesplot)
#install.packages("data.table", repos='https://cran.ma.imperial.ac.uk/')
library(data.table)
#install.packages("h2o", repos='https://cran.ma.imperial.ac.uk/')
library(h2o)
#install.packages("cmdstanr", repos = c('https://stan-dev.r-universe.dev', getOption("repos")))
library(cmdstanr)
set_cmdstan_path("/nfs/research/jlees/leonie/SelectionModel/cmdstan-2.35.0")

source("../TOMBOMBADIL/R_funcs/generate_data.R")
model <- cmdstan_model("../TOMBOMBADIL/models/single_locus.stan", 
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

################# INPUT ###################
# Input arguments
args <- commandArgs(trailingOnly=TRUE)
job_array_filename <- args[1]
row_index <- as.integer(args[2])
output_folder <- args[3]

print(job_array_filename)
print(row_index)

job_array_df = read.table(job_array_filename, header=FALSE)

# load the list of codons
fasta_file_path <- job_array_df[row_index,3] # is where file location of current file is saved (?)
gene_name <- job_array_df[row_index,2]
print(fasta_file_path)
# Line  that passes arguments from job array
#args <- commandArgs(trailingOnly = TRUE)

# Alignment path
#fasta_file_path <- args[1]

# Read alignments
data <- seqinr::read.alignment(file = fasta_file_path, format = "fasta")

data_list <- generate_data(data, cores=2)

results_list <- vector("list", length = data_list$gene_length)

# Pre-allocate vectors to store results
omega_mean <- numeric(length = data_list$gene_length)
omega_q5 <- numeric(length = data_list$gene_length)
omega_q95 <- numeric(length = data_list$gene_length)
kappa_mean <- numeric(length = data_list$gene_length)
kappa_q5 <- numeric(length = data_list$gene_length)
kappa_q95 <- numeric(length = data_list$gene_length)
theta_mean <- numeric(length = data_list$gene_length)
theta_q5 <- numeric(length = data_list$gene_length)
theta_q95 <- numeric(length = data_list$gene_length)

# vector containing the matrix showing how many codons are repeated in each region
v <- apply(data_list$X, MARGIN = 1, FUN = max)

# THIS IS FOR THE WHOLE DATASET
for (i in 1:data_list$gene_length) {
  if(v[i] == data_list$n_samples[i]){
    next
  }
  # Creating list of vectors containing counts
  data_list_singlelocus <- list(n_genomes = data_list$n, #How many genomes
                                n_observed = unname(data_list$X[i,]), #vector of codon counts
                                pi_eq = rep(1/61, 61)) # equilibrium of diff codons
  
  # MCMC (or whichever optimiser you want to use)
  model_fit <- model$sample(data = data_list_singlelocus, threads_per_chain = 1, parallel_chains = 4) 
  results <- model_fit$draws(variables = c("omega", "kappa", "theta"))
  results_list[[i]]<- model_fit$summary() # Store the summary results list, this is eq to tibble value in single locus file
  # For readability I created this new value
  tibble_summary <- results_list[[i]]
  # Extracting means and quantiles for each iteration
  omega_mean[i] <- tibble_summary[tibble_summary$variable == "omega", "mean"][[1]]  # Extracts the first value of column mean
  omega_q5[i]   <- tibble_summary[tibble_summary$variable == "omega", "q5"][[1]] 
  omega_q95[i]  <- tibble_summary[tibble_summary$variable == "omega", "q95"][[1]]
  kappa_mean[i] <- tibble_summary[tibble_summary$variable == "kappa", "mean"][[1]]  # Extracts the first value of column mean
  kappa_q5[i]   <- tibble_summary[tibble_summary$variable == "kappa", "q5"][[1]] 
  kappa_q95[i]  <- tibble_summary[tibble_summary$variable == "kappa", "q95"][[1]]
  theta_mean[i] <- tibble_summary[tibble_summary$variable == "theta", "mean"][[1]]  # Extracts the first value of column mean
  theta_q5[i]   <- tibble_summary[tibble_summary$variable == "theta", "q5"][[1]] 
  theta_q95[i]  <- tibble_summary[tibble_summary$variable == "theta", "q95"][[1]]
}

#mcmc_areas(results)
#mcmc_trace(results)
#mcmc_hist(results)
#model_fit$summary()

results_df <- data.frame(
  codon_position = 1:data_list$gene_length,
  omega_mean = omega_mean,
  omega_q5 = omega_q5,
  omega_q95 = omega_q95,
  kappa_mean = kappa_mean,
  kappa_q5 = kappa_q5,
  kappa_q95 = kappa_q95,
  theta_mean = theta_mean,
  theta_q5 = theta_q5,
  theta_q95 = theta_q95
)
# Save data frame (df) with RDS 
#gene_name <- file_path_sans_ext(basename(fasta_file_path))
saveRDS(results_df, file = paste0(output_folder, gene_name, ".rds"))

if(max(omega_mean) > 1){
  write(paste("max omega found is", max(omega_mean)), file = paste0(output_folder, gene_name, ".txt"))
}


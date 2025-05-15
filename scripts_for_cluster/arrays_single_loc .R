library(cmdstanr)
library(seqinr)
library(data.table)
library(tools)
source("R_funcs/generate_data.R")
# Plot
library(ggplot2)

model <- cmdstan_model("models/single_locus.stan")

# List of possible codons and their amino acid
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

################# INPUT ###################

# Line  that passes arguments from job array
args <- commandArgs(trailingOnly = TRUE)

# Alignment path
fasta_file_path <- args[1]

# Read alignments
alignment <- seqinr::read.alignment(file = fasta_file_path, format = "fasta")

###########################################

# Creating variable that contains function generate_data that calculates matrices with codons (input for the MCMC)
# creates a list that contains all the inputs required by your Stan model.
data_list <- generate_data(data, cores=2)


results_list <- vector("list", length = data_list$gene_length)

# Pre-allocate vectors to store results
omega_mean <- numeric(length = data_list$gene_length)
omega_q5 <- numeric(length = data_list$gene_length)
omega_q95 <- numeric(length = data_list$gene_length)

# vector containing the matrix showing how many codons are repeated in each region
v <- apply(data_list$X, MARGIN = 1, FUN = max)

# THIS IS FOR THE WHOLE DATASET
for (i in 1:data_list$gene_length) {
  if(v[i] == data_list$n_samples[i]){
    next
  }
  print(i) # Shows number of codon you're on
  # Creating list of vectors containing counts
  data_list_singlelocus <- list(n_genomes = data_list$n, # How many genomes
                                n_observed = unname(data_list$X[i,]), # vector of codon counts
                                pi_eq = rep(1/61, 61)) # equilibrium of diff codons
  
  # HMC
  model_fit <- model$sample(data = data_list_singlelocus, threads_per_chain = 1, parallel_chains = 1) # for parallelisation, not needed here
  results <- model_fit$draws(variables = c("omega", "kappa", "theta"))
  results_list[[i]]<- model_fit$summary() # Store the summary results list, this is eq to tibble value in single locus file
  # For readability I created this new value
  tibble_summary <- results_list[[i]]
  # Extracting means and quantiles for each iteration
  omega_mean[i] <- tibble_summary[tibble_summary$variable == "omega", "mean"][[1]]  # Extracts the first value of column mean
  omega_q5[i]   <- tibble_summary[tibble_summary$variable == "omega", "q5"][[1]] 
  omega_q95[i]  <- tibble_summary[tibble_summary$variable == "omega", "q95"][[1]]
  
  print(model_fit$summary())
  print(results)
  print(omega_mean)
  print(omega_q5)
  print(omega_q95)
}

###################### RESULTS & PLOTS ###########################

# Extract gene name automatically from the FASTA filename
gene_name <- file_path_sans_ext(basename(fasta_file_path))
# This removes folder and ".fasta" extension -> you get just the gene name

# Convert means and quantiles into dataframes so that ggplot can handle them.
# Create data frame (assuming omega_mean, omega_q5, omega_q95 exist)
df <- data.frame(
  codon_position = 1:data_list$gene_length,
  omega_mean = omega_mean,
  omega_q5 = omega_q5,
  omega_q95 = omega_q95
)

# Create dot plot of mean omega and confidence intervals
p <- ggplot(df, aes(x = codon_position, y = omega_mean)) +
  geom_point(color = 'blue', size = 2, alpha = 1) +
  geom_errorbar(aes(ymin = omega_q5, ymax = omega_q95), width = 0.2, color = 'deepskyblue', alpha = 0.5) +
  labs(
    title = paste0("Omega Mean Across Codon Positions: ", gene_name),
    x = "Codon Position",
    y = "Omega (ω)"
  ) +
  theme_bw()

# Save plot
ggsave(paste0("results_cluster/", gene_name, ".png"), plot = p)

# Save data frame (df) with RDS 
saveRDS(df, file = paste0("results_cluster/", gene_name, ".rds"))
omega_df <- readRDS(paste0("results_cluster/", gene_name, ".rds")) # Assign to an object so that you can View() it


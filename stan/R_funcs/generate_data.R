

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

generate_data <- function(data, cores = 1){
  
  # Chop each sequence into codons
  res <- lapply(X = data$seq, 
                FUN = function(x){toupper(substring(x, 
                                                    seq(1, nchar(x), 3), 
                                                    seq(3, nchar(x), 3)))})
  # Put codons into l x matrix
  l <- length(res[[1]])
  mx <- matrix(unlist(res), byrow = TRUE, ncol = l)
  
  # Convert matrix to data.table and count codons at each location
  tx <- data.table::as.data.table(mx)
  tx$sample <- 1:nrow(tx)
  tx <- melt(tx, id.vars = "sample")
  tx[, site := as.numeric(variable)]
  tx <- tx[, .(sample, site, codon = value)]
  tx$codon <- factor(tx$codon, levels = tripletNames_noSTO)
  tx <- tx[!is.na(codon)]
  rtx <- tx[, as.vector(table(codon)), "site"]
  
  # count matrix for stan model
  X <- matrix(rtx$V1, byrow = TRUE, ncol = 61)
  colnames(X) <- tripletNames_noSTO
  
  # Create output
  out_list <- list()
  out_list$X <- X
  out_list$l <- nrow(X)
  out_list$n <- length(res)
  out_list$N <- rowSums(out_list$X)
  out_list$pi_eq <- rep(1 / 61, 61)
  out_list$grainsize <- 1
  
  ## Shard maths
  nshards <- cores
  out_list$M <- nshards
  ncodon <- out_list$l
  base_n <- floor(ncodon/ nshards)
  spare <- ncodon - (nshards * base_n)
  n_per_shard <- c(rep(base_n + 1, spare), rep(base_n, nshards - spare))
  
  shard_starts <- 1
  shard_ends <- n_per_shard[1]
  for(i in 2:nshards){
    shard_starts[i] <- shard_ends[i - 1] + 1
    shard_ends[i] <- shard_starts[i] + n_per_shard[i] - 1
  }

  out_list$shard_starts <- shard_starts
  out_list$shard_ends <- shard_ends
  out_list$n_per_shard <- n_per_shard
  out_list$max_per_shard <- max(n_per_shard)
  
  return(out_list)
}

extract_res <- function(fit, mod_name = "unnamed"){
  
  var_names <- c("om", "kap", "muu")
  full_names <- c("om" = "omega", "ka" = "kappa", "mu" = "mu")
  
  draws_df <- data.table::as.data.table(fit$draws(format = "df", variables = var_names))
  
  l <- (dim(draws_df)[2] - 3) / 3
  
  
  measure_vars <- paste0(rep(var_names, rep(l, 3)), rep(paste0("[", 1:l, "]"), 3))
  
  draws_df <- melt(draws_df, measure = measure_vars, variable.name = "site", value.name = "value")
  draws_df[, .iteration := NULL]
  draws_df$siteno <- rep(rep(1:l, rep(max(draws_df$.draw), l)), 3)
  draws_df[, site := substr(site, 1, 2)]
  draws_df[, par := full_names[site]]
  
  draws_df <- draws_df[, .(lq = quantile(value, 0.025),
                           uq = quantile(value, 0.975),
                           q25 = quantile(value, 0.25),
                           med = median(value)), by = c("par", "siteno")]
  
  draws_df[, mod := mod_name]
  
  draws_df[, par := factor(draws_df$par, levels = full_names)]
  
  return(draws_df)  
}

initfn <- function(){
  out <- list(kap = rtruncnorm(n = 1, a = 0, mean = 0, sd = 1),
       th = rnorm(1, log(0.1), 1),
       alp = rnorm(1, log(0.1), 1),
       bet = rnorm(1, log(0.1), 1),
       gam = rnorm(1, log(0.1), 1),
       del = rnorm(1, log(0.1), 1),
       eps = rnorm(1, log(0.1), 1),
       et = rnorm(1, log(0.1), 1),
       lambda = runif(1, 0, 1),
       # om_raw = rnorm(n = data_list$l, mean = -1, sd = 0.25),
       om_raw = rtruncnorm(n = data_list$l, a = 0, mean = 0, sd = 0.1),
       omega = rtruncnorm(n = data_list$l, a = 0, mean = 0, sd = 0.5),
       om_mean = rtruncnorm(n = 1, a = 0, mean = 0, sd = 1),
       om_sd = rtruncnorm(n = 1, a = 0, mean = 0, sd = 1))
  out$lambda[2] <- 1 - out$lambda[1]
  return(out)
}

plot_mutations <- function(mat){
  
  temp <- mat_to_dt(mat)
  
  cod_mut <- temp$dt %>% 
    ggplot() +
    geom_tile(aes(y = codon, x = siteno, fill = value)) +
    theme_bw() +
    labs(y = "Codon", x = "location")
  
  prot_mut <- temp$prot_dt %>%
    ggplot() +
    geom_tile(aes(y = protein, x = siteno, fill = value)) +
    theme_bw() +
    labs(y = "Protein abreviation", x = "location")
  
  if(max(temp$dt$siteno) < 50){
    cod_mut <- cod_mut + geom_text(aes(y = codon, x = siteno, label = value))
    prot_mut <- prot_mut + geom_text(aes(y = protein, x = siteno, label = value))
  }
  
  out <- cod_mut + prot_mut + plot_layout(guide = "collect") & 
    # scale_x_continuous(breaks = seq(1, max(dt$siteno), by = 10)) & 
    scale_fill_gradient(low = "blue", high = "red")
  
  
  
  return(out)
}

mat_to_dt <- function(mat){
  dt <- as.data.table(mat)
  colnames(dt) <- tripletNames_noSTO
  
  dt$siteno <- 1:nrow(dt)
  dt <- melt(dt, id.var = "siteno", variable.name = "codon")
  dt$codno <- match(dt$codon, tripletNames_noSTO)
  
  prot <- unlist(geneticCode)[-c(11, 12, 15)]
  dt$protein <- prot[dt$codno]
  prot_dt <- dt[, .(value = sum(value)), by = .(siteno, protein)][order(siteno)]
  
  return(list("dt" = dt, "prot_dt" = prot_dt))
}


plot_dnds <- function(mat){
  
  temp <- mat_to_dt(mat)
  
  p1 <- temp$dt[, .(value = sum(value)), "siteno"] %>%
    ggplot(aes(x = siteno, y = value)) +
    geom_bar(stat = "identity") +
    labs(x = "", y = "Coverage") +
    coord_cartesian(xlim = c(1, nrow(mat)), expand = FALSE)
  
  xdt <- temp$dt[value > 0, .(cod = .N), "siteno"]
  p2 <-  xdt %>%
    ggplot(aes(x = siteno, y = cod)) + 
    geom_bar(position = position_stack(), stat = "identity") +
    # scale_y_log10() +
    labs(x = "", y = "# unique proteins") +
    # geom_hline(yintercept = 1, col = "red", lty = 2)
    coord_cartesian(ylim = c(1, (max(xdt$cod) + 1)), expand = FALSE) +
    scale_y_continuous(breaks = seq(1, (max(xdt$cod) + 1), 2))
  
  
  ndt <- data.table::merge.data.table(temp$dt[, .(n_codon = sum(value > 0)), "siteno"], 
                                      temp$prot_dt[, .(n_prot = sum(value > 0)), "siteno"], by = "siteno")
  
  p3 <- ndt %>%
    ggplot(aes(x = siteno, y = (n_codon / n_prot))) + 
    geom_bar(stat = "identity") +
    labs(x = "location", y = "(# unique codons / # unique proteins)") +
    coord_cartesian(ylim = c(1, (ndt[, max(n_codon / n_prot)])), expand = FALSE)
  
  out <- p1 / p2 / p3 & cowplot::theme_minimal_hgrid()
  
  return(out)
}

# an_3nt <- function(z){
#   nt <- c("A", "C", "G", "T")
#   stop_codons <- c("TAA", "TAG", "TGA")
#   out <- unique(c(paste0(nt, substr(z, 2, 3)),
#                   paste0(substr(z, 1, 2), nt),
#                   paste0(substr(z, 1, 1), 
#                          nt, 
#                          substr(z, 3, 3))))
#   out <- out[!(out %in% stop_codons)]
#   return(out)
# }



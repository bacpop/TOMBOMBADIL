functions {
#include functions/NY98.stan
}

data {
  row_vector[61] pi_eq;
  int <lower = 1> n_genomes;
  vector[61] n_observed;
  // real <lower = 0> kappa;
  // real <lower = 0> omega;
  // real <lower = 0> theta;
} 

transformed data {
  // These things are here because it's possible to only calculate them once
  // at the start of the MCMC
  vector[61] lp = to_vector(log(pi_eq));
  matrix[61, 61] pimat = diag_matrix(sqrt(to_vector(pi_eq)));
  matrix[61, 61] pimatinv = diag_matrix(inv(sqrt(to_vector(pi_eq))));
  vector[61] ones = rep_vector(1, 61);
  matrix[61, 61] pimult;
  real phi = lgamma(n_genomes + 1) - sum(lgamma(n_observed + 1));
  
  for(j in 1:61){
    for(i in 1:61){
      pimult[i, j] = sqrt(pi_eq[j] / pi_eq[i]);
    }
  }
  
  matrix[61, 61] observed_mat = rep_matrix(to_row_vector(n_observed), 61);
}
 
parameters {
  real <lower = 0> kappa; 
  real <lower = 0> theta; // mu in the paper
  real <lower = 0> omega;
}

model {
  // Find mean mutation rate under neutrality
  matrix[61, 61] A = build_A(kappa, 1, pimat, pimult);
  real meanrate = 0.0 - dot_product(pi_eq, diagonal(A));
  real scale = (theta / 2.0) / meanrate;
  
  // Calculate substitution rate matrix not under neutrality
  matrix[61,61] mutmat = update_A(A, omega, pimult);
  
  // Eigenvectors/values of substitution rate matrix
  matrix[61,61] V = eigenvectors_sym(mutmat);
  vector[61] E = 1 / (1 - 2 * scale * eigenvalues_sym(mutmat));
  matrix[61,61] V_inv = diag_post_multiply(V, E);
  
  // Create m_AB for each ancestral codon
  matrix[61, 61] m_AB;
  for(i in 1:61) {
    matrix[61, 61] Va = rep_matrix(row(V, i), 61);
    m_AB[, i] = rows_dot_product(Va, V_inv);
  }
  
  // Multiply by equilibrium frequencies
  m_AB = (m_AB' * pimatinv)' * pimat;
  
  // Normalise - m_AB / m_AA
  for(i in 1:61){
    m_AB[, i] /= m_AB[i, i];
    m_AB[i, i] = 1.0e-06; // This happens in the C code but not mentioned elsewhere
    for(j in 1:61){
      if(m_AB[i, j] < 0) m_AB[i, j] = 1.0e-06;
    }
  }
  
  // Writing to columns was faster so now we transpose
  m_AB = m_AB';
  
  // Likelihood calculation
  // observed_codon ~ multinomial_dirichlet(probabilities calculated above)
  matrix[61, 61] muti = add_diag(m_AB, 1);
  matrix[61, 61] lgmuti = lgamma(muti);
  vector[61] ttheta = m_AB * ones;
  vector[61] ltheta = log(ttheta);
  vector[61] lgtheta = lgamma(ttheta);
  vector[61] poslp = lgtheta - lgamma(n_genomes + ttheta) - log(n_genomes + ttheta) + ltheta;
  
  matrix[61, 61] gam_mat = lgamma(observed_mat + muti) - lgmuti;

  // Vector of likelihood for each ancestral codon
  vector[61] likposanc = lp;
  likposanc += gam_mat * ones;
  likposanc += poslp + phi;

  real log_lik = log_sum_exp(likposanc);
  // total likelihood
  target += log_lik;
  
  // Parameter priors
  omega ~ exponential(2);
  kappa ~ exponential(2);
  theta ~ exponential(2);
  
  
}

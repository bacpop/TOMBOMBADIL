functions {
#include functions/GTR_mat.stan
#include functions/ref_likelihood.stan
}

data {
  row_vector[61] pi_eq;
  int <lower = 1> l; // gene length
  array[l, 61] int X; // codon frequencies at each locus in the gene
  array [l] int N;
  real om_prior;
  real th_prior;
} 

transformed data{
  vector[61] lp = to_vector(log(pi_eq));
  array[l] matrix[61, 61] obs_mat;
  matrix[61, 61] pimat = diag_matrix(sqrt(to_vector(pi_eq)));
  matrix[61, 61] pimatinv = diag_matrix(inv(sqrt(to_vector(pi_eq))));
  matrix[61, 61] pimult;
  vector[l] phi;
  int grainsize = 1; // for reduce_sum
  
  for(j in 1:61){
    for(i in 1:61){
      pimult[i, j] = sqrt(pi_eq[j] / pi_eq[i]);
    }
  }

  for(i in 1:l){
    phi[i] = lgamma(N[i] + 1) - sum(lgamma(to_vector(X[i, ]) + 1));
    obs_mat[i] = rep_matrix(to_row_vector(X[i, ]), 61);
  }
  
}

parameters {
  // We fit everything logged so that it can vary across
  // whole real line
  // Parameters shared across all locations in gene
  real a, b, c, d, ee, f, th;
  // Omega for each location in gene
  vector [l] om_raw;
} 

transformed parameters {
  // Transform fitted parameters into real space
  real theta = exp(th);
  vector[l] omega = exp(om_raw);
  real alpha = exp(a);
  real beta = exp(b);
  real gamma = exp(c);
  real delta = exp(d);
  real epsilon = exp(ee);
  real eta = exp(f);
  // Find mean mutation rate under neutrality 
  // (same for all locations because all pars shared across locations)
  matrix[61, 61] A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult);
  real meanrate = - dot_product(pi_eq, diagonal(A));
  real scale = (theta / 2.0) / meanrate;
}
 
model {
  // Priors
  om_raw ~ normal(om_prior, 1);
  th ~ normal(th_prior, 1);
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  c ~ normal(0, 1);
  d ~ normal(0, 1);
  ee ~ normal(0, 1);
  f ~ normal(0, 1);
  
  // Likelihood
  target += reduce_sum(rs_GTR,
  obs_mat,
  grainsize,
  A,
  pimult,
  pimatinv,
  pimat,
  scale,
  N,
  omega,
  alpha,
  beta,
  gamma,
  delta,
  epsilon,
  eta,
  lp,
  phi);
}

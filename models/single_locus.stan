functions {
#include functions/NY98.stan
}

data {
  row_vector[61] pi_eq;
  int <lower = 1> n_genomes;
  vector[61] n_observed;
  array[61] int n_observed_array;
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

  //for (i in 1:61) {
    //n_observed_array[i] = round(n_observed[i]);
  //}
}
 
parameters {
  real l_kappa; 
  real l_theta; // mu in the paper
  real l_omega;
}

transformed parameters {
  real kappa = exp(l_kappa);
  real theta = exp(l_theta);
  real omega = exp(l_omega);
}

model {
  print("kappa:");
  print(kappa);

  print("theta:");
  print(theta);

  print("omega:");
  print(omega);

  // Find mean mutation rate under neutrality
  matrix[61, 61] A = build_A(kappa, 1, pimat, pimult);

  print("A 1:");
  print(A[8, ]);

  real meanrate = 0.0 - dot_product(pi_eq, diagonal(A));
  real scale = (theta / 2.0) / meanrate;

  // Calculate substitution rate matrix not under neutrality
  matrix[61,61] mutmat = update_A(A, omega, pimult);

  print("mutmat:");
  print(mutmat);

  // Eigenvectors/values of substitution rate matrix
  // (mjr) could replace calls to eigen{vectors,values}_sym(mutmat) with single call to eigendecompose_sym
  // (mjr) can we rejig this to not need eigenvectors/values at all? I think
  // it's just taken from the original paper but maybe there's a way to make
  // this way faster
  matrix[61,61] V = eigenvectors_sym(mutmat);
  vector[61] E = 1 / (1 - 2 * scale * eigenvalues_sym(mutmat));
  // How does this calculate V_inv? Is it using that mutmat (= theta in paper?) is symmetric? (is it?)
  matrix[61,61] V_inv = diag_post_multiply(V, E);

  // Create m_AB for each ancestral codon
  matrix[61, 61] m_AB;
  for(i in 1:61) {
    matrix[61, 61] Va = rep_matrix(row(V, i), 61);
    m_AB[, i] = rows_dot_product(Va, V_inv);
  }

  print("m_AB 1:");
  print(m_AB[8, ]);

  // Multiply by equilibrium frequencies
  m_AB = (m_AB' * pimatinv)' * pimat;

  print("pimat:");
  print(pimat);

  print("pimatinv:");
  print(pimatinv);

  print("m_AB 2:");
  print(m_AB[8, ]);

  // Agrees with python version up to here!
  
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

  print("m_AB 3:");
  print(m_AB[8, ]);

  
  // Likelihood calculation
  // observed_codon ~ multinomial_dirichlet(probabilities calculated above)

  // prints "1" but surely is 1 + 1e-6 - double check somehow
  matrix[61, 61] muti = add_diag(m_AB, 1);

  print("muti:");
  print(muti);

  for (i in 1:61) {
    real row_sum = 0.0;

    for (j in 1:61) {
      row_sum += muti[i, j];
    }

    //for (j in 1:61) {
      //muti[i, j] /= row_sum;
    //}
  }

  //print("normalised muti:");
  //print(muti);

  matrix[61, 61] lgmuti = lgamma(muti);
  vector[61] ttheta = m_AB * ones;
  vector[61] ltheta = log(ttheta);
  vector[61] lgtheta = lgamma(ttheta);
  vector[61] poslp = lgtheta - lgamma(n_genomes + ttheta) - log(n_genomes + ttheta) + ltheta;
  
  matrix[61, 61] gam_mat = lgamma(observed_mat + muti) - lgmuti;

  print("gam_mat:");
  print(gam_mat);

  // Vector of likelihood for each ancestral codon
  vector[61] likposanc = lp;
  likposanc += gam_mat * ones;
  likposanc += poslp + phi;

  print("likposanc:");
  print(likposanc);

  print("likposanc - lp:");
  print(likposanc - lp);

  real log_lik = log_sum_exp(likposanc);
  
  print("log_lik:");
  print(log_lik);

  // (mjr) attempt to calculate same loglikelihood using built-in stan fns
  vector[61] log_liks_2;
  for (i in 1:61) {
    log_liks_2[i] = dirichlet_multinomial_lpmf(n_observed_array | to_vector(muti[i, ]));
  }

  print("log_liks_2:");
  print(log_liks_2);

  print("log_sum_exp(log_lik_2 + lp):");
  print(log_sum_exp(log_liks_2 + lp));

  // total likelihood
  target += log_lik;
  
  // Parameter priors
  l_omega ~ std_normal();
  l_kappa ~ std_normal();
  l_theta ~ std_normal();
}

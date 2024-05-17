// Likelihood function for map_rect
vector likelihood_func(vector shared_vec, vector diff_vec,
data array[] real x_r, data array[] int x_i){
  // Get size of shard from last element of x_i
  int n = num_elements (x_i);
  int n_sites = x_i[n];
  
  // Reconstruct parameters shared between shards
  real theta = shared_vec[1];
  real kappa = shared_vec[2];
  
  // Reconstruct parameteres varying between shards
  vector[n_sites] omega = diff_vec[1:n_sites];
  
  // Reconstruct pi matrix
  vector[61] pi_eq = to_vector(x_r[1:61]);
  vector[61] lp = to_vector(log(pi_eq));
  matrix[61, 61] pimat = diag_matrix(sqrt(to_vector(pi_eq)));
  matrix[61, 61] pimatinv = diag_matrix(inv(sqrt(to_vector(pi_eq))));
  vector[61] ones = rep_vector(1, 61);
  matrix[61, 61] pimult;
  
  for(j in 1:61){
    for(i in 1:61){
      pimult[i, j] = sqrt(pi_eq[j] / pi_eq[i]);
    }
  }
  
  // Reconstruct codon observations
  array[n_sites] matrix[61, 61] obs_mat;
  array[n_sites] vector[61] obs_vec;
  array[n_sites] int N;
  vector[n_sites] phi;
  for(site in 1:n_sites){
    int j = 1 + ((site - 1) * 61);
    int k = j + 60;
    obs_vec[site] = to_vector(x_i[j:k]);
    obs_mat[site] = rep_matrix(to_row_vector(obs_vec[site]), 61);
    N[site] = sum(x_i[j:k]);
    phi[site] = lgamma(N[site] + 1) - sum(lgamma(obs_vec[site] + 1));
  }

  // Compute parameters shared over all sites
  // Find mean mutation rate under neutrality
  matrix[61, 61] A = build_A(kappa, 1, pimat, pimult);
  real meanrate = 0.0 - dot_product(pi_eq, diagonal(A));
  real scale = (theta / 2.0) / meanrate;
  
  // Compute parameters specific to each site and add up likelihood
  matrix[61, 61] mutmat;
  matrix[61, 61] V;
  matrix[61, 61] V_inv;
  vector[61] E;
  matrix[61, 61] Va;
  matrix[61, 61] m_AB;
  matrix[61, 61] muti;
  matrix[61, 61] lgmuti;
  matrix[61, 61] gam_mat;
  vector[61] ttheta;
  vector[61] ltheta;
  vector[61] lgtheta;
  int Np;
  vector[61] poslp;
  vector[61] likposanc;
  vector[n_sites] lik;
  
  for(site in 1:n_sites) {
    // Calculate substitution rate matrix not under neutrality
    mutmat = update_A(A, omega[site], pimult);
    
    // Eigenvectors/values of substitution rate matrix
    V = eigenvectors_sym(mutmat);
    E = 1 / (1 - 2 * scale * eigenvalues_sym(mutmat));
    V_inv = diag_post_multiply(V, E);
    
    // Create m_AB for each ancestral codon
    for(i in 1:61) {
      Va = rep_matrix(row(V, i), 61);
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
    muti = add_diag(m_AB, 1);
    lgmuti = lgamma(muti);
    ttheta = m_AB * ones;
    ltheta = log(ttheta);
    lgtheta = lgamma(ttheta);
    Np = N[site];
    poslp = lgtheta - lgamma(Np + ttheta) - log(Np + ttheta) + ltheta;
  
    gam_mat = lgamma(obs_mat[site] + muti) - lgmuti;

    likposanc = lp;
    likposanc += gam_mat * ones;
    likposanc += poslp + phi[site];

    lik[site] = log_sum_exp(likposanc);
  }
  return lik;
}

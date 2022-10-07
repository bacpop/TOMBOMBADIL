real rs_GTR(array[] matrix X_slice,
int start, int end,
matrix A,
matrix pimult,
matrix pimatinv,
matrix pimat,
real scale,
array[] int N,
vector omega,
real alpha,
real beta,
real gamma,
real delta,
real epsilon,
real eta,
vector lp,
vector phi) {
  
  matrix[61, 61] mutmat, V, V_inv, m_AB, Va, obs_mat, muti, lgmuti, gam_mat;
  vector[61] E, obs_vec, ttheta, ltheta, lgtheta, poslp, likposanc;
  vector[61] ones = rep_vector(1, 61);
  real lik = 0;
  real Np;
  int z = 1;
  
  // This loops over location in the gene
  for(pos in start:end){
    // Fetches data relevant to this location
    obs_vec = to_vector(row(X_slice[z], 1));
    obs_mat = X_slice[z];
    z += 1;
    
    // Calculate substitution rate matrix not under neutrality for omega at this location
    // here you can multiply the relevant elements by new omega value and recalculate the diagonal
    mutmat = build_GTR(alpha, beta, gamma, delta, epsilon, eta, omega[pos], pimat, pimult);
    
    // Eigenvectors/values of substitution rate matrix
    V = eigenvectors_sym(mutmat);
    E = 1 / (1 - 2 * scale * eigenvalues_sym(mutmat));
    V_inv = diag_post_multiply(V, E);
    
    // Create m_AB for each ancestral codon
    // In this code we avoid looping over ancestral codons using matrix algebra
    // We calculate a matrix called m_AB where each row is the vector alpha_AB
    // from Danny's paper for each potential ancestral codon
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
      for(j in 1:61){if(m_AB[i, j] < 0) m_AB[i, j] = 1.0e-06;}
    }
    
    // Writing to columns was faster so now we transpose
    m_AB = m_AB';
    
    // Likelihood calculation
    
    // These parts just require m_AB or the count data and are therefore
    // the same for all potential ancestral codons at this
    // location in the gene
    muti = add_diag(m_AB, 1);
    lgmuti = lgamma(muti);
    ttheta = m_AB * ones;
    ltheta = log(ttheta);
    lgtheta = lgamma(ttheta);
    Np = N[pos];
    poslp = lgtheta - lgamma(Np + ttheta) - log(Np + ttheta) + ltheta;
    
    // Calculates parts different for each ancestor (in the rows of gam_mat)
    // Again we skip a loop using matrix algebra
    gam_mat = lgamma(obs_mat + muti) - lgmuti;
    
    // Sum together
    likposanc = lp;
    likposanc += gam_mat * ones; // rowSums(gam_mat)
    likposanc += poslp + phi[pos];
    lik += log_sum_exp(likposanc);
  }
  return lik;
}

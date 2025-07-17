functions {
  #include NY98.stan
  #include GTR.stan
  #include likelihoods.stan
  #include constant_likelihood.stan
}

data {
  row_vector[61] pi_eq;
  int <lower = 1> gene_length; // gene length
  array [gene_length] int n_samples;
  int <lower = 1> n_shards; // number of shards
  array[n_shards] int n_per_shard;
  int max_per_shard;
  array[n_shards] int shard_starts;
  array[n_shards] int shard_ends;
  array[n_shards, (max_per_shard * 61) + 1] int obs_array_int;
  int<lower=0, upper=1> GTR; // GTR substition matrix flag
  int<lower=0, upper=1> omega_varies; // flag for omega varying by loci or not
  int<lower=0, upper=1> omega_hierarchical; // flag for varying omega to be iid or hierarchical
  int<lower=0, upper=1> omega_mixture; // flag for mixture model
  int<lower=1> K; // number of mixture components
  vector[K] mixprop; // mixing proportions
  vector[K] mix_mean;
  vector[K] mix_var;
} 

transformed data{
  // integer array of codon observations at each location - x_i
  // This is a no. shards * no. sites in shard * 61 array
  // real array of codon frequencies
  // This is a no. shards * 61 array
  array[n_shards, 61] real obs_array_real;
  
  for(i in 1:n_shards){
    // Each shard needs the same codon frequency information repeated
    obs_array_real[i, 1:61] = to_array_1d(pi_eq);
  }
  int NY98 = GTR == 0 ? 1 : 0;
  int omega_length = omega_varies == 1 || omega_hierarchical == 1 || omega_mixture == 1 ? gene_length : 1;
  vector[K] lmp = log(mixprop);
}

parameters {
  // Parameters shared by all models
  real <lower = 0> theta;
  // Parameters for NY98 substitution model
  vector<lower=0>[NY98] kappa;
  // Parameters for GTR substitution model
  vector <lower = 0>[GTR] alpha;
  vector <lower = 0>[GTR] beta;
  vector <lower = 0>[GTR] gamma;
  vector <lower = 0>[GTR] delta;
  vector <lower = 0>[GTR] epsilon;
  vector <lower = 0>[GTR] eta;
  
  // Omega either vector of length 1 or length of alignment
  vector<lower=0>[omega_length] omega;
  // Hyperparameters for omega if hierarchical
  vector[omega_hierarchical] omega_mean;
  vector<lower =0> [omega_hierarchical] omega_var;
}

transformed parameters {
  // Assign parameters that are shared between shards to a vector
  vector[7] shard_shared_params = rep_vector(0.0, 7);
  shard_shared_params[1] = theta;
  if(GTR == 1){
    shard_shared_params[2] = alpha[1];
    shard_shared_params[3] = beta[1];
    shard_shared_params[4] = gamma[1];
    shard_shared_params[5] = delta[1];
    shard_shared_params[6] = epsilon[1];
    shard_shared_params[7] = eta[1];
  } else {
    shard_shared_params[2] = kappa[1];
  }
  
  // Pack omega into a vector to give to shards depending on its size
  vector[gene_length] omega_vec;
  if(omega_varies == 1 || omega_hierarchical == 1 || omega_mixture == 1){
    omega_vec = omega;
  } else {
    omega_vec = rep_vector(omega[1], gene_length);
  }
  
  // Assign parameters that are different between shards to an array of vectors
  array[max_per_shard] real temp_vec;
  array[n_shards] vector[max_per_shard] shard_diff_params;
  for (i in 1:n_shards){
    temp_vec = append_array(to_array_1d(omega_vec[shard_starts[i]:shard_ends[i]]),
    rep_array(0, max_per_shard - n_per_shard[i]));
    shard_diff_params[i] = to_vector(temp_vec);
  }
  array[gene_length] vector[K] mix_ll;
  if(omega_mixture ==1) {
    for(i in 1:gene_length){
      for(j in 1:K){
        mix_ll[i, j] = lmp[j] + normal_lpdf(omega[i] | mix_mean[j], mix_var[j]);
      }
    }
  }
}

model {
  // Different Dirichlet-Multinomial likelihoods and paramtere priors depending on substitution model used
  if(GTR == 1){
    if(omega_varies == 1 || omega_hierarchical == 1 || omega_mixture == 1){
      target += sum(map_rect(likelihood_GTR, shard_shared_params, shard_diff_params,
      obs_array_real, obs_array_int));
    } else {
      target += sum(map_rect(likelihood_constant_GTR, shard_shared_params, shard_diff_params,
      obs_array_real, obs_array_int));
    }
    
    alpha ~ std_normal() T[0, ];
    beta ~ std_normal() T[0, ];
    gamma ~ std_normal() T[0, ];
    delta ~ std_normal() T[0, ];
    epsilon ~ std_normal() T[0, ];
    eta ~ std_normal() T[0, ];
  } else {
    if(omega_varies == 1 || omega_hierarchical == 1 || omega_mixture == 1){
      target += sum(map_rect(likelihood_NY98, shard_shared_params, shard_diff_params,
      obs_array_real, obs_array_int));
    } else {
      target += sum(map_rect(likelihood_constant_NY98, shard_shared_params, shard_diff_params,
      obs_array_real, obs_array_int));
    }
    
    kappa ~ std_normal() T[0, ];
  }
  
  // Parameter shared by all models
  theta ~ std_normal() T[0, ];
  
  // Differing omega priors depending on model structure
  if(omega_hierarchical == 1){
    omega_mean ~ normal(log(0.5), 1);
    omega_var ~ std_normal() T[0,];
    for(i in 1:gene_length){
      omega[i] ~ lognormal(omega_mean, omega_var);
    }
  } else if(omega_mixture == 1) {
    for(i in 1:gene_length){
      target += log_sum_exp(mix_ll[i]);
    }
  } else {
    omega ~ lognormal(log(0.5), 1);
  }
}

generated quantities {
  vector[gene_length] pr;
  if(omega_mixture == 1){
    for(i in 1:gene_length){
      pr[i] = categorical_logit_rng(mix_ll[i]);
    }
  }
}

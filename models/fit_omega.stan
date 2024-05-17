functions {
#include functions/NY98.stan
#include functions/likelihoods.stan
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
  // real <lower = 0> kappa;
  // real <lower = 0> omega;
  // real <lower = 0> theta;
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
  
}

parameters {
  real <lower = 0> kappa;
  real <lower = 0> theta;

  // How you parameterise omega will depend
  real <lower = 0> omega;
}

transformed parameters {
  // Assign parameters that are shared between shards to a vector
  vector [2] shard_shared_params;
  shard_shared_params[1] = theta;
  shard_shared_params[2] = kappa;
  vector[gene_length] omega_vec = rep_vector(omega, gene_length);
  
  // Assign parameters that are different between shards to an array of vectors
  array[max_per_shard] real temp_vec;
  array[n_shards] vector[max_per_shard] shard_diff_params;
  for (i in 1:n_shards){
    temp_vec = append_array(to_array_1d(omega_vec[shard_starts[i]:shard_ends[i]]),
    rep_array(0, max_per_shard - n_per_shard[i]));
    // temp_vec = rep_vector(0, max_per_shard);
    shard_diff_params[i] = to_vector(temp_vec);
  }
}

model {
  target += sum(map_rect(likelihood_func, shard_shared_params, shard_diff_params,
  obs_array_real, obs_array_int));

  omega ~ normal(0, 1);
  kappa ~ normal(0, 1);
  theta ~ normal(0, 1);
}

generated quantities {
  vector[gene_length] log_lik = map_rect(likelihood_func, shard_shared_params, shard_diff_params,
  obs_array_real, obs_array_int);
}

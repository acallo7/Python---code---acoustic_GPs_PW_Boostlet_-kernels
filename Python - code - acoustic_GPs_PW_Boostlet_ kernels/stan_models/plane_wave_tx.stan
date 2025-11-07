functions {
  matrix cosine_kernel(array[] vector y, array[] vector t, vector sigma_w, array[] real k, array[] real ome) {
    int N = size(y);
    int D = num_elements(y[1]);
    int Ln = num_elements(k);

    matrix[N, N] K;

    for (i in 1:N) {
      for (j in 1:N) {
        if (i == j) {
          K[i, j] = sum(sigma_w) ^ 2;
        } else if (i < j && j <= N) {
          K[i, j] = 0;
          for (d in 1:D) {
            for (n in 1:Ln) {
              for (m in 1:Ln) {
                K[i, j] += (square(sigma_w[d])) * cos((ome[n] * (-t[j][d])) + (k[m] * y[i][d]));
              }
            }
          }
        }
        if (i <= N && j <= N) {
          K[j, i] = K[i, j];
        }
      }
    }
    return K;
  }

  matrix sine_kernel(array[] vector y, array[] vector t, vector sigma_w, array[] real k, array[] real ome) {
    int N = size(y);
    int D = num_elements(y[1]);
    int Ln = num_elements(k);

    matrix[N, N] K;

    for (i in 1:N) {
      for (j in 1:N) {
        if (i == j) {
          K[i, j] = 0;
        } else if (i < j && j <= N) {
          K[i, j] = 0;
          for (d in 1:D) {
            for (n in 1:Ln) {
              for (m in 1:Ln) {
                K[i, j] += (square(sigma_w[d])) * sin((ome[n] * (-t[j][d])) + (k[m] * y[i][d]));
              }
            }
          }
        }
        if (i <= N && j <= N) {
          K[j, i] = K[i, j];
        }
      }
    }
    return K;
  }

  matrix qr_decompose(matrix K) {
    int m = rows(K);
    int n = cols(K);
    matrix[m, m] Q;
    matrix[m, n] R;

    // Perform QR decomposition
    Q = qr_thin_Q(K);
    R = qr_thin_R(K);

    // Return the Q matrix (orthogonal)
    return R;
  }
}

data {
  int<lower=0> N_meas; // Number of measurement positions.
  int<lower=0> N_time; // Time discretisation
  int<lower=0> N_reps; // Repetitions.
  int<lower=0> N_f; // Number of frequency
  int<lower=0> D; // Number of basis elements
  array[N_meas] vector[2 * N_time] h_stan; // Measured Pressure at receiver. Real and imag stacked.
  array[N_meas] vector[1] y; // Spatial positions
  array[N_time] vector[1] t; // Spatial positions
  array[D] vector[2] wave_directions; // possible plane wave directions
  matrix[2 * N_meas, 2 * N_time] Sigma_stan;
  real a;
  real b_log_std;
  real b_log_mean;
  real delta;
  vector[N_f] k;
  vector[N_f] ome;
}

transformed data {
  vector[2 * N_meas] mu = rep_vector(0, 2 * N_meas);
  array[N_meas] vector[D] y_projected;
  array[N_time] vector[D] t_projected;
  array[N_f] real k_array;
  array[N_f] real ome_array;

  for (i in 1:N_meas) {
    for (d in 1:D) {
      y_projected[i, d] = y[i][1] * wave_directions[d][1];
    }
  }

  for (j in 1:N_time) {
    for (d in 1:D) {
      t_projected[j, d] = t[j][1] * wave_directions[d][1];
    }
  }

  for (i in 1:N_f) {
    k_array[i] = k[i];
  }

  for (i in 1:N_f) {
    ome_array[i] = ome[i];
  }
}

parameters {
  vector<lower=0>[D] sigma_w;
  real<lower=0> b_log;
}

transformed parameters {
  matrix[N_meas, N_time] K_self;
  matrix[N_meas, N_time] K_realimag;
  matrix[2 * N_meas, 2 * N_time] K;
  matrix[2*N_meas, 2*N_time] L_K;
  real<lower=0> b = pow(10, -b_log);

  K_self = cosine_kernel(y_projected, t_projected, to_vector(sigma_w), k_array, ome_array);
  K_realimag = sine_kernel(y_projected, t_projected, to_vector(sigma_w), k_array, ome_array);

  K = append_row(append_col(K_self, K_realimag), append_col(K_realimag, K_self)) + Sigma_stan + delta;
  
  L_K = qr_decompose(K);
}

model {
  to_vector(sigma_w) ~ inv_gamma(a, b);
  b_log ~ normal(b_log_mean, b_log_std);
  for (nrep in 1:N_reps){
      h_stan[nrep] ~ multi_normal_cholesky(mu, L_K);
      }
}

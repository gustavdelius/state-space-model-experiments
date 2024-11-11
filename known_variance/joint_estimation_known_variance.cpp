// joint_estimation.cpp
#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(y);       // Observations
  
  // Parameters to estimate
  PARAMETER(a);         // State transition parameter
  PARAMETER_VECTOR(x);  // State variables
  
  // Hyperparameters (known variances)
  DATA_SCALAR(q);       // Process noise variance
  DATA_SCALAR(r);       // Measurement noise variance
  
  int n = y.size();
  Type nll = 0.0;  // Negative log-likelihood
  
  // State equation likelihood
  for(int t = 1; t < n; t++){
    Type mean_xt = a * x[t - 1];
    nll -= dnorm(x[t], mean_xt, sqrt(q), true);
  }
  
  // Observation equation likelihood
  for(int t = 0; t < n; t++){
    nll -= dnorm(y[t], x[t], sqrt(r), true);
  }
  
  // Prior for initial state
  nll -= dnorm(x[0], Type(0), sqrt(q / (1 - a * a)), true);
  
  return nll;
}

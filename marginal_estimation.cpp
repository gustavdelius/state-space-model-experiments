// marginal_estimation.cpp
#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(y);       // Observations
  
  // Parameters to estimate
  PARAMETER(a);         // State transition parameter
  PARAMETER(log_q);             // Log of process noise variance
  PARAMETER(log_r);             // Log of measurement noise variance
  
  Type q = exp(log_q);          // Process noise variance
  Type r = exp(log_r);          // Measurement noise variance
  
  int n = y.size();
  Type nll = 0.0;  // Negative log-likelihood
  
  // Define state-space model matrices
  // Since the model is linear and Gaussian, we can use the Kalman filter
  
  // Initialize variables
  Type x_pred = 0.0;             // Predicted state
  Type P_pred = q / (1 - a * a); // Predicted covariance
  Type x_filt;                   // Filtered state
  Type P_filt;                   // Filtered covariance
  
  for(int t = 0; t < n; t++){
    // Observation prediction
    Type y_pred = x_pred;
    Type S = P_pred + r;
    
    // Update step
    Type K = P_pred / S;  // Kalman gain
    x_filt = x_pred + K * (y[t] - y_pred);
    P_filt = (1 - K) * P_pred;
    
    // Accumulate negative log-likelihood
    nll += 0.5 * (log(2 * M_PI * S) + pow(y[t] - y_pred, 2) / S);
    
    // Predict next state
    x_pred = a * x_filt;
    P_pred = a * a * P_filt + q;
  }
  
  return nll;
}

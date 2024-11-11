# Load necessary libraries
library(TMB)
library(MASS)  # For mvrnorm

# Create synthetic data ----

# Set seed for reproducibility
set.seed(42)

# Model parameters
T <- 500                  # Number of time points
a_true <- 0.8            # True parameter value
q <- 1                   # Process noise variance
r <- 1                   # Measurement noise variance

# Initialize state and observation vectors
x_true <- numeric(T)
y <- numeric(T)

# Initial state
x_true[1] <- rnorm(1, mean = 0, sd = sqrt(q))

# Simulate states and observations
for (t in 2:T) {
  x_true[t] <- a_true * x_true[t - 1] + rnorm(1, mean = 0, sd = sqrt(q))
}
y <- x_true + rnorm(T, mean = 0, sd = sqrt(r))

# Plot the true states and observations
plot(1:T, x_true, type = 'l', col = 'blue', lwd = 2, ylab = 'Value', xlab = 'Time')
points(1:T, y, col = 'red', pch = 16)
legend('topright', legend = c('True State', 'Observations'), col = c('blue', 'red'), lty = c(1, NA), pch = c(NA, 16))



# Compile and Run the Joint Estimation Model ----

# Compile the model
compile("known_variance/joint_estimation_known_variance.cpp")
dyn.load(dynlib("known_variance/joint_estimation_known_variance"))

# Prepare data and parameters for TMB
data_joint <- list(y = y, q = q, r = r)
parameters_joint <- list(a = 0.5, x = rep(0, T))

# Create TMB object
obj_joint <- MakeADFun(data = data_joint, parameters = parameters_joint, 
                       DLL = "joint_estimation_known_variance")

# Optimize the model
opt_joint <- nlminb(start = obj_joint$par, objective = obj_joint$fn, 
                    gradient = obj_joint$gr)

# Report results
report_joint <- sdreport(obj_joint)
summary_joint <- summary(report_joint)
print(summary_joint)

cat("Estimated a (Marginal Estimation):", report_joint$par.fixed["a"], "\n")

# Compile and run the marginal estimation model ----

# Compile the model
compile("known_variance/marginal_estimation_known_variance.cpp")
dyn.load(dynlib("known_variance/marginal_estimation_known_variance"))

# Prepare data and parameters for TMB
data_marginal <- list(y = y, q = q, r = r)
parameters_marginal <- list(a = 0.5)

# Create TMB object
obj_marginal <- MakeADFun(data = data_marginal, 
                          parameters = parameters_marginal, 
                          DLL = "marginal_estimation_known_variance")

# Optimize the model
opt_marginal <- nlminb(start = obj_marginal$par, objective = obj_marginal$fn, 
                       gradient = obj_marginal$gr)

# Report results
report_marginal <- sdreport(obj_marginal)
summary_marginal <- summary(report_marginal)
print(summary_marginal)

# Compare results ----
cat("Marginal Estimation Results:\n")
print(summary_joint)
cat("Estimated a (Marginal Estimation):", report_marginal$par.fixed["a"], "\n")

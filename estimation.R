# Load necessary libraries
library(TMB)

# Create synthetic data ----

# Set seed for reproducibility
set.seed(42)

# Model parameters
T <- 500                  # Number of time points
a_true <- 0.8            # True parameter value
q_true <- 1              # True process noise variance
r_true <- 1              # True measurement noise variance

# Initialize state and observation vectors
x_true <- numeric(T)
y <- numeric(T)

# Initial state (from stationary distribution)
x_true[1] <- rnorm(1, mean = 0, sd = sqrt(q_true / (1 - a_true^2)))

# Simulate states and observations
for (t in 2:T) {
  x_true[t] <- a_true * x_true[t - 1] + rnorm(1, mean = 0, sd = sqrt(q_true))
}
y <- x_true + rnorm(T, mean = 0, sd = sqrt(r_true))

# Plot the true states and observations
plot(1:T, x_true, type = 'l', col = 'blue', lwd = 2, ylab = 'Value', xlab = 'Time')
points(1:T, y, col = 'red', pch = 16)
legend('topright', legend = c('True State', 'Observations'), col = c('blue', 'red'), lty = c(1, NA), pch = c(NA, 16))



# Compile and Run the Joint Estimation Model ----

# Compile the model
compile("joint_estimation.cpp")
dyn.load(dynlib("joint_estimation"))

# Prepare data and parameters for TMB
data_joint <- list(y = y)
parameters_joint <- list(a = 0.5, log_q = log(1), log_r = log(1), x = rep(0, T))

# Create TMB object
obj_joint <- MakeADFun(data = data_joint, parameters = parameters_joint, 
                       DLL = "joint_estimation")

# Optimize the model
opt_joint <- nlminb(start = obj_joint$par, objective = obj_joint$fn, 
                    gradient = obj_joint$gr)

# Report results
report_joint <- sdreport(obj_joint)
summary_joint <- summary(report_joint)
# Print estimates
estimates_joint <- summary_joint[rownames(summary_joint) %in% c("a", "log_q", "log_r"), ]
estimates_joint_exp <- estimates_joint
estimates_joint_exp[rownames(estimates_joint) != "a", 1] <- exp(estimates_joint[rownames(estimates_joint) != "a", 1])

cat("Joint Estimation Results (Parameters):\n")
print(estimates_joint_exp)

# Treat x as random effect ----
# Create TMB object
obj_laplace <- MakeADFun(data = data_joint, parameters = parameters_joint, 
                       random = "x", DLL = "joint_estimation")

# Optimize the model
obj_laplace <- nlminb(start = obj_laplace$par, objective = obj_laplace$fn, 
                    gradient = obj_laplace$gr)

# Report results
report_laplace <- sdreport(obj_laplace)
summary_laplace <- summary(report_laplace)
# Print estimates
estimates_laplace <- summary_joint[rownames(summary_laplace) %in% 
                                     c("a", "log_q", "log_r"), ]
estimates_laplace_exp <- estimates_laplace
estimates_laplace_exp[rownames(estimates_laplace) != "a", 1] <- 
  exp(estimates_laplace[rownames(estimates_laplace) != "a", 1])

cat("Laplace Estimation Results (Parameters):\n")
print(estimates_laplace_exp)
# Compile and run the marginal estimation model ----

# Compile the model
compile("marginal_estimation.cpp")
dyn.load(dynlib("marginal_estimation"))

# Prepare data and parameters for TMB
data_marginal <- list(y = y)
parameters_marginal <- list(a = 0.5, log_q = log(1), log_r = log(1))

# Create TMB object
obj_marginal <- MakeADFun(data = data_marginal, 
                          parameters = parameters_marginal, 
                          DLL = "marginal_estimation")

# Optimize the model
opt_marginal <- nlminb(start = obj_marginal$par, objective = obj_marginal$fn, 
                       gradient = obj_marginal$gr)

# Report results
report_marginal <- sdreport(obj_marginal)
summary_marginal <- summary(report_marginal)
print(summary_marginal)

# Compare results ----
# Extract estimates from joint estimation
estimates_joint <- summary_joint[rownames(summary_joint) %in% c("a", "log_q", "log_r"), ]
estimates_joint_exp <- estimates_joint
estimates_joint_exp[rownames(estimates_joint) != "a", 1] <- exp(estimates_joint[rownames(estimates_joint) != "a", 1])

# Extract estimates from marginal estimation
estimates_marginal <- summary_marginal[rownames(summary_marginal) %in% c("a", "log_q", "log_r"), ]
estimates_marginal_exp <- estimates_marginal
estimates_marginal_exp[rownames(estimates_marginal) != "a", 1] <- exp(estimates_marginal[rownames(estimates_marginal) != "a", 1])

# True values
true_values <- c(a = a_true, q = q_true, r = r_true)

# Create a comparison table
comparison <- data.frame(
  Parameter = c("a", "q", "r"),
  True_Value = c(a_true, q_true, r_true),
  Joint_Estimate = c(
    estimates_joint_exp["a", 1],
    estimates_joint_exp["log_q", 1],
    estimates_joint_exp["log_r", 1]
  ),
  Marginal_Estimate = c(
    estimates_marginal_exp["a", 1],
    estimates_marginal_exp["log_q", 1],
    estimates_marginal_exp["log_r", 1]
  )
)

print(comparison)

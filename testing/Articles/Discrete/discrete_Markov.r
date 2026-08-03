#nolint start: line_length_linter

# this file is for testing consistency of random jump forests on Markov data
# with discrete covariates

# import helper functions and packages
source("testing/Articles/Discrete/Helpers.r")

# defining the data-generating process and simulating data
#--------------------------------------------------------------------------------

{
# ChatGPT suggests the following parameter values for the intensities

# baseline intercepts
alpha0_02 <- 2e-05
alpha0_12 <- 1e-03
alpha0_10 <- 1e-02

# baseline X1-values
alpha1_02 <- 1e-05
alpha1_12 <- 2e-04
alpha1_10 <- 0

# baseline offsets
beta0_01 <- -8.790565
beta0_02 <- -11.227431
beta0_12 <- -7.471461
beta0_10 <- 0.813713

# X1 coefficient (maybe not include this and the alpha1)
beta1_01 <- -0.1
beta1_02 <- 0.35
beta1_12 <- 0.22
beta1_10 <- -0.05

# the intercept for time
gamma1_01 <- 0.04367222
gamma1_02 <- 0.099600
gamma1_12 <- 0.067450
gamma1_10 <- -0.045250

# the X2 coefficient
beta2_01 <- -1.444444e-03
beta2_02 <- -1e-03
beta2_12 <- -5e-04
beta2_10 <- 7e-04

# the X3 coefficient
beta3_01 <- -1.416667e-03
beta3_02 <- -7e-03
beta3_12 <- -4e-04
beta3_10 <- 5e-04

# misc. coefficients
gamma2_01 <- 1.183597e-03
beta4_01 <- -1.111111e-05
beta5_01 <- 2.777778e-06
gamma3_01 <- -9.255633e-06
beta6_01 <- 1.234568e-07

# initial age
x <- 30

# define intensities
mu01 <- function(t, x1, x2, x3) {
    exp(beta0_01 + beta1_01 * x1 + (gamma1_01 + beta2_01 * x2 + beta3_01 * x3) * (t + x)
        + (gamma2_01 + beta4_01 * x2 + beta5_01 * x3) * (t + x)^2
        + (gamma3_01 + beta6_01 * x2) * (t + x)^3)
}
mu02 <- function(t, x1, x2, x3) {
    alpha0_02 + alpha1_02 * x1 + exp(beta0_02 + beta1_02 * x1 + (gamma1_02 + beta2_02 * x2 + beta2_02 * x3) * (t + x))
}
mu12 <- function(t, x1, x2, x3) {
    alpha0_12 + alpha1_12 * x1 + exp(beta0_12 + beta1_12 * x1 + (gamma1_12 + beta2_12 * x2 + beta2_12 * x3) * (t + x))
}
mu10 <- function(t, x1, x2, x3) {
    alpha0_10 + alpha1_10 * x1 + exp(beta0_10 + beta1_10 * x1 + (gamma1_10 + beta2_10 * x2 + beta2_10 * x3) * (t + x))
}

Lambda <- function(t, x1, x2, x3) {
    A <- matrix(c(0, mu01(t, x1, x2, x3), mu02(t, x1, x2, x3), mu10(t, x1, x2, x3), 0, mu12(t, x1, x2, x3), 0, 0, 0),
    nrow = 3, byrow = TRUE)
    diag(A) <- -rowSums(A)
    A
}
}

# now simulate the paths
{
set.seed(2026)
n <- 1000
# signal
X1 <- rbinom(n, 1, 1/2)                                                   # sex, 0: female, 1: male
X2 <- sample(1:4, n, replace = TRUE, prob = c(0.2, 0.4, 0.3, 0.1))        # education level
X3 <- sample(1:5, n, replace = TRUE, prob = c(0.15, 0.3, 0.4, 0.1, 0.05)) # wage level

# noise
X4 <- rnorm(n)
X5 <- rnorm(n)
X6 <- rnorm(n)

R <- runif(n, 40, 120)

sim <- list()
for(i in 1:n){
  # total rate out
  rates <- function(j, t, u){-Lambda(t, X1[i], X2[i], X3[i])[j,j]}
  # jump probabilities upon a jump happening
  mark_dist <- function(j, t, u){
    row <- Lambda(t, X1[i], X2[i], X3[i])[j,]
    rate_out <- -row[j]  # the diagonal is the total intensity out of the state
    row[j] <- 0
    return(row/rate_out)
  }
  sim[[i]] <- sim_path(1L, rates = rates, dists = mark_dist, tn = R[i])
}
}

# censoring rate
sum(R == unlist(lapply(sim, FUN = function(z){tail(z$times, 1)}))) / n  # 0.179

# fit a jump forest
#--------------------------------------------------------------------------------

test_data <- data.frame(X1 = X1, X2 = X2, X3 = X3, X4 = X4, X5 = X5, X6 = X6)
fitted_forest <- jfforest(MM ~ ., data = sim, feature_data = test_data)
print_forest(fitted_forest)

# do hyperparameter tuning here
# (min_node_size in c(10, 20, 50, 100, 200) for each splitting rule, possibly excluding FH)
# then take the best fitting forest and compare to the rest of the models below
# want to compare in regards to error

# VIMP (should do several runs, it seems that JF has a hard time distinguishing noise from signal),
# but the model is also quite complicated, maybe simplify before continuing?
unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "brier", seed = 2026)$vimp)
unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "kl", seed = 2026)$vimp)
unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "spherical", seed = 2026)$vimp)
unlist(jfforest.vimp(fitted_forest, method = "random", loss = "brier", seed = 2026)$vimp)
unlist(jfforest.vimp(fitted_forest, method = "random", loss = "kl", seed = 2026)$vimp)
unlist(jfforest.vimp(fitted_forest, method = "random", loss = "spherical", seed = 2026)$vimp)

# fit Cox proportional hazard model (how exactly?)
#--------------------------------------------------------------------------------

# fit a Poisson regression model (how exactly is this done with covariates?)
#--------------------------------------------------------------------------------

# fit the conditional Aalen-Johansen estimator
#--------------------------------------------------------------------------------
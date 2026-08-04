#nolint start: line_length_linter

# this file is for testing consistency of random jump forests on Markov data
# with discrete covariates

# import helper functions and packages
source("testing/Articles/Discrete/Helpers.r")

# defining the data-generating process and simulating data
#--------------------------------------------------------------------------------

{
# Mortality parameters are calibrated so that, after averaging over the X2/X3
# distribution below, residual life expectancy at age 30 is approximately
# 50.9 years for men (X1 = 1) and 54.2 years for women (X1 = 0). For comparison,
# Statistics Denmark table HISB8 reports 50.92 and 54.50 years in 2024:2025.
# In particular, beta3_02 = -0.007 previously made a four-level increase in X3
# multiply healthy-state mortality at age 100 by exp(-0.007 * 4 * 100) = 0.061.
# The revised value gives a much more moderate multiplier of exp(-0.1) = 0.905.

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
beta0_02 <- -11.5
beta0_12 <- -7.5
beta0_10 <- 0.813713

# X1 coefficient (maybe not include this and the alpha1)
beta1_01 <- -0.1
beta1_02 <- 0.40
beta1_12 <- 0.30
beta1_10 <- -0.05

# the intercept for time
gamma1_01 <- 0.04367222
gamma1_02 <- 0.100000
gamma1_12 <- 0.065000
gamma1_10 <- -0.045250

# the X2 coefficient
beta2_01 <- -1.444444e-03
beta2_02 <- -3e-04
beta2_12 <- -3e-04
beta2_10 <- 7e-04

# the X3 coefficient
beta3_01 <- -1.416667e-03
beta3_02 <- -2.5e-04
beta3_12 <- -2.5e-04
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
mu01 <- function(t, X) {
    exp(beta0_01 + beta1_01 * X[1] + (gamma1_01 + beta2_01 * X[2] + beta3_01 * X[3]) * (t + x)
        + (gamma2_01 + beta4_01 * X[2] + beta5_01 * X[3]) * (t + x)^2
        + (gamma3_01 + beta6_01 * X[2]) * (t + x)^3)
}
mu02 <- function(t, X) {
    alpha0_02 + alpha1_02 * X[1] + exp(beta0_02 + beta1_02 * X[1] + (gamma1_02 + beta2_02 * X[2] + beta3_02 * X[3]) * (t + x))
}
mu12 <- function(t, X) {
    alpha0_12 + alpha1_12 * X[1] + exp(beta0_12 + beta1_12 * X[1] + (gamma1_12 + beta2_12 * X[2] + beta3_12 * X[3]) * (t + x))
}
mu10 <- function(t, X) {
    alpha0_10 + alpha1_10 * X[1] + exp(beta0_10 + beta1_10 * X[1] + (gamma1_10 + beta2_10 * X[2] + beta3_10 * X[3]) * (t + x))
}

Lambda <- function(t, X) {
    A <- matrix(c(0, mu01(t, X), mu02(t, X), mu10(t, X), 0, mu12(t, X), 0, 0, 0),
    nrow = 3, byrow = TRUE)
    diag(A) <- -rowSums(A)
    A
}
}

# now simulate the paths
{
tic()
set.seed(2026)
n <- 50000
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
  rates <- function(j, t, u){-Lambda(t, c(X1[i], X2[i], X3[i]))[j,j]}
  # jump probabilities upon a jump happening
  mark_dist <- function(j, t, u){
    row <- Lambda(t, c(X1[i], X2[i], X3[i]))[j,]
    rate_out <- -row[j]  # the diagonal is the total intensity out of the state
    row[j] <- 0
    return(row/rate_out)
  }
  sim[[i]] <- sim_path(1L, rates = rates, dists = mark_dist, tn = R[i])
  if (i %% 1000 == 0) {
    cat("Finished iteration", i, "/", n, "\n")
  }
}
test_data <- data.frame(X1 = X1, X2 = X2, X3 = X3, X4 = X4, X5 = X5, X6 = X6)
toc()   # takes about 234 seconds for n = 50,000
}

# empirical censoring rate in the current simulation
sum(R == unlist(lapply(sim, FUN = function(z){tail(z$times, 1)}))) / n

# For death time T, survival is the probability of occupying either living
# state: S(t | X) = P(T > t | X) = p0(t | X) + p1(t | X). Consequently,
# E(T | X) is the integral of S(t | X). This function solves p'(t) = p(t) Lambda(t, X)
# with fourth-order Runge--Kutta steps and integrates survival by the trapezoid rule.
average_residual_lifetime <- function(X, initial = c(1, 0, 0), tail_time = 120 - x, max_time = 200, step = 0.02, tolerance = 1e-10) {
    if (length(X) < 3 || length(initial) != 3 || abs(sum(initial) - 1) > 1e-10) {
        stop("X must contain X1, X2 and X3, and initial must be a probability vector of length 3")
    }

    derivative <- function(time, probabilities) {
        drop(probabilities %*% Lambda(time, X))
    }

    time <- 0
    probabilities <- initial
    survival <- sum(probabilities[1:2])
    mean_lifetime <- 0
    tail_probability <- if (tail_time == 0) survival else NA_real_

    while (time < max_time && (time < tail_time || survival > tolerance)) {
        h <- min(step, max_time - time)

        # Limit the step when late-age exit rates become large.
        maximum_exit_rate <- max(-diag(Lambda(time, X)),
                                 -diag(Lambda(time + h, X)))
        if (is.finite(maximum_exit_rate) && maximum_exit_rate > 0) {
            h <- min(h, 0.1 / maximum_exit_rate)
        }

        k1 <- derivative(time, probabilities)
        k2 <- derivative(time + h / 2, probabilities + h * k1 / 2)
        k3 <- derivative(time + h / 2, probabilities + h * k2 / 2)
        k4 <- derivative(time + h, probabilities + h * k3)
        next_probabilities <- probabilities + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        next_survival <- sum(next_probabilities[1:2])

        if (!is.finite(next_survival)) {
            stop("Non-finite survival probability; reduce step or max_time")
        }
        if (is.na(tail_probability) && time + h >= tail_time) {
            fraction <- (tail_time - time) / h
            tail_probability <- survival + fraction * (next_survival - survival)
        }

        mean_lifetime <- mean_lifetime + h * (survival + next_survival) / 2
        time <- time + h
        probabilities <- next_probabilities
        survival <- next_survival
    }

    if (survival > tolerance) {
        warning("The lifetime integral was truncated before the survival tail became negligible")
    }

    c(mean.residual.lifetime = mean_lifetime,
      expected.age.at.death = x + mean_lifetime,
      probability.age.above.120 = tail_probability,
      survival.at.integration.end = survival)
}

# Inspect one individual and all 40 signal-covariate profiles.
average_residual_lifetime(c(X1 = 0, X2 = 2, X3 = 3))
lifetime_profiles <- expand.grid(X1 = 0:1, X2 = 1:4, X3 = 1:5)
lifetime_diagnostics <- t(vapply(seq_len(nrow(lifetime_profiles)), function(i) {
    average_residual_lifetime(as.numeric(lifetime_profiles[i, ]))
}, numeric(4)))
lifetime_diagnostics <- cbind(lifetime_profiles, lifetime_diagnostics)

range(lifetime_diagnostics$mean.residual.lifetime)       # approximately 49.7--56.0 years
max(lifetime_diagnostics$probability.age.above.120)      # approximately 2e-6

profile_weights <- c(0.2, 0.4, 0.3, 0.1)[lifetime_profiles$X2] * c(0.15, 0.3, 0.4, 0.1, 0.05)[lifetime_profiles$X3]
weighted_residual_lifetime <- vapply(0:1, function(sex) {
    use <- lifetime_profiles$X1 == sex
    weighted.mean(lifetime_diagnostics$mean.residual.lifetime[use], profile_weights[use])
}, numeric(1))
names(weighted_residual_lifetime) <- c("women", "men")
weighted_residual_lifetime                              # approximately 54.2 and 50.9 years


# fit a jump forest (hyperparameter tuning)
#--------------------------------------------------------------------------------

num_obs <- 1000
#fitted_forest <- jfforest(MM ~ ., data = sim[1:100], feature_data = test_data[1:100,])
#print_forest(fitted_forest)

min_node_sizes <- c(10, 20, 50, 100, 200)
split_rules <- c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
ibs <- matrix(rep(0, 25), nrow = 5)
kl <- matrix(rep(0, 25), nrow = 5)
spherical <- matrix(rep(0, 25), nrow = 5)
# row: spliting rule, column: minimal node size

for (i in 1:5) {
    for (j in 1:5) {
        fitted_forest <- jfforest(MM ~ ., data = sim[1:num_obs], feature_data = test_data[1:num_obs,],
                                  splitrule = split_rules[i], min_node_size = min_node_sizes[j], seed = 2026)
        ibs[i, j] <- fitted_forest$ibs.normalised
        kl[i, j] <- fitted_forest$ikl.normalised
        spherical[i, j] <- fitted_forest$is.normalised

        # free memory along the way
        rm('fitted_forest')
        gc()

        cat("Finished fitting forest", (i - 1) * 5 + j, "out of 25\n")
    }
}

min(ibs)
ibs # best is logrank with min_node_size = 100

min(kl)
kl  # best is logrank with min_node_size = 200

min(spherical)
spherical   # best is petoprentice with min_node_size = 20

# for plots later
write.table(ibs, file = "testing/Articles/Discrete/Data/tuning_ibs.txt", sep = "\t", row.names = FALSE)
write.table(kl, file = "testing/Articles/Discrete/Data/tuning_kl.txt", sep = "\t", row.names = FALSE)
write.table(spherical, file = "testing/Articles/Discrete/Data/tuning_spherical.txt", sep = "\t", row.names = FALSE)

# now make hyperparameter tuning plots
ibs <- read.table("testing/Articles/Discrete/Data/tuning_ibs.txt", header = TRUE)
colnames(ibs) <- as.factor(min_node_sizes)
ikl <- read.table("testing/Articles/Discrete/Data/tuning_kl.txt", header = TRUE)
colnames(ikl) <- as.factor(min_node_sizes)
spherical <- read.table("testing/Articles/Discrete/Data/tuning_spherical.txt", header = TRUE)
colnames(spherical) <- as.factor(min_node_sizes)

# plot function for the normalised errors
plot_normalised_error <- function(error, ylab, file = NULL, width = 8, height = 5, resolution = 300, legend.pos = "topright") {
    saving <- !is.null(file)
    if (saving) {
        dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
        png(file, width = width, height = height, units = "in", res = resolution)
        on.exit(dev.off())
    }

    values <- as.matrix(error)
    rule_colours <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")

    matplot(min_node_sizes, t(values), type = "o",
            col = rule_colours, lty = 1, lwd = 2, pch = seq_along(split_rules),
            xaxt = "n", xlab = "Minimum node size", ylab = ylab)
    axis(1, at = min_node_sizes, labels = colnames(error), gap.axis = -1)
    legend(legend.pos, legend = split_rules, col = rule_colours,
           lty = 1, lwd = 2, pch = seq_along(split_rules), bty = "n")

    invisible(file)
}

plot_normalised_error(error = ibs, ylab = "Normalised IBS", width = 5, height = 3)

plot_normalised_error(error = ibs, ylab = "Normalised IBS", file = "testing/Articles/Discrete/Plots/tuning_ibs.png", width = 6, height = 6)
plot_normalised_error(error = ikl, ylab = "Normalised KL", file = "testing/Articles/Discrete/Plots/tuning_ikl.png", width = 6, height = 6)
plot_normalised_error(error = spherical, ylab = "Normalised Spherical error", file = "testing/Articles/Discrete/Plots/tuning_is.png", width = 6, height = 6, legend.pos = "bottomright")

# we choose to go with the final choices of logrank with min_node_size = 100 to start
fitted_forest <- jfforest(MM ~ ., data = sim[1:num_obs], feature_data = test_data[1:num_obs,],
                          splitrule = "logrank", min_node_size = 100, seed = 2026)
print_forest(fitted_forest)

# VIMP (should do several runs, it seems that JF has a hard time distinguishing noise from signal),
# but the model is also quite complicated
unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "brier", seed = 2026)$vimp)
unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "kl", seed = 2026)$vimp)
unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "spherical", seed = 2026)$vimp)
unlist(jfforest.vimp(fitted_forest, method = "random", loss = "brier", seed = 2026)$vimp)
unlist(jfforest.vimp(fitted_forest, method = "random", loss = "kl", seed = 2026)$vimp)
unlist(jfforest.vimp(fitted_forest, method = "random", loss = "spherical", seed = 2026)$vimp)

# 'verify' consistency
#--------------------------------------------------------------------------------

length(unique(unlist(lapply(sim, function(z) z$times))))
# 105090 event and censoring times total for 50,000 observations!

# fit forest using num_obs observations with the best hyperparameters found above
# (the original number of event times was 50,000+, we reduce it to 1000, could probably do with even less)
num_obs <- 50000
tic()
final_forest <- jfforest(MM ~ ., data = sim[1:num_obs], feature_data = test_data[1:num_obs,], splitrule = "logrank",
                         min_node_size = 100, seed = 2026, ntrees = 500, save_predictions = FALSE, num_event_times = 1000)
toc()   # only takes about 8 seconds to fit, 8 GB of ram usage though...
print_forest(final_forest)

# notes:
# - should probably increase the minimal node size, 200-250 would likely be more appropriate
# - with so many observations, all the good splits (the categorical features) will run out, meaning that at some point,
#   we are just fitting noise. The total number of combinations possible using the categorical covariates are 2 *4 * 5 = 40

# we know the initial distribution is c(1, 0, 0), so no need to predict it

# we consider the following four combinations of covariates
# man (X_1 = 1) with education level X_2 = 1 and income level X3 = 2
# man (X_1 = 1) with education level X_2 = 4 and income level X3 = 5
# woman (X_1 = 0) with education level X_2 = 1 and income level X3 = 2
# woman (X_1 = 0) with education level X_2 = 4 and income level X3 = 5

new_data <- data.frame(X1 = c(1,1,0,0), X2 = c(1,4,1,4), X3 = c(2,5,2,5),
                       X4 = rep(0, 4), X5 = rep(0, 4), X6 = rep(0, 4))
predictions <- jfforest.predict(final_forest, new_data = new_data)

# predicted occupation probabilities for each state
occprobs1_predict <- occprob_from_data(predictions[[1]], c(1, 0, 0))
occprobs2_predict <- occprob_from_data(predictions[[2]], c(1, 0, 0))
occprobs3_predict <- occprob_from_data(predictions[[3]], c(1, 0, 0))
occprobs4_predict <- occprob_from_data(predictions[[4]], c(1, 0, 0))

event_times <- final_forest$unique.event.times
plot_end <- max(event_times)
true_steps <- max(1L, ceiling(10^4 * plot_end / (120 - x)))

# true occupation probabilities for each state
occprobs1 <- occprob(function(t) {Lambda(t, as.numeric(new_data[1,]))}, plot_end, c(1,0,0), true_steps)
occprobs2 <- occprob(function(t) {Lambda(t, as.numeric(new_data[2,]))}, plot_end, c(1,0,0), true_steps)
occprobs3 <- occprob(function(t) {Lambda(t, as.numeric(new_data[3,]))}, plot_end, c(1,0,0), true_steps)
occprobs4 <- occprob(function(t) {Lambda(t, as.numeric(new_data[4,]))}, plot_end, c(1,0,0), true_steps)

# Prepare predicted and true occupation probabilities on the same time grid.
plot_times <- event_times
true_times <- seq(0, plot_end, length.out = length(occprobs1))

predicted_occprobs <- list(occprobs1_predict, occprobs2_predict,
                           occprobs3_predict, occprobs4_predict)
true_occprobs <- list(occprobs1, occprobs2, occprobs3, occprobs4)
panel_titles <- c("Man: X2 = 1, X3 = 2", "Man: X2 = 4, X3 = 5",
                  "Woman: X2 = 1, X3 = 2", "Woman: X2 = 4, X3 = 5")
state_colours <- c("#0072B2", "#D55E00", "#009E73")

# Draw to the active graphics device when file is NULL; otherwise save a PNG.
plot_occupation_probabilities <- function(file = NULL, width = 11, height = 6.5, resolution = 300) {
    saving <- !is.null(file)
    if (saving) {
        dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
        png(file, width = width, height = height, units = "in", res = resolution)
    }

    old_par <- par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
    on.exit({
        par(old_par)
        if (saving) {
            dev.off()
        }
    })

    for (i in seq_along(predicted_occprobs)) {
        predicted <- do.call(rbind, predicted_occprobs[[i]])
        truth <- do.call(rbind, true_occprobs[[i]])
        truth_at_event_times <- vapply(seq_len(ncol(truth)), function(state) {
            approx(true_times, truth[, state], xout = plot_times)$y
        }, numeric(length(plot_times)))

        matplot(plot_times, predicted,
                type = "l", lty = 1, lwd = 2, col = state_colours,
                ylim = c(0, 1), xlab = "Time", ylab = "Occupation probability",
                main = panel_titles[i])
        matlines(plot_times, truth_at_event_times,
                 lty = 2, lwd = 2, col = state_colours)

        if (i == 1) {
            legend("right", legend = c(paste("State", 1:3), "Predicted", "True"),
                   col = c(state_colours, "black", "black"),
                   lty = c(rep(1, 3), 1, 2), lwd = 2, bty = "n")
        }
    }

    invisible(file)
}

# Display the plot in RStudio, then save the same plot at 300 DPI.
plot_occupation_probabilities()
plot_occupation_probabilities("testing/Articles/Discrete/Plots/discrete_consistency_Markov.png")

# how to plot the errors? should the forest be modified to return the whole vector of scores?
# not relevant to plot errors here, this is just visualisation

# in principle we should test larger node sizes, but internal error prediction is too memory demanding

# fit the conditional Aalen-Johansen estimator
#--------------------------------------------------------------------------------

# fit Cox proportional hazard model (how exactly?)
#--------------------------------------------------------------------------------

# fit a Poisson regression model (how exactly is this done with covariates?)
#--------------------------------------------------------------------------------

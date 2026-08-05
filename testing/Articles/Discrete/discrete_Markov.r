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
toc()   # takes about 230 seconds for n = 50,000
}

# save data
saveRDS(sim, file = "testing/Articles/Discrete/Data/sim.rds")
write.table(test_data, file = "testing/Articles/Discrete/Data/test_data.txt")

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

# read in the data
sim <- readRDS("testing/Articles/Discrete/Data/sim.rds")
test_data <- read.table("testing/Articles/Discrete/Data/test_data.txt", header = TRUE)

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

tuning_colours <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")

plot_score_curves(
    scores = ibs, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(ibs), colours = tuning_colours,
    pch = seq_along(split_rules), ylab = "Normalised IBS", width = 5, height = 3
)
plot_score_curves(
    scores = ibs, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(ibs), colours = tuning_colours,
    pch = seq_along(split_rules), ylab = "Normalised IBS",
    file = "testing/Articles/Discrete/Plots/tuning_ibs.png", width = 6, height = 6
)
plot_score_curves(
    scores = ikl, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(ikl), colours = tuning_colours,
    pch = seq_along(split_rules), ylab = "Normalised KL",
    file = "testing/Articles/Discrete/Plots/tuning_ikl.png", width = 6, height = 6
)
plot_score_curves(
    scores = spherical, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(spherical), colours = tuning_colours,
    pch = seq_along(split_rules), ylab = "Normalised Spherical error",
    legend_position = "bottomright",
    file = "testing/Articles/Discrete/Plots/tuning_is.png", width = 6, height = 6
)

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

# import helper functions and packages
source("testing/Articles/Discrete/Helpers.r")

# read in the data
sim <- readRDS("testing/Articles/Discrete/Data/sim.rds")
test_data <- read.table("testing/Articles/Discrete/Data/test_data.txt", header = TRUE)

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

# Prepare predicted and true occupation probabilities.
true_times <- seq(0, plot_end, length.out = length(occprobs1))

predicted_occprobs <- lapply(
    list(occprobs1_predict, occprobs2_predict, occprobs3_predict, occprobs4_predict),
    function(probabilities) do.call(rbind, probabilities)
)
true_occprobs <- lapply(
    list(occprobs1, occprobs2, occprobs3, occprobs4),
    function(probabilities) do.call(rbind, probabilities)
)
panel_titles <- c("Man: X2 = 1, X3 = 2", "Man: X2 = 4, X3 = 5",
                  "Woman: X2 = 1, X3 = 2", "Woman: X2 = 4, X3 = 5")
state_colours <- c("#0072B2", "#D55E00", "#009E73")

consistency_occupation_curves <- list(
    True = list(times = true_times, values = true_occprobs, type = "l", lty = 1),
    Predicted = list(times = event_times, values = predicted_occprobs, type = "s", lty = 2)
)

# Display the plot in RStudio, then save the same plot at 300 DPI.
plot_panel_curves(
    consistency_occupation_curves, component_labels = paste("State", 0:2),
    panel_titles = panel_titles, component_colours = state_colours,
    ylab = "Occupation probability", ylim = c(0, 1), panel_layout = c(2, 2),
    legend_position = "right", legend_order = c("Predicted", "True")
)
plot_panel_curves(
    consistency_occupation_curves, component_labels = paste("State", 0:2),
    panel_titles = panel_titles, component_colours = state_colours,
    ylab = "Occupation probability", ylim = c(0, 1), panel_layout = c(2, 2),
    legend_position = "right", legend_order = c("Predicted", "True"),
    file = "testing/Articles/Discrete/Plots/discrete_consistency_Markov.png"
)

# Compare the forest and true cumulative transition rates.
consistency_transition_indices <- matrix(
    c(1, 2,
      1, 3,
      2, 1,
      2, 3),
    ncol = 2, byrow = TRUE
)
consistency_transition_names <- c("0 -> 1", "0 -> 2", "1 -> 0", "1 -> 2")
consistency_transition_colours <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7")

consistency_forest_transition_rates <- lapply(predictions, function(prediction) {
    extract_matrix_entries(
        prediction, consistency_transition_indices, consistency_transition_names
    )
})

# Numerically integrate the four true transition intensities.
consistency_true_transition_rates <- lapply(seq_len(nrow(new_data)), function(i) {
    X <- as.numeric(new_data[i, ])
    instantaneous_rates <- t(vapply(true_times, function(time) {
        Lambda(time, X)[consistency_transition_indices]
    }, numeric(nrow(consistency_transition_indices))))

    increments <- sweep(
        (instantaneous_rates[-1, , drop = FALSE] +
         instantaneous_rates[-nrow(instantaneous_rates), , drop = FALSE]) / 2,
        1, diff(true_times), "*"
    )
    cumulative_rates <- matrix(
        0, nrow = nrow(instantaneous_rates), ncol = ncol(instantaneous_rates)
    )
    cumulative_rates[-1, ] <- apply(increments, 2, cumsum)
    cumulative_rates
})

consistency_transition_curves <- list(
    True = list(
        times = true_times, values = consistency_true_transition_rates,
        type = "l", lty = 1
    ),
    Predicted = list(
        times = event_times, values = consistency_forest_transition_rates,
        type = "s", lty = 2
    )
)

plot_panel_curves(
    consistency_transition_curves, component_labels = consistency_transition_names,
    panel_titles = panel_titles, component_colours = consistency_transition_colours,
    ylab = "Cumulative transition rate", xlim = c(0, plot_end),
    include_zero = TRUE, panel_layout = c(2, 2), legend_position = "topleft",
    legend_order = c("Predicted", "True")
)
plot_panel_curves(
    consistency_transition_curves, component_labels = consistency_transition_names,
    panel_titles = panel_titles, component_colours = consistency_transition_colours,
    ylab = "Cumulative transition rate", xlim = c(0, plot_end),
    include_zero = TRUE, panel_layout = c(2, 2), legend_position = "topleft",
    legend_order = c("Predicted", "True"),
    file = "testing/Articles/Discrete/Plots/discrete_consistency_Markov_transition_rates.png"
)

# how to plot the errors? should the forest be modified to return the whole vector of scores?
# not relevant to plot errors here, this is just visualisation

# in principle we should test larger node sizes, but internal error prediction is too memory demanding

# fit and compare to the conditional Aalen-Johansen estimator
#--------------------------------------------------------------------------------

# import helper functions and packages
source("testing/Articles/Discrete/Helpers.r")

# read in the data
sim <- readRDS("testing/Articles/Discrete/Data/sim.rds")
test_data <- read.table("testing/Articles/Discrete/Data/test_data.txt", header = TRUE)

# we choose to work with less data here and we drop the noise variables
num.obs <- 10000
length(unique(unlist(lapply(sim[1:num.obs], function(z) z$times)))) # 10432

fitted_forest <- jfforest(MM ~ ., data = sim[1:num.obs], feature_data = test_data[1:num.obs, 1:3], splitrule = "logrank",
                          min_node_size = 100, seed = 2026, ntrees = 500, save_predictions = FALSE, num_event_times = 1000)
print_forest(fitted_forest)

new_data <- data.frame(X1 = c(1,1,0,0), X2 = c(1,4,1,4), X3 = c(2,5,2,5))
forest_predictions <- jfforest.predict(fitted_forest, new_data = new_data)

# The package only supports a scalar conditioning variable directly. Since the
# signal variables are discrete, fit an ordinary AJ estimator within each exact stratum.
rows_by_profile <- lapply(seq_len(nrow(new_data)), function(i) {
    which(seq_len(nrow(test_data)) <= num.obs &
        test_data$X1 == new_data$X1[i] &
        test_data$X2 == new_data$X2[i] &
        test_data$X3 == new_data$X3[i])
})
lengths(rows_by_profile)    # note: very few observations in groups 2 and 4

AJfits <- lapply(rows_by_profile, function(rows) {aalen_johansen(sim[rows], p = 2)})

# Convert the forest Nelson--Aalen predictions to occupation probabilities.
comparison_forest_occprobs <- lapply(forest_predictions, function(prediction) {
    do.call(rbind, occprob_from_data(prediction, c(1, 0, 0)))
})
comparison_forest_times <- fitted_forest$unique.event.times

# Compute the true probabilities over the complete range covered by either estimator.
comparison_end <- max(comparison_forest_times,unlist(lapply(AJfits, function(fit) fit$t)))
comparison_steps <- max(1L, ceiling(10^4 * comparison_end / (120 - x)))
comparison_true_times <- seq(0, comparison_end, length.out = comparison_steps + 1L)
comparison_true_occprobs <- lapply(seq_len(nrow(new_data)), function(i) {
    do.call(rbind, occprob(
        function(t) Lambda(t, as.numeric(new_data[i, ])),
        comparison_end, c(1, 0, 0), comparison_steps
    ))
})

comparison_AJ_occprobs <- lapply(AJfits, function(fit) {
    do.call(rbind, fit$p)
})

comparison_titles <- c("Man: X2 = 1, X3 = 2", "Man: X2 = 4, X3 = 5",
                       "Woman: X2 = 1, X3 = 2", "Woman: X2 = 4, X3 = 5")
comparison_colours <- c("#0072B2", "#D55E00", "#009E73")

comparison_occupation_curves <- list(
    True = list(
        times = comparison_true_times, values = comparison_true_occprobs,
        type = "l", lty = 1
    ),
    Forest = list(
        times = comparison_forest_times, values = comparison_forest_occprobs,
        type = "s", lty = 2
    ),
    `Conditional AJ` = list(
        times = lapply(AJfits, `[[`, "t"), values = comparison_AJ_occprobs,
        type = "s", lty = 3
    )
)

plot_panel_curves(
    comparison_occupation_curves, component_labels = paste("State", 0:2),
    panel_titles = comparison_titles, component_colours = comparison_colours,
    ylab = "Occupation probability", xlim = c(0, comparison_end),
    ylim = c(0, 1), panel_layout = c(2, 2), legend_position = "right",
    legend_order = c("Forest", "Conditional AJ", "True")
)
plot_panel_curves(
    comparison_occupation_curves, component_labels = paste("State", 0:2),
    panel_titles = comparison_titles, component_colours = comparison_colours,
    ylab = "Occupation probability", xlim = c(0, comparison_end),
    ylim = c(0, 1), panel_layout = c(2, 2), legend_position = "right",
    legend_order = c("Forest", "Conditional AJ", "True"),
    file = "testing/Articles/Discrete/Plots/discrete_comparison_AJ_forest.png"
)

# Compare the four nonzero cumulative transition rates.
comparison_transition_indices <- matrix(
    c(1, 2,
      1, 3,
      2, 1,
      2, 3),
    ncol = 2, byrow = TRUE
)
comparison_transition_names <- c("0 -> 1", "0 -> 2", "1 -> 0", "1 -> 2")
comparison_transition_colours <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7")

comparison_forest_transition_rates <- lapply(forest_predictions, function(prediction) {
    extract_matrix_entries(
        prediction, comparison_transition_indices, comparison_transition_names
    )
})
comparison_AJ_transition_rates <- lapply(AJfits, function(fit) {
    extract_matrix_entries(
        fit$Lambda, comparison_transition_indices, comparison_transition_names
    )
})

# Integrate each true transition intensity over the common plotting grid.
comparison_true_transition_rates <- lapply(seq_len(nrow(new_data)), function(i) {
    X <- as.numeric(new_data[i, ])
    instantaneous_rates <- t(vapply(comparison_true_times, function(time) {
        Lambda(time, X)[comparison_transition_indices]
    }, numeric(nrow(comparison_transition_indices))))

    increments <- sweep(
        (instantaneous_rates[-1, , drop = FALSE] +
         instantaneous_rates[-nrow(instantaneous_rates), , drop = FALSE]) / 2,
        1, diff(comparison_true_times), "*"
    )
    cumulative_rates <- matrix(
        0, nrow = nrow(instantaneous_rates), ncol = ncol(instantaneous_rates)
    )
    cumulative_rates[-1, ] <- apply(increments, 2, cumsum)
    cumulative_rates
})

comparison_transition_curves <- list(
    True = list(
        times = comparison_true_times, values = comparison_true_transition_rates,
        type = "l", lty = 1
    ),
    Forest = list(
        times = comparison_forest_times, values = comparison_forest_transition_rates,
        type = "s", lty = 2
    ),
    `Conditional AJ` = list(
        times = lapply(AJfits, `[[`, "t"), values = comparison_AJ_transition_rates,
        type = "s", lty = 3
    )
)

plot_panel_curves(
    comparison_transition_curves, component_labels = comparison_transition_names,
    panel_titles = comparison_titles, component_colours = comparison_transition_colours,
    ylab = "Cumulative transition rate", xlim = c(0, comparison_end),
    include_zero = TRUE, panel_layout = c(2, 2), legend_position = "topleft",
    legend_order = c("Forest", "Conditional AJ", "True")
)
plot_panel_curves(
    comparison_transition_curves, component_labels = comparison_transition_names,
    panel_titles = comparison_titles, component_colours = comparison_transition_colours,
    ylab = "Cumulative transition rate", xlim = c(0, comparison_end),
    include_zero = TRUE, panel_layout = c(2, 2), legend_position = "topleft",
    legend_order = c("Forest", "Conditional AJ", "True"),
    file = "testing/Articles/Discrete/Plots/discrete_comparison_AJ_forest_transition_rates.png"
)

# number of 1 - > 0 transitions
count_disabled_to_active <- function(path) {
    states <- path$states

    if (length(states) < 2L) {
        return(0L)
    }

    sum(
        head(states, -1L) == 2L &
        tail(states, -1L) == 1L
    )
}

# Total among the observations used for fitting
transition_counts <- vapply(
    sim[seq_len(num.obs)],
    count_disabled_to_active,
    integer(1)
)

sum(transition_counts)            # total number of 1 -> 0 transitions
sum(transition_counts > 0L)       # individuals with at least one
mean(transition_counts)           # transitions per individual
table(transition_counts)          # distribution per individual

disabled_to_active_by_profile <- vapply(rows_by_profile, function(rows) {
    sum(vapply(sim[rows], count_disabled_to_active, integer(1)))
}, integer(1))

names(disabled_to_active_by_profile) <- comparison_titles
disabled_to_active_by_profile

# key takeaway: for the occupation probabilities, it is clear the the random forest performs better in the sense that it subsamples
# data much more efficiently. There is definitely underestimation of the transition 1 -> 0, but this is to be expected since there
# are very few transitions in the different groups. The underestimation is much worse for CAJ than for the jump forest

# fit Cox proportional hazard model
#--------------------------------------------------------------------------------

# Convert the same paths used by the CAJ study to a counting-process data set.
# A same-state final interval is censoring; every other interval ends in the
# destination state recorded by the path. Absolute time is used because the DGP
# is a clock-forward Markov model.
paths_to_cox_data <- function(paths, features, number_of_states = 3L) {
    if (length(paths) != nrow(features)) {
        stop("paths and features must describe the same number of individuals")
    }

    rows <- Map(function(path, id) {
        if (length(path$times) != length(path$states) || length(path$times) < 2L ||
            any(!is.finite(path$times)) || is.unsorted(path$times, strictly = TRUE) ||
            any(path$states < 1L | path$states > number_of_states)) {
            stop("each path must contain valid, strictly increasing times and states")
        }

        number_of_intervals <- length(path$times) - 1L
        cbind(
            data.frame(
                id = rep.int(id, number_of_intervals),
                tstart = head(path$times, -1L), # -1 removes last element
                tstop = tail(path$times, -1L),  # -1 removes first element
                from = head(path$states, -1L),
                to = tail(path$states, -1L)
            ),
            features[rep.int(id, number_of_intervals), , drop = FALSE]
        )
    }, paths, seq_along(paths))

    result <- do.call(rbind, rows)
    rownames(result) <- NULL
    state_levels <- paste0("state", seq_len(number_of_states))
    event_label <- ifelse(
        result$from == result$to, "censor", paste0("state", result$to)
    )
    # The first event-factor level denotes censoring to survival::Surv.
    result$event <- factor(event_label, levels = c("censor", state_levels))
    result$istate <- factor(paste0("state", result$from), levels = state_levels)
    result
}

cox_data <- paths_to_cox_data(
    sim[seq_len(num.obs)],
    test_data[seq_len(num.obs), c("X1", "X2", "X3"), drop = FALSE]
)

# survival::coxph expands the coefficients and baseline hazards by transition in
# a native multi-state fit. Thus both specifications are fitted to all four DGP
# transitions (0 -> 1, 0 -> 2, 1 -> 0 and 1 -> 2) in one call. X1 remains
# linear because it is binary; ns() supplies natural cubic splines for the two
# ordinal predictors.
cox_formulas <- list(
    linear = survival::Surv(tstart, tstop, event) ~ X1 + X2 + X3,
    cubic_spline = survival::Surv(tstart, tstop, event) ~
        X1 + splines::ns(X2, df = 3) + splines::ns(X3, df = 3)
)

tic()
cox_fits <- lapply(cox_formulas, function(cox_formula) {
    fit <- survival::coxph(
        cox_formula, data = cox_data, id = id, istate = istate,
        ties = "breslow", model = TRUE, x = TRUE, singular.ok = FALSE
    )
    if (any(!is.finite(stats::coef(fit)))) {
        stop("a transition-specific Cox model contains non-finite coefficients")
    }
    fit
})
toc()   # about one second

# The four fitted transition models and their event counts.
cox_transition_counts <- with(cox_data, table(istate, event))
cox_transition_counts
lapply(cox_fits, print)

# Use survival's Aalen--Johansen product integral (stype = 1) to turn the fitted
# transition hazards into occupation probabilities. The transition order in a
# coxphms object is model-dependent, so it is matched explicitly.
extract_cox_curves <- function(fit, new_data, horizon = NULL, initial = c(1, 0, 0), state_order = paste0("state", 1:3), transition_order = c("1:2", "1:3", "2:1", "2:3")) {
    predicted <- survival::survfit(
        fit, newdata = new_data, p0 = initial, stype = 1, se.fit = FALSE
    )
    pstate <- predicted$pstate
    cumulative_hazards <- predicted$cumhaz

    # Retain the profile dimension if a survival version drops it for one row.
    if (length(dim(pstate)) == 2L) {
        pstate <- array(pstate, dim = c(nrow(pstate), 1L, ncol(pstate)))
    }
    if (length(dim(cumulative_hazards)) == 2L) {
        cumulative_hazards <- array(
            cumulative_hazards,
            dim = c(nrow(cumulative_hazards), 1L, ncol(cumulative_hazards))
        )
    }

    state_index <- match(state_order, fit$states)
    transition_index <- match(transition_order, colnames(fit$cmap))
    if (anyNA(state_index) || anyNA(transition_index)) {
        stop("could not match the requested Cox state or transition order")
    }
    pstate <- pstate[, , state_index, drop = FALSE]
    cumulative_hazards <- cumulative_hazards[, , transition_index, drop = FALSE]

    number_of_profiles <- nrow(new_data)
    number_of_times <- length(predicted$time)
    occupation_probabilities <- lapply(seq_len(number_of_profiles), function(i) {
        values <- matrix(
            pstate[, i, , drop = FALSE], nrow = number_of_times,
            ncol = length(state_order), dimnames = list(NULL, state_order)
        )
        row_totals <- rowSums(values)
        if (any(!is.finite(row_totals)) || any(row_totals <= 0)) {
            stop("Cox occupation predictions have invalid row totals")
        }
        # Remove the small numerical row-sum drift from the direct product integral.
        sweep(values, 1L, row_totals, "/")
    })
    cumulative_transition_rates <- lapply(seq_len(number_of_profiles), function(i) {
        matrix(
            cumulative_hazards[, i, , drop = FALSE], nrow = number_of_times,
            ncol = length(transition_order), dimnames = list(NULL, transition_order)
        )
    })

    prediction_times <- predicted$time
    if (!length(prediction_times) || prediction_times[1L] > 0) {
        prediction_times <- c(0, prediction_times)
        occupation_probabilities <- lapply(occupation_probabilities, function(values) {
            rbind(stats::setNames(initial, state_order), values)
        })
        cumulative_transition_rates <- lapply(cumulative_transition_rates, function(values) {
            rbind(stats::setNames(rep(0, length(transition_order)), transition_order), values)
        })
    }
    if (!is.null(horizon) && tail(prediction_times, 1L) < horizon) {
        prediction_times <- c(prediction_times, horizon)
        occupation_probabilities <- lapply(occupation_probabilities, function(values) {
            rbind(values, tail(values, 1L))
        })
        cumulative_transition_rates <- lapply(cumulative_transition_rates, function(values) {
            rbind(values, tail(values, 1L))
        })
    }

    list(
        times = prediction_times,
        occupation_probabilities = occupation_probabilities,
        cumulative_transition_rates = cumulative_transition_rates,
        survfit = predicted
    )
}

tic()
cox_predictions <- lapply(cox_fits, extract_cox_curves, new_data = new_data, horizon = comparison_end)
toc()   # about 2.6 seconds to compute predictions

# Compare the Jump Forest and both Cox specifications against the DGP truth.
cox_occupation_curves <- list(
    True = list(
        times = comparison_true_times, values = comparison_true_occprobs,
        type = "l", lty = 1
    ),
    `Jump Forest` = list(
        times = comparison_forest_times, values = comparison_forest_occprobs,
        type = "s", lty = 2
    ),
    `Cox linear` = list(
        times = cox_predictions$linear$times,
        values = cox_predictions$linear$occupation_probabilities,
        type = "s", lty = 3
    ),
    `Cox cubic spline` = list(
        times = cox_predictions$cubic_spline$times,
        values = cox_predictions$cubic_spline$occupation_probabilities,
        type = "s", lty = 4
    )
)

plot_panel_curves(
    cox_occupation_curves, component_labels = paste("State", 0:2),
    panel_titles = comparison_titles, component_colours = comparison_colours,
    ylab = "Occupation probability", xlim = c(0, comparison_end),
    ylim = c(0, 1), panel_layout = c(2, 2), legend_position = "right",
    legend_order = c("Jump Forest", "Cox linear", "Cox cubic spline", "True"),
    legend_cex = 0.8
)
plot_panel_curves(
    cox_occupation_curves, component_labels = paste("State", 0:2),
    panel_titles = comparison_titles, component_colours = comparison_colours,
    ylab = "Occupation probability", xlim = c(0, comparison_end),
    ylim = c(0, 1), panel_layout = c(2, 2), legend_position = "right",
    legend_order = c("Jump Forest", "Cox linear", "Cox cubic spline", "True"),
    legend_cex = 0.8,
    file = "testing/Articles/Discrete/Plots/discrete_comparison_cox_forest.png"
)

cox_transition_curves <- list(
    True = list(
        times = comparison_true_times, values = comparison_true_transition_rates,
        type = "l", lty = 1
    ),
    `Jump Forest` = list(
        times = comparison_forest_times, values = comparison_forest_transition_rates,
        type = "s", lty = 2
    ),
    `Cox linear` = list(
        times = cox_predictions$linear$times,
        values = cox_predictions$linear$cumulative_transition_rates,
        type = "s", lty = 3
    ),
    `Cox cubic spline` = list(
        times = cox_predictions$cubic_spline$times,
        values = cox_predictions$cubic_spline$cumulative_transition_rates,
        type = "s", lty = 4
    )
)

plot_panel_curves(
    cox_transition_curves, component_labels = comparison_transition_names,
    panel_titles = comparison_titles, component_colours = comparison_transition_colours,
    ylab = "Cumulative transition rate", xlim = c(0, comparison_end),
    include_zero = TRUE, panel_layout = c(2, 2), legend_position = "topleft",
    legend_order = c("Jump Forest", "Cox linear", "Cox cubic spline", "True"),
    legend_cex = 0.8
)
plot_panel_curves(
    cox_transition_curves, component_labels = comparison_transition_names,
    panel_titles = comparison_titles, component_colours = comparison_transition_colours,
    ylab = "Cumulative transition rate", xlim = c(0, comparison_end),
    include_zero = TRUE, panel_layout = c(2, 2), legend_position = "topleft",
    legend_order = c("Jump Forest", "Cox linear", "Cox cubic spline", "True"),
    legend_cex = 0.8,
    file = "testing/Articles/Discrete/Plots/discrete_comparison_cox_forest_transition_rates.png"
)

# Time-averaged oracle curve errors provide one metric shared by all three
# methods. With equal state weights, occupation MISE is proportional to the
# integrated Brier-score regret relative to the true probabilities.
step_matrix_at <- function(times, values, evaluation_times) {
    values <- as.matrix(values)
    if (!is.numeric(times) || length(times) != nrow(values) || is.unsorted(times)) {
        stop("times must be sorted and match the rows of values")
    }
    index <- findInterval(evaluation_times, times)
    if (any(index == 0L)) {
        stop("the prediction grid must begin no later than the evaluation grid")
    }
    values[index, , drop = FALSE]
}

curve_error_table <- function(truth_times, truth_values, prediction_times, prediction_values, model, component_names, profile_names) {
    if (length(truth_values) != length(prediction_values) ||
        length(profile_names) != length(truth_values)) {
        stop("truth, predictions and profile_names must have matching panels")
    }
    horizon <- max(truth_times) - min(truth_times)
    if (!is.finite(horizon) || horizon <= 0) {
        stop("truth_times must cover a positive finite interval")
    }

    do.call(rbind, lapply(seq_along(truth_values), function(i) {
        panel_times <- if (is.list(prediction_times)) {
            prediction_times[[i]]
        } else {
            prediction_times
        }
        truth <- as.matrix(truth_values[[i]])
        prediction <- step_matrix_at(
            panel_times, prediction_values[[i]], truth_times
        )
        if (!identical(dim(prediction), dim(truth)) ||
            length(component_names) != ncol(truth)) {
            stop("truth and prediction matrices must have matching dimensions")
        }

        squared_error <- (prediction - truth)^2
        increments <- sweep(
            (squared_error[-1L, , drop = FALSE] +
             squared_error[-nrow(squared_error), , drop = FALSE]) / 2,
            1L, diff(truth_times), "*"
        )
        mise <- colSums(increments) / horizon
        data.frame(
            model = model, profile = profile_names[i], component = component_names,
            MISE = mise, RMSE = sqrt(mise), row.names = NULL
        )
    }))
}

occupation_curve_errors <- rbind(
    curve_error_table(
        comparison_true_times, comparison_true_occprobs,
        comparison_forest_times, comparison_forest_occprobs,
        "Jump Forest", paste("State", 0:2), comparison_titles
    ),
    curve_error_table(
        comparison_true_times, comparison_true_occprobs,
        cox_predictions$linear$times,
        cox_predictions$linear$occupation_probabilities,
        "Cox linear", paste("State", 0:2), comparison_titles
    ),
    curve_error_table(
        comparison_true_times, comparison_true_occprobs,
        cox_predictions$cubic_spline$times,
        cox_predictions$cubic_spline$occupation_probabilities,
        "Cox cubic spline", paste("State", 0:2), comparison_titles
    )
)

transition_curve_errors <- rbind(
    curve_error_table(
        comparison_true_times, comparison_true_transition_rates,
        comparison_forest_times, comparison_forest_transition_rates,
        "Jump Forest", comparison_transition_names, comparison_titles
    ),
    curve_error_table(
        comparison_true_times, comparison_true_transition_rates,
        cox_predictions$linear$times,
        cox_predictions$linear$cumulative_transition_rates,
        "Cox linear", comparison_transition_names, comparison_titles
    ),
    curve_error_table(
        comparison_true_times, comparison_true_transition_rates,
        cox_predictions$cubic_spline$times,
        cox_predictions$cubic_spline$cumulative_transition_rates,
        "Cox cubic spline", comparison_transition_names, comparison_titles
    )
)

summarise_curve_errors <- function(errors) {
    summary <- stats::aggregate(MISE ~ model, data = errors, FUN = mean)
    summary$RMSE <- sqrt(summary$MISE)
    summary[order(summary$MISE), ]
}

occupation_error_summary <- summarise_curve_errors(occupation_curve_errors)
transition_error_summary <- summarise_curve_errors(transition_curve_errors)
occupation_error_summary
transition_error_summary

# The DGP contains time-varying effects of X2 and X3 (and additive hazard
# constants for three transitions), so neither Cox specification is correctly
# proportional. Cubic splines relax covariate shape, but not that PH assumption.


# fit a Poisson regression model (how exactly is this done with covariates?)
#--------------------------------------------------------------------------------



# fit the true model?
#--------------------------------------------------------------------------------

#nolint end

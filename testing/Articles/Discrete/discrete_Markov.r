#nolint start: line_length_linter

# this file is for testing consistency of random jump forests on Markov data
# with discrete covariates

# import helper functions and packages
source("testing/Articles/Helpers.r")

# defining the data-generating process and simulating data
#--------------------------------------------------------------------------------

interactions <- FALSE

{
# Keep both versions of the study reproducible from this script. All generated
# artifacts use the same basenames and are separated by their study directory.
study_name <- if (interactions) "Interactions" else "NoInteractions"
data_directory <- file.path(
    "testing", "Articles", "Discrete", "Data", study_name
)
plot_directory <- file.path(
    "testing", "Articles", "Discrete", "Plots", study_name
)
dir.create(data_directory, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_directory, recursive = TRUE, showWarnings = FALSE)

data_path <- function(filename) file.path(data_directory, filename)
plot_path <- function(filename) file.path(plot_directory, filename)

# These probabilities are used both to simulate X2/X3 and to centre their
# interaction. Independence and centring make the interaction orthogonal to
# both main effects in the population linear predictor.
x2_probabilities <- c(0.2, 0.4, 0.3, 0.1)
x3_probabilities <- c(0.15, 0.3, 0.4, 0.1, 0.05)
x2_centre <- weighted.mean(seq_along(x2_probabilities), x2_probabilities)
x3_centre <- weighted.mean(seq_along(x3_probabilities), x3_probabilities)

# Mortality parameters in the interaction DGP are calibrated so that, after
# averaging over the X2/X3 distribution, residual life expectancy at age 30 is
# 50.92 years for men (X1 = 1) and 54.50 years for women (X1 = 0). These are the
# 2024:2025 values in Statistics Denmark table HISB8. The original coefficients
# remain in force when interactions is FALSE.

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

# Centred X2-by-X3 interaction in both mortality intensities. At absolute age
# 100, beta23_mortality = 0.0015 gives a maximum-versus-minimum interaction
# hazard-ratio contrast of exp(0.0015 * (4.08 - (-3.12)) * 100) = 2.94. The intercept
# adjustments compensate for the interaction's nonlinear effect on population-
# averaged mortality without changing the original no-interaction DGP. They add
# -0.030684 for women and -0.030684 + 0.030758 = 0.000074 for men to the two
# exponential mortality predictors.
beta23_mortality <- if (interactions) 1.5e-03 else 0
mortality_intercept_calibration <- if (interactions) -0.030684 else 0
mortality_sex_calibration <- if (interactions) 0.030758 else 0

# misc. coefficients
gamma2_01 <- 1.183597e-03
beta4_01 <- -1.111111e-05
beta5_01 <- 2.777778e-06
gamma3_01 <- -9.255633e-06
beta6_01 <- 1.234568e-07

# initial age
x <- 30

# define intensities
interaction_23 <- function(X) {
    (X[2] - x2_centre) * (X[3] - x3_centre)
}

mu01 <- function(t, X) {
    exp(beta0_01 + beta1_01 * X[1] + (gamma1_01 + beta2_01 * X[2] + beta3_01 * X[3]) * (t + x)
        + (gamma2_01 + beta4_01 * X[2] + beta5_01 * X[3]) * (t + x)^2
        + (gamma3_01 + beta6_01 * X[2]) * (t + x)^3)
}
mu02 <- function(t, X) {
    alpha0_02 + alpha1_02 * X[1] + exp(
        beta0_02 + mortality_intercept_calibration +
        (beta1_02 + mortality_sex_calibration) * X[1] +
        (gamma1_02 + beta2_02 * X[2] + beta3_02 * X[3] +
         beta23_mortality * interaction_23(X)) * (t + x)
    )
}
mu12 <- function(t, X) {
    alpha0_12 + alpha1_12 * X[1] + exp(
        beta0_12 + mortality_intercept_calibration +
        (beta1_12 + mortality_sex_calibration) * X[1] +
        (gamma1_12 + beta2_12 * X[2] + beta3_12 * X[3] +
         beta23_mortality * interaction_23(X)) * (t + x)
    )
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

# Retain the original evaluation profiles for the no-interaction study. For the
# interaction study, hold sex fixed and evaluate the complete 2-by-2 X2/X3
# corner design. This exposes both positive and negative interaction cells
# instead of confounding the interaction contrast with the sex effect.
evaluation_profiles <- if (interactions) {
    data.frame(
        X1 = rep(0, 4),
        X2 = c(1, 1, 4, 4),
        X3 = c(2, 5, 2, 5)
    )
} else {
    data.frame(
        X1 = c(1, 1, 0, 0),
        X2 = c(1, 4, 1, 4),
        X3 = c(2, 5, 2, 5)
    )
}
evaluation_profile_titles <- paste0(
    ifelse(evaluation_profiles$X1 == 1, "Man", "Woman"),
    ": X2 = ", evaluation_profiles$X2,
    ", X3 = ", evaluation_profiles$X3
)
}

# simulations
#--------------------------------------------------------------------------------

# now simulate the paths
{
tic()
set.seed(2026)
n <- 50000
# signal
X1 <- rbinom(n, 1, 1/2)                                       # sex, 0: female, 1: male
X2 <- sample(1:4, n, replace = TRUE, prob = x2_probabilities) # education level
X3 <- sample(1:5, n, replace = TRUE, prob = x3_probabilities) # wage level

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
saveRDS(sim, file = data_path("sim.rds"))
write.table(test_data, file = data_path("test_data.txt"))

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

range(lifetime_diagnostics$mean.residual.lifetime)
max(lifetime_diagnostics$probability.age.above.120)

profile_weights <- x2_probabilities[lifetime_profiles$X2] *
    x3_probabilities[lifetime_profiles$X3]
weighted_residual_lifetime <- vapply(0:1, function(sex) {
    use <- lifetime_profiles$X1 == sex
    weighted.mean(lifetime_diagnostics$mean.residual.lifetime[use], profile_weights[use])
}, numeric(1))
names(weighted_residual_lifetime) <- c("women", "men")
weighted_residual_lifetime

# Guard the Statistics Denmark calibration and the intended interaction size
# against accidental coefficient changes. Under the interaction DGP, the
# profile range is approximately 47.77--58.36 residual years and the maximum
# probability of surviving beyond age 120 is approximately 1.61e-4.
statistics_denmark_residual_lifetime <- c(women = 54.50, men = 50.92)
if (interactions && any(abs(
    weighted_residual_lifetime - statistics_denmark_residual_lifetime
) > 0.01)) {
    stop("the interaction DGP no longer matches the HISB8 calibration")
}

interaction_profile_values <- outer(
    seq_along(x2_probabilities) - x2_centre,
    seq_along(x3_probabilities) - x3_centre
)
interaction_contrast_age_100 <- exp(
    beta23_mortality * diff(range(interaction_profile_values)) * 100
)
if (interactions && interaction_contrast_age_100 < 2) {
    stop("the X2-by-X3 interaction is no longer substantively large")
}


# fit a jump forest (hyperparameter tuning)
#--------------------------------------------------------------------------------

# read in the data
sim <- readRDS(data_path("sim.rds"))
test_data <- read.table(data_path("test_data.txt"), header = TRUE)

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

best_tuning_configuration <- function(scores, metric) {
    best <- which(scores == min(scores), arr.ind = TRUE)[1, ]
    data.frame(
        metric = metric,
        split_rule = split_rules[best["row"]],
        min_node_size = min_node_sizes[best["col"]],
        score = scores[best["row"], best["col"]]
    )
}
best_tuning_results <- rbind(
    best_tuning_configuration(ibs, "IBS"),
    best_tuning_configuration(kl, "KL"),
    best_tuning_configuration(spherical, "Spherical")
)
best_tuning_results

# for plots later
write.table(ibs, file = data_path("tuning_ibs.txt"), sep = "\t", row.names = FALSE)
write.table(kl, file = data_path("tuning_kl.txt"), sep = "\t", row.names = FALSE)
write.table(spherical, file = data_path("tuning_spherical.txt"), sep = "\t", row.names = FALSE)

# now make hyperparameter tuning plots
ibs <- read.table(data_path("tuning_ibs.txt"), header = TRUE)
colnames(ibs) <- as.factor(min_node_sizes)
ikl <- read.table(data_path("tuning_kl.txt"), header = TRUE)
colnames(ikl) <- as.factor(min_node_sizes)
spherical <- read.table(data_path("tuning_spherical.txt"), header = TRUE)
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
    file = plot_path("tuning_ibs.png"), width = 6, height = 6
)
plot_score_curves(
    scores = ikl, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(ikl), colours = tuning_colours,
    pch = seq_along(split_rules), ylab = "Normalised KL",
    file = plot_path("tuning_ikl.png"), width = 6, height = 6
)
plot_score_curves(
    scores = spherical, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(spherical), colours = tuning_colours,
    pch = seq_along(split_rules), ylab = "Normalised Spherical error",
    legend_position = "bottomright",
    file = plot_path("tuning_is.png"), width = 6, height = 6
)

# without interaction: we choose to go with the final choices of logrank with min_node_size = 100 to start
# with interaction: we go with taroneware and min_node_size = 200
fitted_forest <- jfforest(MM ~ ., data = sim[1:num_obs], feature_data = test_data[1:num_obs,],
                          splitrule = "taroneware", min_node_size = 200, seed = 2026)
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
source("testing/Articles/Helpers.r")

# read in the data
sim <- readRDS(data_path("sim.rds"))
test_data <- read.table(data_path("test_data.txt"), header = TRUE)

length(unique(unlist(lapply(sim, function(z) z$times))))
# 100000+ event and censoring times total for 50,000 observations!

# fit forest using num_obs observations with the best hyperparameters found above
# (the original number of event times was 50,000+, we reduce it to 1000, could probably do with even less)
num_obs <- 50000

tic()
final_forest <- jfforest(MM ~ ., data = sim[1:num_obs], feature_data = test_data[1:num_obs,], splitrule = "taroneware",
                         min_node_size = 200, seed = 2026, ntrees = 500, save_predictions = FALSE, num_event_times = 1000)
toc()   # only takes about 8 seconds to fit, 8 GB of ram usage though...
print_forest(final_forest)

# notes:
# - should probably increase the minimal node size, 200-250 would likely be more appropriate
# - with so many observations, all the good splits (the categorical features) will run out, meaning that at some point,
#   we are just fitting noise. The total number of combinations possible using the categorical covariates are 2 *4 * 5 = 40

# we know the initial distribution is c(1, 0, 0), so no need to predict it

# Add zero-valued noise covariates to the four signal profiles defined above.
new_data <- cbind(
    evaluation_profiles,
    X4 = rep(0, nrow(evaluation_profiles)),
    X5 = rep(0, nrow(evaluation_profiles)),
    X6 = rep(0, nrow(evaluation_profiles))
)
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
panel_titles <- evaluation_profile_titles
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
    file = plot_path("discrete_consistency_Markov.png")
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
    extract_matrix_entries(prediction, consistency_transition_indices, consistency_transition_names)
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
    file = plot_path("discrete_consistency_Markov_transition_rates.png")
)

# how to plot the errors? should the forest be modified to return the whole vector of scores?
# not relevant to plot errors here, this is just visualisation

# in principle we should test larger node sizes, but internal error prediction is too memory demanding

{
# load in data and fit benchmark jump forest to compare all the following models to
#--------------------------------------------------------------------------------

# import helper functions and packages
source("testing/Articles/Helpers.r")

# read in the data
sim <- readRDS(data_path("sim.rds"))
test_data <- read.table(data_path("test_data.txt"), header = TRUE)

# we choose to work with less data here and we drop the noise variables
num.obs <- 10000
forest_honest <- TRUE
forest_min_node_size <- 200
comparison_run_name <- paste0(
    "n", num.obs, "_mns", forest_min_node_size,
    if (isTRUE(forest_honest)) "_honest" else ""
)
analysis_file_suffix <- paste0(
    "_n", num.obs, if (isTRUE(forest_honest)) "_honest" else ""
)
comparison_plot_directory <- file.path(plot_directory, comparison_run_name)
dir.create(
    comparison_plot_directory, recursive = TRUE, showWarnings = FALSE
)
comparison_plot_path <- function(filename) {
    file.path(comparison_plot_directory, filename)
}
length(unique(unlist(lapply(sim[1:num.obs], function(z) z$times)))) # 21065

fitted_forest <- jfforest(MM ~ ., data = sim[1:num.obs], feature_data = test_data[1:num.obs, 1:3], splitrule = "taroneware",
                          min_node_size = forest_min_node_size, seed = 2026, ntrees = 500, save_predictions = FALSE, num_event_times = 1000,
                          honest = forest_honest)
print_forest(fitted_forest)

new_data <- evaluation_profiles
forest_predictions <- jfforest.predict(fitted_forest, new_data = new_data)

# Convert the forest Nelson--Aalen predictions to occupation probabilities.
comparison_forest_occprobs <- lapply(forest_predictions, function(prediction) {
    do.call(rbind, occprob_from_data(prediction, c(1, 0, 0)))
})
comparison_forest_times <- fitted_forest$unique.event.times

# fit and compare to the conditional Aalen-Johansen estimator
#--------------------------------------------------------------------------------

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

# Compute the true probabilities over the complete range covered by either estimator.
comparison_end <- max(comparison_forest_times, unlist(lapply(AJfits, function(fit) fit$t)))
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

comparison_titles <- evaluation_profile_titles
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
    file = comparison_plot_path("discrete_comparison_AJ_forest.png")
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
    file = comparison_plot_path(
        "discrete_comparison_AJ_forest_transition_rates.png"
    )
)

# Time-averaged integrated squared errors against the known DGP curves. The
# detailed tables retain every profile/state or profile/transition result; the
# summaries give each profile and component equal weight. These are oracle
# curve errors for this simulated data set (formal MISE would additionally
# average them over repeated simulated data sets). AJ estimates are carried
# forward from a profile's last observed time to the common plotting horizon,
# consistently with the step curves above.
caj_occupation_curve_errors <- rbind(
    curve_error_table(
        comparison_true_times, comparison_true_occprobs,
        comparison_forest_times, comparison_forest_occprobs,
        "Jump Forest", paste("State", 0:2), comparison_titles
    ),
    curve_error_table(
        comparison_true_times, comparison_true_occprobs,
        lapply(AJfits, `[[`, "t"), comparison_AJ_occprobs,
        "Conditional AJ", paste("State", 0:2), comparison_titles
    )
)

caj_transition_curve_errors <- rbind(
    curve_error_table(
        comparison_true_times, comparison_true_transition_rates,
        comparison_forest_times, comparison_forest_transition_rates,
        "Jump Forest", comparison_transition_names, comparison_titles
    ),
    curve_error_table(
        comparison_true_times, comparison_true_transition_rates,
        lapply(AJfits, `[[`, "t"), comparison_AJ_transition_rates,
        "Conditional AJ", comparison_transition_names, comparison_titles
    )
)

occupation_error_summary_caj <-
    summarise_curve_errors(caj_occupation_curve_errors)
transition_error_summary_caj <-
    summarise_curve_errors(caj_transition_curve_errors)
caj_occupation_curve_errors
caj_transition_curve_errors
occupation_error_summary_caj
transition_error_summary_caj

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
    file = comparison_plot_path("discrete_comparison_cox_forest.png")
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
    file = comparison_plot_path(
        "discrete_comparison_cox_forest_transition_rates.png"
    )
)

cox_occupation_curve_errors <- rbind(
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

cox_transition_curve_errors <- rbind(
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

occupation_error_summary_cox <- summarise_curve_errors(
    cox_occupation_curve_errors
)
transition_error_summary_cox <- summarise_curve_errors(
    cox_transition_curve_errors
)
occupation_error_summary_cox
transition_error_summary_cox

# The DGP contains time-varying effects of X2 and X3 (and additive hazard
# constants for three transitions), so neither Cox specification is correctly
# proportional. With interactions enabled, both specifications also omit the
# X2-by-X3 term. Cubic splines relax the separate covariate shapes, but neither
# that interaction nor the PH assumption.


# fit a Poisson regression model naively using stratified sampling as for the CAJ
#--------------------------------------------------------------------------------

# This comparison is deliberately disabled. With num.obs = 2000 the exact
# covariate strata are extremely small, and the full-sample Poisson regressions
# below are the relevant model-based comparison.
if (FALSE) {

# we fit the Poisson regressions using JumpPoisReg on each group of data like for the CAJ?

rows_by_profile <- lapply(seq_len(nrow(new_data)), function(i) {
    which(seq_len(nrow(test_data)) <= num.obs &
        test_data$X1 == new_data$X1[i] &
        test_data$X2 == new_data$X2[i] &
        test_data$X3 == new_data$X3[i])
})
lengths(rows_by_profile)
# note: very few observations in groups 2 and 4, probably only makes sense to compare groups 1 and 3

# some heuristics to choose the grids

lapply(sim[rows_by_profile[[1]]], function(z) z$times)

# some heuristic (total number of jumps from j to k in the data)
sum(unlist(lapply(sim[rows_by_profile[[1]]], function(z) count_transition_jumps(z, 1, 3, c(0, 10)))))
sum(unlist(lapply(sim[rows_by_profile[[2]]], function(z) count_transition_jumps(z, 1, 2, c(0, 90)))))
sum(unlist(lapply(sim[rows_by_profile[[3]]], function(z) count_transition_jumps(z, 1, 2, c(60, 90)))))

sqrt(300)   # about 17, so 90/15 is probably a suitable binwidth to aim for
pois_fit1 <- fit_markov_poisson(sim[rows_by_profile[[1]]], c(0, 10, 20, 30, 40, 50, 60, 90))
pois_fit3 <- fit_markov_poisson(sim[rows_by_profile[[3]]], c(0, 10, 20, 30, 40, 50, 60, 90))  # to avoid NAs, I had to reduce the number of bins
pois_fit1
pois_fit3

# fit the others anyway
pois_fit2 <- fit_markov_poisson(sim[rows_by_profile[[2]]], c(0, 30, 60, 90))
pois_fit4 <- fit_markov_poisson(sim[rows_by_profile[[4]]], c(0, 30, 60, 90))
pois_fit2
pois_fit4

# also make a benchmark fit without subsampling
pois_fit_all <- fit_markov_poisson(sim, c(seq(0, 60, 60/100), 65, 70, 80, 90))
pois_fit_all

rows_by_profile[[5]] <- 1:num.obs # if one wants to include the benchmark also (I decided to drop this)
pois_fits <- list(pois_fit1, pois_fit2, pois_fit3, pois_fit4, pois_fit_all)

# Evaluate every benchmark on the common horizon and DGP truth constructed in
# the CAJ section. The Poisson fits may have grids extending to 90, but extending
# the error horizon to 90 would change the Jump Forest error and make the model
# summaries incomparable.
if (comparison_end > min(vapply(pois_fits[seq_len(nrow(new_data))], function(fit) {
    tail(fit$t_grid, 1L)
}, numeric(1)))) {
    stop("the common comparison horizon exceeds a fitted Poisson grid")
}

# compute the cumulative transition rates for the fitted OE rates
#predict_markov_poisson(pois_fit1, comparison_true_times, "1->2")

cumulative_markov_poisson(pois_fit1, comparison_true_times)

# we only include the four prediction points (subsamples) for now
comparison_poisson_occprobs <- lapply(seq_len(nrow(new_data)), function(i) {
    do.call(rbind, occprob_from_data(cumulative_markov_poisson(pois_fits[[i]], comparison_true_times), c(1,0,0)))
})

comparison_titles <- evaluation_profile_titles
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
    'Poisson regression' = list(
        # the Poisson regression times are chosen to be the same as the true times
        times = comparison_true_times, values = comparison_poisson_occprobs,
        type = "l", lty = 3
    )
)

plot_panel_curves(
    comparison_occupation_curves, component_labels = paste("State", 0:2),
    panel_titles = comparison_titles, component_colours = comparison_colours,
    ylab = "Occupation probability", xlim = c(0, comparison_end),
    ylim = c(0, 1), panel_layout = c(2, 2), legend_position = "right",
    legend_order = c("Forest", "Poisson regression", "True")
)
plot_panel_curves(
    comparison_occupation_curves, component_labels = paste("State", 0:2),
    panel_titles = comparison_titles, component_colours = comparison_colours,
    ylab = "Occupation probability", xlim = c(0, comparison_end),
    ylim = c(0, 1), panel_layout = c(2, 2), legend_position = "right",
    legend_order = c("Forest", "Poisson regression", "True"),
    file = comparison_plot_path("discrete_comparison_poisson_forest.png")
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
comparison_poisson_transition_rates <- lapply(seq_len(nrow(new_data)), function(i) {
    extract_matrix_entries(
        cumulative_markov_poisson(pois_fits[[i]], comparison_true_times), comparison_transition_indices, comparison_transition_names
    )
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
    `Poisson regression` = list(
        times = comparison_true_times, values = comparison_poisson_transition_rates,
        type = "l", lty = 3
    )
)

plot_panel_curves(
    comparison_transition_curves, component_labels = comparison_transition_names,
    panel_titles = comparison_titles, component_colours = comparison_transition_colours,
    ylab = "Cumulative transition rate", xlim = c(0, comparison_end),
    include_zero = TRUE, panel_layout = c(2, 2), legend_position = "topleft",
    legend_order = c("Forest", "Poisson regression", "True")
)
plot_panel_curves(
    comparison_transition_curves, component_labels = comparison_transition_names,
    panel_titles = comparison_titles, component_colours = comparison_transition_colours,
    ylab = "Cumulative transition rate", xlim = c(0, comparison_end),
    include_zero = TRUE, panel_layout = c(2, 2), legend_position = "topleft",
    legend_order = c("Forest", "Poisson regression", "True"),
    file = comparison_plot_path(
        "discrete_comparison_poisson_forest_transition_rates.png"
    )
)

stratified_poisson_occupation_curve_errors <- rbind(
    curve_error_table(
        comparison_true_times, comparison_true_occprobs,
        comparison_forest_times, comparison_forest_occprobs,
        "Jump Forest", paste("State", 0:2), comparison_titles
    ),
    curve_error_table(
        comparison_true_times, comparison_true_occprobs,
        comparison_true_times, comparison_poisson_occprobs,
        "Stratified Poisson", paste("State", 0:2), comparison_titles
    )
)

stratified_poisson_transition_curve_errors <- rbind(
    curve_error_table(
        comparison_true_times, comparison_true_transition_rates,
        comparison_forest_times, comparison_forest_transition_rates,
        "Jump Forest", comparison_transition_names, comparison_titles
    ),
    curve_error_table(
        comparison_true_times, comparison_true_transition_rates,
        comparison_true_times, comparison_poisson_transition_rates,
        "Stratified Poisson", comparison_transition_names, comparison_titles
    )
)

occupation_error_summary_poisson <- summarise_curve_errors(
    stratified_poisson_occupation_curve_errors
)
transition_error_summary_poisson <- summarise_curve_errors(
    stratified_poisson_transition_curve_errors
)
occupation_error_summary_poisson
transition_error_summary_poisson
}

# fit Poisson regressions to all subjects and all covariates
#--------------------------------------------------------------------------------

# Unlike the exact-stratum analysis above, these models use every one of the
# same num.obs subjects used to train the Jump Forest. Each directed transition
# has its own piecewise-constant baseline and covariate coefficients, and exact
# time at risk enters the Poisson likelihood through offset(log(exposure)).
# This is the piecewise-exponential analogue of the transition-specific Cox
# models: the two specifications differ only in whether X2 and X3 enter linearly
# or through natural cubic splines.

# Use exactly the same horizon and truth grid as the CAJ and Cox comparisons.
# Keeping aliases with Poisson-specific names makes the code below readable
# while guaranteeing that the Jump Forest error is numerically identical.
poisson_regression_end <- comparison_end

# The second living state is reached only after a first jump, so its outgoing
# events are sparse near time zero. Pool years 0--10, use five-year bins through
# year 60, and pool the observed tail. The diagnostics printed below make the
# event counts and exposure behind this choice explicit and make alternative
# grids straightforward to assess.
poisson_regression_time_grid <- sort(unique(c(
    0,
    seq(10, 60, by = 5),
    poisson_regression_end
)))
poisson_regression_time_grid <- poisson_regression_time_grid[
    poisson_regression_time_grid >= 0 &
    poisson_regression_time_grid <= poisson_regression_end
]
if (tail(poisson_regression_time_grid, 1L) < poisson_regression_end) {
    poisson_regression_time_grid <- c(
        poisson_regression_time_grid, poisson_regression_end
    )
}

tic()
poisson_regression_data <- build_markov_poisson_data(
    paths = sim[seq_len(num.obs)],
    features = test_data[
        seq_len(num.obs), c("X1", "X2", "X3"), drop = FALSE
    ],
    time_grid = poisson_regression_time_grid,
    transitions = comparison_transition_indices
)
toc()   # about 2 seconds for n = 10000

poisson_regression_bin_diagnostics <- poisson_regression_data$diagnostics
poisson_regression_bin_diagnostics

poisson_regression_formulas <- list(
    linear = count ~ interval_factor + X1 + X2 + X3 +
        offset(log(exposure)),
    cubic_spline = count ~ interval_factor + X1 +
        splines::ns(X2, df = 3) + splines::ns(X3, df = 3) +
        offset(log(exposure))
)

tic()
poisson_regression_fits <- lapply(
    poisson_regression_formulas,
    function(model_formula) {
        fit_markov_poisson_regression(
            poisson_regression_data, model_formula
        )
    }
)
toc()   # well below one second for the grouped sufficient statistics (about 0.2-0.3 seconds)

poisson_regression_fit_diagnostics <- lapply(poisson_regression_fits, `[[`, "diagnostics")
poisson_regression_fit_diagnostics

# DGP sanity check only: fit one oracle linear Poisson model with X2:X3 and
# likelihood-ratio test it against the unchanged additive linear benchmark.
# The oracle is excluded from every prediction, error summary and comparison
# plot. Summing the two mortality-transition deviances gives a joint two-degree-
# of-freedom check that the generated interaction is empirically detectable.
if (interactions) {
    oracle_interaction_fit <- fit_markov_poisson_regression(
        poisson_regression_data,
        count ~ interval_factor + X1 + X2 + X3 + X2:X3 +
            offset(log(exposure))
    )
    interaction_signal_diagnostic <- do.call(rbind, Map(
        function(additive_fit, oracle_fit, transition) {
            likelihood_ratio_test <- stats::anova(
                additive_fit, oracle_fit, test = "Chisq"
            )
            data.frame(
                transition = transition,
                interaction_coefficient = unname(
                    stats::coef(oracle_fit)["X2:X3"]
                ),
                likelihood_ratio = likelihood_ratio_test$Deviance[2L],
                degrees_of_freedom = 1L,
                p_value = likelihood_ratio_test[["Pr(>Chi)"]][2L]
            )
        },
        poisson_regression_fits$linear$models,
        oracle_interaction_fit$models,
        names(oracle_interaction_fit$models)
    ))
    mortality_rows <- interaction_signal_diagnostic$transition %in%
        c("1->3", "2->3")
    joint_mortality_likelihood_ratio <- sum(
        interaction_signal_diagnostic$likelihood_ratio[mortality_rows]
    )
    interaction_signal_diagnostic <- rbind(
        interaction_signal_diagnostic,
        data.frame(
            transition = "mortality jointly",
            interaction_coefficient = NA_real_,
            likelihood_ratio = joint_mortality_likelihood_ratio,
            degrees_of_freedom = sum(mortality_rows),
            p_value = stats::pchisq(
                joint_mortality_likelihood_ratio,
                df = sum(mortality_rows), lower.tail = FALSE
            )
        )
    )
    write.table(
        interaction_signal_diagnostic,
        file = data_path(paste0(
            "interaction_signal_diagnostic", analysis_file_suffix, ".txt"
        )),
        sep = "\t", row.names = FALSE, quote = FALSE
    )
    interaction_signal_diagnostic
}

poisson_regression_true_times <- comparison_true_times
poisson_regression_true_occprobs <- comparison_true_occprobs
poisson_regression_true_transition_rates <- comparison_true_transition_rates

# Matrix exponentials give the exact occupation probabilities implied by each
# piecewise-constant fitted generator. Predictions are returned on the dense
# truth grid so curve_error_table() compares values directly rather than adding
# a coarse step-function approximation.
tic()
poisson_regression_predictions <- lapply(
    poisson_regression_fits,
    predict_markov_poisson_regression,
    new_data = new_data,
    times = poisson_regression_true_times,
    initial = c(1, 0, 0)
)
toc()

poisson_regression_occupation_curves <- list(
    True = list(
        times = poisson_regression_true_times,
        values = poisson_regression_true_occprobs,
        type = "l", lty = 1
    ),
    `Jump Forest` = list(
        times = comparison_forest_times,
        values = comparison_forest_occprobs,
        type = "s", lty = 2
    ),
    `Poisson linear` = list(
        times = poisson_regression_predictions$linear$times,
        values = poisson_regression_predictions$linear$occupation_probabilities,
        type = "l", lty = 3
    ),
    `Poisson cubic spline` = list(
        times = poisson_regression_predictions$cubic_spline$times,
        values = poisson_regression_predictions$cubic_spline$occupation_probabilities,
        type = "l", lty = 4
    )
)

plot_panel_curves(
    poisson_regression_occupation_curves,
    component_labels = paste("State", 0:2),
    panel_titles = comparison_titles,
    component_colours = comparison_colours,
    ylab = "Occupation probability",
    xlim = c(0, poisson_regression_end), ylim = c(0, 1),
    panel_layout = c(2, 2), legend_position = "right",
    legend_order = c(
        "Jump Forest", "Poisson linear", "Poisson cubic spline", "True"
    ),
    legend_cex = 0.8
)
plot_panel_curves(
    poisson_regression_occupation_curves,
    component_labels = paste("State", 0:2),
    panel_titles = comparison_titles,
    component_colours = comparison_colours,
    ylab = "Occupation probability",
    xlim = c(0, poisson_regression_end), ylim = c(0, 1),
    panel_layout = c(2, 2), legend_position = "right",
    legend_order = c(
        "Jump Forest", "Poisson linear", "Poisson cubic spline", "True"
    ),
    legend_cex = 0.8,
    file = comparison_plot_path(
        "discrete_comparison_poisson_regression_forest.png"
    )
)

poisson_regression_transition_curves <- list(
    True = list(
        times = poisson_regression_true_times,
        values = poisson_regression_true_transition_rates,
        type = "l", lty = 1
    ),
    `Jump Forest` = list(
        times = comparison_forest_times,
        values = comparison_forest_transition_rates,
        type = "s", lty = 2
    ),
    `Poisson linear` = list(
        times = poisson_regression_predictions$linear$times,
        values = poisson_regression_predictions$linear$cumulative_transition_rates,
        type = "l", lty = 3
    ),
    `Poisson cubic spline` = list(
        times = poisson_regression_predictions$cubic_spline$times,
        values = poisson_regression_predictions$cubic_spline$cumulative_transition_rates,
        type = "l", lty = 4
    )
)

plot_panel_curves(
    poisson_regression_transition_curves,
    component_labels = comparison_transition_names,
    panel_titles = comparison_titles,
    component_colours = comparison_transition_colours,
    ylab = "Cumulative transition rate",
    xlim = c(0, poisson_regression_end), include_zero = TRUE,
    panel_layout = c(2, 2), legend_position = "topleft",
    legend_order = c(
        "Jump Forest", "Poisson linear", "Poisson cubic spline", "True"
    ),
    legend_cex = 0.8
)
plot_panel_curves(
    poisson_regression_transition_curves,
    component_labels = comparison_transition_names,
    panel_titles = comparison_titles,
    component_colours = comparison_transition_colours,
    ylab = "Cumulative transition rate",
    xlim = c(0, poisson_regression_end), include_zero = TRUE,
    panel_layout = c(2, 2), legend_position = "topleft",
    legend_order = c(
        "Jump Forest", "Poisson linear", "Poisson cubic spline", "True"
    ),
    legend_cex = 0.8,
    file = comparison_plot_path(
        "discrete_comparison_poisson_regression_forest_transition_rates.png"
    )
)

poisson_regression_occupation_curve_errors <- rbind(
    curve_error_table(
        poisson_regression_true_times, poisson_regression_true_occprobs,
        comparison_forest_times, comparison_forest_occprobs,
        "Jump Forest", paste("State", 0:2), comparison_titles
    ),
    curve_error_table(
        poisson_regression_true_times, poisson_regression_true_occprobs,
        poisson_regression_predictions$linear$times,
        poisson_regression_predictions$linear$occupation_probabilities,
        "Poisson linear", paste("State", 0:2), comparison_titles
    ),
    curve_error_table(
        poisson_regression_true_times, poisson_regression_true_occprobs,
        poisson_regression_predictions$cubic_spline$times,
        poisson_regression_predictions$cubic_spline$occupation_probabilities,
        "Poisson cubic spline", paste("State", 0:2), comparison_titles
    )
)

poisson_regression_transition_curve_errors <- rbind(
    curve_error_table(
        poisson_regression_true_times,
        poisson_regression_true_transition_rates,
        comparison_forest_times, comparison_forest_transition_rates,
        "Jump Forest", comparison_transition_names, comparison_titles
    ),
    curve_error_table(
        poisson_regression_true_times,
        poisson_regression_true_transition_rates,
        poisson_regression_predictions$linear$times,
        poisson_regression_predictions$linear$cumulative_transition_rates,
        "Poisson linear", comparison_transition_names, comparison_titles
    ),
    curve_error_table(
        poisson_regression_true_times,
        poisson_regression_true_transition_rates,
        poisson_regression_predictions$cubic_spline$times,
        poisson_regression_predictions$cubic_spline$cumulative_transition_rates,
        "Poisson cubic spline", comparison_transition_names, comparison_titles
    )
)

occupation_error_summary_poisson_regression <- summarise_curve_errors(
    poisson_regression_occupation_curve_errors
)
transition_error_summary_poisson_regression <- summarise_curve_errors(
    poisson_regression_transition_curve_errors
)
poisson_regression_occupation_curve_errors
poisson_regression_transition_curve_errors
occupation_error_summary_poisson_regression
transition_error_summary_poisson_regression

# final summaries with the MISE and RMSE
#--------------------------------------------------------------------------------

# Collect every benchmark in common detailed and summary tables. The explicit
# checks make a future horizon/grid mismatch fail immediately instead of
# silently reporting several different errors for the same Jump Forest.
forest_occupation_error_tables <- list(
    CAJ = subset(caj_occupation_curve_errors, model == "Jump Forest"),
    Cox = subset(cox_occupation_curve_errors, model == "Jump Forest"),
    poisson_regression = subset(
        poisson_regression_occupation_curve_errors,
        model == "Jump Forest"
    )
)
forest_transition_error_tables <- list(
    CAJ = subset(caj_transition_curve_errors, model == "Jump Forest"),
    Cox = subset(cox_transition_curve_errors, model == "Jump Forest"),
    poisson_regression = subset(
        poisson_regression_transition_curve_errors,
        model == "Jump Forest"
    )
)

same_curve_errors <- function(error_tables) {
    reference <- error_tables[[1L]][
        c("profile", "component", "MISE", "RMSE")
    ]
    all(vapply(error_tables[-1L], function(candidate) {
        isTRUE(all.equal(
            reference,
            candidate[c("profile", "component", "MISE", "RMSE")],
            check.attributes = FALSE
        ))
    }, logical(1)))
}
if (!same_curve_errors(forest_occupation_error_tables) ||
    !same_curve_errors(forest_transition_error_tables)) {
    stop("Jump Forest errors differ across benchmark comparisons")
}

occupation_curve_errors_all <- rbind(
    forest_occupation_error_tables$CAJ,
    subset(caj_occupation_curve_errors, model != "Jump Forest"),
    subset(cox_occupation_curve_errors, model != "Jump Forest"),
    subset(
        poisson_regression_occupation_curve_errors,
        model != "Jump Forest"
    )
)
transition_curve_errors_all <- rbind(
    forest_transition_error_tables$CAJ,
    subset(caj_transition_curve_errors, model != "Jump Forest"),
    subset(cox_transition_curve_errors, model != "Jump Forest"),
    subset(
        poisson_regression_transition_curve_errors,
        model != "Jump Forest"
    )
)
rownames(occupation_curve_errors_all) <- NULL
rownames(transition_curve_errors_all) <- NULL

occupation_error_summary_all <- summarise_curve_errors(
    occupation_curve_errors_all
)
transition_error_summary_all <- summarise_curve_errors(
    transition_curve_errors_all
)

all_curve_errors <- rbind(
    transform(
        occupation_curve_errors_all,
        curve_type = "Occupation probability"
    ),
    transform(
        transition_curve_errors_all,
        curve_type = "Cumulative transition rate"
    )
)
all_curve_errors <- all_curve_errors[
    c("curve_type", "model", "profile", "component", "MISE", "RMSE")
]
rownames(all_curve_errors) <- NULL

all_error_summaries <- rbind(
    transform(
        occupation_error_summary_all,
        curve_type = "Occupation probability"
    ),
    transform(
        transition_error_summary_all,
        curve_type = "Cumulative transition rate"
    )
)
all_error_summaries <- all_error_summaries[
    c("curve_type", "model", "MISE", "RMSE")
]
rownames(all_error_summaries) <- NULL

all_curve_errors
all_error_summaries

}

write.table(
    all_curve_errors,
    file = data_path(paste0("all_curve_errors", analysis_file_suffix, ".txt"))
)
write.table(
    all_error_summaries,
    file = data_path(paste0("all_error_summaries", analysis_file_suffix, ".txt"))
)

# The DGP has time-varying covariate effects. Consequently both Poisson models,
# like the Cox models, remain misspecified: splines relax the separate covariate
# shapes but do not add the X2-by-X3 term or make effects vary over time.

# Do not hard-code a winner: the study-specific rankings are recorded in
# all_error_summaries and the expected-score table below. In the interaction
# study, the key question is whether the forest improves on the unchanged
# additive linear and cubic-spline benchmarks across all four factorial cells.

# pointwise expected Brier, KL and spherical errors
#--------------------------------------------------------------------------------
{
# Score every estimator on the dense DGP grid. The forest and Cox curves
# are step functions, so their most recent prediction is carried forward. The
# resulting scores are theoretical expectations under the known DGP rather than
# empirical IPCW scores; this puts all five estimators on exactly the same basis.
expected_error_times <- comparison_true_times
number_of_score_states <- ncol(comparison_true_occprobs[[1L]])

# These are the defaults used by a Jump Forest fitted without state_weights:
# Brier and spherical errors average over states, whereas KL uses the ordinary
# categorical log score.
expected_error_weights <- list(
    Brier = rep(1 / number_of_score_states, number_of_score_states),
    KL = rep(1, number_of_score_states),
    Spherical = rep(1 / number_of_score_states, number_of_score_states)
)

normalise_score_probabilities <- function(probabilities, label) {
    probabilities <- as.matrix(probabilities)
    storage.mode(probabilities) <- "double"
    tolerance <- 1e-7

    if (ncol(probabilities) != number_of_score_states ||
        any(!is.finite(probabilities)) ||
        any(probabilities < -tolerance) ||
        any(probabilities > 1 + tolerance)) {
        stop(label, " contains invalid occupation probabilities")
    }

    # Remove negligible product-integral drift without concealing a material
    # failure of a prediction to be a probability vector.
    probabilities[probabilities < 0] <- 0
    probabilities[probabilities > 1] <- 1
    probability_totals <- rowSums(probabilities)
    if (any(probability_totals <= 0) ||
        any(abs(probability_totals - 1) > tolerance)) {
        stop(label, " contains rows whose probabilities do not sum to one")
    }
    sweep(probabilities, 1L, probability_totals, "/")
}

expected_error_truth <- lapply(
    seq_along(comparison_true_occprobs),
    function(profile_index) {
        normalise_score_probabilities(
            comparison_true_occprobs[[profile_index]],
            paste("DGP profile", profile_index)
        )
    }
)

expected_error_prediction_sets <- list(
    `Jump Forest` = list(
        times = comparison_forest_times,
        values = comparison_forest_occprobs
    ),
    `Poisson linear` = list(
        times = poisson_regression_predictions$linear$times,
        values = poisson_regression_predictions$linear$occupation_probabilities
    ),
    `Poisson cubic spline` = list(
        times = poisson_regression_predictions$cubic_spline$times,
        values = poisson_regression_predictions$cubic_spline$occupation_probabilities
    ),
    `Cox linear` = list(
        times = cox_predictions$linear$times,
        values = cox_predictions$linear$occupation_probabilities
    ),
    `Cox cubic spline` = list(
        times = cox_predictions$cubic_spline$times,
        values = cox_predictions$cubic_spline$occupation_probabilities
    )
)

align_expected_error_predictions <- function(prediction_set, model_name) {
    if (length(prediction_set$values) != length(expected_error_truth)) {
        stop(model_name, " does not contain one prediction per profile")
    }

    lapply(seq_along(expected_error_truth), function(profile_index) {
        profile_times <- if (is.list(prediction_set$times)) {
            prediction_set$times[[profile_index]]
        } else {
            prediction_set$times
        }
        aligned <- step_matrix_at(
            profile_times,
            prediction_set$values[[profile_index]],
            expected_error_times
        )
        normalise_score_probabilities(
            aligned,
            paste(model_name, "profile", profile_index)
        )
    })
}

expected_error_predictions <- lapply(
    names(expected_error_prediction_sets),
    function(model_name) {
        align_expected_error_predictions(
            expected_error_prediction_sets[[model_name]], model_name
        )
    }
)
names(expected_error_predictions) <- names(expected_error_prediction_sets)

# Return one column per state, matching the state-wise score vectors stored by
# JumpForests. Summing the columns gives the usual total expected score. The
# Bayes act minimises that sum; an individual state contribution can still fall
# below its Bayes-act contribution if another state pays for the improvement.
expected_score_components <- function(truth, prediction, metric, weights) {
    if (!identical(dim(truth), dim(prediction))) {
        stop("truth and prediction must have matching dimensions")
    }

    if (metric == "Brier") {
        unweighted <- truth * (1 - prediction)^2 +
            (1 - truth) * prediction^2
        return(sweep(unweighted, 2L, weights, "*"))
    }

    if (metric == "KL") {
        # This is the same truncation used by probabilityForLogScore() in the
        # C++ score implementation. Multiplication by truth makes a state with
        # zero true probability contribute zero.
        prediction_for_log <- prediction
        prediction_for_log[] <- pmin(
            1 - 1e-15, pmax(1e-15, prediction_for_log)
        )
        return(-sweep(
            truth * log(prediction_for_log), 2L, weights, "*"
        ))
    }

    if (metric == "Spherical") {
        denominator <- sqrt(rowSums(sweep(
            prediction^2, 2L, weights, "*"
        )))
        if (any(!is.finite(denominator)) || any(denominator <= 0)) {
            stop("spherical-score prediction has a zero or invalid norm")
        }
        weighted_prediction <- sweep(prediction, 2L, weights, "*")
        reward <- sweep(weighted_prediction, 1L, denominator, "/")
        return(truth * (1 - reward))
    }

    stop("unknown expected-score metric: ", metric)
}

# Compute the Bayes-act score by state. The helper functions provide the total
# optimal expected errors, so check that the state decomposition sums to those
# totals. optimal_KL() evaluates 0 * log(0) as NaN at boundary probabilities;
# its mathematically continuous value is supplied by an explicit zero term.
optimal_expected_score <- function(truth, metric, weights) {
    optimal_prediction <- truth
    if (metric == "KL") {
        weighted_truth <- sweep(truth, 2L, weights, "*")
        normalising_constant <- rowSums(weighted_truth)
        if (any(normalising_constant <= 0)) {
            stop("KL weights give the truth zero total mass")
        }
        optimal_prediction <- sweep(
            weighted_truth, 1L, normalising_constant, "/"
        )
    }

    if (metric == "KL") {
        # Unlike model scores, the theoretical optimum does not need numerical
        # log-score truncation: define each zero-mass logarithmic term as zero.
        log_optimal_prediction <- matrix(0, nrow(truth), ncol(truth))
        positive_prediction <- optimal_prediction > 0
        log_optimal_prediction[positive_prediction] <- log(optimal_prediction[positive_prediction])
        components <- -weighted_truth * log_optimal_prediction
    } else {
        components <- expected_score_components(
            truth, optimal_prediction, metric, weights
        )
    }
    truth_as_list <- lapply(seq_len(nrow(truth)), function(time_index) {
        truth[time_index, ]
    })
    helper_totals <- switch(
        metric,
        Brier = optimal_brier(weights, truth_as_list),
        KL = optimal_KL(weights, truth_as_list),
        Spherical = optimal_spherical(weights, truth_as_list)
    )
    helper_totals <- as.numeric(unlist(helper_totals, use.names = FALSE))
    component_totals <- rowSums(components)
    boundary_nan <- is.nan(helper_totals)

    if (any(is.infinite(helper_totals)) ||
        any(is.na(helper_totals) & !boundary_nan)) {
        stop(metric, " optimal-score helper returned an unexpected non-finite value")
    }

    if (any(abs(
        helper_totals[!boundary_nan] -
        component_totals[!boundary_nan]
    ) > 1e-10)) {
        stop(metric, " optimal score does not agree with its helper function")
    }
    helper_totals[boundary_nan] <- component_totals[boundary_nan]

    list(components = components, total = helper_totals)
}

# Curves are organised as metric -> method -> profile -> time-by-state matrix.
# Keeping the four profiles separate avoids imposing an arbitrary profile
# weighting and lets every output figure use the same 2-by-2 panel structure as
# the earlier model-comparison plots.
expected_error_curves <- setNames(
    vector("list", length(expected_error_weights)),
    names(expected_error_weights)
)
optimal_expected_error_totals <- expected_error_curves

for (metric in names(expected_error_weights)) {
    metric_weights <- expected_error_weights[[metric]]
    optimal_scores <- lapply(expected_error_truth, function(truth) {
        optimal_expected_score(truth, metric, metric_weights)
    })

    metric_curves <- list(
        `Optimal expected` = lapply(optimal_scores, `[[`, "components")
    )
    for (model_name in names(expected_error_predictions)) {
        metric_curves[[model_name]] <- Map(
            function(truth, prediction) {
                expected_score_components(
                    truth, prediction, metric, metric_weights
                )
            },
            expected_error_truth,
            expected_error_predictions[[model_name]]
        )
    }

    expected_error_curves[[metric]] <- metric_curves
    optimal_expected_error_totals[[metric]] <-
        lapply(optimal_scores, `[[`, "total")
}

expected_error_linetypes <- c(
    `Optimal expected` = "solid",
    `Jump Forest` = "dashed",
    `Poisson linear` = "dotted",
    `Poisson cubic spline` = "dotdash",
    `Cox linear` = "longdash",
    `Cox cubic spline` = "twodash"
)

expected_error_plot_files <- character()
for (metric in names(expected_error_curves)) {
    metric_curves <- expected_error_curves[[metric]]

    for (state_index in seq_len(number_of_score_states)) {
        state_label <- paste("State", state_index - 1L)
        plot_curve_sets <- lapply(names(metric_curves), function(method_name) {
            list(
                times = expected_error_times,
                values = lapply(metric_curves[[method_name]], function(values) {
                    values[, state_index, drop = FALSE]
                }),
                type = "l",
                lty = unname(expected_error_linetypes[method_name]),
                lwd = if (method_name == "Optimal expected") 2.5 else 1.5
            )
        })
        names(plot_curve_sets) <- names(metric_curves)

        maximum_error <- max(unlist(lapply(metric_curves, function(profiles) {
            vapply(profiles, function(values) {
                max(values[, state_index])
            }, numeric(1))
        })))
        error_ylim <- c(0, if (maximum_error > 0) 1.04 * maximum_error else 0.5)

        plot_key <- paste(tolower(metric), state_index - 1L, sep = "_state_")
        plot_file <- comparison_plot_path(
            paste0("discrete_expected_", plot_key, ".png")
        )
        expected_error_plot_files[plot_key] <- plot_file

        plot_panel_curves(
            plot_curve_sets,
            component_labels = state_label,
            panel_titles = comparison_titles,
            component_colours = comparison_colours[state_index],
            ylab = paste("Expected", metric, "error -", state_label),
            xlim = c(0, comparison_end), ylim = error_ylim,
            panel_layout = c(2, 2), legend_position = "topright",
            legend_order = names(metric_curves), legend_cex = 0.65,
            file = plot_file, width = 11, height = 6.5
        )
    }
}

# Store every pointwise state contribution in one long table. This is the data
# underlying the nine plots and retains the metric, model, profile, state and
# evaluation time, so runs with different num.obs can be combined directly.
time_dependent_error_rows <- vector(
    "list",
    length(expected_error_curves) *
        length(expected_error_curves[[1L]]) *
        length(comparison_titles)
)
time_dependent_error_row <- 0L
state_labels <- paste("State", seq_len(number_of_score_states) - 1L)

for (metric in names(expected_error_curves)) {
    for (model_name in names(expected_error_curves[[metric]])) {
        for (profile_index in seq_along(comparison_titles)) {
            time_dependent_error_row <- time_dependent_error_row + 1L
            values <- expected_error_curves[[metric]][[model_name]][[profile_index]]
            time_dependent_error_rows[[time_dependent_error_row]] <- data.frame(
                num_obs = num.obs,
                metric = metric,
                model = model_name,
                profile = comparison_titles[profile_index],
                state = rep(state_labels, each = length(expected_error_times)),
                time = rep(expected_error_times, times = number_of_score_states),
                expected_error = as.vector(values),
                stringsAsFactors = FALSE
            )
        }
    }
}
time_dependent_expected_errors <- do.call(
    rbind, time_dependent_error_rows
)
rownames(time_dependent_expected_errors) <- NULL

# Match JumpForests' normalised integrated multi-state scores: first sum the
# state contributions, integrate over time by the trapezoidal rule, and divide
# by the evaluation horizon. The final summary gives the four profiles equal
# weight, consistently with the earlier benchmark summaries.
normalised_integrated_score <- function(score_components) {
    total_score <- rowSums(score_components)
    horizon <- tail(expected_error_times, 1L) - expected_error_times[1L]
    if (!is.finite(horizon) || horizon <= 0) {
        stop("expected_error_times must cover a positive finite horizon")
    }
    sum(
        (head(total_score, -1L) + tail(total_score, -1L)) *
            diff(expected_error_times) / 2
    ) / horizon
}

normalised_integrated_score_rows <- vector(
    "list",
    length(expected_error_curves) * length(expected_error_curves[[1L]])
)
normalised_integrated_score_row <- 0L
normalised_integrated_scores_by_profile <- list()

for (metric in names(expected_error_curves)) {
    for (model_name in names(expected_error_curves[[metric]])) {
        profile_scores <- vapply(
            expected_error_curves[[metric]][[model_name]],
            normalised_integrated_score,
            numeric(1)
        )
        normalised_integrated_scores_by_profile[[paste(metric, model_name)]] <-
            data.frame(
                num_obs = num.obs,
                metric = metric,
                model = model_name,
                profile = comparison_titles,
                normalised_integrated_score = profile_scores,
                stringsAsFactors = FALSE
            )

        normalised_integrated_score_row <-
            normalised_integrated_score_row + 1L
        normalised_integrated_score_rows[[normalised_integrated_score_row]] <-
            data.frame(
                num_obs = num.obs,
                metric = metric,
                model = model_name,
                normalised_integrated_score = mean(profile_scores),
                stringsAsFactors = FALSE
            )
    }
}
normalised_integrated_scores_by_profile <- do.call(
    rbind, normalised_integrated_scores_by_profile
)
rownames(normalised_integrated_scores_by_profile) <- NULL
normalised_integrated_score_summary <- do.call(
    rbind, normalised_integrated_score_rows
)
rownames(normalised_integrated_score_summary) <- NULL

expected_error_table_files <- c(
    time_dependent = data_path(
        paste0("time_dependent_expected_errors", analysis_file_suffix, ".txt")
    ),
    summary = data_path(
        paste0(
            "normalised_integrated_expected_scores",
            analysis_file_suffix, ".txt"
        )
    )
)
write.table(
    time_dependent_expected_errors,
    file = expected_error_table_files["time_dependent"],
    sep = "\t", row.names = FALSE, quote = FALSE
)
write.table(
    normalised_integrated_score_summary,
    file = expected_error_table_files["summary"],
    sep = "\t", row.names = FALSE, quote = FALSE
)

expected_error_plot_files
normalised_integrated_score_summary
expected_error_table_files
}

normalised_integrated_score_summary

#nolint end

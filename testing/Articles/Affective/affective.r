#nolint start: line_length_linter

# This file is for analysing the Recurrent episodes in affective disorders data from https://multi-state-book.github.io/companion/Ch1.html

# import helper functions and packages
source("testing/Articles/Helpers.r")
library(mstate)
library(tidyverse)
library(etm)

# data preparation
#--------------------------------------------------------------------------------
{
affective <- read.csv("testing/Articles/Affective/affective.csv", sep = ",")
head(affective)
subset(affective, id == 5)
nrow(subset(affective, id == 117))

# 0: 
# 1: hospital
# 2: death
# 3: censoring

# prepare the jump data
ids <- unique(affective$id)
jump_data <- list()
for (i in seq_along(ids)) {
    # extract the lines in the dataset corresponding to the particular patient
    patient_data <- subset(affective, id == ids[i])
    
    # initial time is always zero in the data (as we would hope) and start is always equal
    # to stop except for the initial 0.0 starting time and the final time. The state at the
    # final time is determined by status
    times <- as.numeric(c(0.0, patient_data$stop))
    states <- as.integer(c(patient_data$state[1], patient_data$status))
    
    # make censoring a repeated state as required by Jump Forests
    if (tail(states, 1) == 3) {
        states[length(states)] <- states[length(states) - 1]
    }

    # finally, the Jump Forest package expects the numbering of the states to be 1, 2, ...
    states <- states + 1L

    jump_data[[i]] <- list(times = times, states = states)
}

# ensure that the feature data matches the ordering of the patients in jump_data
feature_data <- affective[match(ids, affective$id), c("bip", "sex", "age", "year"), drop = FALSE]
}

length(jump_data)   # 119 observations
nrow(feature_data)  # 119, concurs

# censoring rate
mean(as.numeric(unlist(lapply(jump_data, function(z) z$states[length(z$states)] == z$states[length(z$states) - 1]))))   # 0.3445378

# tune the jump forest
#--------------------------------------------------------------------------------

# as a benchmark, simply fit the NA estimator with no covariates
null_forest <- jfforest(MM ~ 1, data = jump_data, feature_data = feature_data, min_node_size = 10, seed = 2026)
print_forest(null_forest)   # already better than random guessing

min_node_sizes <- c(2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50)
split_rules <- c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
num_min_node_sizes <- length(min_node_sizes)
num_split_rules <- length(split_rules)

ibs <- matrix(rep(0, num_min_node_sizes * num_split_rules), nrow = num_split_rules)
kl <- matrix(rep(0, num_min_node_sizes * num_split_rules), nrow = num_split_rules)
spherical <- matrix(rep(0, num_min_node_sizes * num_split_rules), nrow = num_split_rules)
# row: spliting rule, column: minimal node size

for (i in 1:num_split_rules) {
    for (j in 1:num_min_node_sizes) {
        fitted_forest <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data, splitrule = split_rules[i], 
                                  min_node_size = min_node_sizes[j], seed = 2026)
        ibs[i, j] <- fitted_forest$ibs.normalised
        kl[i, j] <- fitted_forest$ikl.normalised
        spherical[i, j] <- fitted_forest$is.normalised

        # free memory along the way
        rm('fitted_forest')
        gc()

        cat("Finished fitting forest", (i - 1) * num_min_node_sizes + j, "out of", num_min_node_sizes * num_split_rules ,"\n")
    }
}

best_tuning_configuration <- function(scores, metric) {
    best <- which(scores == min(scores), arr.ind = TRUE)[1, ]
    data.frame(metric = metric, split_rule = split_rules[best["row"]], min_node_size = min_node_sizes[best["col"]], 
               score = scores[best["row"], best["col"]])
}
best_tuning_results <- rbind(
    best_tuning_configuration(ibs, "IBS"),
    best_tuning_configuration(kl, "KL"),
    best_tuning_configuration(spherical, "Spherical")
)
best_tuning_results
jfforest.error(null_forest)

# for plots later
write.table(ibs, file = "testing/Articles/Affective/Data/tuning_ibs.txt", sep = "\t", row.names = FALSE)
write.table(kl, file = "testing/Articles/Affective/Data/tuning_kl.txt", sep = "\t", row.names = FALSE)
write.table(spherical, file = "testing/Articles/Affective/Data/tuning_spherical.txt", sep = "\t", row.names = FALSE)

# now make hyperparameter tuning plots
ibs <- read.table("testing/Articles/Affective/tuning_ibs.txt", header = TRUE)
colnames(ibs) <- as.factor(min_node_sizes)
ikl <- read.table("testing/Articles/Affective/tuning_kl.txt", header = TRUE)
colnames(ikl) <- as.factor(min_node_sizes)
spherical <- read.table("testing/Articles/Affective/tuning_spherical.txt", header = TRUE)
colnames(spherical) <- as.factor(min_node_sizes)

tuning_colours <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")

# save plots
plot_score_curves(
    scores = ibs, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(ibs), colours = tuning_colours, legend_position = "bottomright",
    pch = seq_along(split_rules), ylab = "Normalised IBS", width = 6, height = 6,
    file = "testing/Articles/Affective/Plots/tuning_ibs.png"
)
plot_score_curves(
    scores = ikl, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(ikl), colours = tuning_colours, legend_position = "bottomright",
    pch = seq_along(split_rules), ylab = "Normalised IKL", width = 6, height = 6,
    file = "testing/Articles/Affective/Plots/tuning_ikl.png"
)
plot_score_curves(
    scores = spherical, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(spherical), colours = tuning_colours, legend_position = "bottomright",
    pch = seq_along(split_rules), ylab = "Normalised Integrated spherical error", width = 6, height = 6,
    file = "testing/Articles/Affective/Plots/tuning_spherical.png"
)

# based on the plots, min_node_size = 10 seems best with taroneware splitting
fitted_forest <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data, splitrule = "taroneware", min_node_size = 10, seed = 2026)
print_forest(fitted_forest)

# VIMP
#--------------------------------------------------------------------------------

B <- 500
vimp_permute_brier <- matrix(0, nrow = B, ncol = 4)
vimp_permute_kl <- matrix(0, nrow = B, ncol = 4)
vimp_permute_spherical <- matrix(0, nrow = B, ncol = 4)
vimp_random_brier <- matrix(0, nrow = B, ncol = 4)
vimp_random_kl <- matrix(0, nrow = B, ncol = 4)
vimp_random_spherical <- matrix(0, nrow = B, ncol = 4)

set.seed(67)
tic()
for (b in 1:B) {
    vimp_permute_brier[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "brier")$vimp))
    vimp_permute_kl[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "kl")$vimp))
    vimp_permute_spherical[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "spherical")$vimp))
    vimp_random_brier[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "random", loss = "brier")$vimp))
    vimp_random_kl[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "random", loss = "kl")$vimp))
    vimp_random_spherical[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "random", loss = "spherical")$vimp))
    cat("Finished iteration", b, "of",B, "\n")
}
toc()   # takes close to 20 minutes for B = 500

colnames(vimp_permute_brier) <- c("bip", "sex", "age", "year")
colnames(vimp_permute_kl) <- c("bip", "sex", "age", "year")
colnames(vimp_permute_spherical) <- c("bip", "sex", "age", "year")
colnames(vimp_random_brier) <- c("bip", "sex", "age", "year")
colnames(vimp_random_kl) <- c("bip", "sex", "age", "year")
colnames(vimp_random_spherical) <- c("bip", "sex", "age", "year")

# save results of each run
write.table(vimp_permute_brier, "testing/Articles/Affective/Data/vimp_permute_brier_affective.txt", sep = "\t", row.names = FALSE)
write.table(vimp_permute_kl, "testing/Articles/Affective/Data/vimp_permute_kl_affective.txt", sep = "\t", row.names = FALSE)
write.table(vimp_permute_spherical, "testing/Articles/Affective/Data/vimp_permute_spherical_affective.txt", sep = "\t", row.names = FALSE)
write.table(vimp_random_brier, "testing/Articles/Affective/Data/vimp_random_brier_affective.txt", sep = "\t", row.names = FALSE)
write.table(vimp_random_kl, "testing/Articles/Affective/Data/vimp_random_kl_affective.txt", sep = "\t", row.names = FALSE)
write.table(vimp_random_spherical, "testing/Articles/Affective/Data/vimp_random_spherical_affective.txt", sep = "\t", row.names = FALSE)

# load the results in here
vimp_permute_brier <- read.table("testing/Articles/Affective/vimp_permute_brier_affective.txt", header = TRUE)
vimp_permute_kl <- read.table("testing/Articles/Affective/vimp_permute_kl_affective.txt", header = TRUE)
vimp_permute_spherical <- read.table("testing/Articles/Affective/vimp_permute_spherical_affective.txt", header = TRUE)
vimp_random_brier <- read.table("testing/Articles/Affective/vimp_random_brier_affective.txt", header = TRUE)
vimp_random_kl <- read.table("testing/Articles/Affective/vimp_random_kl_affective.txt", header = TRUE)
vimp_random_spherical <- read.table("testing/Articles/Affective/vimp_random_spherical_affective.txt", header = TRUE)

save <- TRUE
# show plots
plot_vimp("permute", "brier", vimp_permute_brier, path = "testing/Articles/Affective/Plots")
plot_vimp("permute", "kl", vimp_permute_kl, path = "testing/Articles/Affective/Plots")
plot_vimp("permute", "spherical", vimp_permute_spherical, path = "testing/Articles/Affective/Plots")
plot_vimp("random", "brier", vimp_random_brier, path = "testing/Articles/Affective/Plots")
plot_vimp("random", "kl", vimp_random_kl, path = "testing/Articles/Affective/Plots")
plot_vimp("random", "spherical", vimp_random_spherical, path = "testing/Articles/Affective/Plots")

colMeans(vimp_permute_brier)
colMeans(vimp_permute_kl)
colMeans(vimp_permute_spherical)
colMeans(vimp_random_brier)
colMeans(vimp_random_kl)
colMeans(vimp_random_spherical)

# age is certainly the most important variable followed by bip, sex is also somewhat important while year is pretty negligible

# a few exploratory plots?
#--------------------------------------------------------------------------------

# from the VIMP analysis, it seems that age and bip are interesting to study for potential interactions
head(feature_data)
summary(feature_data$age)
new_data <- data.frame(bip = c(1,1,1,0,0,0), age = c(32, 49, 61.5, 32, 49, 61.5), sex = rep(0, 6), year = rep(mean(feature_data$year), 6))
age_bip_predictions <- jfforest.predict(fitted_forest, new_data)

plot_transition_rate(
    predictions = age_bip_predictions,
    event_times = fitted_forest$unique.event.times,
    new_data = new_data,
    from = 2,
    to = 1,
    colour_by = "age",
    linetype_by = "bip",
    max_time = NULL,
    xlab = "Time (months)",
    plot_directory = "testing/Articles/Affective/Plots",
    save = TRUE
)

# there does not seem to be any clear patterns in regards to interactions

# compare to a Cox model
#--------------------------------------------------------------------------------

{

fit_simple_affective_cox <- function(paths, features, ties = "breslow") {
    cox_12_data <- paths_to_transition_cox_data(paths, features, 1L, 2L)
    cox_21_data <- paths_to_transition_cox_data(paths, features, 2L, 1L)
    cox_13_data <- paths_to_transition_cox_data(paths, features, 1L, 3L)
    cox_23_data <- paths_to_transition_cox_data(paths, features, 2L, 3L)

    fit_transition <- function(data) {
        survival::coxph(
            survival::Surv(tstart, tstop, event) ~ bip + sex + age + year,
            data = data, ties = ties,
            model = TRUE, x = TRUE, y = TRUE, singular.ok = FALSE
        )
    }

    fits <- list(
        `1 -> 2` = fit_transition(cox_12_data),
        `2 -> 1` = fit_transition(cox_21_data),
        `1 -> 3` = fit_transition(cox_13_data),
        `2 -> 3` = fit_transition(cox_23_data)
    )

    model <- new_multistate_cox_model(
        transitions = list(
            `1 -> 2` = list(from = 1L, to = 2L, fit = fits[["1 -> 2"]], fixed_values = list()),
            `2 -> 1` = list(from = 2L, to = 1L, fit = fits[["2 -> 1"]], fixed_values = list()),
            `1 -> 3` = list(from = 1L, to = 3L, fit = fits[["1 -> 3"]], fixed_values = list()),
            `2 -> 3` = list(from = 2L, to = 3L, fit = fits[["2 -> 3"]], fixed_values = list())
        ),
        number_of_states = 3L,
        label = "Simple Cox"
    )
    model$fits <- fits
    model$sample_sizes <- data.frame(
        transition = names(fits),
        risk_intervals = c(nrow(cox_12_data), nrow(cox_21_data), nrow(cox_13_data), nrow(cox_23_data)),
        events = c(sum(cox_12_data$event), sum(cox_21_data$event), sum(cox_13_data$event), sum(cox_23_data$event))
    )
    model
}

fit_interaction_affective_cox <- function(paths, features, ties = "breslow") {
    cox_12_data <- paths_to_transition_cox_data(paths, features, 1L, 2L)
    cox_21_data <- paths_to_transition_cox_data(paths, features, 2L, 1L)
    cox_13_data <- paths_to_transition_cox_data(paths, features, 1L, 3L)
    cox_23_data <- paths_to_transition_cox_data(paths, features, 2L, 3L)

    # Only two observed 2 -> 3 events occur in bipolar patients. Penalising the
    # age:bip term prevents separation in the cross-fitting folds while leaving
    # all main effects unpenalised.
    fit_transition <- function(data) {
        survival::coxph(
            survival::Surv(tstart, tstop, event) ~
                bip + sex + age + year +
                survival::ridge(I(age * bip), theta = 1),
            data = data, ties = ties,
            model = TRUE, x = TRUE, y = TRUE, singular.ok = FALSE
        )
    }

    fits <- list(
        `1 -> 2` = fit_transition(cox_12_data),
        `2 -> 1` = fit_transition(cox_21_data),
        `1 -> 3` = fit_transition(cox_13_data),
        `2 -> 3` = fit_transition(cox_23_data)
    )

    model <- new_multistate_cox_model(
        transitions = list(
            `1 -> 2` = list(from = 1L, to = 2L, fit = fits[["1 -> 2"]], fixed_values = list()),
            `2 -> 1` = list(from = 2L, to = 1L, fit = fits[["2 -> 1"]], fixed_values = list()),
            `1 -> 3` = list(from = 1L, to = 3L, fit = fits[["1 -> 3"]], fixed_values = list()),
            `2 -> 3` = list(from = 2L, to = 3L, fit = fits[["2 -> 3"]], fixed_values = list())
        ),
        number_of_states = 3L,
        label = "Interaction Cox"
    )
    model$fits <- fits
    model$sample_sizes <- data.frame(
        transition = names(fits),
        risk_intervals = c(nrow(cox_12_data), nrow(cox_21_data), nrow(cox_13_data), nrow(cox_23_data)),
        events = c(sum(cox_12_data$event), sum(cox_21_data$event), sum(cox_13_data$event), sum(cox_23_data$event))
    )
    model
}

simple_cox_model <- fit_simple_affective_cox(jump_data, feature_data)
interaction_cox_model <- fit_interaction_affective_cox(jump_data, feature_data)
}

simple_cox_model$sample_sizes
interaction_cox_model$sample_sizes
lapply(simple_cox_model$fits, summary)
lapply(interaction_cox_model$fits, summary)

# interesting, no variables are significant on a 5% level for the simple Cox model, although age is close at 0.0721
# for the interaction model with penalisation, age is the only significant variable (p = 0.029), and the interaction
# between age and bip is at 0.053, so borderline significant

# error curves
#--------------------------------------------------------------------------------

# As for EBMT, these are empirical IPCW errors. All three model specifications
# use the same validation folds and marginal reverse-KM censoring distribution.
{

minimum_error_censoring_survival <- 0.05
error_evaluation_times <- sort(unique(c(fitted_forest$unique.event.times, vapply(jump_data, function(path) tail(path$times, 1L), numeric(1)))))
error_censoring_model <- reverse_km_censoring(jump_data)
error_censoring_survival <- censoring_survival_at(error_censoring_model, error_evaluation_times)
error_evaluation_times <- error_evaluation_times[error_censoring_survival >= minimum_error_censoring_survival]

# Every individual in this dataset starts in state 2 (current episode).
initial_distribution <- c(0, 1, 0)
number_of_error_folds <- 5L
error_folds <- stratified_multistate_folds(jump_data, number_of_error_folds, seed = 2026)

# The selected forest hyperparameters are held fixed in the outer folds. A
# fully nested comparison would repeat tuning within each training fold.
jumpforest_crossfit_occupation <- crossfit_jumpforest(
    jump_data,
    feature_data,
    initial = initial_distribution,
    folds = error_folds,
    seed = 2026,
    splitrule = fitted_forest$splitrule,
    min_node_size = fitted_forest$min.node.size,
    ntrees = fitted_forest$num.trees
)
simple_cox_crossfit_occupation <- crossfit_multistate_cox(
    jump_data, feature_data, fit_simple_affective_cox,
    evaluation_times = error_evaluation_times,
    number_of_folds = number_of_error_folds,
    seed = 2026,
    initial = initial_distribution,
    prediction_stype = "exponential",
    folds = error_folds
)
interaction_cox_crossfit_occupation <- crossfit_multistate_cox(
    jump_data, feature_data, fit_interaction_affective_cox,
    evaluation_times = error_evaluation_times,
    number_of_folds = number_of_error_folds,
    seed = 2026,
    initial = initial_distribution,
    prediction_stype = "exponential",
    folds = error_folds
)

error_predictions <- list(
    JumpForest = jumpforest_crossfit_occupation,
    `Simple Cox` = simple_cox_crossfit_occupation,
    `Interaction Cox` = interaction_cox_crossfit_occupation
)

}

affective_error_comparison <- multistate_score_curves(
    error_predictions,
    paths = jump_data,
    evaluation_times = error_evaluation_times,
    minimum_censoring_survival = minimum_error_censoring_survival
)
affective_error_plots <- plot_multistate_error_curves(
    affective_error_comparison,
    xlab = "Time (months)",
    save = TRUE,
    plot_directory = "testing/Articles/Affective/Plots"
)

write.table(
    affective_error_comparison$curves,
    "testing/Articles/Affective/Data/time_dependent_errors_affective.txt",
    sep = "\t", row.names = FALSE
)
write.table(
    affective_error_comparison$integrated,
    "testing/Articles/Affective/Data/integrated_errors_affective.txt",
    sep = "\t", row.names = FALSE
)
affective_error_comparison$integrated

# again they are very close, but the Jump Forest is actually outperformed, possibly because there are very few observations
# and the data seems to lack any strong interactions. (Semi)Parametric models should thus perform quite well. 

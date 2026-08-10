#nolint start: line_length_linter

# This file is for analysing the EBMT platelet-recovery data.

# import helper functions and packages
source("testing/Articles/Helpers.r")
library(mstate)
library(tidyverse)
library(etm)

# data preparation
#--------------------------------------------------------------------------------
{
options(scipen = 1)
data(ebmt3)
head(ebmt3)
unique(ebmt3$drmatch)
unique(ebmt3$tcd)

# Convert the four event-history columns to the trajectory representation used
# by aalen_johansen() and JumpForests. The states are
#   1 = transplant (Tx), 2 = platelet recovery (PR),
#   3 = relapse or death (RelDeath).
jump_data <- lapply(seq_len(nrow(ebmt3)), function(i) {
    states <- 1L
    times <- 0

    # platelet recovery definitely takes place
    if (ebmt3$prstat[i] == 1) {
        states <- c(states, 2L)
        times <- c(times, as.numeric(ebmt3$prtime[i]))
    }
    if (ebmt3$rfsstat[i] == 1) {
        # relapse/death is an observed transition from the current state.
        states <- c(states, 3L)
    } else {
        # repeating the current state at the endpoint denotes censoring.
        states <- c(states, tail(states, 1L))
    }
    times <- c(times, as.numeric(ebmt3$rfstime[i]))

    list(times = times, states = states)
})

head(jump_data)

feature_data <- ebmt3[, 6:9]
head(feature_data)
feature_data$dissub <- as.factor(feature_data$dissub)
feature_data$age <- as.factor(feature_data$age)
feature_data$drmatch <- as.factor(feature_data$drmatch)
feature_data$tcd <- as.factor(feature_data$tcd)

# this concludes data preparation since the features are already in the same order as the patients
}

# tune the jump forest
#--------------------------------------------------------------------------------

# as a benchmark, simply fit the NA estimator with no covariates
null_forest <- jfforest(MM ~ 1, data = jump_data, feature_data = feature_data, min_node_size = 20, seed = 2026)
print_forest(null_forest)

min_node_sizes <- c(5, 10, 25, 50, 100)
split_rules <- c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
ibs <- matrix(rep(0, 25), nrow = 5)
kl <- matrix(rep(0, 25), nrow = 5)
spherical <- matrix(rep(0, 25), nrow = 5)
# row: spliting rule, column: minimal node size

for (i in 1:5) {
    for (j in 1:5) {
        fitted_forest <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data, splitrule = split_rules[i], 
                                  min_node_size = min_node_sizes[j], seed = 2026)
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
    data.frame(metric = metric, split_rule = split_rules[best["row"]], min_node_size = min_node_sizes[best["col"]], 
               score = scores[best["row"], best["col"]])
}
best_tuning_results <- rbind(
    best_tuning_configuration(ibs, "IBS"),
    best_tuning_configuration(kl, "KL"),
    best_tuning_configuration(spherical, "Spherical")
)
best_tuning_results # they all agree! taroneware and min_node_size = 100
write.table(best_tuning_results, "testing/Articles/EBMT/tuning_ebmt3.txt", sep = "\t", row.names = FALSE)

fitted_forest <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data, splitrule = "taroneware", min_node_size = 100, seed = 2026)
print_forest(fitted_forest)

# VIMP
#--------------------------------------------------------------------------------

# all the VIMP results are already saved, so can skip the first part
B <- 100
vimp_permute_brier <- matrix(0, nrow = B, ncol = 4)
vimp_permute_kl <- matrix(0, nrow = B, ncol = 4)
vimp_permute_spherical <- matrix(0, nrow = B, ncol = 4)
vimp_random_brier <- matrix(0, nrow = B, ncol = 4)
vimp_random_kl <- matrix(0, nrow = B, ncol = 4)
vimp_random_spherical <- matrix(0, nrow = B, ncol = 4)

set.seed(67)
tic()
for (b in 62:B) {
    vimp_permute_brier[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "brier")$vimp))
    vimp_permute_kl[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "kl")$vimp))
    vimp_permute_spherical[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "permute", loss = "spherical")$vimp))
    vimp_random_brier[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "random", loss = "brier")$vimp))
    vimp_random_kl[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "random", loss = "kl")$vimp))
    vimp_random_spherical[b, ] <- as.numeric(unlist(jfforest.vimp(fitted_forest, method = "random", loss = "spherical")$vimp))
    cat("Finished iteration", b, "of",B, "\n")
}
toc()

colnames(vimp_permute_brier) <- c("dissub", "age", "drmatch", "tcd")
colnames(vimp_permute_kl) <- c("dissub", "age", "drmatch", "tcd")
colnames(vimp_permute_spherical) <- c("dissub", "age", "drmatch", "tcd")
colnames(vimp_random_brier) <- c("dissub", "age", "drmatch", "tcd")
colnames(vimp_random_kl) <- c("dissub", "age", "drmatch", "tcd")
colnames(vimp_random_spherical) <- c("dissub", "age", "drmatch", "tcd")

# save results of each run
write.table(vimp_permute_brier, "testing/Articles/EBMT/vimp_permute_brier_ebmt3.txt", sep = "\t", row.names = FALSE)
write.table(vimp_permute_kl, "testing/Articles/EBMT/vimp_permute_kl_ebmt3.txt", sep = "\t", row.names = FALSE)
write.table(vimp_permute_spherical, "testing/Articles/EBMT/vimp_permute_spherical_ebmt3.txt", sep = "\t", row.names = FALSE)
write.table(vimp_random_brier, "testing/Articles/EBMT/vimp_random_brier_ebmt3.txt", sep = "\t", row.names = FALSE)
write.table(vimp_random_kl, "testing/Articles/EBMT/vimp_random_kl_ebmt3.txt", sep = "\t", row.names = FALSE)
write.table(vimp_random_spherical, "testing/Articles/EBMT/vimp_random_spherical_ebmt3.txt", sep = "\t", row.names = FALSE)

# load the results in here
vimp_permute_brier <- read.table("testing/Articles/EBMT/vimp_permute_brier_ebmt3.txt", header = TRUE)
vimp_permute_kl <- read.table("testing/Articles/EBMT/vimp_permute_kl_ebmt3.txt", header = TRUE)
vimp_permute_spherical <- read.table("testing/Articles/EBMT/vimp_permute_spherical_ebmt3.txt", header = TRUE)
vimp_random_brier <- read.table("testing/Articles/EBMT/vimp_random_brier_ebmt3.txt", header = TRUE)
vimp_random_kl <- read.table("testing/Articles/EBMT/vimp_random_kl_ebmt3.txt", header = TRUE)
vimp_random_spherical <- read.table("testing/Articles/EBMT/vimp_random_spherical_ebmt3.txt", header = TRUE)

plot_vimp <- function(method, metric, vimp_table, save = FALSE) {
    method <- match.arg(method, c("random", "permute"))
    metric <- match.arg(metric, c("brier", "kl", "spherical"))
    if (!is.logical(save) || length(save) != 1L || is.na(save)) {
        stop("save must be TRUE or FALSE")
    }

    plot_data <- stack(vimp_table)
    names(plot_data) <- c("vimp", "covariate")

    plot_data$covariate <- reorder(
        plot_data$covariate,
        plot_data$vimp,
        FUN = median,
        na.rm = TRUE
    )

    p <- ggplot(plot_data, aes(x = covariate, y = vimp, fill = covariate)) +
        geom_boxplot(
            width = 0.7,
            outlier.shape = 21,
            outlier.fill = "white",
            outlier.size = 1.7,
            na.rm = TRUE
        ) +
        geom_hline(yintercept = 0, colour = "grey35", linewidth = 0.4) +
        coord_flip() +
        scale_fill_manual(values = c(
            dissub = "#66C2A5",
            age = "#FC8D62",
            drmatch = "#8DA0CB",
            tcd = "#E78AC3"
        )) +
        labs(
            title = paste(
                tools::toTitleCase(method),
                tools::toTitleCase(metric),
                "VIMP"
            ),
            x = NULL,
            y = "VIMP",
            fill = "Covariate"
        ) +
        guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
        theme_bw() +
        theme(
            legend.position = "bottom",
            legend.title = element_blank(),
            plot.title = element_text(hjust = 0.5)
        )

    if (save) {
        plot_directory <- "testing/Articles/EBMT/Plots"
        dir.create(plot_directory, recursive = TRUE, showWarnings = FALSE)
        ggsave(
            filename = file.path(
                plot_directory,
                paste0("vimp_", method, "_", metric, ".png")
            ),
            plot = p,
            width = 10,
            height = 6,
            units = "in",
            dpi = 300
        )
    }
    p
}

save <- TRUE
# show plots
plot_vimp("permute", "brier", vimp_permute_brier, save = save)
plot_vimp("permute", "kl", vimp_permute_kl, save = save)
plot_vimp("permute", "spherical", vimp_permute_spherical, save = save)
plot_vimp("random", "brier", vimp_random_brier, save = save)
plot_vimp("random", "kl", vimp_random_kl, save = save)
plot_vimp("random", "spherical", vimp_random_spherical, save = save)

colMeans(vimp_permute_brier)
colMeans(vimp_permute_kl)
colMeans(vimp_permute_spherical)
colMeans(vimp_random_brier)
colMeans(vimp_random_kl)
colMeans(vimp_random_spherical)

# except for KL (and I would not put too much weight on that VIMP score), it seems that 
# the most important variable is age, then dissub and then tcd, while drmatch may actually
# hurt performance somewhat

# a few exploratory plots
#--------------------------------------------------------------------------------

# let's try to visualise potential interactions (age:dissub, age:tcd, dissub:tcd)
head(feature_data)

# note: Putter, Fiocco and Geskus use age = "<=20", dissub = "AML", tcd = "No TCD" and
# drmatch = "No gender mismatch" as baseline

# plot of tcd and all three age groups?
new_data <- data.frame(age = c(">40", "20-40", "<=20", ">40", "20-40", "<=20"), tcd = c("No TCD", "No TCD", "No TCD", "TCD", "TCD", "TCD"), 
                       drmatch = rep("No gender mismatch", 6), dissub = rep("AML", 6))

age_tcd_predictions <- jfforest.predict(fitted_forest, new_data)

plot_transition_rate(
    predictions = age_tcd_predictions,
    event_times = fitted_forest$unique.event.times,
    new_data = new_data,
    from = 1,
    to = 2,
    colour_by = "age",
    linetype_by = "tcd",
    max_time = 500,
    xlab = "Time (days)",
    plot_directory = "testing/Articles/EBMT/Plots",
    save = TRUE
)

# now investigae age:dissub (9 curves)
new_data <- data.frame(age = rep(c(">40", "20-40", "<=20"), 3), tcd = rep("No TCD", 9), 
                       drmatch = rep("No gender mismatch", 9), dissub = c(rep(c("AML"), 3), rep(c("ALL"), 3), rep(c("CML"), 3)))

age_dissub_predictions <- jfforest.predict(fitted_forest, new_data)

plot_transition_rate(
    predictions = age_dissub_predictions,
    event_times = fitted_forest$unique.event.times,
    new_data = new_data,
    from = 2,
    to = 3,
    colour_by = "age",
    linetype_by = "dissub",
    max_time = NULL,
    xlab = "Time (days)",
    plot_directory = "testing/Articles/EBMT/Plots",
    save = TRUE
)

# fitting the Cox model of Putter, Fiocco and Geskus
#--------------------------------------------------------------------------------

{

fit_simple_ebmt_cox <- function(paths, features, ties = "breslow") {
    cox_12_data <- paths_to_transition_cox_data(paths, features, 1L, 2L)
    cox_13_data <- paths_to_transition_cox_data(paths, features, 1L, 3L)
    cox_23_data <- paths_to_transition_cox_data(paths, features, 2L, 3L)

    # Equation (25) has one common baseline for both transitions into state 3.
    # PR is a time-dependent indicator: 0 before and 1 after platelet recovery.
    endpoint_data <- rbind(
        transform(cox_13_data, PR = 0),
        transform(cox_23_data, PR = 1)
    )
    rownames(endpoint_data) <- NULL

    fit_12 <- survival::coxph(
        survival::Surv(tstart, tstop, event) ~
            dissub + age + drmatch + tcd,
        data = cox_12_data, ties = ties,
        model = TRUE, x = TRUE, y = TRUE, singular.ok = FALSE
    )
    fit_endpoint <- survival::coxph(
        survival::Surv(tstart, tstop, event) ~
            dissub + age + drmatch + tcd + PR +
            PR:(dissub + age + drmatch + tcd),
        data = endpoint_data, ties = ties,
        model = TRUE, x = TRUE, y = TRUE, singular.ok = FALSE
    )

    model <- new_multistate_cox_model(
        transitions = list(
            `1 -> 2` = list(
                from = 1L, to = 2L, fit = fit_12,
                fixed_values = list()
            ),
            `1 -> 3` = list(
                from = 1L, to = 3L, fit = fit_endpoint,
                fixed_values = list(PR = 0)
            ),
            `2 -> 3` = list(
                from = 2L, to = 3L, fit = fit_endpoint,
                fixed_values = list(PR = 1)
            )
        ),
        number_of_states = 3L,
        label = "Simple Cox"
    )
    model$fits <- list(`1 -> 2` = fit_12, endpoint = fit_endpoint)
    model$sample_sizes <- data.frame(
        transition = c("1 -> 2", "1 -> 3", "2 -> 3"),
        risk_intervals = c(nrow(cox_12_data), nrow(cox_13_data), nrow(cox_23_data)),
        events = c(sum(cox_12_data$event), sum(cox_13_data$event), sum(cox_23_data$event))
    )
    model
}

fit_interaction_ebmt_cox <- function(paths, features, ties = "breslow") {
    cox_12_data <- paths_to_transition_cox_data(paths, features, 1L, 2L)
    cox_13_data <- paths_to_transition_cox_data(paths, features, 1L, 3L)
    cox_23_data <- paths_to_transition_cox_data(paths, features, 2L, 3L)

    # age:dissub is included in every transition. age:tcd is included only in
    # the two relapse/death transitions, as requested. These are three separate
    # fits, so this model also relaxes equation (25)'s shared endpoint baseline.
    fit_12 <- survival::coxph(
        survival::Surv(tstart, tstop, event) ~
            age * dissub + drmatch + tcd,
        data = cox_12_data, ties = ties,
        model = TRUE, x = TRUE, y = TRUE, singular.ok = FALSE
    )
    fit_13 <- survival::coxph(
        survival::Surv(tstart, tstop, event) ~
            age * dissub + age * tcd + drmatch,
        data = cox_13_data, ties = ties,
        model = TRUE, x = TRUE, y = TRUE, singular.ok = FALSE
    )
    fit_23 <- survival::coxph(
        survival::Surv(tstart, tstop, event) ~
            age * dissub + age * tcd + drmatch,
        data = cox_23_data, ties = ties,
        model = TRUE, x = TRUE, y = TRUE, singular.ok = FALSE
    )

    model <- new_multistate_cox_model(
        transitions = list(
            `1 -> 2` = list(from = 1L, to = 2L, fit = fit_12, fixed_values = list()),
            `1 -> 3` = list(from = 1L, to = 3L, fit = fit_13, fixed_values = list()),
            `2 -> 3` = list(from = 2L, to = 3L, fit = fit_23, fixed_values = list())
        ),
        number_of_states = 3L,
        label = "Interaction Cox"
    )
    model$fits <- list(`1 -> 2` = fit_12, `1 -> 3` = fit_13, `2 -> 3` = fit_23)
    model$sample_sizes <- data.frame(
        transition = c("1 -> 2", "1 -> 3", "2 -> 3"),
        risk_intervals = c(nrow(cox_12_data), nrow(cox_13_data), nrow(cox_23_data)),
        events = c(sum(cox_12_data$event), sum(cox_13_data$event), sum(cox_23_data$event))
    )
    model
}


simple_cox_model <- fit_simple_ebmt_cox(jump_data, feature_data)
interaction_cox_model <- fit_interaction_ebmt_cox(jump_data, feature_data)

simple_cox_model$sample_sizes
interaction_cox_model$sample_sizes
summary(simple_cox_model$fits$`1 -> 2`) # concurs perfectly with the table on page 31 of Putter, Fiocco and Geskus
summary(simple_cox_model$fits$endpoint)
lapply(interaction_cox_model$fits, summary)

# In equation (25), delta is the coefficient of PR. It is the log hazard ratio
# comparing 2 -> 3 with 1 -> 3 at the same time since transplant for the
# reference patient (AML, age <=20, no gender mismatch, no TCD). It is not an
# event/censoring indicator. Delta's covariate-dependent counterpart is the
# vector of PR:covariate coefficients.
delta_summary <- cox_term_summary(simple_cox_model$fits$endpoint, "PR")
delta_summary
}

# fit a Poisson regression model?
#--------------------------------------------------------------------------------

# error curves for all the models
#--------------------------------------------------------------------------------

# The simulation study has a known data-generating truth, whereas EBMT does not.
# Therefore these are empirical IPCW errors. All three specifications use the
# same validation folds and the same marginal reverse-KM censoring distribution,
# avoiding both an in-sample advantage and mismatched validation samples.

{

# Include both transition and censoring times: predictions are constant at a
# pure censoring time, but empirical IPCW weights can jump there. Truncate the
# grid before prediction once marginal G falls below 0.05; this avoids unstable
# late-tail weights and defines the integration horizon used by every model.
minimum_error_censoring_survival <- 0.05
error_evaluation_times <- sort(unique(c(fitted_forest$unique.event.times, vapply(jump_data, function(path) tail(path$times, 1L), numeric(1)))))
error_censoring_model <- reverse_km_censoring(jump_data)
error_censoring_survival <- censoring_survival_at(
    error_censoring_model, error_evaluation_times
)
error_evaluation_times <- error_evaluation_times[
    error_censoring_survival >= minimum_error_censoring_survival
]
initial_distribution <- c(1, 0, 0)
number_of_error_folds <- 5L
error_folds <- stratified_multistate_folds(
    jump_data, number_of_error_folds, seed = 2026
)

# The selected forest hyperparameters are held fixed in the outer folds. A
# fully nested performance study would repeat the tuning inside each training
# fold; this comparison evaluates the already selected specification.
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
    jump_data, feature_data, fit_simple_ebmt_cox,
    evaluation_times = error_evaluation_times,
    number_of_folds = number_of_error_folds,
    seed = 2026,
    initial = initial_distribution,
    prediction_stype = "aalen-johansen",
    folds = error_folds
)
interaction_cox_crossfit_occupation <- crossfit_multistate_cox(
    jump_data, feature_data, fit_interaction_ebmt_cox,
    evaluation_times = error_evaluation_times,
    number_of_folds = number_of_error_folds,
    seed = 2026,
    initial = initial_distribution,
    prediction_stype = "aalen-johansen",
    folds = error_folds
)

error_predictions <- list(
    JumpForest = jumpforest_crossfit_occupation,
    `Simple Cox` = simple_cox_crossfit_occupation,
    `Interaction Cox` = interaction_cox_crossfit_occupation
)
}

ebmt_error_comparison <- multistate_score_curves(
    error_predictions,
    paths = jump_data,
    evaluation_times = error_evaluation_times,
    minimum_censoring_survival = minimum_error_censoring_survival
)
ebmt_error_plots <- plot_multistate_error_curves(
    ebmt_error_comparison,
    xlab = "Time (days)",
    save = TRUE,
    plot_directory = "testing/Articles/EBMT/Plots"
)

write.table(
    ebmt_error_comparison$curves,
    "testing/Articles/EBMT/time_dependent_errors_ebmt3.txt",
    sep = "\t", row.names = FALSE
)
write.table(
    ebmt_error_comparison$integrated,
    "testing/Articles/EBMT/integrated_errors_ebmt3.txt",
    sep = "\t", row.names = FALSE
)
ebmt_error_comparison$integrated

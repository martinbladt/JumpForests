#nolint start: line_length_linter

# This file is for analysing the EBMT platelet-recovery data.

# import helper functions and packages
source("testing/Articles/Discrete/Helpers.r")
library(mstate)
library(tidyverse)
library(etm)

# data preparation
#--------------------------------------------------------------------------------
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

# Plot one predicted cumulative transition rate for all covariate profiles.
plot_transition_rate <- function(predictions, event_times, new_data, from, to, colour_by, linetype_by, max_time = NULL, save = FALSE) {
    if (!is.data.frame(new_data)) {
        stop("new_data must be a data frame")
    }
    if (!all(c(colour_by, linetype_by) %in% names(new_data))) {
        stop("colour_by and linetype_by must name columns in new_data")
    }
    if (!is.numeric(event_times) || length(event_times) == 0L ||
        any(!is.finite(event_times)) || is.unsorted(event_times, strictly = TRUE)) {
        stop("event_times must be a non-empty, strictly increasing numeric vector")
    }
    if (!is.list(predictions) || length(predictions) != nrow(new_data)) {
        stop("predictions must contain one prediction for each row of new_data")
    }
    if (!is.null(max_time) &&
        (!is.numeric(max_time) || length(max_time) != 1L ||
         !is.finite(max_time) || max_time <= 0)) {
        stop("max_time must be NULL or one positive finite number")
    }
    if (!is.logical(save) || length(save) != 1L || is.na(save)) {
        stop("save must be TRUE or FALSE")
    }

    valid_state <- function(state) {
        is.numeric(state) && length(state) == 1L && is.finite(state) &&
            state >= 1 && state == as.integer(state)
    }
    if (!valid_state(from) || !valid_state(to) || from == to) {
        stop("from and to must be distinct positive integers")
    }
    from <- as.integer(from)
    to <- as.integer(to)

    add_zero <- event_times[1L] > 0
    plot_data <- do.call(rbind, lapply(seq_along(predictions), function(i) {
        predicted_matrices <- predictions[[i]]
        if (length(predicted_matrices) != length(event_times)) {
            stop("each prediction must contain one matrix per event time")
        }

        transition_rate <- vapply(predicted_matrices, function(rate_matrix) {
            if (!is.matrix(rate_matrix) ||
                from > nrow(rate_matrix) || to > ncol(rate_matrix)) {
                stop("from or to is outside the predicted transition matrix")
            }
            rate_matrix[from, to]
        }, numeric(1))

        data.frame(
            time = if (add_zero) c(0, event_times) else event_times,
            transition_rate = if (add_zero) c(0, transition_rate) else transition_rate,
            profile = i,
            colour_value = as.character(new_data[[colour_by]][i]),
            linetype_value = as.character(new_data[[linetype_by]][i])
        )
    }))

    plot_data$colour_value <- factor(
        plot_data$colour_value,
        levels = unique(as.character(new_data[[colour_by]]))
    )
    plot_data$linetype_value <- factor(
        plot_data$linetype_value,
        levels = unique(as.character(new_data[[linetype_by]]))
    )

    p <- ggplot(
        plot_data,
        aes(
            x = time,
            y = transition_rate,
            colour = colour_value,
            linetype = linetype_value,
            group = profile
        )
    ) +
        geom_step(linewidth = 0.9, direction = "hv") +
        labs(
            title = paste0("Predicted transition rate: ", from, " → ", to),
            x = "Time (days)",
            y = "Cumulative transition rate",
            colour = colour_by,
            linetype = linetype_by
        ) +
        guides(
            colour = guide_legend(order = 1),
            linetype = guide_legend(order = 2)
        ) +
        theme_classic() +
        theme(
            legend.position = "bottom",
            plot.title = element_text(hjust = 0.5)
        )

    if (!is.null(max_time)) {
        p <- p + coord_cartesian(xlim = c(0, max_time))
    }

    if (save) {
        plot_directory <- "testing/Articles/EBMT/Plots"
        dir.create(plot_directory, recursive = TRUE, showWarnings = FALSE)
        ggsave(
            filename = file.path(
                plot_directory,
                paste0(
                    "transition_rate_", from, "_", to, "_",
                    colour_by, "_", linetype_by, ".png"
                )
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

plot_transition_rate(
    predictions = age_tcd_predictions,
    event_times = fitted_forest$unique.event.times,
    new_data = new_data,
    from = 1,
    to = 2,
    colour_by = "age",
    linetype_by = "tcd",
    max_time = NULL,
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
    save = TRUE
)

# fitting the Cox model of Putter, Fiocco and Geskus
#--------------------------------------------------------------------------------

# fit a Poisson regression model?
#--------------------------------------------------------------------------------

# error curves for all the models
#--------------------------------------------------------------------------------

# fit a Cox model as in Putter Fiocco and Geskus (pure linear, no interaction)
# fit another Cox model that allows for interactions between all variables
# summarise errors (Brier, KL, spherical) and plot the time-dependent error curves
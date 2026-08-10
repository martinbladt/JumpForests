#nolint start: line_length_linter

# This file is for analysing the 1.7 Bone marrow transplantation in acute leukemia data
# from https://multi-state-book.github.io/companion/Ch1.html

# import helper functions and packages
source("testing/Articles/Helpers.r")
library(mstate)
library(tidyverse)
library(etm)

# data preparation
#--------------------------------------------------------------------------------

{
bmt <- read.csv("testing/Articles/BMT/Data/bmt.csv", sep = ",")
head(bmt)

# four-state model with
# 0: BMT
# 1: GvHD
# 2: Relapse
# 3: Death
# possible transitions are 0->1, 0->2, 0->3, 1->2, 1->3 and 2->3

# prepare jump data
# important note: there are only the following possible paths:
# 0->1->2->3, 0->1->3, 0->2->3 and 0->3
jump_data <- list()
for (i in 1:nrow(bmt)) {
    patient <- as.numeric(bmt[i, ])
    times <- numeric()
    states <- integer()

    if (patient[8] == 1) {
        # the path is 0->1->2->3
        if (patient[6] == 1) {
            times <- c(0.0, patient[7], patient[5], patient[3])
            states <- c(1L, 2L, 3L, 4L)
        # the path is 0->1->3
        } else {
            times <- c(0.0, patient[7], patient[3])
            states <- c(1L, 2L, 4L)
        }
    } else {
        # the path is 0->2->3
        if (patient[6] == 1) {
            times <- c(0.0, patient[5], patient[3])
            states <- c(1L, 3L, 4L)
        # the path is 0->3
        } else {
            times <- c(0.0, patient[3])
            states <- c(1L, 4L)
        }
    }

    # JumpForests requires strictly increasing times. For the seven patients
    # with an event at the death/censoring time, move the earlier event back by
    # a negligible amount while preserving the recorded follow-up endpoint.
    if (any(diff(times) < 0)) {
        stop("Events are not chronologically ordered for patient ", bmt$id[i])
    }
    for (j in rev(which(diff(times) == 0))) {
        times[j] <- times[j + 1L] - 1e-8
    }

    # make the last state a repetition of the previous one in case of censoring
    if (patient[4] == 0) {
        states[length(states)] <- states[length(states) - 1]
    }

    # resolve here

    # finally, update jump_data
    jump_data[[i]] <- list(times = times, states = states)
}

# for an example of a censoring time occuring at the same time as a jump
#bmt[421,]
#jump_data[421]

feature_data <- bmt[, c(11, 12, 13, 14)]    # remove 'team' (255 levels)

}

# number of event times
length(unique(unlist(lapply(jump_data, function(z) z$times))))  # approx 1500, may reduce to 1000 no problem when tuning
sort(unique(unlist(lapply(jump_data, function(z) z$times))))    # especially since they are nice and evenly spaced out

# tune the forest
#--------------------------------------------------------------------------------

null_forest <- jfforest(MM ~ 1, data = jump_data, feature_data = feature_data, min_node_size = 40, seed = 2026, num_event_times = 1000)
print_forest(null_forest)   # simple baseline is 3/16 = 0.1875, so 0.1646 is already better

min_node_sizes <- c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200)
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
                                  min_node_size = min_node_sizes[j], seed = 2026, num_event_times = 1000)
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
best_tuning_results # definitely approxlogrank

# for plots later
write.table(ibs, file = "testing/Articles/BMT/Data/tuning_ibs.txt", sep = "\t", row.names = FALSE)
write.table(kl, file = "testing/Articles/BMT/Data/tuning_kl.txt", sep = "\t", row.names = FALSE)
write.table(spherical, file = "testing/Articles/BMT/Data/tuning_spherical.txt", sep = "\t", row.names = FALSE)

# now make hyperparameter tuning plots
ibs <- read.table("testing/Articles/BMT/Data/tuning_ibs.txt", header = TRUE)
colnames(ibs) <- as.factor(min_node_sizes)
ikl <- read.table("testing/Articles/BMT/Data/tuning_kl.txt", header = TRUE)
colnames(ikl) <- as.factor(min_node_sizes)
spherical <- read.table("testing/Articles/BMT/Data/tuning_spherical.txt", header = TRUE)
colnames(spherical) <- as.factor(min_node_sizes)

tuning_colours <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")

# save plots
plot_score_curves(
    scores = ibs, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(ibs), colours = tuning_colours, legend_position = "bottomright",
    pch = seq_along(split_rules), ylab = "Normalised IBS", width = 6, height = 6,
    file = "testing/Articles/BMT/Plots/tuning_ibs.png"
)
plot_score_curves(
    scores = ikl, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(ikl), colours = tuning_colours, legend_position = "bottomright",
    pch = seq_along(split_rules), ylab = "Normalised IKL", width = 6, height = 6,
    file = "testing/Articles/BMT/Plots/tuning_ikl.png"
)
plot_score_curves(
    scores = spherical, x = min_node_sizes, series_labels = split_rules,
    x_tick_labels = colnames(spherical), colours = tuning_colours, legend_position = "bottomright",
    pch = seq_along(split_rules), ylab = "Normalised Integrated spherical error", width = 6, height = 6,
    file = "testing/Articles/BMT/Plots/tuning_spherical.png"
)

# based on the plots and the summary, we go with approxlogrank and min_node_size = 100, since
# Brier is quite a lot better for 100 compared to 200, but the others are extremely close for
# 100 and 200

fitted_forest <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data, splitrule = "approxlogrank", 
                                  min_node_size = 100, seed = 2026, num_event_times = 1000)

# VIMP
#--------------------------------------------------------------------------------

B <- 100
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
toc()   # takes approximately 7200 seconds (2 hours) for B = 100 (about 1 minute and 15 seconds for one iteration)

colnames(vimp_permute_brier) <- c("sex", "age", "all", "bmonly")
colnames(vimp_permute_kl) <- c("sex", "age", "all", "bmonly")
colnames(vimp_permute_spherical) <- c("sex", "age", "all", "bmonly")
colnames(vimp_random_brier) <- c("sex", "age", "all", "bmonly")
colnames(vimp_random_kl) <- c("sex", "age", "all", "bmonly")
colnames(vimp_random_spherical) <- c("sex", "age", "all", "bmonly")

# save results of each run
write.table(vimp_permute_brier, "testing/Articles/BMT/Data/vimp_permute_brier_bmt.txt", sep = "\t", row.names = FALSE)
write.table(vimp_permute_kl, "testing/Articles/BMT/Data/vimp_permute_kl_bmt.txt", sep = "\t", row.names = FALSE)
write.table(vimp_permute_spherical, "testing/Articles/BMT/Data/vimp_permute_spherical_bmt.txt", sep = "\t", row.names = FALSE)
write.table(vimp_random_brier, "testing/Articles/BMT/Data/vimp_random_brier_bmt.txt", sep = "\t", row.names = FALSE)
write.table(vimp_random_kl, "testing/Articles/BMT/Data/vimp_random_kl_bmt.txt", sep = "\t", row.names = FALSE)
write.table(vimp_random_spherical, "testing/Articles/BMT/Data/vimp_random_spherical_bmt.txt", sep = "\t", row.names = FALSE)

# AM HERE
#nolint start: line_length_linter

# this file is for analysing the CSL1 liver cirrhosis dataset (https://publicifsv.sund.ku.dk/~linearpredictors/?page=datasets&dataset=Csl)

# import helper functions and packages
source("testing/Articles/Discrete/Helpers.r")
library(mstate)
library(tidyverse)
library(etm)

# data preparation
#--------------------------------------------------------------------------------

data(sir.cont, package = "etm")
head(sir.cont)
length(unique(sir.cont$id)) # number of patients = 747

subset(sir.cont, id == 710)

unique(sir.cont$from)   # all observations start in either 0 or 1
unique(sir.cont$to)     # all possible transitions are 0 -> 1, 0 -> 2, 1 -> 2 and 1 -> 0 (reactivation is possible)

# apply jump data censoring convention
for (i in 1:length(sir.cont$to)) {
    # censoring is encoded as a repetition of the previous state
    if (sir.cont$to[i] == "cens") {
        sir.cont$to[i] <- sir.cont$from[i]
    }
}

# recode states to 1, 2, 3 instead of 0, 1, 2
sir.cont$from <- sir.cont$from + 1
sir.cont$from <- as.integer(sir.cont$from)
sir.cont$to <- as.integer(sir.cont$to)
View(sir.cont)

jump_data <- list()
ids <- unique(sir.cont$id)
for (i in seq_along(ids)) {
    # extract the lines in the dataset corresponding to the particular patient
    patient_data <- subset(sir.cont, id == ids[i])

    # initial time is always 0, and the from state is always the previous to state
    states <- c(patient_data$from[1], patient_data$to)
    times <- c(0.0, patient_data$time)

    jump_data[[i]] <- list(times = times, states = states)
}

# ensure that the feature data matches the ordering of the patients in jump_data
feature_data <- sir.cont[match(ids, sir.cont$id), c("age", "sex"), drop = FALSE]

head(jump_data)

# tune the jump forest
#--------------------------------------------------------------------------------

# as a benchmark, simply fit the NA estimator with no covariates
null_forest <- jfforest(MM ~ 1, data = jump_data, feature_data = feature_data, min_node_size = 20, seed = 2026)

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
best_tuning_results
jfforest.error(null_forest)

# the null model is basically as good as the best models based on the above (simple) tuning,
# maybe not the most relevant example

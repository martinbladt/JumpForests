# Biomedical applications
# Run from manuscript/. Edit the switches below to run selected sections.

# 1. Helpers and packages -------------------------------------------------
source("helpers/setup.r")
library(mstate)
library(tidyverse)
library(etm)

# 2. Settings -------------------------------------------------------------
datasets_to_run <- c("CPHHolter", "BMT", "EBMT", "PROVA")
run_analyses <- TRUE
run_tuning <- FALSE
make_comparison_plots <- TRUE
# Set make_comparison_plots=FALSE when results for all four datasets are absent.

# 3. Copenhagen Holter (cphholter) ----------------------------------------
if (run_analyses && "CPHHolter" %in% datasets_to_run) {
    manuscript_prepare_real_dirs("CPHHolter")
    # data preparation
    #--------------------------------------------------------------------------------

    {
    cphholter <- read.csv("data/biomedical_applications/cphholter.csv", sep = ",")
    head(cphholter)

    # four-state model with
    # 1: No event
    # 2: AF
    # 3: Stroke
    # 4: Dead
    # possible transitions are 1->2, 1->3, 1->4, 2->3, 2->4, 3->2 and 3->4

    # Convert the wide event records to the trajectory representation used by
    # JumpForests. When AF and stroke occur on the same day, AF is ordered first,
    # following the convention used for the Copenhagen Holter data in the book.
    jump_data <- lapply(seq_len(nrow(cphholter)), function(i) {
        events <- data.frame(
            time = numeric(),
            state = integer(),
            priority = integer()
        )

        if (cphholter$afib[i] == 1L) {
            events <- rbind(events, data.frame(
                time = cphholter$timeafib[i],
                state = 2L,
                priority = 1L
            ))
        }
        if (cphholter$stroke[i] == 1L) {
            events <- rbind(events, data.frame(
                time = cphholter$timestroke[i],
                state = 3L,
                priority = 2L
            ))
        }

        if (nrow(events) > 0L) {
            events <- events[order(events$time, events$priority), , drop = FALSE]
        }

        times <- c(0, events$time, cphholter$timedeath[i])
        states <- c(1L, events$state)
        if (cphholter$death[i] == 1L) {
            states <- c(states, 4L)
        } else {
            # Repeating the current state at the endpoint denotes censoring.
            states <- c(states, tail(states, 1L))
        }

        if (any(!is.finite(times)) || any(diff(times) < 0)) {
            stop("Invalid event chronology for subject ", cphholter$id[i])
        }

        # JumpForests requires strictly increasing times. Five subjects have two
        # recorded events on the same day, so move the earlier event back by a
        # negligible amount while preserving the recorded follow-up endpoint.
        for (j in rev(which(diff(times) == 0))) {
            times[j] <- times[j + 1L] - 1e-8
        }

        if (times[1L] != 0 || any(diff(times) <= 0)) {
            stop("Could not resolve tied event times for subject ", cphholter$id[i])
        }

        list(times = times, states = states)
    })

    # Retain the baseline covariates in the same subject order as jump_data.
    # JumpForests requires complete feature rows, so replace missing continuous
    # measurements by their non-missing column medians.
    feature_data <- cphholter[, 8:17]
    continuous_features <- c("chol", "bmi", "probnp", "sbp")

    # how many values are missing? continuous features:
    feature_data$chol[is.na(feature_data$chol)]     # 7
    feature_data$bmi[is.na(feature_data$bmi)]       # 2
    feature_data$probnp[is.na(feature_data$probnp)] # 9
    feature_data$sbp[is.na(feature_data$sbp)]       # 3

    # categorical features:
    feature_data$sex[is.na(feature_data$sex)]
    feature_data$age[is.na(feature_data$age)]
    feature_data$smoker[is.na(feature_data$smoker)]
    feature_data$esvea[is.na(feature_data$esvea)]
    feature_data$diabet[is.na(feature_data$diabet)]
    feature_data$aspirin[is.na(feature_data$aspirin)]
    # no missing categorical values!

    # since so few values are missing, we simply impute the missing continuous features using the median
    for (feature in continuous_features) {
        feature_data[[feature]][is.na(feature_data[[feature]])] <- median(
            feature_data[[feature]],
            na.rm = TRUE
        )
    }
    }

    # tune the forest
    #--------------------------------------------------------------------------------

    nrow(feature_data)  # 678 observations in total (medium size)
    sqrt(678)           # 26

    null_forest <- jfforest(MM ~ 1, data = jump_data, feature_data = feature_data, min_node_size = 30, seed = 2026)
    JumpForests:::print_forest(null_forest)   # simple baseline is 3/16 = 0.1875, so 0.08325329 is already much better
                                              # maybe we should not expect that great of a performance from JumpForest? We'll see

    min_node_sizes <- c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100)
    split_rules <- c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
    best_tuning_results <- if (run_tuning) tune_helper(
        jump_data = jump_data, feature_data = feature_data,
        min_node_sizes = min_node_sizes, split_rules = split_rules,
        path = "results/biomedical_applications/CPHHolter/Data"
    )
    if (run_tuning) print(best_tuning_results)

    # now make hyperparameter tuning plots
    if (run_tuning) tune_plot_helper(
        input_path = "results/biomedical_applications/CPHHolter/Data",
        output_path = "results/biomedical_applications/CPHHolter/Plots",
        min_node_sizes = min_node_sizes, split_rules = split_rules,
        legend_positions = c("bottomright", "topright", "bottomright")
    )

    # based on the plots and the summary, definitely petoprentice splitting rule, and for the sake of robustness,
    # we go with min_node_size = 15 (I am actually still a bit worried that this may be too small, we'll see)
    fitted_forest <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data, splitrule = "petoprentice", 
                                      min_node_size = 15, seed = 2026)
    JumpForests:::print_forest(fitted_forest)   # an average depth of almost 8 is quite deep

    # VIMP
    #--------------------------------------------------------------------------------

    head(feature_data)  # 10 covariates

    vimp_results <- vimp_helper(
        fitted_forest = fitted_forest, variable_names = names(feature_data), B = 100,
        path = "results/biomedical_applications/CPHHolter/Data"
    )
    vimp_plot_helper(
        input_path = "results/biomedical_applications/CPHHolter/Data",
        output_path = "results/biomedical_applications/CPHHolter/Plots"
    )

    # age is without a doubt the most important variable, followed by smoker, probnp, sex and sbp/esvea, the
    # rest are borderline important and will not be considered in detail at all below

    # Investigation: Is the incidence of stroke in esvea patients increased over and above what is explained
    # by AF? Also confirm that ESVEA increases the incidence of AF
    #--------------------------------------------------------------------------------

    # Warning: as we found out below, there are very few transitions 2->3 and 3->2. In fact, only 9 events of type 2->3
    # and 7 of type 3->2. Thus, it may be very hard to conclude anything about the effect of AF, and we may need a to test
    # for several minimal node sizes to see if the conclusion changes (and maybe we would have to redo the runs below, but
    # let's see)

    summary(feature_data$age)
    summary(feature_data$probnp)

    # Use medians for covariates that are not varied in a comparison. For the
    # binary covariates these medians are observed 0/1 values. Age and proBNP are
    # compared at their lower and upper quartiles.
    reference_profile <- as.data.frame(as.list(vapply(
        feature_data, stats::median, numeric(1)
    )))
    age_levels <- unname(stats::quantile(feature_data$age, c(0.25, 0.75)))
    probnp_levels <- round(unname(stats::quantile(
        feature_data$probnp, c(0.25, 0.75)
    )), 2)

    make_reference_profiles <- function(varied_values) {
        profiles <- reference_profile[
            rep.int(1L, nrow(varied_values)), , drop = FALSE
        ]
        profiles[names(varied_values)] <- varied_values
        rownames(profiles) <- NULL
        profiles
    }

    # Compare ESVEA = 0 and ESVEA = 1 separately for the direct No event -> Stroke
    # transition and for AF -> Stroke. All other covariates are held at their
    # median values.
    esvea_profiles <- make_reference_profiles(data.frame(esvea = c(0, 1)))
    esvea_predictions <- jfforest.predict(fitted_forest, esvea_profiles)

    plot_transition_rate(
        predictions = esvea_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = esvea_profiles,
        from = 1,
        to = 3,
        colour_by = "esvea",
        linetype_by = "esvea",
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/CPHHolter/Plots",
        save = TRUE
    )
    plot_transition_rate(
        predictions = esvea_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = esvea_profiles,
        from = 2,
        to = 3,
        colour_by = "esvea",
        linetype_by = "esvea",
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/CPHHolter/Plots",
        save = TRUE
    )

    # Put both stroke transitions on a common y scale within ESVEA profiles so the
    # state-specific rates can also be compared directly.
    plot_transition_rate_comparison(
        predictions = esvea_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = esvea_profiles,
        transitions = data.frame(from = c(1, 2), to = c(3, 3)),
        transition_labels = c("No event -> Stroke", "AF -> Stroke"),
        profile_labels = c("ESVEA = 0", "ESVEA = 1"),
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/CPHHolter/Plots",
        filename = "transition_rate_comparison_stroke_esvea.pdf",
        ncol = 2,
        save = TRUE
    )

    # For this median reference profile, ESVEA = 1 has higher predicted cumulative
    # transition rates for both No event -> Stroke and AF -> Stroke. The latter is
    # based on only nine observed 2 -> 3 transitions and is therefore exploratory.

    # Pairwise age:smoker comparison at the lower and upper age quartiles.
    age_smoker_profiles <- make_reference_profiles(expand.grid(
        age = age_levels,
        smoker = c(0, 1),
        KEEP.OUT.ATTRS = FALSE
    ))
    age_smoker_predictions <- jfforest.predict(
        fitted_forest, age_smoker_profiles
    )

    plot_transition_rate(
        predictions = age_smoker_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = age_smoker_profiles,
        from = 1,
        to = 3,
        colour_by = "age",
        linetype_by = "smoker",
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/CPHHolter/Plots",
        save = TRUE
    )
    plot_transition_rate(
        predictions = age_smoker_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = age_smoker_profiles,
        from = 2,
        to = 3,
        colour_by = "age",
        linetype_by = "smoker",
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/CPHHolter/Plots",
        save = TRUE
    )

    # For No event -> Stroke, age has the dominant association and smoking adds a
    # smaller increase at both ages. AF -> Stroke displays a much less additive
    # pattern, particularly the high curve for age 60 smokers; interpret this only
    # as a signal to investigate because the transition is very sparse.

    # Pairwise age:proBNP comparison at the lower and upper quartiles of both
    # continuous covariates, with smoker fixed at its median value.
    age_probnp_profiles <- make_reference_profiles(expand.grid(
        age = age_levels,
        probnp = probnp_levels,
        KEEP.OUT.ATTRS = FALSE
    ))
    age_probnp_predictions <- jfforest.predict(
        fitted_forest, age_probnp_profiles
    )

    plot_transition_rate(
        predictions = age_probnp_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = age_probnp_profiles,
        from = 1,
        to = 3,
        colour_by = "age",
        linetype_by = "probnp",
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/CPHHolter/Plots",
        save = TRUE
    )
    plot_transition_rate(
        predictions = age_probnp_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = age_probnp_profiles,
        from = 2,
        to = 3,
        colour_by = "age",
        linetype_by = "probnp",
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/CPHHolter/Plots",
        save = TRUE
    )

    # Older age and higher proBNP both increase No event -> Stroke, with only modest
    # visual evidence that their effects interact. The AF -> Stroke curves are more
    # irregular and do not support a stable interaction conclusion.

    # Pairwise smoker:proBNP comparison at the lower and upper proBNP quartiles,
    # with age fixed at its median value.
    smoker_probnp_profiles <- make_reference_profiles(expand.grid(
        smoker = c(0, 1),
        probnp = probnp_levels,
        KEEP.OUT.ATTRS = FALSE
    ))
    smoker_probnp_predictions <- jfforest.predict(
        fitted_forest, smoker_probnp_profiles
    )

    plot_transition_rate(
        predictions = smoker_probnp_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = smoker_probnp_profiles,
        from = 1,
        to = 3,
        colour_by = "probnp",
        linetype_by = "smoker",
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/CPHHolter/Plots",
        save = TRUE
    )
    plot_transition_rate(
        predictions = smoker_probnp_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = smoker_probnp_profiles,
        from = 2,
        to = 3,
        colour_by = "probnp",
        linetype_by = "smoker",
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/CPHHolter/Plots",
        save = TRUE
    )

    # The smoker separation for No event -> Stroke is larger at low proBNP and is
    # absent at high proBNP, suggesting a possible smoker:proBNP interaction. The
    # AF -> Stroke curves also differ non-additively, but are too sparse for a firm
    # conclusion without stability checks across forest configurations.

    # mean errors across many runs compared to a transition-wise Cox model
    #--------------------------------------------------------------------------------

    # The Cox comparator uses transition-specific null models when a training split
    # has too few events to estimate all ten covariate coefficients.
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data, jump_data = jump_data, B = 100, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/CPHHolter/Data/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/CPHHolter/Data/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/CPHHolter/Data/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)

    # load in data here
    ibs_cphholter_harmonised <- read.table("results/biomedical_applications/CPHHolter/Data/oob_errors_ibs_harmonised.txt", header = TRUE)
    ikl_cphholter_harmonised <- read.table("results/biomedical_applications/CPHHolter/Data/oob_errors_ikl_harmonised.txt", header = TRUE)
    ispherical_cphholter_harmonised <- read.table("results/biomedical_applications/CPHHolter/Data/oob_errors_ispherical_harmonised.txt", header = TRUE)

    colMeans(ibs_cphholter_harmonised)

    colMeans(ikl_cphholter_harmonised)

    colMeans(ispherical_cphholter_harmonised)

    # adding noise: errors across runs compared to Cox and VIMP showcase
    #--------------------------------------------------------------------------------

    head(feature_data)
    nrow(feature_data)

    {
    set.seed(2026)
    # add 10 noise variables
    noise10 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 10), nrow = nrow(feature_data)))
    colnames(noise10) <- paste0("N", seq_len(ncol(noise10)))
    feature_data_10 <- cbind(feature_data, noise10)

    # add 25 noise variables
    noise25 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 25), nrow = nrow(feature_data)))
    colnames(noise25) <- paste0("N", seq_len(ncol(noise25)))
    feature_data_25 <- cbind(feature_data, noise25)

    # add 50 noise variables
    noise50 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 50), nrow = nrow(feature_data)))
    colnames(noise50) <- paste0("N", seq_len(ncol(noise50)))
    feature_data_50 <- cbind(feature_data, noise50)
    }

    # the forest with 10 noise variables

    # tune
    if (run_tuning) tune_helper(jump_data = jump_data, feature_data = feature_data_10,
                min_node_sizes = c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100),
                split_rules = c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice"),
                path = "results/biomedical_applications/CPHHolter/Data/Noise10")
    # love it when they agree, petoprentice with min_node_size = 15 it is

    if (run_tuning) tune_plot_helper(
        input_path = "results/biomedical_applications/CPHHolter/Data/Noise10",
        output_path = "results/biomedical_applications/CPHHolter/Plots/Noise10",
        min_node_sizes = c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100),
        split_rules = c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
    )
    # gehan seems to be best, but min_node_size = 5 and 10 are too small. I would actually, based on the intuition from earlier, go with 20,
    # for the separate VIMP exploration

    # choose the final forest to be fitted (only relevant for VIMP)
    fitted_forest_10 <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data_10, splitrule = "gehan",
                                 min_node_size = 20, seed = 2026)
    JumpForests:::print_forest(fitted_forest_10)
    # the scores are extremely close to the optima found above anyway

    # now do the error runs
    {
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_10, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/CPHHolter/Data/Noise10/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/CPHHolter/Data/Noise10/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/CPHHolter/Data/Noise10/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)
    }


    # for 10 noise variables in particular, it would be nice to try VIMP
    vimp_helper(fitted_forest = fitted_forest_10, variable_names = colnames(feature_data_10), B = 500,
                path = "results/biomedical_applications/CPHHolter/Data/Noise10")
    vimp_plot_helper(input_path = "results/biomedical_applications/CPHHolter/Data/Noise10", output_path = "results/biomedical_applications/CPHHolter/Plots/Noise10")
    # takes about 5 seconds per iteration when min_node_size = 20, so about 42 minutes total for B = 500
    # seems to work quite well, could still be worthwhile to try larger minimal node sizes

    # the forest with 25 noise variables

    # now do the error runs
    {
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_25, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/CPHHolter/Data/Noise25/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/CPHHolter/Data/Noise25/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/CPHHolter/Data/Noise25/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)
    }


    # the forest with 50 noise variables

    # now do the error runs
    {
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_50, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/CPHHolter/Data/Noise50/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/CPHHolter/Data/Noise50/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/CPHHolter/Data/Noise50/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)
    }
}

# 4. Bone marrow transplantation (bmt) ------------------------------------
if (run_analyses && "BMT" %in% datasets_to_run) {
    manuscript_prepare_real_dirs("BMT")
    # data preparation
    #--------------------------------------------------------------------------------

    {
    bmt <- read.csv("data/biomedical_applications/bmt.csv", sep = ",")
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
    JumpForests:::print_forest(null_forest)   # simple baseline is 3/16 = 0.1875, so 0.1646 is already better

    min_node_sizes <- c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200)
    split_rules <- c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
    best_tuning_results <- if (run_tuning) tune_helper(
        jump_data = jump_data, feature_data = feature_data,
        min_node_sizes = min_node_sizes, split_rules = split_rules,
        path = "results/biomedical_applications/BMT/Data", num_event_times = 1000
    )
    if (run_tuning) print(best_tuning_results)

    # now make hyperparameter tuning plots
    if (run_tuning) tune_plot_helper(
        input_path = "results/biomedical_applications/BMT/Data",
        output_path = "results/biomedical_applications/BMT/Plots",
        min_node_sizes = min_node_sizes, split_rules = split_rules
    )

    # based on the plots and the summary, we go with approxlogrank and min_node_size = 100, since
    # Brier is quite a lot better for 100 compared to 200, but the others are extremely close for
    # 100 and 200

    fitted_forest <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data, splitrule = "approxlogrank", 
                                      min_node_size = 100, seed = 2026, num_event_times = 1000)

    # VIMP
    #--------------------------------------------------------------------------------

    vimp_results <- vimp_helper(
        fitted_forest = fitted_forest, variable_names = names(feature_data), B = 100,
        path = "results/biomedical_applications/BMT/Data"
    )
    vimp_plot_helper(
        input_path = "results/biomedical_applications/BMT/Data/",
        output_path = "results/biomedical_applications/BMT/Plots/"
    )

    # most important variables seem to be "age", "bmonly" and "all" with "sex" somewhat degrading performance

    # a few exploratory plots
    #--------------------------------------------------------------------------------

    head(feature_data)
    summary(feature_data$age)

    # investigate age:bmonly
    new_data <- data.frame(sex = rep(0, 6), all = rep(0, 6), age = c(19, 32, 44, 19, 32, 44), bmonly = c(1,1,1,0,0,0))
    age_bmonly_predictions <- jfforest.predict(fitted_forest, new_data)

    plot_transition_rate(
        predictions = age_bmonly_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = new_data,
        from = 1,
        to = 4,
        colour_by = "age",
        linetype_by = "bmonly",
        max_time = NULL,
        xlab = "Time (months)",
        plot_directory = "results/biomedical_applications/BMT/Plots",
        save = TRUE
    )

    # For BMT -> death, the age curves have broadly similar bmonly separation, so
    # there is little visual evidence of an age:bmonly interaction. This conclusion
    # is transition-specific; the relapse and GvHD -> death plots show some
    # non-parallelism and should not be described as having no interaction.

    # investigate age:all
    new_data <- data.frame(sex = rep(0, 6), bmonly = rep(0, 6), age = c(19, 32, 44, 19, 32, 44), all = c(1,1,1,0,0,0))
    age_all_predictions <- jfforest.predict(fitted_forest, new_data)

    plot_transition_rate(
        predictions = age_all_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = new_data,
        from = 1,
        to = 3,
        colour_by = "age",
        linetype_by = "all",
        max_time = NULL,
        xlab = "Time (months)",
        plot_directory = "results/biomedical_applications/BMT/Plots",
        save = TRUE
    )

    plot_transition_rate(
        predictions = age_all_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = new_data, from = 2, to = 3,
        colour_by = "age", linetype_by = "all",
        xlab = "Time (months)",
        plot_directory = "results/biomedical_applications/BMT/Plots", save = TRUE
    )

    # There is visual evidence of age:all interaction for BMT -> relapse: the age
    # separation is substantially larger for all = 1 than for all = 0. Separate
    # plots of 1 -> 3 and 2 -> 3 have different y scales, so use the direct
    # comparisons below to assess the effect of being in the GvHD state.

    # investigate bmonly:all
    new_data <- data.frame(sex = rep(0, 4), age = rep(32, 4), bmonly = c(1,1,0,0), all = c(1,0,1,0))
    bmonly_all_predictions <- jfforest.predict(fitted_forest, new_data)

    plot_transition_rate(
        predictions = bmonly_all_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = new_data,
        from = 1,
        to = 3,
        colour_by = "bmonly",
        linetype_by = "all",
        max_time = NULL,
        xlab = "Time (months)",
        plot_directory = "results/biomedical_applications/BMT/Plots",
        save = TRUE
    )

    plot_transition_rate(
        predictions = bmonly_all_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = new_data, from = 2, to = 3,
        colour_by = "bmonly", linetype_by = "all",
        xlab = "Time (months)",
        plot_directory = "results/biomedical_applications/BMT/Plots", save = TRUE
    )

    # There is a bmonly:all interaction for BMT -> relapse but little such
    # interaction for GvHD -> relapse. For these profiles, relapse rates are higher
    # from BMT than from GvHD, whereas death rates are markedly higher from GvHD.
    # These are descriptive state-specific associations, not causal effects of GvHD.

    # Compare BMT and GvHD transition rates directly for every row of new_data. The
    # facets use a common y scale, so within-profile and across-profile differences
    # can both be read from the plots.
    profile_labels <- paste0(
        "bmonly=", new_data$bmonly, ", all=", new_data$all
    )

    plot_transition_rate_comparison(
        predictions = bmonly_all_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = new_data,
        transitions = data.frame(from = c(1, 2), to = c(3, 3)),
        transition_labels = c("BMT -> relapse", "GvHD -> relapse"),
        profile_labels = profile_labels,
        max_time = NULL,
        xlab = "Time (months)",
        plot_directory = "results/biomedical_applications/BMT/Plots",
        filename = "transition_rate_comparison_relapse_bmonly_all.pdf",
        ncol = 2,
        save = TRUE
    )

    # across the possible interactions for bmonly and all, relapse is higher from BMT (but should be more thoroughly investigated)
    # also, relapse is higher for all = 1

    plot_transition_rate_comparison(
        predictions = bmonly_all_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = new_data,
        transitions = data.frame(from = c(1, 2), to = c(4, 4)),
        transition_labels = c("BMT -> death", "GvHD -> death"),
        profile_labels = profile_labels,
        max_time = NULL,
        xlab = "Time (months)",
        plot_directory = "results/biomedical_applications/BMT/Plots",
        filename = "transition_rate_comparison_death_bmonly_all.pdf",
        ncol = 2,
        save = TRUE
    )

    # across the possible interactions for bmonly and all, death transition rate is higher from GvHD (but should be more thoroughly investigated)
    # also, the death transition rate is higher for all = 1, concurs with the higher relapse rate for all = 1

    # age:all again
    new_data <- data.frame(
        sex = rep(0, 6),
        bmonly = rep(0, 6),
        age = rep(c(19, 32, 44), each = 2),
        all = rep(c(0, 1), 3)
    )
    age_all_predictions <- jfforest.predict(fitted_forest, new_data)
    profile_labels <- paste0(
        "age=", new_data$age, ", all=", new_data$all
    )

    plot_transition_rate_comparison(
        predictions = age_all_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = new_data,
        transitions = data.frame(from = c(1, 2), to = c(3, 3)),
        transition_labels = c("BMT -> relapse", "GvHD -> relapse"),
        profile_labels = profile_labels,
        max_time = NULL,
        xlab = "Time (months)",
        plot_directory = "results/biomedical_applications/BMT/Plots",
        filename = "transition_rate_comparison_relapse_age_all.pdf",
        ncol = 2,
        save = TRUE
    )

    # a bit more interesting with relapse: higher for BMT, but only very slightly for all = 0 (interesting!), and the age does not
    # have that big of an effect

    plot_transition_rate_comparison(
        predictions = age_all_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = new_data,
        transitions = data.frame(from = c(1, 2), to = c(4, 4)),
        transition_labels = c("BMT -> death", "GvHD -> death"),
        profile_labels = profile_labels,
        max_time = NULL,
        xlab = "Time (months)",
        plot_directory = "results/biomedical_applications/BMT/Plots",
        filename = "transition_rate_comparison_death_age_all.pdf",
        ncol = 2,
        save = TRUE
    )

    # death pattern is consistent: higher for GvHD than for BMT, and age and all do not seem to have a big effect

    # baseline fit without covariates
    # AalenJohansen::aalen_johansen() identifies follow-up endpoints by matching
    # their times. The BMT times are rounded and heavily tied, so give tied
    # endpoints deterministic negligible offsets. Within a tie, observed events
    # precede censorings. This avoids removing subjects from the risk set at an
    # unrelated intermediate event with the same recorded time.
    null_jump_data <- jump_data
    endpoint_times <- vapply(
        null_jump_data, function(path) tail(path$times, 1L), numeric(1)
    )
    endpoint_censored <- vapply(null_jump_data, function(path) {
        tail(path$states, 1L) == tail(path$states, 2L)[1L]
    }, logical(1))
    endpoint_rank <- integer(length(null_jump_data))
    for (endpoint_time in unique(endpoint_times)) {
        tied <- which(endpoint_times == endpoint_time)
        tied <- tied[order(endpoint_censored[tied], tied)]
        endpoint_rank[tied] <- seq_along(tied)
    }
    for (i in seq_along(null_jump_data)) {
        endpoint <- length(null_jump_data[[i]]$times)
        null_jump_data[[i]]$times[endpoint] <-
            null_jump_data[[i]]$times[endpoint] + endpoint_rank[i] * 1e-8
    }

    null_fit <- AalenJohansen::aalen_johansen(null_jump_data, p = 3)

    # The Aalen--Johansen output has one entry per observed endpoint and therefore
    # can contain tied times. Keep the final cumulative estimate at each time so it
    # has the strictly increasing grid expected by the plotting helper.
    null_keep <- !duplicated(null_fit$t, fromLast = TRUE)
    null_event_times <- null_fit$t[null_keep]
    null_predictions <- list(null_fit$Lambda[null_keep])
    null_new_data <- data.frame(fit = "No covariates")

    plot_transition_rate_comparison(
        predictions = null_predictions,
        event_times = null_event_times,
        new_data = null_new_data,
        transitions = data.frame(from = c(1, 2), to = c(4, 4)),
        transition_labels = c("BMT -> death", "GvHD -> death"),
        profile_labels = "No covariates",
        max_time = NULL,
        xlab = "Time (months)",
        plot_directory = "results/biomedical_applications/BMT/Plots",
        filename = "transition_rate_comparison_death_null.pdf",
        save = TRUE
    )

    plot_transition_rate_comparison(
        predictions = null_predictions,
        event_times = null_event_times,
        new_data = null_new_data,
        transitions = data.frame(from = c(1, 2), to = c(3, 3)),
        transition_labels = c("BMT -> relapse", "GvHD -> relapse"),
        profile_labels = "No covariates",
        max_time = NULL,
        xlab = "Time (months)",
        plot_directory = "results/biomedical_applications/BMT/Plots",
        filename = "transition_rate_comparison_relapse_null.pdf",
        save = TRUE
    )

    # okay, so overall (no covariates), the trend is higher death transition rate from GvHD and higher relapse rate for BMT

    # conclusions: Across all covariate combinations, the death transition rate from the GvHD state is higher. Relapse is more
    # interesting. It is higher in general from the BMT state than the GvHD state by a solid margin for most combinations except
    # for age and all = 0! all = 1 shows substantially higher relapse rates from BMT, but only slightly larger rates from GvHD
    # compared to all = 0. 

    # potential selling point: You can make explore the data in a very interesting way with the JumpForest package. You can investigate
    # interactions for all transitions of interest without needing to specify a model up front and afterwards you fit a more
    # interpretable model for the particular transitions of interest. Also, it makes it easy to check visually for differences in
    # transitions from states of interest like for GvHD and BMT above.

    # mean errors across many runs compared to a transition-wise Cox model
    #--------------------------------------------------------------------------------

    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data, jump_data = jump_data, B = 100, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/BMT/Data/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/BMT/Data/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/BMT/Data/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)

    # load in data here
    ibs_bmt_harmonised <- read.table("results/biomedical_applications/BMT/Data/oob_errors_ibs_harmonised.txt", header = TRUE)
    ikl_bmt_harmonised <- read.table("results/biomedical_applications/BMT/Data/oob_errors_ikl_harmonised.txt", header = TRUE)
    ispherical_bmt_harmonised <- read.table("results/biomedical_applications/BMT/Data/oob_errors_ispherical_harmonised.txt", header = TRUE)

    colMeans(ibs_bmt_harmonised)

    colMeans(ikl_bmt_harmonised)

    colMeans(ispherical_bmt_harmonised)

    # Yes, we outperform Cox on all fronts here! And it actually makes sense. We have 2009 observations, and hence non-parametric
    # methods should begin to shine.

    # adding noise: errors across runs compared to Cox and VIMP showcase
    #--------------------------------------------------------------------------------

    head(feature_data)
    nrow(feature_data)

    {
    set.seed(2026)
    # add 10 noise variables
    noise10 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 10), nrow = nrow(feature_data)))
    colnames(noise10) <- paste0("N", seq_len(ncol(noise10)))
    feature_data_10 <- cbind(feature_data, noise10)

    # add 25 noise variables
    noise25 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 25), nrow = nrow(feature_data)))
    colnames(noise25) <- paste0("N", seq_len(ncol(noise25)))
    feature_data_25 <- cbind(feature_data, noise25)

    # add 50 noise variables
    noise50 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 50), nrow = nrow(feature_data)))
    colnames(noise50) <- paste0("N", seq_len(ncol(noise50)))
    feature_data_50 <- cbind(feature_data, noise50)
    }

    # the forest with 10 noise variables

    # tune
    if (run_tuning) tune_helper(jump_data = jump_data, feature_data = feature_data_10,
                min_node_sizes = c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200),
                split_rules = c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice"),
                path = "results/biomedical_applications/BMT/Data/Noise10")
    # apparently min_node_size = 5 with petoprentice is best according to IBS and spherical errors
    # I would not choose so deep trees in practice. I will run the VIMP with both suggested configurations

    if (run_tuning) tune_plot_helper(
        input_path = "results/biomedical_applications/BMT/Data/Noise10",
        output_path = "results/biomedical_applications/BMT/Plots/Noise10",
        min_node_sizes = c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200),
        split_rules = c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
    )

    # choose the final forest to be fitted (only relevant for VIMP)
    fitted_forest_10 <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data_10, splitrule = "logrank",
                                 min_node_size = 100, seed = 2026, num_event_times = 1000)
    JumpForests:::print_forest(fitted_forest_10)

    # now do the error runs
    {
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_10, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/BMT/Data/Noise10/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/BMT/Data/Noise10/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/BMT/Data/Noise10/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)
    }


    # for 10 noise variables in particular, it would be nice to try VIMP
    vimp_helper(fitted_forest = fitted_forest_10, variable_names = colnames(feature_data_10), B = 100,
                path = "results/biomedical_applications/BMT/Data/Noise10")
    vimp_plot_helper(input_path = "results/biomedical_applications/BMT/Data/Noise10", output_path = "results/biomedical_applications/BMT/Plots/Noise10")
    # YES, it works exactly as intended! All the noise variables are deemed as unimportant
    # Takes a very long time when min_node_size = 5, about 37 seconds for an iteration (an hour for 100)
    # for min_node_size = 100, it only takes about 15 seconds per iteration (about 25 minutes for 100)

    # IMPORTANT: As I suspected, the min_node_size = 5 forest performs quite a loss worse. The correct ordering of
    # the variables is kept, but the VIMP values are much stronger concentrated around zero for min_node_size = 100
    # exactly as they should be. This highlights the importance of having sufficiently large nodes, despite what the
    # naive tuning may say. There also seems to be a tendency for the noised up data to prefer deeper trees. This is
    # important to keep in mind. Namely that while the errors are better for smaller minimal node sizes, it is usually
    # better to choose a more moderate size. The rule of thumb with a minimal node size slightly larger than 
    # sqrt(n) is actually quite excellent and it is even supported by theory.

    # the forest with 25 noise variables

    # now do the error runs
    {
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_25, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/BMT/Data/Noise25/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/BMT/Data/Noise25/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/BMT/Data/Noise25/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)

    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_50, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/BMT/Data/Noise50/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/BMT/Data/Noise50/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/BMT/Data/Noise50/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)
    }

    # Noise50 is computed in the preceding block.
}

# 5. Platelet recovery (ebmt3) --------------------------------------------
if (run_analyses && "EBMT" %in% datasets_to_run) {
    manuscript_prepare_real_dirs("EBMT")
    # data preparation
    #--------------------------------------------------------------------------------
    {
    options(scipen = 1)
    ebmt3 <- read.csv("data/biomedical_applications/ebmt3.csv")
    # CSVs do not retain factor metadata. Restore the mstate::ebmt3 levels
    # so the model matrices and reference categories match the source data.
    ebmt3$dissub <- factor(ebmt3$dissub, levels = c("AML", "ALL", "CML"))
    ebmt3$age <- factor(ebmt3$age, levels = c("<=20", "20-40", ">40"))
    ebmt3$drmatch <- factor(ebmt3$drmatch,
                          levels = c("No gender mismatch", "Gender mismatch"))
    ebmt3$tcd <- factor(ebmt3$tcd, levels = c("No TCD", "TCD"))
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
            # relapse/death is an observed transition from the current state
            states <- c(states, 3L)
        } else {
            # repeating the current state at the endpoint denotes censoring
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
    JumpForests:::print_forest(null_forest)

    min_node_sizes <- c(5, 10, 25, 50, 100)
    split_rules <- c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
    best_tuning_results <- if (run_tuning) tune_helper(
        jump_data = jump_data, feature_data = feature_data,
        min_node_sizes = min_node_sizes, split_rules = split_rules,
        path = "results/biomedical_applications/EBMT/Data"
    )
    if (run_tuning) {
        print(best_tuning_results)
        write.table(best_tuning_results, "results/biomedical_applications/EBMT/tuning_ebmt3.txt", sep = "\t", row.names = FALSE)
    }

    fitted_forest <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data, splitrule = "taroneware", min_node_size = 100, seed = 2026)
    JumpForests:::print_forest(fitted_forest)

    # VIMP
    #--------------------------------------------------------------------------------

    vimp_results <- vimp_helper(
        fitted_forest = fitted_forest, variable_names = names(feature_data), B = 100,
        path = "results/biomedical_applications/EBMT/Data"
    )
    vimp_plot_helper(
        input_path = "results/biomedical_applications/EBMT/Data",
        output_path = "results/biomedical_applications/EBMT/Plots"
    )

    lapply(vimp_results, colMeans)

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
        plot_directory = "results/biomedical_applications/EBMT/Plots",
        save = TRUE
    )

    for (from in c(1L, 2L)) {
        plot_transition_rate(
            predictions = age_tcd_predictions,
            event_times = fitted_forest$unique.event.times,
            new_data = new_data, from = from, to = 3L,
            colour_by = "age", linetype_by = "tcd",
            max_time = 500, xlab = "Time (days)",
            plot_directory = "results/biomedical_applications/EBMT/Plots", save = TRUE
        )
    }

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
        plot_directory = "results/biomedical_applications/EBMT/Plots",
        save = TRUE
    )

    plot_transition_rate(
        predictions = age_dissub_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = new_data, from = 1L, to = 3L,
        colour_by = "age", linetype_by = "dissub",
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/EBMT/Plots", save = TRUE
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
    error_censoring_survival <- censoring_survival_at(error_censoring_model, error_evaluation_times)
    error_evaluation_times <- error_evaluation_times[error_censoring_survival >= minimum_error_censoring_survival]
    initial_distribution <- c(1, 0, 0)
    number_of_error_folds <- 5L
    error_folds <- stratified_multistate_folds(jump_data, number_of_error_folds, seed = 2026)

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
        plot_directory = "results/biomedical_applications/EBMT/Plots"
    )

    write.table(
        ebmt_error_comparison$curves,
        "results/biomedical_applications/EBMT/time_dependent_errors_ebmt3.txt",
        sep = "\t", row.names = FALSE
    )
    write.table(
        ebmt_error_comparison$integrated,
        "results/biomedical_applications/EBMT/integrated_errors_ebmt3.txt",
        sep = "\t", row.names = FALSE
    )
    ebmt_error_comparison$integrated

    # mean errors across many runs compared to a transition-wise Cox model
    #--------------------------------------------------------------------------------

    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data, jump_data = jump_data, B = 100, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/EBMT/Data/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/EBMT/Data/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/EBMT/Data/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)

    # load in the data
    ibs_ebmt_harmonised <- read.table("results/biomedical_applications/EBMT/Data/oob_errors_ibs_harmonised.txt", header = TRUE)
    ikl_ebmt_harmonised <- read.table("results/biomedical_applications/EBMT/Data/oob_errors_ikl_harmonised.txt", header = TRUE)
    ispherical_ebmt_harmonised <- read.table("results/biomedical_applications/EBMT/Data/oob_errors_ispherical_harmonised.txt", header = TRUE)

    colMeans(ibs_ebmt_harmonised)

    colMeans(ikl_ebmt_harmonised)

    colMeans(ispherical_ebmt_harmonised)

    # adding noise: errors across runs compared to Cox and VIMP showcase
    #--------------------------------------------------------------------------------

    head(feature_data)
    nrow(feature_data)

    {
    set.seed(2026)
    # add 10 noise variables
    noise10 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 10), nrow = nrow(feature_data)))
    colnames(noise10) <- paste0("N", seq_len(ncol(noise10)))
    feature_data_10 <- cbind(feature_data, noise10)

    # add 25 noise variables
    noise25 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 25), nrow = nrow(feature_data)))
    colnames(noise25) <- paste0("N", seq_len(ncol(noise25)))
    feature_data_25 <- cbind(feature_data, noise25)

    # add 50 noise variables
    noise50 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 50), nrow = nrow(feature_data)))
    colnames(noise50) <- paste0("N", seq_len(ncol(noise50)))
    feature_data_50 <- cbind(feature_data, noise50)
    }

    # the forest with 10 noise variables

    # tune
    if (run_tuning) tune_helper(jump_data = jump_data, feature_data = feature_data_10,
                min_node_sizes = c(5, 10, 25, 50, 100),
                split_rules = c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice"),
                path = "results/biomedical_applications/EBMT/Data/Noise10")
    # apparently min_node_size = 5 with approxlogrank is best according to IBS and spherical errors
    # I would not choose so deep trees in practice. I will run VIMP with min_node_size = 100 instead

    if (run_tuning) tune_plot_helper(
        input_path = "results/biomedical_applications/EBMT/Data/Noise10",
        output_path = "results/biomedical_applications/EBMT/Plots/Noise10",
        min_node_sizes = c(5, 10, 25, 50, 100),
        split_rules = c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
    )

    # choose the final forest to be fitted (only relevant for VIMP)
    fitted_forest_10 <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data_10, splitrule = "approxlogrank",
                                 min_node_size = 100, seed = 2026, num_event_times = 1000)
    JumpForests:::print_forest(fitted_forest_10)
    # looks good for min_node_size = 100

    # now do the error runs
    {
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_10, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/EBMT/Data/Noise10/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/EBMT/Data/Noise10/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/EBMT/Data/Noise10/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)
    }


    # for 10 noise variables in particular, it would be nice to try VIMP
    vimp_helper(fitted_forest = fitted_forest_10, variable_names = colnames(feature_data_10), B = 100,
                path = "results/biomedical_applications/EBMT/Data/Noise10")
    vimp_plot_helper(input_path = "results/biomedical_applications/EBMT/Data/Noise10", output_path = "results/biomedical_applications/EBMT/Plots/Noise10")
    # for min_node_size = 100, it takes about 9 seconds for one iteration, about 15 minutes total
    # we could run the VIMP for min_node_size = 5 also, and I think we would confirm that 100 is the better choice for accurate VIMP

    # the forest with 25 noise variables

    # now do the error runs
    {
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_25, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/EBMT/Data/Noise25/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/EBMT/Data/Noise25/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/EBMT/Data/Noise25/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)
    }

    # the forest with 50 noise variables

    # now do the error runs
    {
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_50, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/EBMT/Data/Noise50/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/EBMT/Data/Noise50/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/EBMT/Data/Noise50/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)
    }

    colMeans(result_harmonised$ispherical)
}

# 6. Liver cirrhosis trial (prova) ----------------------------------------
if (run_analyses && "PROVA" %in% datasets_to_run) {
    manuscript_prepare_real_dirs("PROVA")
    # data preparation
    #--------------------------------------------------------------------------------

    {
    prova <- read.csv("data/biomedical_applications/prova.csv", sep = ",")  # for VSCode

    # check for missing values
    length(prova$coag[is.na(prova$coag)]) # 14 missing values for coag (column 11)
    prova$coag[is.na(prova$sex)]
    prova$coag[is.na(prova$age)]
    prova$coag[is.na(prova$bili)] # some missing values, 10 (column 10)
    prova$coag[is.na(prova$varsize)]

    # the covariates with missing values are not of interest, so could remove them
    # or simply impute them with the median

    # prepare jump data (simple illness death model without recovery)

    jump_data <- lapply(seq_len(nrow(prova)), function(i) {
      states <- 1L
      times <- 0
      
      # bleeding takes place
      if (prova$bleed[i] == 1) {
        states <- c(states, 2L)
        times <- c(times, as.numeric(prova$timebleed[i]))
      }
      if (prova$death[i] == 1) {
        # death is an observed transition from the current state
        states <- c(states, 3L)
      } else {
        # repeating the current state at the endpoint denotes censoring
        states <- c(states, tail(states, 1L))
      }
      times <- c(times, as.numeric(prova$timedeath[i]))
      
      list(times = times, states = states)
    })

    # retain all covariates and replace missing values by the non-missing column median
    feature_data <- prova[, 6:12]
    for (feature in c("bili", "coag")) {
      feature_data[[feature]][is.na(feature_data[[feature]])] <- median(feature_data[[feature]], na.rm = TRUE)
    }
    }

    # tune the forest
    #--------------------------------------------------------------------------------

    nrow(prova)
    sqrt(nrow(prova))
    null_forest <- jfforest(MM ~ 1, data = jump_data, feature_data = feature_data, min_node_size = 20, seed = 2026)
    JumpForests:::print_forest(null_forest)

    min_node_sizes <- c(2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50)
    split_rules <- c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
    best_tuning_results <- if (run_tuning) tune_helper(
        jump_data = jump_data, feature_data = feature_data,
        min_node_sizes = min_node_sizes, split_rules = split_rules,
        path = "results/biomedical_applications/PROVA/Data"
    )
    if (run_tuning) print(best_tuning_results)
    jfforest.error(null_forest)

    # now make hyperparameter tuning plots
    if (run_tuning) tune_plot_helper(
        input_path = "results/biomedical_applications/PROVA/Data",
        output_path = "results/biomedical_applications/PROVA/Plots",
        min_node_sizes = min_node_sizes, split_rules = split_rules
    )
    # best choice seems to be min_node_size = 2 with petoprentice, but choosing such a small minimum node size
    # will be problematic later, especially with rare events and since we have so little data. A minimum node size
    # of about 15-20 (close to the square root of the data size) should be more appropriate. The plots also indicate
    # a local minimum close to 15 and 20. We go with the latter, since we have to be able to capture rare events.
    # petoprentice seems to be very strong for this particular dataset
    fitted_forest <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data, splitrule = "petoprentice", min_node_size = 20, seed = 2026)

    # VIMP
    #--------------------------------------------------------------------------------

    vimp_results <- vimp_helper(
        fitted_forest = fitted_forest, variable_names = names(feature_data), B = 500,
        path = "results/biomedical_applications/PROVA/Data"
    )
    vimp_plot_helper(
        input_path = "results/biomedical_applications/PROVA/Data",
        output_path = "results/biomedical_applications/PROVA/Plots"
    )

    # the most important covariate is coag, followed by bili, sex, varsize, age
    # beta and in some cases scle are borderline important and beta seems to degrade performance? interesting

    JumpForests:::print_forest(fitted_forest)

    # as a test, let us fit a forest without the age covariate
    fitted_forest_noage <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data[,-4], splitrule = "petoprentice", min_node_size = 2, seed = 2026)
    JumpForests:::print_forest(fitted_forest_noage) # no, performs worse, worth a try

    # exploratory plots for the four treatment groups and the effect of bleeding
    #--------------------------------------------------------------------------------

    summary(feature_data$age)
    summary(feature_data$coag)
    summary(feature_data$bili)
    summary(feature_data$varsize)

    # first I want to see how higher levels of coag and bili on their own affect the transition rates
    # (just for the women in the control group beta = 0 = scle)
    coag_bili <- data.frame(coag = c(49, 65, 81.75, 49, 65, 81.75, 49, 65, 81.75), bili = c(13.25, 13.25, 13.25, 23, 23, 23, 43.75, 43.75, 43.75),
                           age = rep(54, 9), beta = rep(0, 9), scle = rep(0, 9), sex = rep(0, 9), varsize = rep(2.0, 9))
    coag_bili_predictions <- jfforest.predict(fitted_forest, coag_bili)

    profile_labels <- paste0("coag=", coag_bili$coag, ", bili=", coag_bili$bili)
    plot_transition_rate_comparison(
        predictions = coag_bili_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = coag_bili,
        transitions = data.frame(from = c(1, 2), to = c(3, 3)),
        transition_labels = c("no bleeding -> death", "bleeding -> death"),
        profile_labels = profile_labels,
        max_time = NULL,
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/PROVA/Plots",
        filename = "transition_rate_comparison_death_coag_bili.pdf",
        ncol = 3,
        save = TRUE
    )

    # okay, the bleeding->death transition is higher across the board (not surprising)
    # there does not seem to be a strong interaction effect between bili and coag, except for the highest coag values
    # it seems that higher coagulation decreases the bleeding transition rate, which I guess makes sense

    # now check the treatment groups
    treatment <- data.frame(beta = c(1,1,0,0), scle = c(1,0,1,0), coag = rep(65, 4), bili = rep(23, 4), 
                            age = rep(54, 4), sex = rep(0, 4), varsize = rep(2.0, 4))
    treatment_predictions <- jfforest.predict(fitted_forest, treatment)

    profile_labels <- paste0("beta=", treatment$beta, ", scle=", treatment$scle)
    plot_transition_rate_comparison(
        predictions = treatment_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = treatment,
        transitions = data.frame(from = c(1, 2), to = c(3, 3)),
        transition_labels = c("no bleeding -> death", "bleeding -> death"),
        profile_labels = profile_labels,
        max_time = NULL,
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/PROVA/Plots",
        filename = "transition_rate_comparison_death_treatment.pdf",
        ncol = 2,
        save = TRUE
    )

    # the treatments seem to have quite little effect, in fact the scle treatment seems to be somewhat hurtful
    # but one should be careful in concluding anything here, in fact bleeding seems to be extremely important
    # for the risk of death, so we should rather focus on the no bleeding -> bleeding transition

    plot_transition_rate_comparison(
        predictions = treatment_predictions,
        event_times = fitted_forest$unique.event.times,
        new_data = treatment,
        transitions = data.frame(from = 1, to = 2),
        transition_labels = c("no bleeding -> bleeding"),
        profile_labels = profile_labels,
        max_time = NULL,
        xlab = "Time (days)",
        plot_directory = "results/biomedical_applications/PROVA/Plots",
        filename = "transition_rate_comparison_bleeding_treatment.pdf",
        ncol = 2,
        save = TRUE
    )

    # honestly quite hard to spot a difference

    # let us follow Munch et al and not consider beta at all (beta = 0)
    # it would be interesting to investigate whether there is a difference in efficiency of treatment based
    # on the baseline covariates, especially since coag and bili are so important (likely a lot more than the treatments)

    # Compare sclerotherapy at the lower and upper quartiles of coag and bili. All
    # other covariates are held fixed, including beta = 0.
    treatment_coag_bili <- expand.grid(
        scle = c(0, 1),
        bili = c(13.25, 43.75),
        coag = c(49, 81.75),
        KEEP.OUT.ATTRS = FALSE
    )
    treatment_coag_bili$beta <- 0
    treatment_coag_bili$age <- 54
    treatment_coag_bili$sex <- 0
    treatment_coag_bili$varsize <- 2
    predictions_treatment_coag_bili <- jfforest.predict(fitted_forest, treatment_coag_bili)

    # Put the two treatment curves in the same panel. Faceting by bili and coag
    # makes a change in the treatment separation across panels visible as an
    # interaction.
    plot_scle_interaction <- function(from, to, transition_label, filename) {
        plot_data <- do.call(rbind, lapply(seq_len(nrow(treatment_coag_bili)), function(i) {
            cumulative_rate <- vapply(
                predictions_treatment_coag_bili[[i]],
                function(rate_matrix) rate_matrix[from, to],
                numeric(1)
            )
            data.frame(
                time = fitted_forest$unique.event.times,
                cumulative_rate = cumulative_rate,
                scle = treatment_coag_bili$scle[i],
                bili = treatment_coag_bili$bili[i],
                coag = treatment_coag_bili$coag[i]
            )
        }))

        plot_data$scle <- factor(
            plot_data$scle,
            levels = c(0, 1),
            labels = c("No sclerotherapy", "Sclerotherapy")
        )
        plot_data$bili <- factor(
            plot_data$bili,
            levels = sort(unique(treatment_coag_bili$bili)),
            labels = paste0("bili = ", sort(unique(treatment_coag_bili$bili)))
        )
        plot_data$coag <- factor(
            plot_data$coag,
            levels = sort(unique(treatment_coag_bili$coag)),
            labels = paste0("coag = ", sort(unique(treatment_coag_bili$coag)))
        )

        p <- ggplot(
            plot_data,
            aes(x = time, y = cumulative_rate, colour = scle, group = scle)
        ) +
            geom_step(linewidth = 0.9, direction = "hv") +
            facet_grid(rows = vars(bili), cols = vars(coag)) +
            scale_colour_manual(values = c("#0072B2", "#D55E00")) +
            labs(
                title = NULL,
                x = "Time (days)",
                y = "Cumulative transition rate",
                colour = NULL
            ) +
            theme_classic() +
            theme(
                legend.position = "bottom",
                plot.title = element_text(hjust = 0.5)
            )

        ggsave(
            filename = file.path("results/biomedical_applications/PROVA/Plots", filename),
            plot = p,
            width = 10,
            height = 7,
            units = "in",
            dpi = 300
        )
        p
    }

    scle_bleeding_plot <- plot_scle_interaction(
        from = 1, to = 2,
        transition_label = "no bleeding -> bleeding",
        filename = "scle_interaction_bleeding_coag_bili.pdf"
    )
    scle_death_without_bleeding_plot <- plot_scle_interaction(
        from = 1, to = 3,
        transition_label = "no bleeding -> death",
        filename = "scle_interaction_death_without_bleeding_coag_bili.pdf"
    )
    scle_death_after_bleeding_plot <- plot_scle_interaction(
        from = 2, to = 3,
        transition_label = "bleeding -> death",
        filename = "scle_interaction_death_after_bleeding_coag_bili.pdf"
    )

    # alright, from these plots, scle seems to work (and no effect at all on no bleeding -> bleeding,
    # which also makes sense)

    # mean errors across many runs compared to a transition-wise Cox model
    #--------------------------------------------------------------------------------

    # for this particular dataset, some imputation was needed. We choose to make one imputation across the whole
    # dataset and not do imputation in every training sample.

    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data, jump_data = jump_data, B = 100, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/PROVA/Data/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/PROVA/Data/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/PROVA/Data/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)

    # adding noise: errors across runs compared to Cox and VIMP showcase
    #--------------------------------------------------------------------------------

    head(feature_data)
    nrow(feature_data)

    {
    set.seed(2026)
    # add 10 noise variables
    noise10 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 10), nrow = nrow(feature_data)))
    colnames(noise10) <- paste0("N", seq_len(ncol(noise10)))
    feature_data_10 <- cbind(feature_data, noise10)

    # add 25 noise variables
    noise25 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 25), nrow = nrow(feature_data)))
    colnames(noise25) <- paste0("N", seq_len(ncol(noise25)))
    feature_data_25 <- cbind(feature_data, noise25)

    # add 50 noise variables
    noise50 <- as.data.frame(matrix(rnorm(nrow(feature_data) * 50), nrow = nrow(feature_data)))
    colnames(noise50) <- paste0("N", seq_len(ncol(noise50)))
    feature_data_50 <- cbind(feature_data, noise50)
    }

    # the forest with 10 noise variables

    # tune
    if (run_tuning) tune_helper(jump_data = jump_data, feature_data = feature_data_10,
                min_node_sizes = c(10, 15, 20, 25, 30, 35, 40, 45, 50),
                split_rules = c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice"),
                path = "results/biomedical_applications/PROVA/Data/Noise10")
    # love it when they agree, petoprentice with min_node_size = 15 it is

    if (run_tuning) tune_plot_helper(
        input_path = "results/biomedical_applications/PROVA/Data/Noise10",
        output_path = "results/biomedical_applications/PROVA/Plots/Noise10",
        min_node_sizes = c(10, 15, 20, 25, 30, 35, 40, 45, 50),
        split_rules = c("logrank", "gehan", "taroneware", "approxlogrank", "petoprentice")
    )

    # choose the final forest to be fitted (only relevant for VIMP)
    fitted_forest_10 <- jfforest(MM ~ ., data = jump_data, feature_data = feature_data_10, splitrule = "petoprentice",
                                 min_node_size = 15, seed = 2026)
    JumpForests:::print_forest(fitted_forest_10)

    # now do the error runs
    {
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_10, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/PROVA/Data/Noise10/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/PROVA/Data/Noise10/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/PROVA/Data/Noise10/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)
    }


    # for 10 noise variables in particular, it would be nice to try VIMP
    vimp_helper(fitted_forest = fitted_forest_10, variable_names = colnames(feature_data_10), B = 500,
                path = "results/biomedical_applications/PROVA/Data/Noise10")
    vimp_plot_helper(input_path = "results/biomedical_applications/PROVA/Data/Noise10", output_path = "results/biomedical_applications/PROVA/Plots/Noise10")
    # YES, it works exactly as intended! All the noise variables are deemed as unimportant

    # the forest with 25 noise variables

    # now do the error runs
    {
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_25, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/PROVA/Data/Noise25/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/PROVA/Data/Noise25/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/PROVA/Data/Noise25/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)
    }


    # the forest with 50 noise variables

    # now do the error runs
    {
    result_harmonised <- compute_dataset_errors_harmonised(feature_data = feature_data_50, jump_data = jump_data, B = 100, penalised = TRUE, null_model = TRUE)

    write.table(result_harmonised$ibs, file = "results/biomedical_applications/PROVA/Data/Noise50/oob_errors_ibs_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ikl, file = "results/biomedical_applications/PROVA/Data/Noise50/oob_errors_ikl_harmonised.txt", sep = "\t", row.names = FALSE)
    write.table(result_harmonised$ispherical, file = "results/biomedical_applications/PROVA/Data/Noise50/oob_errors_ispherical_harmonised.txt", sep = "\t", row.names = FALSE)
    }
}

# 7. Combined absolute and relative error grids ---------------------------
if (make_comparison_plots) {
    for (metric in c("ibs", "ikl", "is")) {
        plot_dataset_noise_error_grid(
            error = metric, input_path = "results/biomedical_applications",
            output_path = "results/biomedical_applications/Plots", harmonised = TRUE
        )
        plot_dataset_noise_relative_error_grid(
            error = metric, input_path = "results/biomedical_applications",
            output_path = "results/biomedical_applications/Plots", harmonised = TRUE
        )
    }
}

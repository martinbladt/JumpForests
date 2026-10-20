# Study helpers. Loading this file does not run a study.

threshold_interaction_cox_formulas <- list(
    cox_x1_x2 = survival::Surv(tstart, tstop, event) ~ X1 * X2 + X3,
    cox_all_interactions = survival::Surv(tstart, tstop, event) ~ X1 * X2 * X3
)

sim_threshold_interaction_data <- function(n, seed = 2026) {
    set.seed(seed)
    X1 <- rbinom(n, 1L, 1 / 3)
    X2 <- runif(n, -1, 1)
    X3 <- runif(n)
    feature_data <- data.frame(X1, X2, X3)

    A <- matrix(c(-0.4, 0.25, 0.15, 0.2, -0.45, 0.25, 0, 0, 0), nrow = 3L, byrow = TRUE)

    sim <- vector("list", n)
    for (i in 1:n) {
    intensity <- (1 + 3 * X1[i] * (abs(X2[i]) <= 1 / 2) * (1 + 2 * X3[i])) * A
    rates <- function(j, t, u) -intensity[j, j]
    mark_dist <- function(j, t, u) {
        probabilities <- intensity[j, ]
        probabilities[j] <- 0
        probabilities / -intensity[j, j]
    }
    sim[[i]] <- AalenJohansen::sim_path(
        1L, rates = rates, dists = mark_dist,
        abs = c(FALSE, FALSE, TRUE), bs = -diag(intensity)
    )
    }
    list(jump_data = sim, feature_data = feature_data)
}


load_threshold_interaction_data <- function(n, path) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
    cache_path <- file.path(path, "simulated_data_seed2026.rds")
    if (file.exists(cache_path)) {
        result <- readRDS(cache_path)
        if (!is.list(result) || !is.data.frame(result$feature_data) ||
            nrow(result$feature_data) != n || length(result$jump_data) != n) {
            stop("cached simulated data are incompatible with n = ", n)
        }
        return(result)
    }
    result <- sim_threshold_interaction_data(n, seed = 2026)
    saveRDS(result, cache_path)
    result
}


run_threshold_interaction_harmonised_study <- function(data, path, B = 100,
                                         seed = 2026, num_event_times = 0) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
    metrics <- c("ibs", "ikl", "ispherical")
    model_names <- c("logrank", "rsf", "cox", names(threshold_interaction_cox_formulas))
    output_paths <- file.path(
        path, paste0("oob_errors_", metrics, "_harmonised.txt")
    )
    configuration_path <- file.path(path, "harmonised_study_configuration.rds")
    configuration <- list(
        n = as.integer(nrow(data$feature_data)), B = as.integer(B),
        seed = as.integer(seed), num_event_times = as.integer(num_event_times),
        models = model_names, splitrule = "logrank",
        jf_min_node_size = "floor(sqrt(training_subjects)) + 1",
        rsf_min_node_size = "floor(sqrt(transition_training_intervals)) + 1",
        cox_formulas = lapply(threshold_interaction_cox_formulas, function(formula) {
            paste(deparse(formula), collapse = "")
        })
    )
    if (all(file.exists(output_paths))) {
        if (!file.exists(configuration_path) ||
            !identical(readRDS(configuration_path), configuration)) {
            stop("Existing threshold-interaction tables have a different or unknown configuration in ", path)
        }
        saved <- lapply(output_paths, utils::read.table,
                        header = TRUE, check.names = FALSE)
        if (!all(vapply(saved, function(table) {
            nrow(table) == B && identical(names(table), model_names)
        }, logical(1)))) {
            stop("Existing threshold-interaction tables have incompatible rows or models in ", path)
        }
        message("Keeping existing threshold-interaction harmonised tables in ", path)
        return(invisible(setNames(saved, metrics)))
    }
    if (any(file.exists(output_paths)) || file.exists(configuration_path)) {
        stop("Incomplete threshold-interaction tables in ", path, "; inspect before rerunning")
    }

    result <- compute_dataset_errors_harmonised(
        feature_data = data$feature_data, jump_data = data$jump_data,
        B = B, seed = seed, rsf = TRUE, num_event_times = num_event_times,
        cox_formulas = threshold_interaction_cox_formulas
    )
    for (i in seq_along(metrics)) {
        write.table(result[[metrics[i]]], file = output_paths[i],
                    sep = "\t", row.names = FALSE)
    }
    saveRDS(configuration, configuration_path)
    invisible(result)
}


run_threshold_interaction_noise_interactions <- function(data, path, seed = 2026) {
    validate_jump_paths(data$jump_data)
    if (!is.data.frame(data$feature_data) ||
        nrow(data$feature_data) != length(data$jump_data) ||
        !identical(names(data$feature_data), c("X1", "X2", "X3")) ||
        anyNA(data$feature_data)) {
        stop("data must be the cached threshold-interaction sample with X1, X2 and X3")
    }
    if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed) ||
        seed < 0 || seed > .Machine$integer.max || seed != as.integer(seed)) {
        stop("seed must be a non-negative integer")
    }
    source_paths <- file.path(path, paste0(
        "ibs_errors_", c("jump_forest", "cox", "cox_lasso"), ".txt"
    ))
    if (!all(file.exists(source_paths))) {
        stop("the three completed original noise-study tables are required in ", path)
    }
    original <- lapply(source_paths, utils::read.table,
                       header = TRUE, sep = "\t", check.names = FALSE)
    B <- nrow(original[[1L]])
    table_names <- names(original[[1L]])
    if (B < 1L || length(table_names) < 1L ||
        !identical(table_names, paste0("noise_", seq_along(table_names) - 1L)) ||
        !all(vapply(original, function(table) {
            nrow(table) == B && identical(names(table), table_names) &&
                all(vapply(table, is.numeric, logical(1))) &&
                all(is.finite(as.matrix(table))) && all(as.matrix(table) >= 0)
        }, logical(1)))) {
        stop("original tables must contain matching, complete noise_0,... columns")
    }

    number_of_observations <- nrow(data$feature_data)
    training_size <- floor(0.8 * number_of_observations)
    number_of_noise_variables <- length(table_names) - 1L
    # Replay both independent RNG resets from compute_noise_ibs().
    set.seed(seed)
    if (number_of_noise_variables > 0L) {
        noise_data <- as.data.frame(matrix(
            rnorm(number_of_observations * number_of_noise_variables),
            nrow = number_of_observations
        ))
        noise_names <- paste0("noise_", seq_len(number_of_noise_variables))
        names(noise_data) <- tail(
            make.unique(c(names(data$feature_data), noise_names)),
            number_of_noise_variables
        )
    } else {
        noise_data <- data.frame(row.names = seq_len(number_of_observations))
    }
    set.seed(seed)
    training_splits <- replicate(
        B, sample(seq_len(number_of_observations), size = training_size),
        simplify = FALSE
    )
    # The original noise study used the full training transition-time grid.
    # It can be reconstructed directly without fitting another forest.
    evaluation_grids <- lapply(training_splits, function(indices) {
        transition_times <- unlist(lapply(data$jump_data[indices], function(p) {
            changed <- head(p$states, -1L) != tail(p$states, -1L)
            tail(p$times, -1L)[changed]
        }))
        sort(unique(c(0, transition_times)))
    })
    formulas <- list(
        cox_x1_x2 = survival::Surv(tstart, tstop, event) ~ X1 * X2 + X3,
        cox_all_interactions = survival::Surv(tstart, tstop, event) ~ X1 * X2 * X3
    )
    output_paths <- file.path(path, paste0("ibs_errors_", names(formulas), ".txt"))
    checkpoint_path <- file.path(path, "ibs_errors_cox_interactions_checkpoint.rds")
    object_md5 <- function(object) {
        temporary <- tempfile(fileext = ".rds")
        on.exit(unlink(temporary))
        saveRDS(object, temporary, compress = FALSE)
        unname(tools::md5sum(temporary))
    }
    configuration <- list(
        version = 1L, seed = as.integer(seed), rng_kind = RNGkind(),
        B = B, noise_columns = table_names,
        data_md5 = object_md5(data),
        design_md5 = object_md5(list(noise_data, training_splits)),
        original_tables_md5 = unname(tools::md5sum(source_paths)),
        formulas = lapply(formulas, function(f) paste(deparse(f), collapse = ""))
    )
    errors <- setNames(lapply(formulas, function(f) {
        matrix(NA_real_, nrow = B, ncol = length(table_names),
               dimnames = list(NULL, table_names))
    }), names(formulas))
    if (file.exists(checkpoint_path)) {
        checkpoint <- readRDS(checkpoint_path)
        if (!identical(checkpoint$configuration, configuration) ||
            !identical(names(checkpoint$errors), names(errors)) ||
            !all(vapply(checkpoint$errors, function(table) {
                is.matrix(table) && is.numeric(table) &&
                    identical(dim(table), c(B, length(table_names))) &&
                    identical(colnames(table), table_names) &&
                    all(is.finite(table[!is.na(table)])) &&
                    all(table[!is.na(table)] >= 0)
            }, logical(1)))) {
            stop("the interaction checkpoint does not match this dataset and study")
        }
        errors <- checkpoint$errors
    } else {
        if (any(file.exists(output_paths))) {
            stop("interaction tables exist without their checkpoint; inspect before rerunning")
        }
        # Verify the reconstruction against saved main-effect Cox scores at
        # the first and last noise levels. These are two small reference fits,
        # not a repeat of the original study; no JF or Cox Lasso is fitted.
        first_training <- training_splits[[1L]]
        first_test <- setdiff(seq_len(number_of_observations), first_training)
        for (noise_index in unique(c(1L, length(table_names)))) {
            features <- cbind(data$feature_data,
                              noise_data[seq_len(noise_index - 1L)])
            reference <- cox_test_errors(
                features, data$jump_data, first_training, first_test,
                evaluation_times = evaluation_grids[[1L]]
            )$IBS
            if (!isTRUE(all.equal(as.numeric(reference),
                                 as.numeric(original[[2L]][1L, noise_index]),
                                 tolerance = 1e-8))) {
                stop("reconstructed Cox score disagrees with the saved ",
                     table_names[noise_index], " score; check the dataset, seed and scoring code")
            }
        }
        message("Verified the replay against saved Cox scores at the first and last noise levels")
    }
    write_checkpoint <- function() {
        staged <- tempfile(pattern = ".threshold-interaction-cox-", tmpdir = path)
        on.exit(unlink(staged))
        saveRDS(list(configuration = configuration, errors = errors),
                staged, compress = FALSE)
        if (!file.rename(staged, checkpoint_path)) {
            stop("could not save interaction-study checkpoint")
        }
    }
    write_checkpoint()

    for (noise_index in seq_along(table_names)) {
        added_noise <- names(noise_data)[seq_len(noise_index - 1L)]
        features <- cbind(data$feature_data, noise_data[added_noise])
        noise_formulas <- lapply(formulas, function(f) {
            stats::reformulate(
                c(attr(stats::terms(f), "term.labels"), added_noise),
                response = "survival::Surv(tstart, tstop, event)",
                env = environment(f)
            )
        })
        for (b in seq_len(B)) {
            pending <- names(errors)[vapply(errors, function(table) {
                is.na(table[b, noise_index])
            }, logical(1))]
            if (!length(pending)) next
            training_indices <- training_splits[[b]]
            test_indices <- setdiff(seq_len(number_of_observations), training_indices)
            for (model in pending) {
                score <- cox_test_errors(
                    features, data$jump_data, training_indices, test_indices,
                    evaluation_times = evaluation_grids[[b]],
                    transition_formula = noise_formulas[[model]]
                )$IBS
                if (length(score) != 1L || !is.finite(score) || score < 0) {
                    stop("invalid IBS for ", model, ", split ", b,
                         ", ", table_names[noise_index])
                }
                errors[[model]][b, noise_index] <- score
                write_checkpoint()
            }
            cat("Completed threshold-interaction Cox extension at noise level", noise_index - 1L,
                "split", b, "of", B, "\n")
        }
    }
    for (i in seq_along(errors)) {
        staged <- tempfile(pattern = ".threshold-interaction-table-", tmpdir = path)
        write.table(errors[[i]], file = staged, sep = "\t", row.names = FALSE)
        if (!file.rename(staged, output_paths[i])) {
            unlink(staged)
            stop("could not save interaction table: ", output_paths[i])
        }
    }
    message("Saved both threshold-interaction Cox extension tables in ", path)
    invisible(lapply(errors, as.data.frame))
}


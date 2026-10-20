# Study helpers. Loading this file does not run a study.

simulate_proportional_hazards <- function(n, seed = 2026, prob_start_state2 = 0.60,
                         follow_up = 8, signal_scale = 1) {
    if (!is.numeric(n) || length(n) != 1L || !is.finite(n) ||
        n < 1 || n > .Machine$integer.max || n != floor(n)) {
        stop("n must be a positive integer")
    }
    if (!is.numeric(prob_start_state2) || length(prob_start_state2) != 1L ||
        !is.finite(prob_start_state2) ||
        prob_start_state2 < 0 || prob_start_state2 > 1) {
        stop("prob_start_state2 must be a probability")
    }
    if (!is.numeric(follow_up) || length(follow_up) != 1L ||
        is.na(follow_up) || follow_up <= 0) {
        stop("follow_up must be positive (Inf is allowed)")
    }
    if (!is.numeric(signal_scale) || length(signal_scale) != 1L ||
        !is.finite(signal_scale) || signal_scale <= 0) {
        stop("signal_scale must be finite and positive")
    }

    set.seed(seed)
    X1 <- rbinom(n, 1L, 1 / 2)
    X2 <- runif(n, -1, 1)
    X3 <- runif(n)
    feature_data <- data.frame(X1, X2, X3)
    initial_states <- 1L + rbinom(n, 1L, prob_start_state2)

    risk_12 <- exp(signal_scale * (
        0.80 * (X1 - 0.5) + 0.60 * X2 + 1.10 * (X3 - 0.5)
    ))
    risk_13 <- exp(signal_scale * 0.45 * (X1 - 0.5))
    risk_23 <- exp(signal_scale * 0.35 * X2)
    risk_multipliers <- c(risk_12, risk_13, risk_23)
    if (any(!is.finite(risk_multipliers)) || any(risk_multipliers <= 0)) {
        stop("signal_scale produces invalid transition rates")
    }

    # Exact inverse cumulative-hazard sampling, using independent clocks for
    # the two competing exits from state 1. The baseline cumulative hazards are
    # H12(t) = .012 * ((1+t)^2 - 1), H13(t) = .32 * (sqrt(1+t) - 1), and
    # H23(t) = (18/325) * ((1+t)^(3/2) - 1).
    sim <- vector("list", n)
    for (i in seq_len(n)) {
        times <- 0
        states <- initial_states[i]
        entry_2 <- 0

        if (initial_states[i] == 1L) {
            time_12 <- expm1(0.5 * log1p(rexp(1L) / (0.012 * risk_12[i])))
            time_13 <- expm1(2 * log1p(rexp(1L) / (0.32 * risk_13[i])))
            exit_1 <- min(time_12, time_13)
            if (exit_1 >= follow_up) {
                sim[[i]] <- list(times = c(0, follow_up), states = c(1L, 1L))
                next
            }
            if (time_13 < time_12) {
                sim[[i]] <- list(times = c(0, time_13), states = c(1L, 3L))
                next
            }
            entry_2 <- time_12
            times <- c(times, entry_2)
            states <- c(states, 2L)
        }

        # Solve risk_23 * (H23(time_23) - H23(entry_2)) = Exp(1).
        # This keeps the baseline on study time after entry into state 2.
        waiting_23 <- (1 + entry_2) * expm1((2 / 3) * log1p(
            rexp(1L) / ((18 / 325) * risk_23[i] * (1 + entry_2)^(3 / 2))
        ))
        time_23 <- entry_2 + waiting_23
        sim[[i]] <- list(
            times = c(times, min(time_23, follow_up)),
            states = c(states, if (time_23 < follow_up) 3L else 2L)
        )
    }
    list(jump_data = sim, feature_data = feature_data)
}


predict_proportional_hazards_forest <- function(forest, new_data, test_paths) {
    predictions <- jfforest.predict(forest, new_data = new_data)
    number_of_states <- as.integer(forest$num.states)
    initial <- lapply(test_paths, function(path) {
        as.numeric(seq_len(number_of_states) == path$states[1L])
    })
    structure(
        list(
            times = forest$unique.event.times,
            values = nelson_aalen_to_occupation_predictions(
                predictions, forest$unique.event.times, initial,
                label = "JF"
            )
        ),
        class = c("occupation_prediction_set", "list")
    )
}


run_proportional_hazards_vimp <- function(
        n, path, B = 100,
        noise_counts = c(0L, 10L, 25L, 50L),
        num_permutations = 1L, seed = 2026,
        num_event_times = 500L) {
    positive_integer <- function(value, name) {
        if (!is.numeric(value) || length(value) != 1L ||
            !is.finite(value) || value < 1L ||
            value > .Machine$integer.max || value != floor(value)) {
            stop(name, " must be a positive integer")
        }
        as.integer(value)
    }
    non_negative_integer <- function(value, name) {
        if (!is.numeric(value) || length(value) != 1L ||
            !is.finite(value) || value < 0L ||
            value > .Machine$integer.max || value != floor(value)) {
            stop(name, " must be a non-negative integer")
        }
        as.integer(value)
    }

    n <- positive_integer(n, "n")
    B <- positive_integer(B, "B")
    num_permutations <- positive_integer(
        num_permutations, "num_permutations"
    )
    num_event_times <- non_negative_integer(
        num_event_times, "num_event_times"
    )
    seed <- non_negative_integer(seed, "seed")
    splitrule <- "logrank"
    if (!is.numeric(noise_counts) || length(noise_counts) == 0L ||
        any(!is.finite(noise_counts)) || any(noise_counts < 0L) ||
        any(noise_counts > .Machine$integer.max) ||
        any(noise_counts != floor(noise_counts)) ||
        anyDuplicated(noise_counts)) {
        stop("noise_counts must contain unique non-negative integers")
    }
    noise_counts <- sort(as.integer(noise_counts))
    if (!is.character(path) || length(path) != 1L || is.na(path) ||
        !nzchar(path) || !grepl("[.]txt$", path, ignore.case = TRUE)) {
        stop("path must be one non-empty path ending in .txt")
    }
    if (!requireNamespace("glmnet", quietly = TRUE)) {
        stop("the proportional-hazards VIMP study requires the glmnet package")
    }

    training_size <- floor(0.8 * n)
    if (training_size < 2L || training_size >= n) {
        stop("n must be large enough to create a non-empty 80/20 split")
    }
    min_node_size <- as.integer(floor(sqrt(training_size)) + 1L)
    full_min_node_size <- as.integer(floor(sqrt(n)) + 1L)
    output_directory <- dirname(path)
    if (!dir.exists(output_directory) &&
        !dir.create(output_directory, recursive = TRUE)) {
        stop("could not create output directory: ", output_directory)
    }

    transition_matrix <- matrix(
        c(1L, 2L, 1L, 3L, 2L, 3L), ncol = 2L, byrow = TRUE,
        dimnames = list(NULL, c("from", "to"))
    )
    transition_names <- c("1 -> 2", "1 -> 3", "2 -> 3")
    model_names <- c(
        "jump_forest", "cox", "cox_lasso", "jump_forest_oob"
    )
    noise_grid <- paste(noise_counts, collapse = ",")
    # Increment when changing simulate_proportional_hazards's defaults or the scoring protocol.
    dgp_id <- "cox_sparse_mixed_time_inhomogeneous_v1"

    fit_cox_system <- function(training_paths, training_features,
                               penalised = FALSE) {
        transition_formula <- stats::reformulate(
            names(training_features),
            response = "survival::Surv(tstart, tstop, event)"
        )
        fitted_transitions <- lapply(
            seq_len(nrow(transition_matrix)), function(transition_index) {
                from <- transition_matrix[transition_index, "from"]
                to <- transition_matrix[transition_index, "to"]
                transition_data <- paths_to_transition_cox_data(
                    training_paths, training_features, from, to
                )
                fitted_transition <- if (penalised) {
                    fit_transition_glmnet(
                        transition_data, transition_formula
                    )
                } else {
                    fit_transition_cox(
                        transition_data, transition_formula,
                        ties = "breslow"
                    )
                }
                list(
                    from = from, to = to, fit = fitted_transition,
                    fixed_values = list()
                )
            }
        )
        names(fitted_transitions) <- transition_names
        new_multistate_cox_model(
            fitted_transitions, number_of_states = 3L,
            label = if (penalised) "Cox Lasso" else "Cox"
        )
    }

    coefficient_matrix <- function(model, variable_names) {
        coefficients <- matrix(
            0, nrow = length(variable_names), ncol = length(transition_names),
            dimnames = list(variable_names, transition_names)
        )
        for (transition_index in seq_along(model$transitions)) {
            fit <- model$transitions[[transition_index]]$fit
            fitted_coefficients <- if (
                inherits(fit, "penalised_transition_cox")
            ) {
                fit$coefficients
            } else {
                stats::coef(fit)
            }
            common_variables <- intersect(
                variable_names, names(fitted_coefficients)
            )
            coefficients[common_variables, transition_index] <-
                fitted_coefficients[common_variables]
        }
        coefficients
    }

    predict_cox_system <- function(model, new_data, test_paths,
                                    evaluation_times) {
        initial_states <- vapply(
            test_paths, function(path) path$states[1L], integer(1)
        )
        prediction_values <- vector("list", length(test_paths))
        for (initial_state in sort(unique(initial_states))) {
            rows <- which(initial_states == initial_state)
            initial <- numeric(model$number_of_states)
            initial[initial_state] <- 1
            predictions <- predict_multistate_cox(
                model, new_data[rows, , drop = FALSE], evaluation_times,
                initial = initial, stype = "exponential"
            )
            prediction_values[rows] <- predictions$values
        }
        structure(
            list(times = evaluation_times, values = prediction_values),
            class = c("occupation_prediction_set", "list")
        )
    }

    # Construct the marginal-KM weighted normalised IBS once per split and
    # reuse it for every model and permutation on that split.
    make_ibs_scorer <- function(training_paths, test_paths,
                                evaluation_times) {
        number_of_test_observations <- length(test_paths)
        number_of_times <- length(evaluation_times)
        censoring_model <- reverse_km_censoring(training_paths)
        censoring_on_grid <- censoring_survival_at(
            censoring_model, evaluation_times
        )
        endpoint_times <- vapply(
            test_paths, function(path) tail(path$times, 1L), numeric(1)
        )
        censored <- vapply(test_paths, function(path) {
            tail(path$states, 1L) == tail(head(path$states, -1L), 1L)
        }, logical(1))
        ipcw <- matrix(
            0, nrow = number_of_times,
            ncol = number_of_test_observations
        )
        for (observation in seq_along(test_paths)) {
            before_endpoint <- evaluation_times < endpoint_times[observation]
            usable <- before_endpoint & censoring_on_grid > 0
            ipcw[usable, observation] <- 1 / (
                number_of_test_observations * censoring_on_grid[usable]
            )
            if (!censored[observation]) {
                after_event <-
                    evaluation_times >= endpoint_times[observation]
                censoring_before_event <- censoring_survival_at(
                    censoring_model, endpoint_times[observation],
                    left_limit = TRUE
                )
                if (censoring_before_event > 0) {
                    ipcw[after_event, observation] <- 1 / (
                        number_of_test_observations *
                            censoring_before_event
                    )
                }
            }
        }
        observed_states <- t(observed_states_at(
            test_paths, evaluation_times
        ))

        function(prediction_set) {
            if (length(prediction_set$values) !=
                number_of_test_observations) {
                stop("a prediction set has the wrong number of observations")
            }
            aligned_probabilities <- lapply(
                seq_len(number_of_test_observations), function(observation) {
                    prediction_times <- if (
                        is.list(prediction_set$times)
                    ) {
                        prediction_set$times[[observation]]
                    } else {
                        prediction_set$times
                    }
                    step_occupation_probabilities_at(
                        prediction_times,
                        prediction_set$values[[observation]],
                        evaluation_times
                    )
                }
            )
            if (any(vapply(
                aligned_probabilities, ncol, integer(1)
            ) != 3L)) {
                stop("a prediction set does not contain three states")
            }

            squared_norm <- observed_probability <- matrix(
                0, nrow = number_of_times,
                ncol = number_of_test_observations
            )
            for (state in seq_len(3L)) {
                state_probability <- vapply(
                    aligned_probabilities,
                    function(probabilities) probabilities[, state],
                    numeric(number_of_times)
                )
                squared_norm <- squared_norm + state_probability^2
                belongs_to_state <- observed_states == state &
                    !is.na(observed_states)
                observed_probability[belongs_to_state] <-
                    state_probability[belongs_to_state]
            }
            known <- ipcw > 0 & !is.na(observed_states)
            brier_loss <- matrix(
                0, nrow = number_of_times,
                ncol = number_of_test_observations
            )
            brier_loss[known] <- (
                1 - 2 * observed_probability[known] +
                    squared_norm[known]
            ) / 3
            brier_curve <- rowSums(ipcw * brier_loss)
            normalised_trapezoid(evaluation_times, brier_curve)
        }
    }

    rank_importance <- function(importance) {
        if (any(!is.finite(importance))) {
            stop("a VIMP calculation returned a non-finite value")
        }
        rank(-importance, ties.method = "average")
    }

    make_result_rows <- function(
            run, number_of_noise, model, variable_names,
            importance, importance_type, training_fraction,
            baseline_ibs = NA_real_, selected = NULL) {
        number_of_variables <- length(variable_names)
        missing_logical <- rep(NA, number_of_variables)
        get_logical <- function(values, transition) {
            if (is.null(values)) missing_logical else values[, transition]
        }
        importance_ranks <- rank_importance(importance)
        data.frame(
            run = run,
            dgp_id = dgp_id,
            sample_size = n,
            noise_count = number_of_noise,
            model = model,
            importance_type = importance_type,
            training_fraction = training_fraction,
            variable = variable_names,
            true_global = variable_names %in% c("X1", "X2", "X3"),
            true_1_2 = variable_names %in% c("X1", "X2", "X3"),
            true_1_3 = variable_names == "X1",
            true_2_3 = variable_names == "X2",
            importance = importance,
            importance_rank = importance_ranks,
            top_three = importance_ranks <= 3,
            baseline_ibs = rep(baseline_ibs, number_of_variables),
            selected_1_2 = get_logical(selected, "1 -> 2"),
            selected_1_3 = get_logical(selected, "1 -> 3"),
            selected_2_3 = get_logical(selected, "2 -> 3"),
            selected_any = if (is.null(selected)) {
                missing_logical
            } else {
                rowSums(selected) > 0L
            },
            forest_min_node_size = if (model == "jump_forest_oob") {
                full_min_node_size
            } else {
                min_node_size
            },
            forest_splitrule = splitrule,
            num_event_times = num_event_times,
            permutation_repetitions = num_permutations,
            study_seed = seed,
            number_of_runs = B,
            noise_grid = noise_grid,
            stringsAsFactors = FALSE,
            check.names = FALSE
        )
    }

    expected_columns <- names(make_result_rows(
        run = 1L, number_of_noise = 0L,
        model = model_names[1L], variable_names = "X1",
        importance = 0, importance_type = "test_delta_ibs",
        training_fraction = training_size / n
    ))
    results <- NULL
    if (file.exists(path)) {
        results <- utils::read.table(
            path, header = TRUE, sep = "\t", check.names = FALSE
        )
        if (!identical(names(results), expected_columns)) {
            stop(paste0(
                "the existing output table has an incompatible format; ",
                "use a new path for the proportional-hazards DGP"
            ))
        }
        compatible <-
            all(results$dgp_id == dgp_id) &&
            all(results$sample_size == n) &&
            all(results$forest_min_node_size == ifelse(
                results$model == "jump_forest_oob",
                full_min_node_size, min_node_size
            )) &&
            all(results$forest_splitrule == splitrule) &&
            all(results$num_event_times == num_event_times) &&
            all(results$permutation_repetitions == num_permutations) &&
            all(results$study_seed == seed) &&
            all(results$number_of_runs == B) &&
            all(results$noise_grid == noise_grid)
        if (!compatible) {
            stop("the existing output table belongs to a different study")
        }
        if (any(results$run < 1L | results$run > B) ||
            any(!results$noise_count %in% noise_counts) ||
            anyDuplicated(results[c(
                "run", "noise_count", "model", "variable"
            )])) {
            stop("the existing output table contains invalid or duplicate rows")
        }
    }

    pair_complete <- function(run, number_of_noise, variable_names) {
        if (is.null(results)) {
            return(FALSE)
        }
        pair_rows <- results[
            results$run == run &
                results$noise_count == number_of_noise,
            , drop = FALSE
        ]
        if (nrow(pair_rows) == 0L) {
            return(FALSE)
        }
        complete <- nrow(pair_rows) ==
            length(model_names) * length(variable_names) &&
            setequal(unique(pair_rows$model), model_names) &&
            all(vapply(model_names, function(model) {
                setequal(
                    pair_rows$variable[pair_rows$model == model],
                    variable_names
                )
            }, logical(1)))
        if (!complete) {
            stop(
                "the existing table contains an incomplete result for run ",
                run, " and noise_count ", number_of_noise
            )
        }
        TRUE
    }

    write_checkpoint <- function(results) {
        model_order <- match(results$model, model_names)
        results <- results[order(
            results$run, results$noise_count, model_order,
            results$importance_rank, results$variable
        ), , drop = FALSE]
        rownames(results) <- NULL
        staged_path <- tempfile(
            pattern = ".proportional-hazards-vimp-", tmpdir = output_directory,
            fileext = ".txt"
        )
        on.exit(unlink(staged_path), add = TRUE)
        utils::write.table(
            results, file = staged_path, sep = "\t",
            row.names = FALSE, quote = FALSE
        )
        staged_results <- utils::read.table(
            staged_path, header = TRUE, sep = "\t", check.names = FALSE
        )
        if (!identical(names(staged_results), names(results)) ||
            nrow(staged_results) != nrow(results)) {
            stop("the staged VIMP table failed validation")
        }
        # The staged file is in the same directory, so this replacement is an
        # atomic checkpoint on the study's Linux filesystem.
        if (!file.rename(staged_path, path)) {
            stop("could not update the VIMP study table: ", path)
        }
        staged_results
    }

    set.seed(seed)
    random_seed <- function(number) {
        sample.int(.Machine$integer.max, number, replace = TRUE)
    }
    data_seeds <- random_seed(B)
    noise_seeds <- random_seed(B)
    split_seeds <- random_seed(B)
    external_forest_seeds <- random_seed(B)
    full_forest_seeds <- random_seed(B)
    permutation_seeds <- random_seed(B)
    oob_vimp_seeds <- random_seed(B)

    for (run in seq_len(B)) {
        pending <- vapply(
            seq_along(noise_counts), function(noise_index) {
                variable_names <- c(
                    "X1", "X2", "X3",
                    if (noise_counts[noise_index] > 0L) {
                        paste0("noise_", seq_len(noise_counts[noise_index]))
                    }
                )
                !pair_complete(
                    run, noise_counts[noise_index], variable_names
                )
            }, logical(1)
        )
        if (!any(pending)) {
            next
        }

        simulated_data <- simulate_proportional_hazards(n, seed = data_seeds[run])
        if (!identical(
            names(simulated_data$feature_data), c("X1", "X2", "X3")
        )) {
            stop("simulate_proportional_hazards must return feature columns X1, X2 and X3")
        }
        set.seed(noise_seeds[run])
        maximum_noise <- max(noise_counts)
        noise_data <- if (maximum_noise == 0L) {
            data.frame(row.names = seq_len(n))
        } else {
            noise_matrix <- matrix(
                stats::rnorm(n * maximum_noise), nrow = n
            )
            noise_data <- as.data.frame(noise_matrix)
            names(noise_data) <- paste0(
                "noise_", seq_len(maximum_noise)
            )
            noise_data
        }
        set.seed(split_seeds[run])
        training_indices <- sample(seq_len(n), size = training_size)
        test_indices <- base::setdiff(seq_len(n), training_indices)
        training_paths <- simulated_data$jump_data[training_indices]
        test_paths <- simulated_data$jump_data[test_indices]
        all_variable_names <- c(
            "X1", "X2", "X3",
            if (maximum_noise > 0L) {
                paste0("noise_", seq_len(maximum_noise))
            }
        )
        # Use the same row permutations for variables shared by different
        # noise levels, making comparisons across the nested noise sets paired.
        set.seed(permutation_seeds[run])
        permutation_rows <- lapply(
            seq_len(num_permutations), function(permutation) {
                setNames(lapply(
                    all_variable_names, function(variable) {
                        sample.int(length(test_indices))
                    }
                ), all_variable_names)
            }
        )
        set.seed(oob_vimp_seeds[run])
        internal_vimp_call_seeds <- random_seed(num_permutations)

        for (noise_index in which(pending)) {
            number_of_noise <- noise_counts[noise_index]
            analysis_features <- simulated_data$feature_data
            if (number_of_noise > 0L) {
                analysis_features <- cbind(
                    analysis_features,
                    noise_data[seq_len(number_of_noise)]
                )
            }
            variable_names <- names(analysis_features)
            training_features <- analysis_features[
                training_indices, , drop = FALSE
            ]
            test_features <- analysis_features[
                test_indices, , drop = FALSE
            ]

            fitted_forest <- jfforest(
                MM ~ ., data = training_paths,
                feature_data = training_features,
                splitrule = splitrule,
                min_node_size = min_node_size,
                seed = external_forest_seeds[run],
                save_predictions = FALSE,
                num_event_times = num_event_times
            )
            evaluation_times <- fitted_forest$unique.event.times
            score_ibs <- make_ibs_scorer(
                training_paths, test_paths, evaluation_times
            )
            fitted_cox <- fit_cox_system(
                training_paths, training_features, penalised = FALSE
            )
            fitted_cox_lasso <- fit_cox_system(
                training_paths, training_features, penalised = TRUE
            )

            raw_cox_coefficients <- coefficient_matrix(
                fitted_cox, variable_names
            )
            raw_lasso_coefficients <- coefficient_matrix(
                fitted_cox_lasso, variable_names
            )
            lasso_selected <- raw_lasso_coefficients != 0

            baseline_predictions <- list(
                jump_forest = predict_proportional_hazards_forest(
                    fitted_forest, test_features, test_paths
                ),
                cox = predict_cox_system(
                    fitted_cox, test_features, test_paths,
                    evaluation_times
                ),
                cox_lasso = predict_cox_system(
                    fitted_cox_lasso, test_features, test_paths,
                    evaluation_times
                )
            )
            baseline_ibs <- vapply(
                baseline_predictions, score_ibs, numeric(1)
            )

            external_importance <- matrix(
                0, nrow = length(variable_names), ncol = 3L,
                dimnames = list(
                    variable_names,
                    c("jump_forest", "cox", "cox_lasso")
                )
            )
            cox_active <- rowSums(raw_cox_coefficients != 0) > 0L
            lasso_active <- rowSums(lasso_selected) > 0L
            for (variable_index in seq_along(variable_names)) {
                variable <- variable_names[variable_index]
                for (permutation in seq_len(num_permutations)) {
                    permuted_features <- test_features
                    permuted_features[[variable]] <- test_features[[variable]][
                        permutation_rows[[permutation]][[variable]]
                    ]
                    external_importance[variable, "jump_forest"] <-
                        external_importance[variable, "jump_forest"] +
                        score_ibs(predict_proportional_hazards_forest(
                            fitted_forest, permuted_features, test_paths
                        )) - baseline_ibs["jump_forest"]
                    if (cox_active[variable_index]) {
                        external_importance[variable, "cox"] <-
                            external_importance[variable, "cox"] +
                            score_ibs(predict_cox_system(
                                fitted_cox, permuted_features, test_paths,
                                evaluation_times
                            )) - baseline_ibs["cox"]
                    }
                    if (lasso_active[variable_index]) {
                        external_importance[variable, "cox_lasso"] <-
                            external_importance[variable, "cox_lasso"] +
                            score_ibs(predict_cox_system(
                                fitted_cox_lasso, permuted_features,
                                test_paths, evaluation_times
                            )) - baseline_ibs["cox_lasso"]
                    }
                }
            }
            external_importance <-
                external_importance / num_permutations

            full_forest <- jfforest(
                MM ~ ., data = simulated_data$jump_data,
                feature_data = analysis_features,
                splitrule = splitrule,
                min_node_size = full_min_node_size,
                seed = full_forest_seeds[run],
                save_predictions = TRUE,
                num_event_times = num_event_times
            )
            internal_vimp <- vapply(
                internal_vimp_call_seeds, function(vimp_seed) {
                    values <- as.numeric(unlist(jfforest.vimp(
                        full_forest, method = "permute",
                        loss = "brier", seed = vimp_seed
                    )$vimp))
                    if (length(values) != length(variable_names)) {
                        stop("jfforest.vimp returned the wrong number of values")
                    }
                    values
                }, numeric(length(variable_names))
            )
            if (is.null(dim(internal_vimp))) {
                internal_vimp <- matrix(
                    internal_vimp, nrow = length(variable_names)
                )
            }
            mean_internal_vimp <- rowMeans(internal_vimp)

            new_rows <- rbind(
                make_result_rows(
                    run, number_of_noise, "jump_forest", variable_names,
                    external_importance[, "jump_forest"],
                    "test_delta_ibs", training_size / n,
                    baseline_ibs["jump_forest"]
                ),
                make_result_rows(
                    run, number_of_noise, "cox", variable_names,
                    external_importance[, "cox"],
                    "test_delta_ibs", training_size / n,
                    baseline_ibs["cox"]
                ),
                make_result_rows(
                    run, number_of_noise, "cox_lasso", variable_names,
                    external_importance[, "cox_lasso"],
                    "test_delta_ibs", training_size / n,
                    baseline_ibs["cox_lasso"],
                    selected = lasso_selected
                ),
                make_result_rows(
                    run, number_of_noise, "jump_forest_oob",
                    variable_names, mean_internal_vimp,
                    "internal_oob_brier", 1
                )
            )
            results <- if (is.null(results)) {
                new_rows
            } else {
                rbind(results, new_rows)
            }
            results <- write_checkpoint(results)
            rm(
                fitted_forest, full_forest, fitted_cox,
                fitted_cox_lasso, baseline_predictions
            )
            gc(verbose = FALSE)
            cat(
                "Completed proportional-hazards VIMP run", run, "of", B,
                "with", number_of_noise, "noise variables\n"
            )
        }
    }

    invisible(results)
}


plot_proportional_hazards_vimp_top_three <- function(
        vimp_results, path = NULL, include_zero = FALSE,
        confidence_level = 0.95) {
    if (!is.numeric(confidence_level) || length(confidence_level) != 1L ||
        !is.finite(confidence_level) || confidence_level <= 0 ||
        confidence_level >= 1) {
        stop("confidence_level must be a number strictly between zero and one")
    }
    if (!is.null(path) &&
        (!is.character(path) || length(path) != 1L || is.na(path) ||
         !nzchar(path))) {
        stop("path must be NULL or one non-empty character string")
    }
    signal_results <- prepare_proportional_hazards_signal_recovery(
        vimp_results, include_zero = include_zero
    )

    model_sources <- c(
        "jump_forest", "cox", "cox_lasso", "jump_forest_oob"
    )
    model_labels <- c(
        jump_forest = "JF (held-out)",
        cox = "Cox",
        cox_lasso = "Cox Lasso",
        jump_forest_oob = "JF (internal OOB)"
    )
    model_colours <- c(
        "JF (held-out)" = "#0072B2",
        "Cox" = "#777777",
        "Cox Lasso" = "#009E73",
        "JF (internal OOB)" = "#0072B2"
    )
    model_shapes <- c(
        "JF (held-out)" = 0,
        "Cox" = 1,
        "Cox Lasso" = 2,
        "JF (internal OOB)" = 18
    )
    model_linetypes <- c(
        "JF (held-out)" = "solid",
        "Cox" = "solid",
        "Cox Lasso" = "solid",
        "JF (internal OOB)" = "dashed"
    )

    signal_groups <- split(
        signal_results,
        interaction(
            signal_results$run, signal_results$noise_count,
            signal_results$model_source, drop = TRUE
        )
    )
    recovery <- do.call(rbind, lapply(signal_groups, function(group) {
        data.frame(
            run = group$run[1L],
            noise_count = group$noise_count[1L],
            model = group$model_source[1L],
            recovered = all(group$top_three),
            stringsAsFactors = FALSE
        )
    }))

    summary_groups <- split(
        recovery,
        interaction(recovery$noise_count, recovery$model, drop = TRUE)
    )
    plot_data <- do.call(rbind, lapply(summary_groups, function(group) {
        successes <- sum(group$recovered)
        number_of_runs <- nrow(group)
        proportion <- successes / number_of_runs
        z_value <- stats::qnorm(1 - (1 - confidence_level) / 2)
        denominator <- 1 + z_value^2 / number_of_runs
        centre <- (
            proportion + z_value^2 / (2 * number_of_runs)
        ) / denominator
        half_width <- z_value * sqrt(
            proportion * (1 - proportion) / number_of_runs +
                z_value^2 / (4 * number_of_runs^2)
        ) / denominator
        data.frame(
            noise_count = group$noise_count[1L],
            model = group$model[1L],
            successes = successes,
            number_of_runs = number_of_runs,
            percentage = 100 * proportion,
            lower = 100 * max(0, centre - half_width),
            upper = 100 * min(1, centre + half_width),
            stringsAsFactors = FALSE
        )
    }))
    if (!include_zero) {
        plot_data <- plot_data[plot_data$noise_count > 0, , drop = FALSE]
    }
    if (nrow(plot_data) == 0L) {
        stop("no noise levels remain after applying include_zero")
    }

    ordered_models <- unname(model_labels[model_sources])
    plot_data$model <- factor(
        unname(model_labels[plot_data$model]), levels = ordered_models
    )
    dodge <- ggplot2::position_dodge(width = 1.1)
    line_layer <- if (length(unique(plot_data$noise_count)) > 1L) {
        ggplot2::geom_line(position = dodge, linewidth = 0.8)
    }
    plot <- ggplot2::ggplot(
        plot_data,
        ggplot2::aes(
            x = noise_count, y = percentage, colour = model,
            shape = model, linetype = model, group = model
        )
    ) +
        line_layer +
        ggplot2::geom_errorbar(
            ggplot2::aes(ymin = lower, ymax = upper),
            position = dodge, width = 0.8, linewidth = 0.45
        ) +
        ggplot2::geom_point(position = dodge, size = 2.5) +
        ggplot2::scale_colour_manual(values = model_colours) +
        ggplot2::scale_shape_manual(values = model_shapes) +
        ggplot2::scale_linetype_manual(values = model_linetypes) +
        ggplot2::scale_x_continuous(
            breaks = sort(unique(plot_data$noise_count))
        ) +
        ggplot2::scale_y_continuous(
            limits = c(0, 100), breaks = seq(0, 100, by = 20),
            labels = function(value) paste0(value, "%")
        ) +
        ggplot2::labs(
            title = NULL,
            x = "Number of noise variables",
            y = "Runs with X1, X2 and X3 all in the top three",
            colour = "Model", shape = "Model", linetype = "Model"
        ) +
        ggplot2::theme_bw() +
        ggplot2::theme(
            legend.position = "bottom",
            plot.title = ggplot2::element_text(hjust = 0.5)
        )

    if (!is.null(path)) {
        if (!dir.exists(dirname(path)) &&
            !dir.create(dirname(path), recursive = TRUE)) {
            stop("could not create plot directory: ", dirname(path))
        }
        ggplot2::ggsave(
            filename = path, plot = plot,
            width = 10, height = 6, units = "in", dpi = 300
        )
    }

    plot
}


prepare_proportional_hazards_signal_recovery <- function(
        vimp_results, include_zero = FALSE) {
    if (is.character(vimp_results)) {
        if (length(vimp_results) != 1L || is.na(vimp_results) ||
            !nzchar(vimp_results) || !file.exists(vimp_results)) {
            stop("vimp_results must be a data frame or one existing file path")
        }
        vimp_results <- utils::read.table(
            vimp_results, header = TRUE, sep = "\t", check.names = FALSE
        )
    }
    required_columns <- c(
        "run", "noise_count", "model", "variable", "true_global",
        "top_three"
    )
    if (!is.data.frame(vimp_results) || nrow(vimp_results) == 0L ||
        !all(required_columns %in% names(vimp_results))) {
        stop(
            "vimp_results must be a non-empty proportional-hazards VIMP result table with ",
            "run, noise_count, model, variable, true_global and top_three"
        )
    }
    if (!is.logical(include_zero) || length(include_zero) != 1L ||
        is.na(include_zero)) {
        stop("include_zero must be TRUE or FALSE")
    }
    if (is.numeric(vimp_results$top_three) &&
        all(vimp_results$top_three %in% c(0, 1))) {
        vimp_results$top_three <- as.logical(vimp_results$top_three)
    }
    if (!is.logical(vimp_results$top_three) ||
        anyNA(vimp_results$top_three) ||
        !is.logical(vimp_results$true_global) ||
        anyNA(vimp_results$true_global) ||
        !is.numeric(vimp_results$run) || anyNA(vimp_results$run) ||
        any(!is.finite(vimp_results$run)) || any(vimp_results$run < 1) ||
        any(vimp_results$run != floor(vimp_results$run)) ||
        !is.numeric(vimp_results$noise_count) ||
        anyNA(vimp_results$noise_count) ||
        any(!is.finite(vimp_results$noise_count)) ||
        any(vimp_results$noise_count < 0) ||
        any(vimp_results$noise_count != floor(vimp_results$noise_count)) ||
        !is.character(vimp_results$model) ||
        anyNA(vimp_results$model) ||
        !is.character(vimp_results$variable) || anyNA(vimp_results$variable)) {
        stop("vimp_results contains invalid recovery columns")
    }

    # Optional metadata is absent from some historical tables. When present,
    # do not silently pool different DGPs or study settings in the same curve.
    study_columns <- intersect(c(
        "dgp_id", "sample_size", "forest_splitrule",
        "num_event_times", "permutation_repetitions", "study_seed",
        "number_of_runs", "noise_grid"
    ), names(vimp_results))
    for (column in study_columns) {
        if (anyNA(vimp_results[[column]]) ||
            length(unique(vimp_results[[column]])) != 1L) {
            stop("vimp_results mixes studies or has invalid metadata: ", column)
        }
    }
    # The held-out forest uses 80% of the observations, whereas internal OOB
    # VIMP uses a full-data forest. Their square-root node sizes can differ.
    if ("forest_min_node_size" %in% names(vimp_results)) {
        node_sizes <- split(vimp_results$forest_min_node_size, vimp_results$model)
        if (!all(vapply(node_sizes, function(x) {
            is.numeric(x) && all(is.finite(x)) && all(x >= 1) &&
                all(x == floor(x)) && length(unique(x)) == 1L
        }, logical(1)))) {
            stop("forest_min_node_size must be a single positive integer within each model")
        }
    }

    signal_variables <- c("X1", "X2", "X3")
    model_sources <- c(
        "jump_forest", "cox", "cox_lasso", "jump_forest_oob"
    )
    model_labels <- c(
        jump_forest = "JF (held-out)",
        cox = "Cox",
        cox_lasso = "Cox Lasso",
        jump_forest_oob = "JF (internal OOB)"
    )
    missing_models <- setdiff(model_sources, unique(vimp_results$model))
    if (length(missing_models) > 0L) {
        stop(
            "vimp_results is missing models: ",
            paste(missing_models, collapse = ", ")
        )
    }

    model_results <- vimp_results[
        vimp_results$model %in% model_sources, , drop = FALSE
    ]
    if (!include_zero) {
        model_results <- model_results[
            model_results$noise_count > 0, , drop = FALSE
        ]
    }
    if (any(
        model_results$true_global !=
            model_results$variable %in% signal_variables
    )) {
        stop("true_global is inconsistent with X1, X2 and X3")
    }
    signal_results <- model_results[
        model_results$variable %in% signal_variables, required_columns,
        drop = FALSE
    ]
    if (nrow(signal_results) == 0L) {
        stop(paste0(
            "no noise levels remain after applying include_zero; ",
            "if only zero-noise runs are complete, use include_zero = TRUE"
        ))
    }
    if (anyDuplicated(signal_results[c(
        "run", "noise_count", "model", "variable"
    )])) {
        stop("signal rankings contain duplicate run/noise/model/variable rows")
    }

    group_key <- interaction(
        signal_results$run, signal_results$noise_count,
        signal_results$model, drop = TRUE
    )
    complete_groups <- vapply(
        split(signal_results$variable, group_key),
        function(variables) {
            length(variables) == length(signal_variables) &&
                setequal(variables, signal_variables)
        }, logical(1)
    )
    expected_groups <- unique(model_results[c("run", "noise_count", "model")])
    if (!all(complete_groups) || length(complete_groups) != nrow(expected_groups)) {
        stop("every run/noise/model combination must contain X1, X2 and X3")
    }
    for (number_of_noise in sort(unique(signal_results$noise_count))) {
        run_sets <- lapply(model_sources, function(model_source) {
            sort(unique(signal_results$run[
                signal_results$noise_count == number_of_noise &
                    signal_results$model == model_source
            ]))
        })
        if (!all(vapply(
            run_sets[-1L], identical, logical(1), run_sets[[1L]]
        ))) {
            stop("models must be evaluated on the same runs at each noise level")
        }
    }

    signal_results$model_source <- signal_results$model
    signal_results$model <- factor(
        unname(model_labels[signal_results$model_source]),
        levels = unname(model_labels[model_sources])
    )
    signal_results$variable <- factor(
        signal_results$variable, levels = signal_variables
    )
    signal_results
}


plot_proportional_hazards_vimp_signal_top_three <- function(
        vimp_results, path = NULL, include_zero = FALSE,
        confidence_level = 0.95) {
    if (!is.numeric(confidence_level) || length(confidence_level) != 1L ||
        !is.finite(confidence_level) || confidence_level <= 0 ||
        confidence_level >= 1) {
        stop("confidence_level must be a number strictly between zero and one")
    }
    if (!is.null(path) &&
        (!is.character(path) || length(path) != 1L || is.na(path) ||
         !nzchar(path))) {
        stop("path must be NULL or one non-empty character string")
    }
    signal_results <- prepare_proportional_hazards_signal_recovery(
        vimp_results, include_zero = include_zero
    )

    summary_groups <- split(
        signal_results,
        interaction(
            signal_results$noise_count, signal_results$model_source,
            signal_results$variable, drop = TRUE
        )
    )
    plot_data <- do.call(rbind, lapply(summary_groups, function(group) {
        successes <- sum(group$top_three)
        number_of_runs <- nrow(group)
        proportion <- successes / number_of_runs
        z_value <- stats::qnorm(1 - (1 - confidence_level) / 2)
        denominator <- 1 + z_value^2 / number_of_runs
        centre <- (
            proportion + z_value^2 / (2 * number_of_runs)
        ) / denominator
        half_width <- z_value * sqrt(
            proportion * (1 - proportion) / number_of_runs +
                z_value^2 / (4 * number_of_runs^2)
        ) / denominator
        data.frame(
            noise_count = group$noise_count[1L],
            model = group$model[1L],
            model_source = group$model_source[1L],
            variable = group$variable[1L],
            successes = successes,
            number_of_runs = number_of_runs,
            percentage = 100 * proportion,
            lower = 100 * max(0, centre - half_width),
            upper = 100 * min(1, centre + half_width)
        )
    }))
    plot_data$model <- factor(
        plot_data$model, levels = levels(signal_results$model)
    )
    plot_data$variable <- factor(
        plot_data$variable, levels = levels(signal_results$variable)
    )

    model_colours <- c(
        "JF (held-out)" = "#0072B2",
        "Cox" = "#777777",
        "Cox Lasso" = "#009E73",
        "JF (internal OOB)" = "#0072B2"
    )
    model_shapes <- c(
        "JF (held-out)" = 0, "Cox" = 1,
        "Cox Lasso" = 2, "JF (internal OOB)" = 18
    )
    model_linetypes <- c(
        "JF (held-out)" = "solid", "Cox" = "solid",
        "Cox Lasso" = "solid", "JF (internal OOB)" = "dashed"
    )
    dodge <- ggplot2::position_dodge(width = 1.1)
    line_layer <- if (length(unique(plot_data$noise_count)) > 1L) {
        ggplot2::geom_line(position = dodge, linewidth = 0.8)
    }
    plot <- ggplot2::ggplot(
        plot_data,
        ggplot2::aes(
            x = noise_count, y = percentage, colour = model,
            shape = model, linetype = model, group = model
        )
    ) +
        line_layer +
        ggplot2::geom_errorbar(
            ggplot2::aes(ymin = lower, ymax = upper),
            position = dodge, width = 0.8, linewidth = 0.4
        ) +
        ggplot2::geom_point(position = dodge, size = 2.2) +
        ggplot2::facet_wrap(~variable, nrow = 1L) +
        ggplot2::scale_colour_manual(values = model_colours) +
        ggplot2::scale_shape_manual(values = model_shapes) +
        ggplot2::scale_linetype_manual(values = model_linetypes) +
        ggplot2::scale_x_continuous(
            breaks = sort(unique(plot_data$noise_count))
        ) +
        ggplot2::scale_y_continuous(
            limits = c(0, 100), breaks = seq(0, 100, by = 20),
            labels = function(value) paste0(value, "%")
        ) +
        ggplot2::labs(
            title = NULL,
            x = "Number of noise variables",
            y = "Runs in which the variable was in the top three",
            colour = "Model", shape = "Model", linetype = "Model"
        ) +
        ggplot2::theme_bw() +
        ggplot2::theme(
            legend.position = "bottom",
            plot.title = ggplot2::element_text(hjust = 0.5)
        )

    if (!is.null(path)) {
        if (!dir.exists(dirname(path)) &&
            !dir.create(dirname(path), recursive = TRUE)) {
            stop("could not create plot directory: ", dirname(path))
        }
        ggplot2::ggsave(
            filename = path, plot = plot,
            width = 12, height = 5.5, units = "in", dpi = 300
        )
    }
    plot
}


plot_proportional_hazards_vimp_mean_true_top_three <- function(
        vimp_results, path = NULL, include_zero = FALSE,
        confidence_level = 0.95) {
    if (!is.numeric(confidence_level) || length(confidence_level) != 1L ||
        !is.finite(confidence_level) || confidence_level <= 0 ||
        confidence_level >= 1) {
        stop("confidence_level must be a number strictly between zero and one")
    }
    if (!is.null(path) &&
        (!is.character(path) || length(path) != 1L || is.na(path) ||
         !nzchar(path))) {
        stop("path must be NULL or one non-empty character string")
    }
    signal_results <- prepare_proportional_hazards_signal_recovery(
        vimp_results, include_zero = include_zero
    )

    run_groups <- split(
        signal_results,
        interaction(
            signal_results$run, signal_results$noise_count,
            signal_results$model_source, drop = TRUE
        )
    )
    run_data <- do.call(rbind, lapply(run_groups, function(group) {
        data.frame(
            run = group$run[1L],
            noise_count = group$noise_count[1L],
            model = group$model[1L],
            model_source = group$model_source[1L],
            number_recovered = sum(group$top_three)
        )
    }))
    summary_groups <- split(
        run_data,
        interaction(run_data$noise_count, run_data$model_source, drop = TRUE)
    )
    plot_data <- do.call(rbind, lapply(summary_groups, function(group) {
        number_of_runs <- nrow(group)
        mean_recovered <- mean(group$number_recovered)
        # An early checkpoint can contain just one run at a noise level.
        # Show its mean, but there is no estimable t interval yet.
        half_width <- if (number_of_runs > 1L) {
            stats::qt(1 - (1 - confidence_level) / 2, df = number_of_runs - 1L) *
                stats::sd(group$number_recovered) / sqrt(number_of_runs)
        } else NA_real_
        data.frame(
            noise_count = group$noise_count[1L],
            model = group$model[1L],
            model_source = group$model_source[1L],
            number_of_runs = number_of_runs,
            mean_recovered = mean_recovered,
            lower = max(0, mean_recovered - half_width),
            upper = min(3, mean_recovered + half_width)
        )
    }))
    plot_data$model <- factor(
        plot_data$model, levels = levels(signal_results$model)
    )

    model_colours <- c(
        "JF (held-out)" = "#0072B2",
        "Cox" = "#777777",
        "Cox Lasso" = "#009E73",
        "JF (internal OOB)" = "#0072B2"
    )
    model_shapes <- c(
        "JF (held-out)" = 0, "Cox" = 1,
        "Cox Lasso" = 2, "JF (internal OOB)" = 18
    )
    model_linetypes <- c(
        "JF (held-out)" = "solid", "Cox" = "solid",
        "Cox Lasso" = "solid", "JF (internal OOB)" = "dashed"
    )
    dodge <- ggplot2::position_dodge(width = 1.1)
    line_layer <- if (length(unique(plot_data$noise_count)) > 1L) {
        ggplot2::geom_line(position = dodge, linewidth = 0.8)
    }
    plot <- ggplot2::ggplot(
        plot_data,
        ggplot2::aes(
            x = noise_count, y = mean_recovered, colour = model,
            shape = model, linetype = model, group = model
        )
    ) +
        line_layer +
        ggplot2::geom_errorbar(
            data = plot_data[is.finite(plot_data$lower), , drop = FALSE],
            ggplot2::aes(ymin = lower, ymax = upper),
            position = dodge, width = 0.8, linewidth = 0.4
        ) +
        ggplot2::geom_point(position = dodge, size = 2.5) +
        ggplot2::scale_colour_manual(values = model_colours) +
        ggplot2::scale_shape_manual(values = model_shapes) +
        ggplot2::scale_linetype_manual(values = model_linetypes) +
        ggplot2::scale_x_continuous(
            breaks = sort(unique(plot_data$noise_count))
        ) +
        ggplot2::scale_y_continuous(
            limits = c(0, 3), breaks = seq(0, 3, by = 0.5)
        ) +
        ggplot2::labs(
            title = NULL,
            x = "Number of noise variables",
            y = "Mean number of true variables in the top three",
            colour = "Model", shape = "Model", linetype = "Model"
        ) +
        ggplot2::theme_bw() +
        ggplot2::theme(
            legend.position = "bottom",
            plot.title = ggplot2::element_text(hjust = 0.5)
        )

    if (!is.null(path)) {
        if (!dir.exists(dirname(path)) &&
            !dir.create(dirname(path), recursive = TRUE)) {
            stop("could not create plot directory: ", dirname(path))
        }
        ggplot2::ggsave(
            filename = path, plot = plot,
            width = 10, height = 6, units = "in", dpi = 300
        )
    }
    plot
}


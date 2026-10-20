# Study helpers. Loading this file does not run a study.

compute_noise_ibs <- function(
        feature_data, jump_data, number_of_noise_variables,
        B = 100, seed = 2026) {
    validate_jump_paths(jump_data)
    if (!is.data.frame(feature_data) ||
        nrow(feature_data) != length(jump_data)) {
        stop("feature_data must have one row per jump path")
    }
    if (anyNA(feature_data) || is.null(names(feature_data)) ||
        any(names(feature_data) == "") || anyDuplicated(names(feature_data))) {
        stop("feature_data must have complete, uniquely named columns")
    }
    if (!is.numeric(number_of_noise_variables) ||
        length(number_of_noise_variables) != 1L ||
        !is.finite(number_of_noise_variables) ||
        number_of_noise_variables < 0L ||
        number_of_noise_variables != as.integer(number_of_noise_variables)) {
        stop("number_of_noise_variables must be a non-negative integer")
    }
    splitrule <- "logrank"
    if (!is.numeric(B) || length(B) != 1L || !is.finite(B) ||
        B < 1L || B != as.integer(B)) {
        stop("B must be a positive integer")
    }
    if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed) ||
        seed < 0L || seed > .Machine$integer.max ||
        seed != as.integer(seed)) {
        stop("seed must be an integer between zero and .Machine$integer.max")
    }
    if (!requireNamespace("glmnet", quietly = TRUE)) {
        stop(
            "the L1 Cox models require the glmnet package; install it with ",
            "install.packages(\"glmnet\")"
        )
    }

    number_of_noise_variables <- as.integer(number_of_noise_variables)
    B <- as.integer(B)
    number_of_observations <- nrow(feature_data)
    training_size <- floor(0.8 * number_of_observations)
    if (training_size < 1L || training_size >= number_of_observations) {
        stop("feature_data must contain enough observations for an 80/20 split")
    }
    min_node_size <- as.integer(floor(sqrt(training_size)) + 1L)

    set.seed(seed)
    if (number_of_noise_variables > 0L) {
        noise_data <- as.data.frame(matrix(
            rnorm(number_of_observations * number_of_noise_variables),
            nrow = number_of_observations
        ))
        noise_names <- paste0("noise_", seq_len(number_of_noise_variables))
        names(noise_data) <- tail(
            make.unique(c(names(feature_data), noise_names)),
            number_of_noise_variables
        )
    } else {
        noise_data <- data.frame(row.names = seq_len(number_of_observations))
    }

    # Generate the splits separately so they do not depend on how many noise
    # variables were requested or on random-number use during model fitting.
    set.seed(seed)
    training_splits <- replicate(
        B,
        sample(seq_len(number_of_observations), size = training_size),
        simplify = FALSE
    )

    noise_counts <- 0:number_of_noise_variables
    table_names <- paste0("noise_", noise_counts)
    empty_error_table <- matrix(
        NA_real_, nrow = B, ncol = length(noise_counts),
        dimnames = list(NULL, table_names)
    )
    errors <- list(
        jump_forest = empty_error_table,
        cox = empty_error_table,
        cox_lasso = empty_error_table
    )
    cox_null_transitions <- matrix(
        vector("list", B * length(noise_counts)),
        nrow = B, ncol = length(noise_counts),
        dimnames = list(NULL, table_names)
    )
    cox_lasso_null_transitions <- matrix(
        vector("list", B * length(noise_counts)),
        nrow = B, ncol = length(noise_counts),
        dimnames = list(NULL, table_names)
    )

    for (noise_index in seq_along(noise_counts)) {
        number_of_noise <- noise_counts[noise_index]
        analysis_features <- feature_data
        if (number_of_noise > 0L) {
            analysis_features <- cbind(
                analysis_features,
                noise_data[seq_len(number_of_noise)]
            )
        }

        for (b in seq_len(B)) {
            training_indices <- training_splits[[b]]
            test_indices <- base::setdiff(
                seq_len(number_of_observations), training_indices
            )
            training_features <- analysis_features[
                training_indices, , drop = FALSE
            ]
            test_features <- analysis_features[
                test_indices, , drop = FALSE
            ]

            fitted_forest <- jfforest(
                MM ~ .,
                data = jump_data[training_indices],
                feature_data = training_features,
                splitrule = splitrule,
                min_node_size = min_node_size,
                seed = b,
                save_predictions = FALSE
            )
            evaluation_times <- fitted_forest$unique.event.times
            forest_predictions <- predict_proportional_hazards_forest(
                fitted_forest, test_features, jump_data[test_indices]
            )
            forest_error <- marginal_km_prediction_errors(
                forest_predictions,
                training_paths = jump_data[training_indices],
                test_paths = jump_data[test_indices],
                evaluation_times = evaluation_times,
                number_of_states = as.integer(fitted_forest$num.states),
                label = "JF"
            )
            errors$jump_forest[b, noise_index] <-
                forest_error["IBS"]
            rm(fitted_forest, forest_predictions)

            cox_error <- cox_test_errors(
                feature_data = analysis_features,
                jump_data = jump_data,
                training_indices = training_indices,
                test_indices = test_indices,
                evaluation_times = evaluation_times
            )
            errors$cox[b, noise_index] <- cox_error$IBS
            cox_null_transitions[[b, noise_index]] <-
                cox_error$null_transitions

            cox_lasso_error <- cox_test_errors(
                feature_data = analysis_features,
                jump_data = jump_data,
                training_indices = training_indices,
                test_indices = test_indices,
                evaluation_times = evaluation_times,
                penalised = TRUE
            )
            errors$cox_lasso[b, noise_index] <- cox_lasso_error$IBS
            cox_lasso_null_transitions[[b, noise_index]] <-
                cox_lasso_error$null_transitions

            cat(
                "Completed noise level", number_of_noise,
                "iteration", b, "of", B, "\n"
            )
        }
    }

    result <- lapply(errors, as.data.frame)
    attr(result, "cox_null_transitions") <- cox_null_transitions
    attr(result, "cox_lasso_null_transitions") <-
        cox_lasso_null_transitions
    result
}


plot_noise_ibs <- function(
        ibs_errors, path = NULL, noise_counts = NULL) {
    model_labels <- noise_study_model_labels(ibs_errors)
    model_sources <- names(model_labels)
    model_colours <- article_model_colours[unname(model_labels)]
    error_tables <- ibs_errors[model_sources]
    valid_table <- vapply(error_tables, function(error_table) {
        (is.data.frame(error_table) || is.matrix(error_table)) &&
            nrow(error_table) > 0L && ncol(error_table) > 0L &&
            all(vapply(as.data.frame(error_table), is.numeric, logical(1))) &&
            all(is.finite(as.matrix(error_table)))
    }, logical(1))
    if (!all(valid_table)) {
        stop("each model error table must contain finite numeric values")
    }

    reference_dimensions <- dim(error_tables[[1L]])
    reference_names <- colnames(error_tables[[1L]])
    same_structure <- vapply(error_tables[-1L], function(error_table) {
        identical(dim(error_table), reference_dimensions) &&
            identical(colnames(error_table), reference_names)
    }, logical(1))
    if (!all(same_structure)) {
        stop("the model error tables must have identical dimensions and columns")
    }
    if (is.null(reference_names) ||
        !all(grepl("^noise_[0-9]+$", reference_names))) {
        stop("error-table columns must be named noise_0, noise_1, ...")
    }
    available_noise_counts <- as.integer(sub(
        "^noise_", "", reference_names
    ))
    if (anyDuplicated(available_noise_counts) ||
        !identical(
            available_noise_counts,
            seq.int(0L, max(available_noise_counts))
        )) {
        stop("noise columns must run consecutively from noise_0")
    }
    noise_counts_selected <- !is.null(noise_counts)
    if (!noise_counts_selected) {
        noise_counts <- available_noise_counts
    } else {
        if (!is.numeric(noise_counts) || length(noise_counts) == 0L ||
            anyNA(noise_counts) || any(!is.finite(noise_counts)) ||
            any(noise_counts < 0L) ||
            any(noise_counts != floor(noise_counts)) ||
            anyDuplicated(noise_counts)) {
            stop("noise_counts must contain unique non-negative integers")
        }
        noise_counts <- as.integer(noise_counts)
        unavailable_counts <- setdiff(
            noise_counts, available_noise_counts
        )
        if (length(unavailable_counts) > 0L) {
            stop(
                "noise_counts contains unavailable values: ",
                paste(unavailable_counts, collapse = ", ")
            )
        }
        noise_counts <- available_noise_counts[
            available_noise_counts %in% noise_counts
        ]
    }
    selected_columns <- match(noise_counts, available_noise_counts)
    error_tables <- lapply(error_tables, function(error_table) {
        error_table[, selected_columns, drop = FALSE]
    })
    if (!is.null(path) &&
        (!is.character(path) || length(path) != 1L ||
         is.na(path) || !grepl("[.]pdf$", path, ignore.case = TRUE))) {
        stop("path must be NULL or one PDF file path")
    }

    plot_data <- do.call(rbind, lapply(model_sources, function(model_source) {
        error_table <- error_tables[[model_source]]
        data.frame(
            noise = rep(noise_counts, each = nrow(error_table)),
            run = rep(seq_len(nrow(error_table)), times = ncol(error_table)),
            model = unname(model_labels[model_source]),
            error = unname(unlist(error_table, use.names = FALSE))
        )
    }))
    mean_data <- stats::aggregate(
        error ~ noise + model, data = plot_data, FUN = mean
    )
    ordered_models <- unname(model_labels[model_sources])
    plot_data$model <- factor(plot_data$model, levels = ordered_models)
    mean_data$model <- factor(mean_data$model, levels = ordered_models)

    noise_step <- if (length(noise_counts) > 1L) min(diff(noise_counts)) else 1
    dodge <- ggplot2::position_dodge(width = 0.75 * noise_step)
    plot <- ggplot2::ggplot(
        plot_data,
        ggplot2::aes(x = noise, y = error, colour = model)
    ) +
        ggplot2::geom_boxplot(
            ggplot2::aes(group = interaction(noise, model)),
            width = 0.65 * noise_step,
            position = dodge,
            fill = "white",
            linewidth = 0.5,
            outlier.shape = NA
        ) +
        ggplot2::geom_line(
            data = mean_data,
            ggplot2::aes(group = model, linetype = model),
            position = dodge,
            linewidth = 0.75
        ) +
        ggplot2::geom_point(
            data = mean_data,
            position = dodge,
            ggplot2::aes(shape = model),
            fill = "white",
            size = 2,
            stroke = 0.8
        ) +
        ggplot2::scale_colour_manual(values = model_colours) +
        ggplot2::scale_shape_manual(values = article_model_shapes) +
        ggplot2::scale_linetype_manual(values = article_model_linetypes) +
        ggplot2::scale_x_continuous(
            breaks = if (noise_counts_selected) {
                noise_counts
            } else {
                pretty(noise_counts)
            }
        ) +
        ggplot2::labs(
            title = NULL,
            x = "Number of noise variables",
            y = "Normalised IBS",
            colour = "Model", shape = "Model", linetype = "Model"
        ) +
        article_theme() +
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
            filename = path,
            plot = plot,
            width = 10,
            height = 6,
            units = "in",
            dpi = 300
        )
    }

    plot
}


plot_noise_ibs_relative_to_JF <- function(
        ibs_errors, path = NULL, noise_counts = NULL) {
    model_labels <- noise_study_model_labels(ibs_errors)
    model_sources <- names(model_labels)
    ordered_models <- unname(model_labels[names(model_labels) != "jump_forest"])
    model_colours <- article_model_colours[ordered_models]
    error_tables <- ibs_errors[model_sources]
    valid_tables <- vapply(error_tables, function(tab) {
        (is.data.frame(tab) || is.matrix(tab)) &&
            nrow(tab) > 0L && ncol(tab) > 0L &&
            all(vapply(as.data.frame(tab), is.numeric, logical(1))) &&
            all(is.finite(as.matrix(tab))) && all(as.matrix(tab) >= 0)
    }, logical(1))
    if (!all(valid_tables)) {
        stop("each IBS table must contain finite, nonnegative numeric values")
    }
    error_tables <- lapply(error_tables, as.matrix)
    reference <- error_tables$jump_forest
    # Unnamed matrix rows and the default data-frame row names both mean
    # pairing by row position. Explicit run labels must agree across models.
    run_names <- function(tab) {
        if (is.null(rownames(tab))) as.character(seq_len(nrow(tab))) else rownames(tab)
    }
    if (!all(vapply(error_tables, function(tab) {
        identical(dim(tab), dim(reference)) &&
            identical(colnames(tab), colnames(reference)) &&
            identical(run_names(tab), run_names(reference))
    }, logical(1)))) {
        stop("IBS tables must have matching dimensions, noise columns and paired row order")
    }
    column_names <- colnames(reference)
    if (is.null(column_names) || anyNA(column_names) ||
        !all(grepl("^noise_[0-9]+$", column_names))) {
        stop("IBS columns must be named noise_0, noise_1, ...")
    }
    available_noise <- suppressWarnings(as.integer(sub("^noise_", "", column_names)))
    if (anyNA(available_noise) || anyDuplicated(available_noise)) {
        stop("IBS columns must identify unique non-negative integer noise levels")
    }
    if (is.null(noise_counts)) {
        noise_counts <- sort(available_noise)
    } else {
        if (!is.numeric(noise_counts) || length(noise_counts) == 0L ||
            any(!is.finite(noise_counts)) || any(noise_counts < 0) ||
            any(noise_counts != floor(noise_counts)) || anyDuplicated(noise_counts) ||
            any(!noise_counts %in% available_noise)) {
            stop("noise_counts must contain unique non-negative integers available in the IBS tables")
        }
        noise_counts <- sort(noise_counts)
    }
    selected <- match(noise_counts, available_noise)
    error_tables <- lapply(error_tables, function(tab) tab[, selected, drop = FALSE])
    reference <- error_tables$jump_forest
    if (any(reference <= 0)) {
        stop("JF IBS must be strictly positive at every selected run and noise level")
    }
    if (!is.null(path) &&
        (!is.character(path) || length(path) != 1L || is.na(path) ||
         !grepl("[.]pdf$", path, ignore.case = TRUE))) {
        stop("path must be NULL or one PDF file path")
    }

    model_labels <- model_labels[names(model_labels) != "jump_forest"]
    relative_raw <- do.call(rbind, lapply(names(model_labels), function(model) {
        # Compute ratios within paired cells before any averaging.
        relative <- 100 * (error_tables[[model]] / reference - 1)
        if (any(!is.finite(relative))) stop("relative IBS differences must be finite")
        do.call(rbind, lapply(seq_along(noise_counts), function(j) {
            data.frame(
                noise = noise_counts[j], model = unname(model_labels[model]),
                run = seq_len(nrow(relative)), error = relative[, j]
            )
        }))
    }))
    relative_summary <- stats::aggregate(error ~ noise + model, relative_raw, mean)
    names(relative_summary)[names(relative_summary) == "error"] <- "mean"
    relative_raw$model <- factor(relative_raw$model, levels = ordered_models)
    relative_summary$model <- factor(relative_summary$model, levels = ordered_models)

    model_linetypes <- article_model_linetypes
    x_span <- max(1, diff(range(noise_counts)))
    noise_step <- if (length(noise_counts) > 1L) min(diff(noise_counts)) else 1
    dodge <- ggplot2::position_dodge(width = 0.75 * noise_step)
    line_layer <- if (length(noise_counts) > 1L) {
        ggplot2::geom_line(ggplot2::aes(linetype = model), position = dodge,
                           linewidth = 0.8)
    }
    plot <- ggplot2::ggplot(
        relative_summary,
        ggplot2::aes(x = noise, y = mean, colour = model, group = model)
    ) +
        ggplot2::geom_hline(
            yintercept = 0, colour = "#333333", linetype = "dashed",
            linewidth = 0.7, show.legend = FALSE
        ) +
        ggplot2::geom_boxplot(
            data = relative_raw,
            ggplot2::aes(y = error, group = interaction(noise, model)),
            width = 0.65 * noise_step, position = dodge,
            fill = "white", linewidth = 0.5, outlier.shape = NA
        ) +
        line_layer +
        ggplot2::geom_point(ggplot2::aes(shape = model), position = dodge,
                            size = 2.6, stroke = 0.8) +
        ggplot2::scale_colour_manual(values = model_colours, limits = ordered_models) +
        ggplot2::scale_shape_manual(values = article_model_shapes, limits = ordered_models) +
        ggplot2::scale_linetype_manual(values = model_linetypes, limits = ordered_models) +
        ggplot2::scale_x_continuous(
            breaks = noise_counts,
            expand = ggplot2::expansion(add = c(0.02, 0.05) * x_span)
        ) +
        ggplot2::scale_y_continuous(
            labels = function(x) paste0(
                ifelse(x > 0, "+", ""), format(x, trim = TRUE, scientific = FALSE), "%"
            ),
            breaks = function(limits) pretty(limits, n = 8),
            expand = ggplot2::expansion(mult = c(0.08, 0.07))
        ) +
        ggplot2::labs(
            title = NULL,
            x = "Number of noise variables", y = "Relative IBS difference (%)",
            colour = "Model", shape = "Model", linetype = "Model"
        ) +
        ggplot2::theme_minimal(base_size = 12) +
        ggplot2::theme(
            legend.position = "bottom",
            axis.text.x = ggplot2::element_text(size = 11),
            axis.text.y = ggplot2::element_text(size = 10),
            panel.grid.minor = ggplot2::element_blank(),
            panel.grid.major.x = ggplot2::element_blank(),
            panel.grid.major.y = ggplot2::element_line(colour = "#E8E8E8", linewidth = 0.35),
            plot.title.position = "panel",
            axis.title.x = ggplot2::element_text(margin = ggplot2::margin(t = 12)),
            axis.title.y = ggplot2::element_text(margin = ggplot2::margin(r = 10)),
            plot.margin = ggplot2::margin(16, 18, 12, 12)
        )

    if (!is.null(path)) {
        if (!dir.exists(dirname(path)) && !dir.create(dirname(path), recursive = TRUE)) {
            stop("could not create plot directory: ", dirname(path))
        }
        ggplot2::ggsave(path, plot, width = 10, height = 6,
                        units = "in", bg = "white")
    }
    plot
}


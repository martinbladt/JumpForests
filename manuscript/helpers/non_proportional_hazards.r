# Helpers for prediction under non-proportional hazards. No fitting on source.
source("helpers/coverage/coverage_dgp.r")

non_proportional_hazards_cox_formulas <- list(
    cox_correct_interactions = survival::Surv(tstart, tstop, event) ~
        X1 * factor(X2) + X3
)

run_non_proportional_hazards_prediction <- function(n, B = 100, seed = 2026,
                                        num_event_times = 250,
                                        path = file.path(
        "results/non_proportional_hazards/Data",
        paste0("n", n)
    )) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
    metrics <- c("ibs", "ikl", "ispherical")
    output_paths <- file.path(
        path, paste0("oob_errors_", metrics, "_harmonised.txt")
    )
    configuration_path <- file.path(path, "harmonised_study_configuration.rds")
    configuration <- list(
        n = as.integer(n), B = as.integer(B), seed = as.integer(seed),
        num_event_times = as.integer(num_event_times),
        parameters = non_proportional_hazards_parameters,
        cox_formulas = lapply(non_proportional_hazards_cox_formulas, function(formula) {
            paste(deparse(formula), collapse = "")
        })
    )
    if (all(file.exists(output_paths))) {
        if (!file.exists(configuration_path) ||
            !identical(readRDS(configuration_path), configuration)) {
            stop("Existing non-proportional-hazards tables have a different or unknown configuration in ", path)
        }
        message("Keeping existing non-proportional-hazards harmonised tables in ", path)
        return(invisible(output_paths))
    }
    if (any(file.exists(output_paths)) || file.exists(configuration_path)) {
        stop("Incomplete non-proportional-hazards tables in ", path, "; inspect before rerunning")
    }

    cache_path <- file.path(path, paste0("simulated_data_seed", seed, ".rds"))
    if (file.exists(cache_path)) {
        cached <- readRDS(cache_path)
        if (!identical(cached$parameters, non_proportional_hazards_parameters) ||
            !identical(cached$n, as.integer(n)) ||
            !identical(cached$seed, as.integer(seed))) {
            stop("Incompatible simulated-data cache at ", cache_path)
        }
        simulated <- cached$data
    } else {
        simulated <- do.call(
            simulate_non_proportional_hazards,
            c(list(n = n, seed = seed), non_proportional_hazards_parameters)
        )
        saveRDS(list(
            n = as.integer(n), seed = as.integer(seed),
            parameters = non_proportional_hazards_parameters, data = simulated
        ), cache_path)
    }

    result <- compute_dataset_errors_harmonised(
        feature_data = simulated$feature_data,
        jump_data = simulated$jump_data,
        B = B, seed = seed, num_event_times = num_event_times,
        cox_formulas = non_proportional_hazards_cox_formulas
    )
    for (metric in metrics) {
        write.table(
            result[[metric]],
            file = output_paths[match(metric, metrics)],
            sep = "\t", row.names = FALSE
        )
    }
    saveRDS(configuration, configuration_path)
    invisible(result)
}


plot_non_proportional_hazards_ibs_by_size <- function(
    sample_sizes,
    input_path = "results/non_proportional_hazards/Data",
    output_path = "results/non_proportional_hazards/Plots"
) {
    model_columns <- c("logrank", "cox", "cox_correct_interactions")
    model_labels <- c(
        logrank = "JF",
        cox = "Cox",
        cox_correct_interactions = "Cox interactions"
    )
    raw <- do.call(rbind, lapply(sample_sizes, function(n) {
        table_path <- file.path(
            input_path, paste0("n", n), "oob_errors_ibs_harmonised.txt"
        )
        scores <- read.table(table_path, header = TRUE, sep = "\t",
                             check.names = FALSE)
        if (!identical(names(scores), model_columns) ||
            any(!is.finite(as.matrix(scores)))) {
            stop("Invalid IBS scores in ", table_path)
        }
        data.frame(
            n = rep.int(n, nrow(scores) * ncol(scores)),
            split = rep(seq_len(nrow(scores)), times = ncol(scores)),
            model = rep(names(scores), each = nrow(scores)),
            ibs = unlist(scores, use.names = FALSE)
        )
    }))
    raw$model <- factor(unname(model_labels[raw$model]),
                        levels = unname(model_labels))
    means <- stats::aggregate(ibs ~ n + model, data = raw, FUN = mean)
    counts <- stats::aggregate(ibs ~ n + model, data = raw, FUN = length)
    names(counts)[3L] <- "splits"
    summary <- merge(means, counts, by = c("n", "model"))
    names(summary)[3L] <- "mean_ibs"
    summary <- summary[order(summary$n, summary$model), ]

    colours <- article_model_colours[unname(model_labels)]
    shapes <- article_model_shapes[unname(model_labels)]
    linetypes <- article_model_linetypes[unname(model_labels)]
    raw$n_factor <- factor(raw$n, levels = sample_sizes)
    summary$n_factor <- factor(summary$n, levels = sample_sizes)
    dodge <- ggplot2::position_dodge(width = 0.6)
    figure <- ggplot2::ggplot(
        raw, ggplot2::aes(x = n_factor, y = ibs, colour = model)
    ) +
        ggplot2::geom_boxplot(
            ggplot2::aes(group = interaction(n_factor, model)),
            position = dodge, width = 0.5, fill = "white",
            linewidth = 0.5, outlier.shape = NA
        ) +
        ggplot2::geom_line(
            data = summary, ggplot2::aes(y = mean_ibs, group = model,
                                         linetype = model),
            position = dodge, linewidth = 0.9
        ) +
        ggplot2::geom_point(
            data = summary, ggplot2::aes(y = mean_ibs, shape = model),
            position = dodge, size = 2.5
        ) +
        ggplot2::scale_colour_manual(values = colours) +
        ggplot2::scale_shape_manual(values = shapes) +
        ggplot2::scale_linetype_manual(values = linetypes) +
        ggplot2::labs(
            title = NULL, subtitle = NULL,
            x = "Sample size n", y = "Normalised integrated Brier score",
            colour = "Model", shape = "Model", linetype = "Model"
        ) +
        article_theme() +
        ggplot2::theme(axis.text.x = ggplot2::element_text(
            size = 11, angle = 45, hjust = 1
        ))

    dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
    ggplot2::ggsave(file.path(output_path, "ibs_by_sample_size.pdf"),
                    figure, width = 10, height = 6)
    invisible(figure)
}


# Reporting helpers for saved study results. Loaded by helpers/setup.r; no models are
# fitted here. The plotting and VIMP preparation functions are defined in the study helper files.

noise_study_model_labels <- function(ibs_errors) {
    required <- c("jump_forest", "cox", "cox_lasso")
    if (!is.list(ibs_errors) || anyDuplicated(names(ibs_errors)) ||
        !all(required %in% names(ibs_errors))) {
        stop("ibs_errors must contain jump_forest, cox and cox_lasso error tables")
    }
    labels <- c(jump_forest = "JF", article_model_labels[c(
        "cox", "cox_lasso", "cox_x1_x2", "cox_all_interactions"
    )])
    labels[names(labels) %in% names(ibs_errors)]
}

read_threshold_interaction_noise_errors <- function(
        path = "results/threshold_interaction/Data",
        require_interactions = TRUE) {
    base_models <- c("jump_forest", "cox", "cox_lasso")
    extra_models <- c("cox_x1_x2", "cox_all_interactions")
    paths <- setNames(
        file.path(path, paste0("ibs_errors_", c(base_models, extra_models), ".txt")),
        c(base_models, extra_models)
    )
    if (!all(file.exists(paths[base_models]))) {
        stop("the original three noise-study score tables are required")
    }
    have_interactions <- file.exists(paths[extra_models])
    if ((require_interactions || any(have_interactions)) && !all(have_interactions)) {
        stop("both completed interaction Cox score tables are required")
    }
    if (all(have_interactions)) {
        checkpoint_path <- file.path(path, "ibs_errors_cox_interactions_checkpoint.rds")
        if (file.exists(checkpoint_path)) {
            checkpoint <- readRDS(checkpoint_path)
            if (!identical(checkpoint$configuration$original_tables_md5,
                           unname(tools::md5sum(paths[base_models])))) {
                stop("interaction results were fitted against different original score tables")
            }
        }
    }
    lapply(paths[file.exists(paths)], function(file) {
        utils::read.table(file, header = TRUE, sep = "\t", check.names = FALSE)
    })
}

plot_saved_threshold_interaction_noise_study <- function(
        input_path = "results/threshold_interaction/Data",
        output_path = "results/threshold_interaction/Plots",
        noise_counts = seq.int(0L, 50L, by = 5L),
        require_interactions = TRUE) {
    errors <- read_threshold_interaction_noise_errors(input_path, require_interactions)
    plots <- list(
        absolute = plot_noise_ibs(
            errors, path = file.path(output_path, "IBS_noise_plot.pdf"),
            noise_counts = noise_counts
        ),
        relative = plot_noise_ibs_relative_to_JF(
            errors, path = file.path(output_path, "IBS_noise_relative_to_JF.pdf"),
            noise_counts = noise_counts
        ),
        combined = plot_noise_ibs_panels(
            errors, path = file.path(output_path, "IBS_noise_panels_threshold_interaction.pdf"),
            noise_counts = noise_counts
        )
    )
    invisible(plots)
}

# A signal is recovered when it is among the three highest-ranked VIMP
# variables in that run. This is distinct from nonzero coefficient selection.
write_proportional_hazards_vimp_tables <- function(
        vimp_results,
        output_path = "results/proportional_hazards/Tables") {
    if (is.character(vimp_results)) {
        if (length(vimp_results) != 1L || is.na(vimp_results) ||
            !file.exists(vimp_results)) {
            stop("vimp_results must be a data frame or one existing file path")
        }
        vimp_results <- utils::read.table(
            vimp_results, header = TRUE, sep = "\t", check.names = FALSE
        )
    }
    signal <- prepare_proportional_hazards_signal_recovery(vimp_results, include_zero = FALSE)
    noise_counts <- c(10L, 25L, 50L)
    if (!setequal(unique(signal$noise_count), noise_counts)) {
        stop("the VIMP tables require noise levels 10, 25 and 50")
    }
    run_ids <- sort(unique(signal$run))
    if ("number_of_runs" %in% names(vimp_results)) {
        B <- unique(vimp_results$number_of_runs)
        if (!is.numeric(B) || length(B) != 1L || !is.finite(B) ||
            B < 1 || B != floor(B) || !setequal(run_ids, seq_len(B))) {
            stop("the VIMP study is incomplete: some expected runs are missing")
        }
    }
    model_labels <- levels(signal$model)
    for (model in model_labels) {
        for (noise in noise_counts) {
            observed_runs <- sort(unique(signal$run[
                signal$model == model & signal$noise_count == noise
            ]))
            if (!identical(observed_runs, run_ids)) {
                stop("every model and noise level must contain the same complete set of runs")
            }
        }
    }

    # Aggregate within runs first; joint recovery cannot be inferred from
    # the three marginal recovery rates.
    run_recovery <- stats::aggregate(
        top_three ~ run + noise_count + model, data = signal, FUN = sum
    )
    names(run_recovery)[names(run_recovery) == "top_three"] <- "recovered"
    summary_rows <- individual_rows <- list()
    summary_body <- individual_body <- character()
    row_line <- function(cells) paste0(paste(cells, collapse = " & "), " \\\\")
    percent <- function(x) paste0(sprintf("%.1f", x), "\\%")
    for (model in model_labels) {
        mean_recovered <- all_three_percent <- numeric(length(noise_counts))
        individual_percent <- matrix(NA_real_, nrow = 3L, ncol = length(noise_counts))
        for (j in seq_along(noise_counts)) {
            noise <- noise_counts[j]
            counts <- run_recovery$recovered[
                run_recovery$model == model & run_recovery$noise_count == noise
            ]
            mean_recovered[j] <- mean(counts)
            all_three_percent[j] <- 100 * mean(counts == 3L)
            for (k in 1:3) {
                individual_percent[k, j] <- 100 * mean(signal$top_three[
                    signal$model == model & signal$noise_count == noise &
                        signal$variable == paste0("X", k)
                ])
            }
        }
        summary_rows[[model]] <- data.frame(
            model = model, noise_count = noise_counts,
            mean_recovered = mean_recovered, all_three_percent = all_three_percent,
            runs = length(run_ids)
        )
        individual_rows[[model]] <- data.frame(
            model = model, variable = rep(paste0("X", 1:3), times = length(noise_counts)),
            noise_count = rep(noise_counts, each = 3L),
            recovery_percent = as.vector(individual_percent), runs = length(run_ids)
        )
        if (length(summary_body)) {
            summary_body <- c(summary_body, "\\addlinespace")
            individual_body <- c(individual_body, "\\addlinespace")
        }
        summary_body <- c(summary_body,
            row_line(c(paste0("\\multirow{2}{*}{", model, "}"),
                       "Mean recovered", sprintf("%.2f", mean_recovered))),
            row_line(c("", "All three", percent(all_three_percent)))
        )
        for (k in 1:3) {
            individual_body <- c(individual_body, row_line(c(
                if (k == 1L) paste0("\\multirow{3}{*}{", model, "}") else "",
                paste0("$X_", k, "$"),
                percent(individual_percent[k, ])
            )))
        }
    }
    study_description <- paste0(length(run_ids), " runs")
    if ("sample_size" %in% names(vimp_results)) {
        study_description <- paste0(study_description, " with $n = ",
                                    unique(vimp_results$sample_size), "$")
    }
    make_table <- function(caption, label, second_column, body) {
        c(
            "% Generated from saved proportional-hazards VIMP rankings; requires \\usepackage{booktabs,multirow}.",
            "\\begin{table}[tbp]", "\\centering", "\\small",
            paste0("\\caption{", caption, "}"), paste0("\\label{", label, "}"),
            "\\begin{tabular}{llrrr}", "\\toprule",
            row_line(c("", "", "\\multicolumn{3}{c}{Number of noise variables}")),
            "\\cmidrule(lr){3-5}",
            row_line(c("Model", second_column, noise_counts)), "\\midrule",
            body, "\\bottomrule", "\\end{tabular}", "\\end{table}"
        )
    }
    summary_tex <- make_table(
        paste0("Signal recovery in the proportional-hazards VIMP study (", study_description, "). ",
               "A signal is recovered when it ranks among the three most important variables. ",
               "Mean recovered is the mean number of recovered signals out of ",
               "$X_1,X_2,X_3$; all three is the percentage of runs recovering them jointly. ",
               "JF (held-out), Cox and Cox Lasso use held-out permutation importance; ",
               "JF (internal OOB) uses internal out-of-bag importance."),
        "tab:proportional-hazards-vimp-summary", "Summary", summary_body
    )
    individual_tex <- make_table(
        paste0("Individual signal recovery in the proportional-hazards VIMP study (", study_description, "). ",
               "Entries are the percentages of runs in which the indicated signal ",
               "ranks among the three most important variables. ",
               "JF (held-out), Cox and Cox Lasso use held-out permutation importance; ",
               "JF (internal OOB) uses internal out-of-bag importance."),
        "tab:proportional-hazards-vimp-individual", "Signal", individual_body
    )
    dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
    paths <- file.path(output_path, c("vimp_recovery_summary.tex", "vimp_recovery_individual.tex"))
    writeLines(summary_tex, paths[1L])
    writeLines(individual_tex, paths[2L])
    message("Saved proportional-hazards VIMP tables in ", output_path)
    invisible(list(
        summary = do.call(rbind, summary_rows),
        individual = do.call(rbind, individual_rows), paths = paths
    ))
}

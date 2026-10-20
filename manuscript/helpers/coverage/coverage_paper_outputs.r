# Reporting only: never simulates data, tunes, or refits a forest.
# This helper deliberately requires the averaged leaf Markov variance study.
# All manuscript entry points and helpers use manuscript/ as the working directory.
.coverage_paper_directory <- normalizePath("helpers/coverage", mustWork = TRUE)
source(file.path(.coverage_paper_directory, "coverage_report_figures.r"))
source(file.path(.coverage_paper_directory, "coverage_dgp.r"))
source(file.path(.coverage_paper_directory, "coverage_inference.r"))

.coverage_paper_config <- function(path) {
    if (!requireNamespace("data.table", quietly = TRUE)) stop("data.table is required")
    config <- readRDS(file.path(path, "configuration.rds"))$config
    if (!identical(config$variance_estimator, "scaled_mean_leaf_na") ||
        !identical(config$swr, FALSE)) {
        stop("These outputs require the new scaled_mean_leaf_na variance estimator with sampling without replacement")
    }
    config
}

.coverage_paper_write_table <- function(table, path) {
    dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
    staged <- tempfile(".coverage-table-", tmpdir = dirname(path))
    on.exit(unlink(staged))
    data.table::fwrite(table, staged, sep = "\t", na = "NA")
    if (!file.rename(staged, path)) stop("Could not save table: ", path)
    invisible(path)
}

# Read only one n/arm at a time, discarding other profiles and unused columns
# immediately. The saved reporting grid includes all DGP switch times. Plotting
# this grid does not change the exact event-time uniform coverage calculations.
read_coverage_curve_data <- function(
    path, n, arm, profiles = coverage_profiles(FALSE),
    noise_profile = c(X4 = 0L, X5 = 1L), replicates = NULL,
    grid = c("report", "events")) {
    config <- .coverage_paper_config(path)
    grid <- match.arg(grid)
    if (length(n) != 1L || !n %in% config$n || length(arm) != 1L ||
        !arm %in% c("without_noise", "with_noise")) stop("Invalid n or noise setting")
    if (is.null(replicates)) replicates <- seq_len(config$N)
    if (!length(replicates) || anyNA(replicates) || anyDuplicated(replicates) ||
        any(!replicates %in% seq_len(config$N))) stop("Invalid replicate numbers")
    required <- paste0("X", 1:3)
    if (!all(required %in% names(profiles))) stop("profiles must contain X1, X2, X3")
    selected <- unique(as.data.frame(profiles)[required])
    if (!nrow(selected)) stop("Select at least one covariate profile")
    support <- coverage_profiles(FALSE)
    key <- function(z) do.call(paste, c(z[required], sep = ";"))
    if (any(!key(selected) %in% key(support))) stop("Profile outside the DGP support")
    if (!all(c("X4", "X5") %in% names(noise_profile)) ||
        !noise_profile[["X4"]] %in% 0:1 || !noise_profile[["X5"]] %in% 1:3) {
        stop("noise_profile must specify X4 in 0:1 and X5 in 1:3")
    }
    if (arm == "with_noise") {
        selected$X4 <- noise_profile[["X4"]]
        selected$X5 <- noise_profile[["X5"]]
    }
    profile_ids <- do.call(paste, c(lapply(names(selected), function(name) {
        paste0(name, "=", selected[[name]])
    }), sep = ";"))
    columns <- c("n", "arm", "replicate", "profile_id", paste0("X", 1:5),
                 "transition", "time", "estimate", "true", "variance",
                 "pointwise_lower", "pointwise_upper",
                 "simultaneous_lower", "simultaneous_upper")
    results <- vector("list", length(replicates))
    for (i in seq_along(replicates)) {
        replicate <- replicates[i]
        directory <- file.path(path, paste0("n", n), sprintf("replicate_%03d", replicate))
        # These small records identify the actual in-bag size and forest factor.
        diagnostic <- data.table::fread(file.path(directory, paste0(arm, "_diagnostics.tsv")))
        expected_factor <- coverage_leaf_variance_factor(
            n, diagnostic$subsample_size, diagnostic$ntrees)
        if (nrow(diagnostic) != 1L || diagnostic$n != n || diagnostic$replicate != replicate ||
            diagnostic$arm != arm || diagnostic$variance_estimator != "scaled_mean_leaf_na" ||
            abs(diagnostic$leaf_variance_factor - expected_factor) > 1e-12) {
            stop("Variance metadata mismatch in ", directory)
        }
        if (grid == "report") {
            table <- data.table::as.data.table(readRDS(file.path(directory, paste0(arm, "_report_grid.rds"))))
        } else {
            table <- data.table::as.data.table(readRDS(file.path(directory, paste0(arm, ".rds")))$event_time_results)
        }
        if (any(table$n != n) || any(table$arm != arm) || any(table$replicate != replicate) ||
            any(table[model == "JF", variance_estimator] != "scaled_mean_leaf_na")) {
            stop("Saved result metadata mismatch in ", directory)
        }
        table <- table[model == "JF" & profile_id %in% profile_ids, ..columns]
        if (!setequal(unique(table$profile_id), profile_ids)) stop("Missing requested profiles in ", directory)
        counts <- table[, .(last_time = max(time)), by = c("profile_id", "transition")]
        if (nrow(counts) != 4L * length(profile_ids) || any(counts$last_time != config$end_time)) {
            stop("Incomplete transition curves in ", directory)
        }
        if (any(!is.finite(table$estimate))) stop("Non-finite JF estimates in ", directory)
        results[[i]] <- table
        if (i %% 25L == 0L || i == length(replicates)) {
            message("Read n=", n, " ", arm, ": ", i, "/", length(replicates), " saved replicates")
        }
    }
    result <- data.table::rbindlist(results)
    data.table::setorderv(result, c("X1", "X3", "X2", "transition", "replicate", "time"))
    result
}

.coverage_transition_curve_plot <- function(curves, transition_value, band_replicate, config) {
    z <- data.table::copy(curves[transition == transition_value])
    selected <- z[replicate == band_replicate]
    if (!nrow(selected)) stop("The selected confidence-band replicate is unavailable")
    if (any(!is.finite(as.matrix(selected[, .(pointwise_lower, pointwise_upper,
                                            simultaneous_lower, simultaneous_upper)])))) {
        stop("The selected confidence bands contain non-finite limits")
    }
    profile_label <- function(d) {
        label <- paste0("X1 = ", d$X1, ", X2 = ", d$X2, ", X3 = ", d$X3)
        if (unique(d$arm) == "with_noise") {
            label <- paste0(label, "\nX4 = ", d$X4, ", X5 = ", d$X5)
        }
        label
    }
    labels <- unique(profile_label(z))
    z[, profile := factor(profile_label(z), levels = labels)]
    selected[, profile := factor(profile_label(selected), levels = labels)]
    truth <- selected[, .(time, true, profile)]
    # Straight segments reproduce the piecewise-linear true cumulative rate;
    # estimator and interval endpoints are displayed as right-continuous steps.
    level <- format(100 * config$level, trim = TRUE)
    estimate_label <- paste0("JF estimates (", data.table::uniqueN(z$replicate), " runs)")
    selected_label <- paste0("JF, replicate ", band_replicate)
    point_label <- paste0(level, "% pointwise limits")
    uniform_label <- paste0(level, "% uniform limits")
    legend <- c("True cumulative rate", estimate_label, selected_label, point_label, uniform_label)
    point <- data.table::rbindlist(list(
        selected[, .(time, profile, value = pointwise_lower, side = "lower")],
        selected[, .(time, profile, value = pointwise_upper, side = "upper")]))
    uniform <- data.table::rbindlist(list(
        selected[, .(time, profile, value = simultaneous_lower, side = "lower")],
        selected[, .(time, profile, value = simultaneous_upper, side = "upper")]))
    ggplot2::ggplot() +
        ggplot2::geom_step(data = z, ggplot2::aes(time, estimate, group = replicate,
                            colour = estimate_label, linetype = estimate_label),
                            linewidth = 0.25, alpha = 1, direction = "hv") +
        ggplot2::geom_step(data = uniform, ggplot2::aes(time, value, group = side,
                            colour = uniform_label, linetype = uniform_label),
                            linewidth = 0.65, direction = "hv") +
        ggplot2::geom_step(data = point, ggplot2::aes(time, value, group = side,
                            colour = point_label, linetype = point_label),
                            linewidth = 0.65, direction = "hv") +
        ggplot2::geom_step(data = selected, ggplot2::aes(time, estimate,
                            colour = selected_label, linetype = selected_label),
                            linewidth = 0.5, direction = "hv") +
        ggplot2::geom_line(data = truth, ggplot2::aes(time, true,
                            colour = "True cumulative rate", linetype = "True cumulative rate"),
                            linewidth = 1.05) +
        ggplot2::facet_wrap(ggplot2::vars(profile), ncol = 4, scales = "free_y") +
        ggplot2::scale_colour_manual(values = setNames(c("#111111", "#999999", rep("#0072B2", 3)), legend),
                                      breaks = legend, drop = FALSE) +
        ggplot2::scale_linetype_manual(values = setNames(c("solid", "solid", "solid", "22", "13"), legend),
                                        breaks = legend, drop = FALSE) +
        ggplot2::scale_x_continuous(breaks = pretty(c(0, config$end_time), n = 4),
                                    expand = ggplot2::expansion(mult = c(0.01, 0.02))) +
        ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0.02, 0.06))) +
        ggplot2::labs(x = paste0("Time (n = ", format(unique(z$n), big.mark = ",", trim = TRUE), ")"),
                      y = bquote(Lambda[.(gsub("->", "", transition_value))](t~"|"~x))) +
        .coverage_plot_theme() +
        ggplot2::theme(legend.text = ggplot2::element_text(size = 10),
                       legend.key.width = grid::unit(1.4, "cm")) +
        ggplot2::guides(colour = ggplot2::guide_legend(nrow = 2, byrow = TRUE),
                        linetype = ggplot2::guide_legend(nrow = 2, byrow = TRUE))
}

plot_coverage_transition_curves <- function(
    path, output_path = file.path(path, "Plots"), n = NULL,
    arms = c("without_noise", "with_noise"), profiles = coverage_profiles(FALSE),
    noise_profile = c(X4 = 0L, X5 = 1L), band_replicate = 1L,
    replicates = NULL, grid = c("report", "events")) {
    config <- .coverage_paper_config(path)
    if (!requireNamespace("ggplot2", quietly = TRUE)) stop("ggplot2 is required")
    grid <- match.arg(grid)
    if (is.null(n)) n <- config$n
    if (is.null(replicates)) replicates <- seq_len(config$N)
    if (length(band_replicate) != 1L || !band_replicate %in% replicates) {
        stop("band_replicate must be one of the plotted replicates")
    }
    dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
    files <- character()
    manifest <- list()
    for (size in n) for (arm_value in arms) {
        curves <- read_coverage_curve_data(path, size, arm_value, profiles,
                                                noise_profile, replicates, grid)
        for (transition_value in coverage_transition_names) {
            plot <- .coverage_transition_curve_plot(curves, transition_value, band_replicate, config)
            name <- paste0("transition_curves_JF_", gsub("->", "_to_", transition_value),
                           "_", arm_value, "_n", size)
            file <- file.path(output_path, paste0(name, ".pdf"))
            height <- max(4.5, 2.5 * ceiling(data.table::uniqueN(curves$profile_id) / 4) + 1.5)
            files[name] <- .coverage_save_pdf(plot, file, width = 14, height = height)
            manifest[[length(manifest) + 1L]] <- data.frame(
                file = basename(file), n = size, arm = arm_value, transition = transition_value,
                model = "JF", variance_estimator = config$variance_estimator,
                curves_per_profile = length(replicates), band_replicate = band_replicate,
                profiles = data.table::uniqueN(curves$profile_id), time_grid = grid,
                X4 = if (arm_value == "with_noise") noise_profile[["X4"]] else NA,
                X5 = if (arm_value == "with_noise") noise_profile[["X5"]] else NA)
        }
        rm(curves)
        gc(verbose = FALSE)
        message("Saved four transition PDFs for n=", size, ", ", arm_value)
    }
    .coverage_paper_write_table(data.table::rbindlist(manifest),
                               file.path(path, "transition_curve_plot_manifest.tsv"))
    invisible(files)
}

.coverage_latex_table <- function(data, row_keys, caption, label, path, sizes, digits) {
    group_label <- c(overall = "Overall", `X1=0` = "$X_1=0$", `X1=1` = "$X_1=1$")
    headers <- c(group = "Group", transition = "Transition", model = "Model", time = "$t$")
    data <- data.table::copy(data)
    data[, group_order := match(group, names(group_label))]
    data[, model_order := match(model, c("JF", "NA"))]
    order_keys <- c("group_order", if ("transition" %in% row_keys) "transition", "model_order",
                    if ("time" %in% row_keys) "time")
    data.table::setorderv(data, c(order_keys, "n"))
    keys <- unique(data[, ..row_keys])
    row_end <- " \\\\"
    lines <- c("% Generated by write_coverage_latex_tables(); no additional LaTeX packages needed.",
               "\\begin{table}[htbp]", "\\centering", "\\small", paste0("\\caption{", caption, "}"),
               paste0("\\label{", label, "}"),
               paste0("\\begin{tabular}{", paste(rep("l", length(row_keys)), collapse = ""),
                      paste(rep("r", length(sizes)), collapse = ""), "}"), "\\hline",
               paste0(paste(c(unname(headers[row_keys]), paste0("$n=", sizes, "$")), collapse = " & "), row_end),
               "\\hline")
    previous_group <- NULL
    for (i in seq_len(nrow(keys))) {
        key <- keys[i]
        matched <- rep(TRUE, nrow(data))
        for (column in row_keys) matched <- matched & data[[column]] == key[[column]]
        rows <- data[matched]
        if (nrow(rows) != length(sizes) || anyDuplicated(rows$n) || !setequal(rows$n, sizes)) {
            stop("Incomplete or duplicated coverage-table row")
        }
        if (!is.null(previous_group) && previous_group != key$group) lines <- c(lines, "\\hline")
        previous_group <- key$group
        text <- vapply(row_keys, function(column) {
            value <- as.character(key[[column]])
            switch(column, group = unname(group_label[value]),
                   transition = paste0("$", gsub("->", "\\\\to ", value), "$"), value)
        }, character(1))
        rates <- rows$coverage[match(sizes, rows$n)]
        cells <- ifelse(is.finite(rates), formatC(100 * rates, format = "f", digits = digits), "--")
        lines <- c(lines, paste0(paste(c(text, cells), collapse = " & "), row_end))
    }
    lines <- c(lines, "\\hline", "\\end{tabular}", "\\end{table}")
    staged <- tempfile(".coverage-table-", tmpdir = dirname(path))
    on.exit(unlink(staged))
    writeLines(lines, staged)
    if (!file.rename(staged, path)) stop("Could not save LaTeX table: ", path)
    invisible(path)
}

# Export the settings actually selected on the saved pilots. Reading these
# small tables does not load evaluation datasets or run any model fits.
write_coverage_tuning_latex_table <- function(
    path, output_path = file.path(path, "Tables"), digits = 4L) {
    config <- .coverage_paper_config(path)
    if (!is.numeric(digits) || length(digits) != 1L || !is.finite(digits) ||
        digits < 0 || digits > 10 || digits != floor(digits)) {
        stop("digits must be an integer between 0 and 10")
    }
    arms <- c("without_noise", "with_noise")
    rule_labels <- c(logrank = "Logrank", gehan = "Gehan",
                     taroneware = "Tarone--Ware", approxlogrank = "Approx. logrank",
                     petoprentice = "Peto--Prentice")
    expected <- expand.grid(n = sort(config$n), arm = arms,
                            stringsAsFactors = FALSE)
    columns <- c("n", "arm", "split_rule", "min_node_size", "oob_ibs")
    tuning <- data.table::rbindlist(lapply(seq_len(nrow(expected)), function(i) {
        file <- file.path(path, paste0("n", expected$n[i]), "pilot",
                          paste0(expected$arm[i], "_selected.tsv"))
        if (!file.exists(file)) stop("Missing selected pilot result: ", file)
        selected <- data.table::fread(file)
        if (!all(c(columns, "status") %in% names(selected)) || nrow(selected) != 1L) {
            stop("Expected one selected pilot result with the required columns: ", file)
        }
        if (!isTRUE(selected$n == expected$n[i]) ||
            !identical(selected$arm, expected$arm[i]) ||
            !identical(selected$status, "ok") ||
            !selected$split_rule %in% names(rule_labels) ||
            !is.numeric(selected$oob_ibs) || !is.finite(selected$oob_ibs) ||
            !is.numeric(selected$min_node_size) || !is.finite(selected$min_node_size) ||
            selected$min_node_size < 1 || selected$min_node_size != floor(selected$min_node_size)) {
            stop("Invalid selected pilot result: ", file)
        }
        selected[, ..columns]
    }))
    tuning <- tuning[order(n, match(arm, arms))]
    caption <- paste0(
        "Selected JF tuning parameters for the coverage study. For each sample size, ",
        "the two feature sets use the same independent pilot dataset. Settings minimise ",
        "the normalised out-of-bag integrated Brier score (OOB IBS) over the supplied ",
        "grid and are held fixed for the evaluation datasets. Without noise uses ",
        "$X_1,X_2,X_3$; with noise additionally includes $X_4,X_5$.")
    row_end <- " \\\\"
    lines <- c(
        "% Generated by write_coverage_tuning_latex_table(); no additional LaTeX packages needed.",
        "\\begin{table}[htbp]", "\\centering", "\\small",
        paste0("\\caption{", caption, "}"), "\\label{tab:coverage-tuning}",
        "\\begin{tabular}{rllrr}", "\\hline",
        paste0("$n$ & Noise & Splitting rule & Min. node size & OOB IBS", row_end),
        "\\hline")
    for (i in seq_len(nrow(tuning))) {
        row <- tuning[i]
        if (i > 1L && tuning$n[i] != tuning$n[i - 1L]) lines <- c(lines, "\\hline")
        cells <- c(as.character(row$n), if (row$arm == "with_noise") "With" else "Without",
                   unname(rule_labels[row$split_rule]), as.character(row$min_node_size),
                   formatC(row$oob_ibs, format = "f", digits = digits, decimal.mark = "."))
        lines <- c(lines, paste0(paste(cells, collapse = " & "), row_end))
    }
    lines <- c(lines, "\\hline", "\\end{tabular}", "\\end{table}")
    dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
    .coverage_paper_write_table(tuning, file.path(output_path, "coverage_tuning_selected.tsv"))
    file <- file.path(output_path, "coverage_tuning.tex")
    staged <- tempfile(".coverage-table-", tmpdir = output_path)
    on.exit(unlink(staged))
    writeLines(lines, staged)
    if (!file.rename(staged, file)) stop("Could not save LaTeX table: ", file)
    invisible(file)
}

write_coverage_latex_tables <- function(
    path, output_path = file.path(path, "Tables"), times = c(1.2, 2, 3.3, 4),
    models = c("JF", "NA"), groups = c("overall", "X1=0", "X1=1"), digits = 1L) {
    config <- .coverage_paper_config(path)
    if (!length(models) || any(!models %in% c("JF", "NA")) || anyDuplicated(models) ||
        !length(groups) || any(!groups %in% c("overall", "X1=0", "X1=1")) || anyDuplicated(groups)) {
        stop("Invalid model or group selection")
    }
    point <- data.table::fread(file.path(path, "pointwise_by_group.tsv"))
    uniform <- data.table::fread(file.path(path, "simultaneous_by_group.tsv"))
    point <- point[model %in% models & group %in% groups]
    uniform <- uniform[model %in% models & group %in% groups]
    if (any(point$datasets_completed != config$N) || any(uniform$datasets_completed != config$N)) {
        stop("The coverage summaries do not contain all requested datasets")
    }
    sizes <- sort(config$n)
    keys <- c("n", "arm", "model", "group", "transition")
    expected <- length(sizes) * 2L * length(models) * length(groups) * 4L
    if (nrow(uniform) != expected || anyDuplicated(uniform[, ..keys])) {
        stop("Missing or duplicated transition/group summaries")
    }
    if (anyDuplicated(point[, c(keys, "time"), with = FALSE])) stop("Duplicated pointwise summaries")
    times <- sort(unique(times))
    if (!length(times) || any(!is.finite(times)) || any(times <= 0 | times > config$end_time)) {
        stop("times must be positive times up to the study endpoint")
    }
    available <- sort(unique(point$time))
    matched <- vapply(times, function(t) {
        j <- which.min(abs(available - t))
        if (abs(available[j] - t) > 1e-9) stop("Requested time is absent from the saved grid: ", t)
        available[j]
    }, numeric(1))
    selected <- point[time %in% matched]
    if (nrow(selected) != expected * length(matched)) stop("Incomplete pointwise summaries at selected times")
    selected[, `:=`(coverage = pointwise_coverage, coverage_type = "pointwise_at_time")]
    average <- point[anchor == FALSE, .(coverage = mean(pointwise_coverage),
                                reporting_times = .N), by = keys]
    average[, coverage_type := "pointwise_grid_average"]
    whole <- data.table::copy(uniform)
    whole[, `:=`(coverage = simultaneous_coverage, coverage_type = "uniform_whole_interval")]
    # Averages here must agree with the independently saved per-curve summary.
    check <- merge(average, uniform[, c(keys, "mean_pointwise_coverage"), with = FALSE], by = keys)
    if (any(abs(check$coverage - check$mean_pointwise_coverage) > 1e-10)) {
        stop("Pointwise grid average disagrees with the saved curve summaries")
    }
    dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
    .coverage_paper_write_table(selected, file.path(output_path, "coverage_pointwise_at_times.tsv"))
    .coverage_paper_write_table(average, file.path(output_path, "coverage_pointwise_grid_average.tsv"))
    .coverage_paper_write_table(whole, file.path(output_path, "coverage_uniform.tsv"))
    files <- character()
    nominal <- format(100 * config$level, trim = TRUE)
    for (arm_value in c("without_noise", "with_noise")) {
        features <- if (arm_value == "with_noise") "$X_1,\\ldots,X_5$" else "$X_1,X_2,X_3$"
        context <- paste0(" Nominal level: ", nominal, "\\%; $N=", config$N,
                          "$ datasets per sample size, using ", features,
                          ". Rates average equally over the covariate profiles in each group; unavailable intervals count as noncoverage.")
        for (type in c("pointwise_grid_average", "uniform")) {
            data <- if (type == "uniform") whole[arm == arm_value] else average[arm == arm_value]
            description <- if (type == "uniform") {
                paste0("Uniform coverage (\\%) of the entire cumulative transition-rate curve on $[0,",
                       config$end_time, "]$. Coverage is checked at all event times and their left limits.")
            } else {
                "Average pointwise coverage (\\%) of cumulative transition rates over the saved reporting times, excluding the deterministic anchor at zero. This is a time-grid average of pointwise coverage."
            }
            name <- paste0("coverage_", type, "_", arm_value)
            file <- file.path(output_path, paste0(name, ".tex"))
            files[name] <- .coverage_latex_table(data, c("group", "transition", "model"),
                paste0(description, context), paste0("tab:", gsub("_", "-", name)), file, sizes, digits)
        }
        for (transition_value in coverage_transition_names) {
            data <- selected[arm == arm_value & transition == transition_value]
            name <- paste0("coverage_pointwise_", gsub("->", "_to_", transition_value), "_", arm_value)
            file <- file.path(output_path, paste0(name, ".tex"))
            description <- paste0("Pointwise coverage (\\%) for transition $",
                                  gsub("->", "\\\\to ", transition_value), "$ at the indicated times.")
            files[name] <- .coverage_latex_table(data, c("group", "model", "time"),
                paste0(description, context), paste0("tab:", gsub("_", "-", name)), file, sizes, digits)
        }
    }
    invisible(files)
}

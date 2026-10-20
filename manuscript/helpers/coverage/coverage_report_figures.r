# PDF diagnostics for completed non-proportional-hazards coverage studies. Safe to source:
# call plot_coverage_study(path) explicitly after study summaries exist.

# Preserve every byte of the detailed coverage summary in a smaller file for
# Git. Stream the copy so this does not load the whole table into R's memory.
# The original TSV and all study checkpoints remain untouched locally.
export_coverage_tables_for_git <- function(path) {
    source <- file.path(path, "pointwise_by_profile.tsv")
    if (!file.exists(source)) return(invisible(character()))
    destination <- paste0(source, ".gz")
    before <- file.info(source)[, c("size", "mtime")]
    staged <- tempfile(".coverage-table-", tmpdir = path, fileext = ".tsv.gz")
    on.exit(unlink(staged), add = TRUE)
    input <- file(source, open = "rb")
    on.exit(close(input), add = TRUE)
    output <- gzfile(staged, open = "wb", compression = 6L)
    on.exit(if (!is.null(output)) close(output), add = TRUE)
    repeat {
        block <- readBin(input, what = "raw", n = 1024L * 1024L)
        if (!length(block)) break
        writeBin(block, output)
    }
    close(output)
    output <- NULL
    if (!identical(before, file.info(source)[, c("size", "mtime")])) {
        stop("Coverage summary changed during export; export it again: ", source)
    }
    if (!file.rename(staged, destination)) stop("Could not export table: ", destination)
    invisible(destination)
}

.coverage_plot_theme <- function() {
    ggplot2::theme_bw(base_size = 12) +
        ggplot2::theme(
            panel.grid.major = ggplot2::element_blank(),
            panel.grid.minor = ggplot2::element_blank(),
            panel.border = ggplot2::element_rect(colour = "#333333", fill = NA,
                                                linewidth = 0.45),
            strip.background = ggplot2::element_rect(fill = "white", colour = NA),
            strip.text = ggplot2::element_text(face = "bold", size = 10),
            axis.text = ggplot2::element_text(colour = "#111111", size = 11),
            axis.title = ggplot2::element_text(size = 12),
            legend.position = "bottom", legend.title = ggplot2::element_blank(),
            legend.text = ggplot2::element_text(size = 12),
            legend.key.width = grid::unit(1.4, "cm"),
            panel.spacing = grid::unit(0.9, "lines"),
            plot.margin = ggplot2::margin(8, 10, 5, 8))
}

.coverage_plot_scales <- function() {
    list(
        ggplot2::scale_colour_manual(values = c(JF = "#0072B2", "NA" = "#111111"),
                                     breaks = c("JF", "NA"), drop = FALSE),
        ggplot2::scale_linetype_manual(values = c(JF = "solid", "NA" = "dashed"),
                                       breaks = c("JF", "NA"), drop = FALSE),
        ggplot2::scale_shape_manual(values = c(JF = 15, "NA" = 16),
                                    breaks = c("JF", "NA"), drop = FALSE),
        ggplot2::scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.25),
                                    labels = function(x) paste0(round(100 * x), "%"),
                                    expand = ggplot2::expansion(mult = c(0.02, 0.02)))
    )
}

.coverage_save_pdf <- function(plot, file, width, height) {
    staged <- tempfile(".coverage-plot-", tmpdir = dirname(file), fileext = ".pdf")
    on.exit(unlink(staged))
    ggplot2::ggsave(staged, plot = plot, device = "pdf", width = width,
                    height = height, units = "in", useDingbats = FALSE)
    if (!file.rename(staged, file)) stop("Could not save PDF: ", file)
    file
}

plot_coverage_study <- function(path, output_path = file.path(path, "Plots")) {
    if (!requireNamespace("ggplot2", quietly = TRUE) ||
        !requireNamespace("data.table", quietly = TRUE)) {
        stop("ggplot2 and data.table are required to draw coverage plots")
    }
    point_path <- file.path(path, "pointwise_by_group.tsv")
    simultaneous_path <- file.path(path, "simultaneous_by_group.tsv")
    if (!file.exists(point_path) || !file.exists(simultaneous_path)) {
        stop("Create coverage summary tables before plotting")
    }
    # fread preserves the quoted model name "NA"; base read.delim would
    # otherwise interpret that model label as a missing value.
    pointwise <- data.table::fread(point_path)
    simultaneous <- data.table::fread(simultaneous_path)
    required <- c("n", "arm", "model", "group", "transition")
    if (!all(c(required, "time", "anchor", "pointwise_coverage") %in% names(pointwise)) ||
        !all(c(required, "simultaneous_coverage") %in% names(simultaneous))) {
        stop("The input tables do not have the expected coverage-summary columns")
    }
    configuration <- file.path(path, "configuration.rds")
    level <- if (file.exists(configuration)) readRDS(configuration)$config$level else 0.95
    if (length(level) != 1L || !is.finite(level) || level <= 0 || level >= 1) {
        stop("Invalid confidence level in saved configuration")
    }
    pointwise <- pointwise[model %in% c("JF", "NA") &
                               group %in% c("X1=0", "X1=1") &
                               !anchor & is.finite(pointwise_coverage)]
    simultaneous <- simultaneous[model %in% c("JF", "NA") &
                                     group %in% c("X1=0", "X1=1") &
                                     is.finite(simultaneous_coverage)]
    transitions <- c("1->2", "1->3", "2->1", "2->3")
    for (table in list(pointwise, simultaneous)) {
        table[, model := factor(model, levels = c("JF", "NA"))]
        table[, transition := factor(transition, levels = transitions)]
    }
    dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
    files <- character()
    for (arm_value in c("without_noise", "with_noise")) {
        for (group_value in c("X1=0", "X1=1")) {
            curves <- data.table::copy(pointwise[arm == arm_value & group == group_value])
            if (!nrow(curves)) next
            sizes <- sort(unique(curves$n))
            curves[, sample_size := factor(paste0("n = ", n),
                                           levels = paste0("n = ", sizes))]
            data.table::setorderv(curves, c("n", "transition", "model", "time"))
            # Show a few shapes on each curve for black/white readability.
            # Marking every reporting time would hide the line and its changes.
            markers <- curves[, .SD[unique(round(seq(1, .N, length.out = min(7, .N))))],
                                by = c("n", "transition", "model")]
            plot <- ggplot2::ggplot(curves, ggplot2::aes(
                x = time, y = pointwise_coverage, colour = model,
                linetype = model, shape = model, group = model)) +
                ggplot2::geom_hline(yintercept = level, colour = "#777777",
                                     linetype = "dotdash", linewidth = 0.45) +
                ggplot2::geom_line(linewidth = 0.65) +
                ggplot2::geom_point(data = markers, size = 1.8, stroke = 0.4) +
                ggplot2::facet_grid(rows = ggplot2::vars(sample_size),
                                    cols = ggplot2::vars(transition), scales = "free_x") +
                ggplot2::scale_x_continuous(breaks = function(x) pretty(x, n = 3)) +
                .coverage_plot_scales() +
                ggplot2::labs(x = "Time", y = "Pointwise coverage") +
                .coverage_plot_theme()
            name <- paste0("pointwise_coverage_", arm_value, "_", gsub("=", "_", group_value))
            file <- file.path(output_path, paste0(name, ".pdf"))
            files[name] <- .coverage_save_pdf(plot, file, width = 10.5, height = 8)
        }
        curves <- data.table::copy(simultaneous[arm == arm_value])
        if (!nrow(curves)) next
        curves[, covariate_group := factor(group, levels = c("X1=0", "X1=1"),
                                          labels = c("X1 = 0", "X1 = 1"))]
        data.table::setorderv(curves, c("group", "transition", "model", "n"))
        sizes <- sort(unique(curves$n))
        plot <- ggplot2::ggplot(curves, ggplot2::aes(
            x = n, y = simultaneous_coverage, colour = model,
            linetype = model, shape = model, group = model)) +
            ggplot2::geom_hline(yintercept = level, colour = "#777777",
                                 linetype = "dotdash", linewidth = 0.45) +
            ggplot2::geom_line(linewidth = 0.7) +
            ggplot2::geom_point(size = 2.2, stroke = 0.4) +
            ggplot2::facet_grid(rows = ggplot2::vars(covariate_group),
                                cols = ggplot2::vars(transition)) +
            ggplot2::scale_x_continuous(breaks = sizes,
                                        labels = format(sizes, big.mark = ",", trim = TRUE,
                                                        scientific = FALSE),
                                        expand = ggplot2::expansion(mult = c(0.12, 0.12))) +
            .coverage_plot_scales() +
            ggplot2::labs(x = "Sample size", y = "Simultaneous coverage") +
            .coverage_plot_theme() +
            ggplot2::theme(axis.text.x = ggplot2::element_text(
                size = 9, angle = 45, hjust = 1, vjust = 1))
        name <- paste0("simultaneous_coverage_", arm_value)
        file <- file.path(output_path, paste0(name, ".pdf"))
        files[name] <- .coverage_save_pdf(plot, file, width = 10.5, height = 6)
    }
    if (!length(files)) stop("No coverage results for the two X1 groups were available")
    export_coverage_tables_for_git(path)
    invisible(files)
}

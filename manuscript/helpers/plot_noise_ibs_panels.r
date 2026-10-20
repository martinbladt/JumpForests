# Load after the noise-study plotting functions. This uses their returned
# plots without changing either helper or their standalone output files.
# Example:
# plot_noise_ibs_panels(
#     ibs_errors, noise_counts = seq.int(0L, 50L, by = 5L),
#     path = "results/proportional_hazards/Plots/IBS_noise_panels.pdf"
# )
plot_noise_ibs_panels <- function(
        ibs_errors, path = NULL, noise_counts = NULL) {
    if (!is.null(path) &&
        (!is.character(path) || length(path) != 1L || is.na(path) ||
         !grepl("[.]pdf$", path, ignore.case = TRUE))) {
        stop("path must be NULL or one PDF file path")
    }

    relative <- plot_noise_ibs_relative_to_JF(
        ibs_errors, noise_counts = noise_counts
    )
    absolute <- plot_noise_ibs(ibs_errors, noise_counts = noise_counts)
    model_labels <- unname(noise_study_model_labels(ibs_errors))
    colours <- article_model_colours[model_labels]

    # The common legend replaces the relative plot's direct endpoint labels.
    relative$layers <- Filter(function(layer) {
        !inherits(layer$geom, "GeomText") &&
            !inherits(layer$geom, "GeomSegment")
    }, relative$layers)
    selected_noise <- sort(unique(relative$data$noise))
    if (length(selected_noise) == 1L) {
        absolute$layers <- Filter(function(layer) {
            !inherits(layer$geom, "GeomLine")
        }, absolute$layers)
    }
    common_theme <- ggplot2::theme_minimal(base_size = 12) +
        ggplot2::theme(
            legend.position = "none",
            axis.text.x = ggplot2::element_text(size = 11),
            axis.text.y = ggplot2::element_text(size = 10),
            panel.grid.minor = ggplot2::element_blank(),
            panel.grid.major.x = ggplot2::element_blank(),
            panel.grid.major.y = ggplot2::element_line(
                colour = "#E8E8E8", linewidth = 0.35
            ),
            axis.title.x = ggplot2::element_text(margin = ggplot2::margin(t = 10)),
            axis.title.y = ggplot2::element_text(margin = ggplot2::margin(r = 8)),
            plot.margin = ggplot2::margin(8, 12, 8, 8)
        )
    style_panel <- function(plot) {
        # Replacing scales is intentional: Cox has the same grey in both panels.
        suppressMessages(plot +
            ggplot2::scale_colour_manual(values = colours) +
            ggplot2::scale_x_continuous(
                breaks = selected_noise,
                expand = ggplot2::expansion(mult = 0.04, add = 0.5)
            ) +
            ggplot2::labs(title = NULL, subtitle = NULL, caption = NULL) +
            common_theme)
    }
    absolute <- style_panel(absolute)
    relative <- style_panel(relative)

    # Keep symbols in the common legend so it also works in greyscale.
    legend_data <- data.frame(
        model = factor(rep(model_labels, each = 2L), levels = model_labels),
        x = rep(c(0, 1), length(model_labels)), y = 0
    )
    legend_plot <- ggplot2::ggplot(
        legend_data, ggplot2::aes(x, y, colour = model,
                                  shape = model, linetype = model)
    ) +
        ggplot2::geom_line(linewidth = 0.8) +
        ggplot2::geom_point(size = 2.5) +
        ggplot2::scale_colour_manual(values = colours, name = NULL) +
        ggplot2::scale_shape_manual(values = article_model_shapes, name = NULL) +
        ggplot2::scale_linetype_manual(values = article_model_linetypes, name = NULL) +
        ggplot2::guides(colour = ggplot2::guide_legend(nrow = 1L)) +
        ggplot2::theme_minimal(base_size = 12) +
        ggplot2::theme(legend.position = "bottom")
    # Building grobs needs a device; avoid creating Rplots.pdf in batch use.
    if (grDevices::dev.cur() == 1L) {
        grDevices::pdf(NULL)
        on.exit(grDevices::dev.off(), add = TRUE)
    }
    legend_table <- ggplot2::ggplotGrob(legend_plot)
    legend_indices <- which(grepl("^guide-box", legend_table$layout$name))
    legend_indices <- legend_indices[vapply(
        legend_table$grobs[legend_indices], inherits, logical(1), "gtable"
    )]
    if (length(legend_indices) != 1L) stop("could not extract the common legend")
    legend <- legend_table$grobs[[legend_indices]]

    panels <- lapply(list(absolute, relative), ggplot2::ggplotGrob)
    # Align the plotting regions, including their top and bottom axes.
    common_heights <- grid::unit.pmax(panels[[1L]]$heights, panels[[2L]]$heights)
    common_widths <- grid::unit.pmax(panels[[1L]]$widths, panels[[2L]]$widths)
    for (i in seq_along(panels)) {
        panels[[i]]$heights <- common_heights
        panels[[i]]$widths <- common_widths
    }
    combined <- gtable::gtable(
        widths = grid::unit(c(1, 1), "null"),
        heights = grid::unit.c(
            grid::unit(1, "null"), sum(legend$heights)
        )
    )
    combined <- gtable::gtable_add_grob(
        combined, panels, t = 1L, l = c(1L, 2L),
        name = c("ibs_panel", "relative_panel")
    )
    combined <- gtable::gtable_add_grob(
        combined, legend, t = 2L, l = 1L, r = 2L, name = "shared_legend"
    )
    if (!is.null(path)) {
        if (!dir.exists(dirname(path)) && !dir.create(dirname(path), recursive = TRUE)) {
            stop("could not create plot directory: ", dirname(path))
        }
        ggplot2::ggsave(path, combined, width = 12, height = 4.8,
                        units = "in", bg = "white")
    }
    combined
}

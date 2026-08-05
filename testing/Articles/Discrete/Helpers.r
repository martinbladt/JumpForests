#nolint start: line_length_linter

devtools::load_all()
library(AalenJohansen)
library(survival)
library(eha)
#remotes::install_github("martinbladt/JumpPoisReg")
library(JumpPoisReg)
library(tictoc)
library(ggplot2)
library(dplyr)

# Helper functions for computing relevant quantities for numerical studies
#--------------------------------------------------------------------------------

# computes a list of the smooth product integral 
# (A: intensity matrix, s: starting time, t: end time, n: number of steps (h = (t - s)/n))
prodint <- function (A, s, t, n){
  x0 <- s
  y0 <- diag(nrow(A(s)))
  res <- list(y0)
  h <- (t - s)/n
  for (i in 1:n) {
    s1 <- h * y0 %*% A(x0)
    s2 <- h * (y0 + s1 / 2) %*% A (x0 + h / 2)
    s3 <- h * (y0 + s2 / 2) %*% A (x0 + h / 2)
    s4 <- h * (y0 + s3) %*% A(x0 + h)
    y0 <- y0 + s1/6 + s2/3 + s3 /3 + s4/6
    x0 <- x0 + h
    res[[i + 1]] <- y0
  }
  return(res)
}

# computes a list of the product integral from a provided Nelson-Aalen estimator
# (na: Nelson-Aalen estimator provided as a list)
prodint_from_data <- function(na) {
  y0 <- diag(nrow(na[[1]]))  # initialise with identity matrix
  res <- list(y0)
  for (i in 2:length(na)) {
    y0 <- y0 + y0 %*% (na[[i]] - na[[i - 1]])
    res[[i]] <- y0
  }
  return(res)
}

# computes a list of occupation probabilities from 0 to t
# (A: intensity matrix, t: end time, init: vector of initial probabilities, n: number of steps (h = t/n))
occprob <- function(A, t, init, n) {
  # fist compute the product integral
  res <- prodint(A, 0, t, n)
  # now multiply by initial probabilities
  return(lapply(res, function(z) init %*% z))
}
# computes a list of occupation probabilities from 0 to t based on a Nelson-Aalen estimator (a list)
# (na: a Nelson-Aalen estimator as a list, init: vector of initial probabilities)
occprob_from_data <- function(na, init) {
  # first compute the product integral
  res <- prodint_from_data(na)
  # now multiply by initial probabilities
  return(lapply(res, function(z) init %*% z))
}

# computes the matrix of transition probabilities in a semi-Markov illness-death model without reactivation
# (A: intensity matrix (a function of both time and duration), s: starting time, t: end time,
# u: initial duration, z: upper bound on the final duration, n: maximum number of
# subdivisions used by the numerical integrations)
transition_probs_semiMarkov <- function(A, s, t, u, z, n = 100L) {
    A_su <- A(s, u)

    integrate_scalar <- function(f, lower, upper) {
      if (lower >= upper) {
        return(0)
      }
      stats::integrate(
        function(x) vapply(x, f, numeric(1)),
        lower = lower,
        upper = upper,
        subdivisions = n,
        stop.on.error = TRUE
      )$value
    }

    mu01 <- function(time, duration) A(time, duration)[1L, 2L]
    mu02 <- function(time, duration) A(time, duration)[1L, 3L]
    mu12 <- function(time, duration) A(time, duration)[2L, 3L]

    # Survival in state 0 from s to end. The duration argument is included even
    # though mu01 and mu02 do not depend on it in the model considered here.
    S0 <- function(end) {
      exp(-integrate_scalar(function(time) {
          duration <- u + time - s
          mu01(time, duration) + mu02(time, duration)
        },
        s,
        end
      ))
    }

    # Survival in state 1, either from the initial state (entry = s and
    # initial_duration = u) or following a 0 -> 1 transition (initial_duration = 0).
    S1 <- function(entry, end, initial_duration = 0) {
      exp(-integrate_scalar(
        function(time) mu12(time, initial_duration + time - entry),
        entry,
        end
      ))
    }

    # Probability of being in state 1 at `end`, having started in state 0 at s.
    # If min_entry is supplied, only paths whose 0 -> 1 transition occurred at
    # or after that time are counted.
    P01 <- function(end, min_entry = s) {
      lower <- max(s, min_entry)
      integrate_scalar(
        function(entry) {
          S0(entry) * mu01(entry, u + entry - s) * S1(entry, end)
        },
        lower,
        end
      )
    }

    cutoff <- max(s, t - z)
    p00_t <- S0(t)
    p01_t <- P01(t)
    p11_t <- S1(s, t, u)

    # Death has final duration <= z exactly when it occurs in [cutoff, t].
    # Since death is absorbing, this probability equals the probability of being
    # alive at cutoff minus the probability of being alive at t.
    p02 <- S0(cutoff) + P01(cutoff) - p00_t - p01_t
    p12 <- S1(s, cutoff, u) - p11_t

    result <- matrix(0, nrow = 3L, ncol = 3L)
    result[1L, 1L] <- if (u + t - s <= z) p00_t else 0
    result[1L, 2L] <- P01(t, cutoff)
    result[1L, 3L] <- p02
    result[2L, 2L] <- if (u + t - s <= z) p11_t else 0
    result[2L, 3L] <- p12
    result[3L, 3L] <- as.numeric(u + t - s <= z)

    # Remove negligible negative values caused by subtracting nearly equal
    # probabilities in the death column.
    result[result < 0 & result > -sqrt(.Machine$double.eps)] <- 0
    result
}

# computes the vector of optimal theoretical expected Brier scores
# (occ_prob: list of occupation probabilities, c: weights)
optimal_brier <- function(c, occ_prob) {
    return(lapply(occ_prob, function(z) (z * (1 - z)) %*% c))
}

# computes the vector of optimal theoretical expected Kullback-Leibler scores
# (occ_prob: list of occupation probabilities, c: weights)
optimal_KL <- function(c, occ_prob) {
    # Kullback-Leibler error is only proper for uniform weights, otherwise the optimum has to be found
    summands <- lapply(occ_prob, function(z) c * z * log(c * z / as.numeric(z %*% c)))
    return(lapply(summands, function(z) -sum(z)))
}

# computes the vector of optimal theoretical expected spherical scores
# (occ_prob: list of occupation probabilities, c: weights)
optimal_spherical <- function(c, occ_prob) {
    return(lapply(occ_prob, function(z) 1 - sqrt((z * z) %*% c)))
}

# Helper functions for plotting
#--------------------------------------------------------------------------------

.open_plot_device <- function(file, width, height, resolution) {
    if (is.null(file)) {
        return(FALSE)
    }
    if (!is.character(file) || length(file) != 1L || is.na(file) || !nzchar(file)) {
        stop("file must be NULL or a non-empty file path")
    }
    if (!is.numeric(width) || length(width) != 1L || !is.finite(width) || width <= 0 ||
        !is.numeric(height) || length(height) != 1L || !is.finite(height) || height <= 0 ||
        !is.numeric(resolution) || length(resolution) != 1L ||
        !is.finite(resolution) || resolution <= 0) {
        stop("width, height and resolution must be positive finite numbers")
    }

    directory <- dirname(file)
    if (!dir.exists(directory) && !dir.create(directory, recursive = TRUE)) {
        stop("could not create plot directory: ", directory)
    }

    extension <- tolower(tools::file_ext(file))
    switch(
        extension,
        png = grDevices::png(file, width = width, height = height,
                             units = "in", res = resolution),
        jpg =,
        jpeg = grDevices::jpeg(file, width = width, height = height,
                               units = "in", res = resolution),
        tif =,
        tiff = grDevices::tiff(file, width = width, height = height,
                               units = "in", res = resolution),
        pdf = grDevices::pdf(file, width = width, height = height),
        svg = grDevices::svg(file, width = width, height = height),
        stop("unsupported plot extension: .", extension)
    )
    TRUE
}

# Plot one score curve per row of a score matrix. The columns correspond to x.
plot_score_curves <- function(scores, x, series_labels = rownames(scores), x_tick_labels = x, colours = NULL, pch = NULL,
                              xlab = "Minimum node size", ylab = NULL, legend_position = "topright", file = NULL,
                              width = 8, height = 5, resolution = 300, ...) {
    scores <- as.matrix(scores)
    storage.mode(scores) <- "double"
    if (!length(scores) || nrow(scores) < 1L || ncol(scores) < 1L) {
        stop("scores must be a non-empty matrix")
    }
    if (!is.numeric(x) || length(x) != ncol(scores) || any(!is.finite(x))) {
        stop("x must contain one finite numeric value per score column")
    }
    if (length(x_tick_labels) != length(x)) {
        stop("x_tick_labels must have the same length as x")
    }

    number_of_series <- nrow(scores)
    if (is.null(series_labels)) {
        series_labels <- paste("Series", seq_len(number_of_series))
    }
    if (length(series_labels) != number_of_series) {
        stop("series_labels must contain one label per score row")
    }
    if (is.null(colours)) {
        colours <- grDevices::hcl.colors(number_of_series, palette = "Dark 3")
    }
    if (length(colours) != number_of_series) {
        stop("colours must contain one colour per score row")
    }
    if (is.null(pch)) {
        pch <- seq_len(number_of_series)
    }
    if (length(pch) != number_of_series) {
        stop("pch must contain one plotting symbol per score row")
    }

    device_opened <- .open_plot_device(file, width, height, resolution)
    if (device_opened) {
        on.exit(grDevices::dev.off(), add = TRUE)
    }

    graphics::matplot(
        x, t(scores), type = "o", col = colours, lty = 1, lwd = 2,
        pch = pch, xaxt = "n", xlab = xlab, ylab = ylab, ...
    )
    graphics::axis(1, at = x, labels = x_tick_labels, gap.axis = -1)
    if (!is.null(legend_position)) {
        graphics::legend(
            legend_position, legend = series_labels, col = colours,
            lty = 1, lwd = 2, pch = pch, bty = "n"
        )
    }

    invisible(file)
}

.normalise_panel_limits <- function(limits, number_of_panels, name) {
    if (is.null(limits)) {
        return(NULL)
    }
    if (is.numeric(limits)) {
        limits <- rep(list(limits), number_of_panels)
    }
    if (!is.list(limits) || length(limits) != number_of_panels ||
        any(vapply(limits, function(x) {
            !is.numeric(x) || length(x) != 2L || any(!is.finite(x)) || x[1L] >= x[2L]
        }, logical(1)))) {
        stop(name, " must be a finite increasing pair or one such pair per panel")
    }
    limits
}

.expand_constant_range <- function(x) {
    if (x[1L] < x[2L]) {
        return(x)
    }
    padding <- if (x[1L] == 0) 0.5 else abs(x[1L]) * 0.04
    x + c(-padding, padding)
}

# Plot several methods on one or more panels. Each entry of curve_sets is a list with `times`
# (one common vector or a vector per panel), `values` (a list of time-by-component matrices), 
# and optional `type`, `lty`, `lwd`, and `label`.
plot_panel_curves <- function(curve_sets, component_labels, panel_titles = NULL, component_colours = NULL, xlab = "Time", ylab = NULL,
                              xlim = NULL, ylim = NULL, include_zero = FALSE, panel_layout = NULL, mar = c(4, 4, 3, 1),
                              legend_panel = 1L, legend_position = "topright", legend_order = names(curve_sets), legend_cex = 1,
                              file = NULL, width = 11, height = 6.5, resolution = 300) {
    if (!is.list(curve_sets) || !length(curve_sets)) {
        stop("curve_sets must be a non-empty named list")
    }
    if (is.null(names(curve_sets)) || any(!nzchar(names(curve_sets))) ||
        anyDuplicated(names(curve_sets))) {
        stop("curve_sets must have unique non-empty names")
    }

    normalised_sets <- vector("list", length(curve_sets))
    names(normalised_sets) <- names(curve_sets)
    number_of_panels <- NULL
    number_of_components <- NULL

    for (method_index in seq_along(curve_sets)) {
        method <- curve_sets[[method_index]]
        if (!is.list(method) || is.null(method$times) || is.null(method$values)) {
            stop("each curve set must contain times and values")
        }

        values <- method$values
        if (is.matrix(values) || is.data.frame(values)) {
            values <- list(values)
        }
        if (!is.list(values) || !length(values)) {
            stop("each values entry must be a non-empty list of matrices")
        }
        values <- lapply(values, function(value) {
            value <- as.matrix(value)
            storage.mode(value) <- "double"
            if (!nrow(value) || !ncol(value)) {
                stop("curve value matrices must be non-empty")
            }
            value
        })

        if (is.null(number_of_panels)) {
            number_of_panels <- length(values)
            number_of_components <- ncol(values[[1L]])
        }
        if (length(values) != number_of_panels ||
            any(vapply(values, ncol, integer(1)) != number_of_components)) {
            stop("all methods must provide the same panels and components")
        }

        times <- method$times
        if (is.numeric(times)) {
            times <- rep(list(times), number_of_panels)
        }
        if (!is.list(times) || length(times) != number_of_panels) {
            stop("times must be one numeric vector or one vector per panel")
        }
        for (panel_index in seq_len(number_of_panels)) {
            panel_times <- times[[panel_index]]
            if (!is.numeric(panel_times) || length(panel_times) != nrow(values[[panel_index]]) ||
                any(!is.finite(panel_times)) || is.unsorted(panel_times)) {
                stop("each time vector must be finite, sorted, and match its value matrix")
            }
        }

        type <- if (is.null(method$type)) "l" else method$type
        lty <- if (is.null(method$lty)) method_index else method$lty
        lwd <- if (is.null(method$lwd)) 2 else method$lwd
        label <- if (is.null(method$label)) names(curve_sets)[method_index] else method$label
        if (!is.character(type) || length(type) != 1L ||
            !(is.numeric(lty) || is.character(lty)) || length(lty) != 1L ||
            !is.numeric(lwd) || length(lwd) != 1L || lwd <= 0 ||
            !is.character(label) || length(label) != 1L) {
            stop("type, lty, lwd and label must be scalar plotting specifications")
        }

        normalised_sets[[method_index]] <- list(
            times = times, values = values, type = type,
            lty = lty, lwd = lwd, label = label
        )
    }

    if (length(component_labels) != number_of_components) {
        stop("component_labels must contain one label per matrix column")
    }
    if (is.null(component_colours)) {
        component_colours <- grDevices::hcl.colors(number_of_components, palette = "Dark 3")
    }
    if (length(component_colours) != number_of_components) {
        stop("component_colours must contain one colour per matrix column")
    }
    if (is.null(panel_titles)) {
        panel_titles <- rep("", number_of_panels)
    }
    if (length(panel_titles) != number_of_panels) {
        stop("panel_titles must contain one title per panel")
    }
    if (is.null(panel_layout)) {
        panel_layout <- grDevices::n2mfrow(number_of_panels)
    }
    if (!is.numeric(panel_layout) || length(panel_layout) != 2L ||
        any(panel_layout < 1) || prod(panel_layout) < number_of_panels) {
        stop("panel_layout must be a positive rows-by-columns pair with enough panels")
    }
    if (!is.null(legend_panel) &&
        (!is.numeric(legend_panel) || length(legend_panel) != 1L ||
         legend_panel < 1L || legend_panel > number_of_panels)) {
        stop("legend_panel must be NULL or the index of a panel")
    }
    if (!all(legend_order %in% names(normalised_sets)) || anyDuplicated(legend_order)) {
        stop("legend_order must contain unique curve-set names")
    }

    x_limits <- .normalise_panel_limits(xlim, number_of_panels, "xlim")
    y_limits <- .normalise_panel_limits(ylim, number_of_panels, "ylim")
    if (is.null(x_limits)) {
        x_limits <- lapply(seq_len(number_of_panels), function(panel_index) {
            .expand_constant_range(range(unlist(lapply(normalised_sets, function(method) {
                method$times[[panel_index]]
            })), finite = TRUE))
        })
    }
    if (is.null(y_limits)) {
        y_limits <- lapply(seq_len(number_of_panels), function(panel_index) {
            values <- unlist(lapply(normalised_sets, function(method) {
                method$values[[panel_index]]
            }))
            if (include_zero) {
                values <- c(0, values)
            }
            limits <- range(values, finite = TRUE)
            if (any(!is.finite(limits))) {
                stop("each panel must contain at least one finite value")
            }
            .expand_constant_range(limits)
        })
    }

    device_opened <- .open_plot_device(file, width, height, resolution)
    old_par <- NULL
    on.exit({
        if (!is.null(old_par)) {
            graphics::par(old_par)
        }
        if (device_opened) {
            grDevices::dev.off()
        }
    }, add = TRUE)
    old_par <- graphics::par(mfrow = panel_layout, mar = mar)

    for (panel_index in seq_len(number_of_panels)) {
        graphics::plot(
            NA_real_, NA_real_, type = "n", xlim = x_limits[[panel_index]],
            ylim = y_limits[[panel_index]], xlab = xlab, ylab = ylab,
            main = panel_titles[panel_index]
        )
        for (method in normalised_sets) {
            graphics::matlines(
                method$times[[panel_index]], method$values[[panel_index]],
                type = method$type, col = component_colours,
                lty = method$lty, lwd = method$lwd
            )
        }

        if (!is.null(legend_panel) && panel_index == legend_panel &&
            !is.null(legend_position)) {
            legend_methods <- normalised_sets[legend_order]
            graphics::legend(
                legend_position,
                legend = c(component_labels, vapply(legend_methods, `[[`, "", "label")),
                col = c(component_colours, rep("black", length(legend_methods))),
                lty = c(
                    rep(normalised_sets[[1L]]$lty, number_of_components),
                    unlist(lapply(legend_methods, `[[`, "lty"), use.names = FALSE)
                ),
                lwd = c(
                    rep(normalised_sets[[1L]]$lwd, number_of_components),
                    vapply(legend_methods, `[[`, numeric(1), "lwd")
                ),
                bty = "n", cex = legend_cex
            )
        }
    }

    invisible(file)
}

# Extract selected row-column entries from every matrix in a list.
extract_matrix_entries <- function(matrices, indices, labels = NULL) {
    if (!is.list(matrices)) {
        stop("matrices must be a list")
    }
    indices <- as.matrix(indices)
    if (!is.numeric(indices) || ncol(indices) != 2L || nrow(indices) < 1L ||
        any(!is.finite(indices)) || any(indices < 1L) || any(indices %% 1 != 0)) {
        stop("indices must be a non-empty two-column matrix of positive integers")
    }
    storage.mode(indices) <- "integer"
    if (!is.null(labels) && length(labels) != nrow(indices)) {
        stop("labels must contain one label per selected entry")
    }

    if (!length(matrices)) {
        result <- matrix(numeric(0), nrow = 0L, ncol = nrow(indices))
    } else {
        result <- t(vapply(matrices, function(value) {
            value <- as.matrix(value)
            if (any(indices[, 1L] > nrow(value)) || any(indices[, 2L] > ncol(value))) {
                stop("indices fall outside at least one matrix")
            }
            value[indices]
        }, numeric(nrow(indices))))
    }
    colnames(result) <- labels
    result
}


# Testing the helper functions
#--------------------------------------------------------------------------------

# Keep the numerical examples local so sourcing this file cannot overwrite a
# study's intensity functions.
test_helper_functions <- function() {
    mu01 <- function(x) {
        if (x <= 65) (0.0004 + 10^(4.54 + 0.06 * x - 10)) else 0
    }
    mu10 <- function(x) {
        if (x <= 65) (2.0058 * exp(-0.117 * x)) else 0
    }
    mu02 <- function(x) {
        0.0005 + 10^(5.88 + 0.038 * x - 10)
    }
    mu12 <- function(x) {
        if (x <= 65) 2 * mu02(x) else mu02(x)
    }
    intensity_matrix <- function(t) {
        matrix(c(-mu01(t) - mu02(t), mu01(t), mu02(t),
                 mu10(t), -mu10(t) - mu12(t), mu12(t),
                 0, 0, 0), nrow = 3, byrow = TRUE)
    }

    probabilities <- occprob(intensity_matrix, 100, c(1, 0, 0), 10^4)
    weights <- rep(1 / 3, 3)
    uniform <- list(rep(1 / 3, 3))
    list(
        brier = optimal_brier(weights, probabilities),
        brier_uniform = optimal_brier(weights, uniform),
        spherical = optimal_spherical(weights, probabilities),
        spherical_uniform = optimal_spherical(weights, uniform),
        KL = optimal_KL(weights, probabilities),
        KL_uniform = optimal_KL(weights, uniform)
    )
}

#nolint end

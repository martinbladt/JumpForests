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

# Poisson regression helpers
#--------------------------------------------------------------------------------

# Convert continuously observed multi-state paths to the grouped sufficient
# statistics for a piecewise-exponential Poisson likelihood. On a grid interval
# (a, b], a path contributes its exact overlap with (a, b] as exposure in the
# state occupied at the start of the path segment. A jump contributes one event
# to the interval containing its endpoint; a final same-state segment is
# censoring and therefore contributes exposure but no event.
#
# Aggregating by transition, interval and covariate profile is likelihood-exact:
# all rows in such a cell have the same Poisson mean per unit exposure. This is
# much smaller than retaining one row per individual/interval, especially for
# the discrete covariates used in the simulation study.
build_markov_poisson_data <- function(paths, features, time_grid, transitions) {
    features <- as.data.frame(features)
    if (!is.list(paths) || !length(paths) || nrow(features) != length(paths)) {
        stop("paths and features must describe the same positive number of individuals")
    }
    if (is.null(names(features)) || any(!nzchar(names(features))) ||
        anyDuplicated(names(features))) {
        stop("features must have unique, non-empty column names")
    }
    reserved_names <- c(
        "id", "tstart", "tstop", "from", "to", "interval",
        "interval_factor", "transition", "exposure", "count"
    )
    if (any(names(features) %in% reserved_names)) {
        stop("feature names conflict with columns used by the Poisson data builder")
    }
    if (anyNA(features)) {
        stop("features cannot contain missing values")
    }

    time_grid <- as.numeric(time_grid)
    if (length(time_grid) < 2L || any(!is.finite(time_grid)) ||
        is.unsorted(time_grid, strictly = TRUE)) {
        stop("time_grid must contain at least two finite, strictly increasing values")
    }

    transitions <- as.data.frame(transitions)
    if (ncol(transitions) != 2L || !nrow(transitions)) {
        stop("transitions must be a non-empty two-column object")
    }
    transition_values <- as.matrix(transitions)
    if (!is.numeric(transition_values) ||
        any(!is.finite(transition_values)) ||
        any(transition_values %% 1 != 0)) {
        stop("transition states must be finite integers")
    }
    transitions <- data.frame(
        from = as.integer(transition_values[, 1L]),
        to = as.integer(transition_values[, 2L])
    )
    if (any(transitions$from < 1L) || any(transitions$to < 1L) ||
        any(transitions$from == transitions$to) ||
        anyDuplicated(transitions)) {
        stop("transitions must contain unique, positive, off-diagonal state pairs")
    }
    transitions$transition <- paste(transitions$from, transitions$to, sep = "->")

    # First put each path into the familiar counting-process representation.
    # Keeping this construction here makes the helper independent of the Cox
    # comparison's local data-conversion function.
    interval_rows <- Map(function(path, id) {
        times <- path$times
        states <- path$states
        if (length(times) != length(states) || length(times) < 2L ||
            !is.numeric(times) || !is.numeric(states) ||
            any(!is.finite(times)) || any(!is.finite(states)) ||
            any(states %% 1 != 0) ||
            is.unsorted(times, strictly = TRUE) || any(states < 1L)) {
            stop("each path must contain valid, strictly increasing times and states")
        }
        times <- as.numeric(times)
        states <- as.integer(states)
        number_of_segments <- length(times) - 1L
        data.frame(
            id = rep.int(id, number_of_segments),
            tstart = head(times, -1L),
            tstop = tail(times, -1L),
            from = head(states, -1L),
            to = tail(states, -1L)
        )
    }, paths, seq_along(paths))
    intervals <- do.call(rbind, interval_rows)
    rownames(intervals) <- NULL

    # Restrict follow-up administratively to the fitted grid. A segment crossing
    # the final grid point contributes exposure up to that point, but its later
    # endpoint is not counted as an event.
    support_start <- time_grid[1L]
    support_end <- tail(time_grid, 1L)
    intervals$split_start <- pmax(intervals$tstart, support_start)
    intervals$split_stop <- pmin(intervals$tstop, support_end)
    intervals <- intervals[
        intervals$split_stop > intervals$split_start,
        , drop = FALSE
    ]
    if (!nrow(intervals)) {
        stop("no path exposure overlaps time_grid")
    }

    number_of_intervals <- length(time_grid) - 1L
    first_interval <- findInterval(intervals$split_start, time_grid)
    last_interval <- pmin.int(
        findInterval(intervals$split_stop, time_grid),
        number_of_intervals
    )
    pieces_per_segment <- last_interval - first_interval + 1L
    if (any(pieces_per_segment < 1L)) {
        stop("could not map at least one path segment to time_grid")
    }

    source_row <- rep.int(seq_len(nrow(intervals)), pieces_per_segment)
    interval <- sequence(pieces_per_segment) +
        rep.int(first_interval - 1L, pieces_per_segment)
    lower <- time_grid[interval]
    upper <- time_grid[interval + 1L]
    exposure <- pmin(intervals$split_stop[source_row], upper) -
        pmax(intervals$split_start[source_row], lower)

    # A stop exactly at a grid boundary belongs to the interval on its left,
    # matching the conventional (a, b] event-counting convention. Any zero-
    # overlap piece created at that boundary is removed below.
    event <- intervals$from[source_row] != intervals$to[source_row] &
        intervals$tstop[source_row] > support_start &
        intervals$tstop[source_row] <= support_end &
        intervals$tstop[source_row] > lower &
        intervals$tstop[source_row] <= upper

    positive_exposure <- exposure > 0
    pieces <- data.frame(
        from = intervals$from[source_row[positive_exposure]],
        to = intervals$to[source_row[positive_exposure]],
        interval = interval[positive_exposure],
        exposure = exposure[positive_exposure],
        event = event[positive_exposure]
    )
    pieces <- cbind(
        pieces,
        features[
            intervals$id[source_row[positive_exposure]],
            , drop = FALSE
        ]
    )
    rownames(pieces) <- NULL

    covariate_names <- names(features)
    exposure_groups <- c("from", "interval", covariate_names)
    exposure_table <- stats::aggregate(
        pieces["exposure"], by = pieces[exposure_groups], FUN = sum
    )

    # Every requested origin state needs positive risk time in every interval.
    # With no exposure a rate is unidentified; silently replacing it by zero
    # would be a statistical error, so ask the caller to merge bins or shorten
    # the fitted horizon instead.
    origin_exposure <- stats::aggregate(
        exposure_table["exposure"],
        by = exposure_table[c("from", "interval")],
        FUN = sum
    )
    required_origin_intervals <- expand.grid(
        from = unique(transitions$from),
        interval = seq_len(number_of_intervals)
    )
    exposure_check <- merge(
        required_origin_intervals, origin_exposure,
        by = c("from", "interval"), all.x = TRUE, sort = FALSE
    )
    if (anyNA(exposure_check$exposure) || any(exposure_check$exposure <= 0)) {
        missing_rows <- exposure_check[
            is.na(exposure_check$exposure) | exposure_check$exposure <= 0,
            c("from", "interval"), drop = FALSE
        ]
        stop(
            "zero origin-state exposure in: ",
            paste(
                paste0("state ", missing_rows$from, ", interval ",
                       missing_rows$interval),
                collapse = "; "
            ),
            ". Merge those bins or shorten time_grid."
        )
    }

    # Duplicate each origin-state exposure cell for all requested destinations.
    # Both cause-specific transition models use the same at-risk time.
    poisson_data <- do.call(rbind, lapply(seq_len(nrow(transitions)), function(i) {
        rows <- exposure_table[
            exposure_table$from == transitions$from[i],
            , drop = FALSE
        ]
        rows$to <- transitions$to[i]
        rows$transition <- transitions$transition[i]
        rows
    }))

    allowed_event <- pieces$event &
        paste(pieces$from, pieces$to, sep = "->") %in% transitions$transition
    if (any(allowed_event)) {
        event_pieces <- pieces[allowed_event, , drop = FALSE]
        event_groups <- c("from", "to", "interval", covariate_names)
        event_table <- stats::aggregate(
            list(count = rep.int(1L, nrow(event_pieces))),
            by = event_pieces[event_groups], FUN = sum
        )
        poisson_data <- merge(
            poisson_data, event_table,
            by = event_groups, all.x = TRUE, sort = FALSE
        )
        poisson_data$count[is.na(poisson_data$count)] <- 0
    } else {
        poisson_data$count <- 0
    }

    poisson_data$interval_factor <- factor(
        poisson_data$interval,
        levels = seq_len(number_of_intervals)
    )
    transition_order <- match(poisson_data$transition, transitions$transition)
    poisson_data <- poisson_data[
        order(transition_order, poisson_data$interval),
        c(
            "transition", "from", "to", "interval", "interval_factor",
            covariate_names, "count", "exposure"
        ),
        drop = FALSE
    ]
    rownames(poisson_data) <- NULL

    diagnostics <- stats::aggregate(
        poisson_data[c("count", "exposure")],
        by = poisson_data[c("transition", "from", "to", "interval")],
        FUN = sum
    )
    diagnostics$interval_start <- time_grid[diagnostics$interval]
    diagnostics$interval_end <- time_grid[diagnostics$interval + 1L]
    diagnostics$crude_rate <- diagnostics$count / diagnostics$exposure
    diagnostics <- diagnostics[
        order(match(diagnostics$transition, transitions$transition),
              diagnostics$interval),
        , drop = FALSE
    ]
    rownames(diagnostics) <- NULL

    structure(
        list(
            data = poisson_data,
            diagnostics = diagnostics,
            time_grid = time_grid,
            transitions = transitions,
            covariates = covariate_names
        ),
        class = "markov_poisson_data"
    )
}

# Fit one piecewise-exponential Poisson GLM per directed transition. The formula
# must contain interval_factor and offset(log(exposure)); all coefficients,
# including covariate effects, are therefore transition-specific.
#
# If an interval has positive exposure but no events for a transition, its
# saturated-baseline MLE is exactly zero (the log baseline tends to -Inf). Such
# an interval contains no information about the covariate coefficients after
# profiling its baseline, so it is left out of glm() and stored for prediction
# as an exact zero rate. This is distinct from a zero-exposure interval, which
# build_markov_poisson_data() rejects as unidentified.
fit_markov_poisson_regression <- function(poisson_data, formula) {
    if (!inherits(poisson_data, "markov_poisson_data")) {
        stop("poisson_data must come from build_markov_poisson_data()")
    }
    if (!inherits(formula, "formula") ||
        !identical(as.character(formula[[2L]]), "count")) {
        stop("formula must be a formula with count as its response")
    }
    formula_variables <- all.vars(formula)
    formula_terms <- stats::terms(formula, specials = "offset")
    if (!all(c("interval_factor", "exposure") %in% formula_variables) || !length(attr(formula_terms, "specials")$offset)) {
        stop("formula must contain interval_factor and an exposure offset")
    }

    models <- vector("list", nrow(poisson_data$transitions))
    zero_event_intervals <- vector("list", length(models))
    names(models) <- names(zero_event_intervals) <- poisson_data$transitions$transition
    fit_diagnostics <- vector("list", length(models))

    for (i in seq_along(models)) {
        transition_name <- names(models)[i]
        transition_data <- poisson_data$data[poisson_data$data$transition == transition_name, , drop = FALSE]
        interval_totals <- stats::aggregate(
            transition_data[c("count", "exposure")],
            by = transition_data["interval"], FUN = sum
        )
        interval_totals <- interval_totals[
            order(interval_totals$interval), , drop = FALSE
        ]
        if (!identical(interval_totals$interval, seq_len(length(poisson_data$time_grid) - 1L)) || any(interval_totals$exposure <= 0)) {
            stop("transition ", transition_name," does not have positive exposure in every interval")
        }

        zero_intervals <- interval_totals$interval[interval_totals$count == 0]
        positive_event_data <- transition_data[!transition_data$interval %in% zero_intervals, , drop = FALSE]
        if (!nrow(positive_event_data)) {
            stop("transition ", transition_name, " has no observed events")
        }
        positive_event_data$interval_factor <- droplevels(positive_event_data$interval_factor)

        model <- stats::glm(
            formula = formula,
            family = stats::poisson(link = "log"),
            data = positive_event_data,
            model = TRUE, x = TRUE, y = TRUE,
            singular.ok = FALSE,
            control = stats::glm.control(maxit = 100L)
        )
        if (!isTRUE(model$converged) || any(!is.finite(stats::coef(model)))) {
            stop("Poisson GLM for transition ", transition_name, " did not converge to finite coefficients")
        }

        models[[i]] <- model
        zero_event_intervals[[i]] <- zero_intervals
        fit_diagnostics[[i]] <- data.frame(
            transition = transition_name,
            events = sum(transition_data$count),
            exposure = sum(transition_data$exposure),
            zero_event_intervals = if (length(zero_intervals)) {
                paste(zero_intervals, collapse = ",")
            } else {
                "none"
            },
            coefficients = length(stats::coef(model)),
            converged = model$converged
        )
    }

    structure(
        list(
            models = models,
            zero_event_intervals = zero_event_intervals,
            formula = formula,
            time_grid = poisson_data$time_grid,
            transitions = poisson_data$transitions,
            diagnostics = do.call(rbind, fit_diagnostics)
        ),
        class = "markov_poisson_regression"
    )
}

# Assemble one row-generator matrix per time interval from a matrix whose rows
# are intervals and whose columns follow transition_indices. Diagonal entries
# are set to minus the sum of the off-diagonal rates, so row vectors evolve as
# p'(t) = p(t) Q(t).
generator_matrices_from_rates <- function(rates, transition_indices, number_of_states = NULL) {
    rates <- as.matrix(rates)
    storage.mode(rates) <- "double"
    transition_indices <- as.matrix(transition_indices)
    if (!nrow(rates) || !ncol(rates) || any(!is.finite(rates)) || any(rates < 0)) {
        stop("rates must be a non-empty matrix of finite non-negative values")
    }
    if (!is.numeric(transition_indices) || ncol(transition_indices) != 2L || nrow(transition_indices) != ncol(rates) ||
        any(!is.finite(transition_indices)) || any(transition_indices %% 1 != 0) || any(transition_indices < 1L) ||
        any(transition_indices[, 1L] == transition_indices[, 2L]) || anyDuplicated(as.data.frame(transition_indices))) {
        stop("transition_indices must give one unique off-diagonal pair per rate column")
    }
    storage.mode(transition_indices) <- "integer"
    minimum_number_of_states <- max(transition_indices)
    if (is.null(number_of_states)) {
        number_of_states <- minimum_number_of_states
    }
    if (!is.numeric(number_of_states) || length(number_of_states) != 1L || !is.finite(number_of_states) || number_of_states %% 1 != 0 ||
        number_of_states < minimum_number_of_states) {
        stop("number_of_states must contain every state in transition_indices")
    }
    number_of_states <- as.integer(number_of_states)

    lapply(seq_len(nrow(rates)), function(i) {
        generator <- matrix(0, nrow = number_of_states, ncol = number_of_states)
        generator[transition_indices] <- rates[i, ]
        diag(generator) <- -rowSums(generator)
        generator
    })
}

.validate_piecewise_generators <- function(rate_matrices, time_grid) {
    time_grid <- as.numeric(time_grid)
    if (length(time_grid) < 2L || any(!is.finite(time_grid)) ||
        is.unsorted(time_grid, strictly = TRUE)) {
        stop("time_grid must contain at least two finite, strictly increasing values")
    }
    if (!is.list(rate_matrices) ||
        length(rate_matrices) != length(time_grid) - 1L) {
        stop("rate_matrices must contain one generator per grid interval")
    }

    matrices <- lapply(rate_matrices, function(generator) {
        generator <- as.matrix(generator)
        storage.mode(generator) <- "double"
        if (!nrow(generator) || nrow(generator) != ncol(generator) ||
            any(!is.finite(generator))) {
            stop("every generator must be a finite, non-empty square matrix")
        }
        generator
    })
    dimensions <- vapply(matrices, nrow, integer(1))
    if (length(unique(dimensions)) != 1L) {
        stop("all generator matrices must have the same dimensions")
    }

    matrices <- lapply(matrices, function(generator) {
        scale <- max(1, max(abs(generator)))
        tolerance <- 1e-10 * scale
        off_diagonal <- generator
        diag(off_diagonal) <- 0
        if (any(off_diagonal < -tolerance) || any(diag(generator) > tolerance) || any(abs(rowSums(generator)) > tolerance)) {
            stop("each rate matrix must be a valid row-generator")
        }
        # Remove only roundoff-sized violations, then restore the diagonal so
        # the generator has exactly zero row sums before exponentiation.
        off_diagonal[off_diagonal < 0] <- 0
        diag(off_diagonal) <- -rowSums(off_diagonal)
        off_diagonal
    })

    list(matrices = matrices, time_grid = time_grid,
         number_of_states = dimensions[1L])
}

.markov_matrix_exponential <- function(generator, duration) {
    number_of_states <- nrow(generator)
    if (!is.numeric(duration) || length(duration) != 1L || !is.finite(duration) || duration < 0) {
        stop("duration must be one finite non-negative number")
    }
    if (duration == 0) {
        return(diag(number_of_states))
    }
    if (!requireNamespace("Matrix", quietly = TRUE)) {
        stop("the recommended Matrix package is required for matrix exponentials")
    }

    transition_matrix <- as.matrix(Matrix::expm(
        Matrix::Matrix(generator * duration, sparse = FALSE)
    ))
    if (any(!is.finite(transition_matrix))) {
        stop("matrix exponentiation produced non-finite transition probabilities")
    }
    tolerance <- 1e-9
    if (any(transition_matrix < -tolerance) ||
        any(abs(rowSums(transition_matrix) - 1) > tolerance)) {
        stop("matrix exponentiation produced a non-stochastic transition matrix")
    }
    transition_matrix[transition_matrix < 0] <- 0
    sweep(transition_matrix, 1L, rowSums(transition_matrix), "/")
}

.validate_piecewise_prediction_times <- function(times, time_grid) {
    times <- as.numeric(times)
    tolerance <- 100 * .Machine$double.eps *
        max(1, max(abs(time_grid)))
    if (!length(times) || any(!is.finite(times)) || is.unsorted(times) ||
        min(times) < time_grid[1L] - tolerance ||
        max(times) > tail(time_grid, 1L) + tolerance) {
        stop("times must be finite, sorted and inside the fitted time grid")
    }
    times[times < time_grid[1L]] <- time_grid[1L]
    times[times > tail(time_grid, 1L)] <- tail(time_grid, 1L)
    times
}

# Exact occupation probabilities under piecewise-constant generator matrices.
# The matrix exponential is the exact transition matrix within a grid interval;
# products are ordered chronologically for the row-vector convention. Prefix
# products avoid repeating completed intervals when many prediction times are
# requested on a dense grid.
piecewise_markov_occupation <- function(rate_matrices, time_grid, times, initial) {
    validated <- .validate_piecewise_generators(rate_matrices, time_grid)
    rate_matrices <- validated$matrices
    time_grid <- validated$time_grid
    times <- .validate_piecewise_prediction_times(times, time_grid)
    number_of_states <- validated$number_of_states

    initial <- as.numeric(initial)
    tolerance <- 1e-10
    if (length(initial) != number_of_states || any(!is.finite(initial)) || any(initial < -tolerance) || abs(sum(initial) - 1) > tolerance) {
        stop("initial must be a probability vector matching the generators")
    }
    initial[initial < 0] <- 0
    initial <- initial / sum(initial)

    number_of_intervals <- length(rate_matrices)
    prefix_products <- vector("list", number_of_intervals)
    prefix_products[[1L]] <- diag(number_of_states)
    if (number_of_intervals > 1L) {
        for (i in 2:number_of_intervals) {
            previous_transition <- .markov_matrix_exponential(
                rate_matrices[[i - 1L]], time_grid[i] - time_grid[i - 1L]
            )
            prefix_products[[i]] <-
                prefix_products[[i - 1L]] %*% previous_transition
        }
    }

    occupation <- t(vapply(times, function(time) {
        interval <- min(findInterval(time, time_grid), number_of_intervals)
        within_interval <- .markov_matrix_exponential(rate_matrices[[interval]], time - time_grid[interval])
        as.numeric(initial %*% prefix_products[[interval]] %*% within_interval)
    }, numeric(number_of_states)))

    if (any(occupation < -tolerance) ||
        any(abs(rowSums(occupation) - 1) > tolerance)) {
        stop("piecewise propagation produced invalid occupation probabilities")
    }
    occupation[occupation < 0] <- 0
    sweep(occupation, 1L, rowSums(occupation), "/")
}

# Integrate piecewise-constant generators exactly. The returned list has the
# same shape as one Jump Forest Nelson--Aalen prediction: one full cumulative
# generator matrix at every requested time. Its off-diagonals are the cumulative
# transition rates; diagonals are minus their row sums.
cumulative_piecewise_rates <- function(rate_matrices, time_grid, times) {
    validated <- .validate_piecewise_generators(rate_matrices, time_grid)
    rate_matrices <- validated$matrices
    time_grid <- validated$time_grid
    times <- .validate_piecewise_prediction_times(times, time_grid)
    number_of_states <- validated$number_of_states
    number_of_intervals <- length(rate_matrices)

    cumulative_at_start <- vector("list", number_of_intervals)
    cumulative_at_start[[1L]] <- matrix(
        0, nrow = number_of_states, ncol = number_of_states
    )
    if (number_of_intervals > 1L) {
        for (i in 2:number_of_intervals) {
            cumulative_at_start[[i]] <- cumulative_at_start[[i - 1L]] + rate_matrices[[i - 1L]] * (time_grid[i] - time_grid[i - 1L])
        }
    }

    lapply(times, function(time) {
        interval <- min(findInterval(time, time_grid), number_of_intervals)
        cumulative_at_start[[interval]] + rate_matrices[[interval]] * (time - time_grid[interval])
    })
}

# Predict transition-specific rates for new covariate profiles, assemble a
# coherent piecewise-constant generator, then compute exact occupation
# probabilities and cumulative rates on the requested grid.
predict_markov_poisson_regression <- function(object, new_data, times, initial = NULL) {
    if (!inherits(object, "markov_poisson_regression")) {
        stop("object must come from fit_markov_poisson_regression()")
    }
    new_data <- as.data.frame(new_data)
    if (!nrow(new_data) || anyNA(new_data)) {
        stop("new_data must contain at least one complete covariate profile")
    }
    times <- .validate_piecewise_prediction_times(times, object$time_grid)

    transition_indices <- as.matrix(
        object$transitions[c("from", "to")]
    )
    number_of_states <- max(transition_indices)
    if (is.null(initial)) {
        initial <- c(1, rep(0, number_of_states - 1L))
    }
    number_of_intervals <- length(object$time_grid) - 1L
    number_of_transitions <- nrow(object$transitions)

    interval_rates <- lapply(seq_len(nrow(new_data)), function(profile) {
        rates <- matrix(
            0, nrow = number_of_intervals, ncol = number_of_transitions,
            dimnames = list(NULL, object$transitions$transition)
        )
        for (j in seq_len(number_of_transitions)) {
            model <- object$models[[j]]
            zero_intervals <- object$zero_event_intervals[[j]]
            fitted_intervals <- base::setdiff(
                seq_len(number_of_intervals), zero_intervals
            )
            prediction_data <- new_data[
                rep.int(profile, length(fitted_intervals)),
                , drop = FALSE
            ]
            prediction_data$interval_factor <- factor(
                fitted_intervals,
                levels = levels(model$model$interval_factor)
            )
            prediction_data$exposure <- 1
            predicted_rates <- stats::predict(
                model, newdata = prediction_data, type = "response"
            )
            if (any(!is.finite(predicted_rates)) || any(predicted_rates < 0)) {
                stop("Poisson prediction produced invalid transition rates")
            }
            rates[fitted_intervals, j] <- predicted_rates
        }
        rates
    })

    generator_matrices <- lapply(interval_rates, function(rates) {
        generator_matrices_from_rates(
            rates, transition_indices, number_of_states
        )
    })
    occupation_probabilities <- lapply(generator_matrices, function(generators) {
        piecewise_markov_occupation(
            generators, object$time_grid, times, initial
        )
    })
    cumulative_transition_rates <- lapply(generator_matrices, function(generators) {
        extract_matrix_entries(
            cumulative_piecewise_rates(generators, object$time_grid, times),
            transition_indices, object$transitions$transition
        )
    })

    list(
        times = times,
        occupation_probabilities = occupation_probabilities,
        cumulative_transition_rates = cumulative_transition_rates,
        interval_rates = interval_rates,
        generator_matrices = generator_matrices
    )
}

# Miscellaneous helper functions
#--------------------------------------------------------------------------------

# counts the number of transitions in a path from state 'from' to state 'to' in time_interval
count_transition_jumps <- function(path, from, to, time_interval = c(-Inf, Inf)) {
    if (!is.numeric(time_interval) ||
        length(time_interval) != 2L ||
        anyNA(time_interval) ||
        time_interval[1L] >= time_interval[2L]) {
        stop("time_interval must be an increasing numeric pair")
    }

    states <- path$states
    times <- path$times

    if (length(states) < 2L || length(states) != length(times)) {
        return(0L)
    }

    from_states <- head(states, -1L)
    to_states <- tail(states, -1L)
    jump_times <- tail(times, -1L)

    sum(
        from_states == from &
        to_states == to &
        jump_times > time_interval[1L] &
        jump_times <= time_interval[2L]
    )
}

# computes the cumulative transition rate for a piecewise constant estimator function
# result is a vector of the cumulative transition rate at times (of size length(times))
# (times: vector of times, oe: a vector of the oe rates at times)
cumulative_transititon_rate_PR <- function(times, oe) {
  n <- length(times)
  if (n != length(oe)) {
    stop("Number of times must be the same as the number of OE rates in oe")
  }

  cum_trans_rate <- rep(times[1] * oe[1], n)
  for (i in 2:n) {
    cum_trans_rate[i] <- cum_trans_rate[i - 1] + (times[i] - times[i - 1]) * oe[i]
  }
  cum_trans_rate
}

# computes the cumulative transition rate for a piecewise constant estimator function
# result is a list of the cumulative transition rate matrices at times (of size length(times))
# (object: a fit object from JumpPoisReg, times: a vector of event times for the returned estimator)
cumulative_markov_poisson <- function(object, times) {
    if (!inherits(object, "jump_pois_markov_fit")) {
        stop("object must come from fit_markov_poisson()")
    }

    times <- as.numeric(times)
    time_grid <- object$t_grid

    if (!length(times) || any(!is.finite(times)) || is.unsorted(times)) {
        stop("times must be a non-empty, sorted, finite vector")
    }
    if (!isTRUE(all.equal(times[1L], time_grid[1L]))) {
        stop("times must start at the first fitted grid point")
    }
    if (min(times) < time_grid[1L] ||
        max(times) > tail(time_grid, 1L)) {
        stop("times must remain inside the fitted Poisson grid")
    }

    transitions <- object$transitions[
        order(object$transitions$from, object$transitions$to),
        ,
        drop = FALSE
    ]

    lower <- head(time_grid, -1L)
    upper <- tail(time_grid, -1L)
    number_of_intervals <- length(lower)

    cumulative_rates <- vapply(
        seq_len(nrow(transitions)),
        function(j) {
            transition <- transitions$transition[j]

            rate_table <- object$rates[
                object$rates$transition == transition,
                ,
                drop = FALSE
            ]
            rate_table <- rate_table[order(rate_table$interval), ]

            if (!identical(
                as.integer(rate_table$interval),
                seq_len(number_of_intervals)
            )) {
                stop("The fitted rate table does not match the time grid")
            }

            rates <- rate_table$rate
            if (any(!is.finite(rates)) || any(rates < 0)) {
                stop(
                    "Transition ", transition,
                    " has missing or invalid fitted rates"
                )
            }

            vapply(times, function(time) {
                interval_exposure <- pmax(0, pmin(time, upper) - lower)
                sum(rates * interval_exposure)
            }, numeric(1))
        },
        numeric(length(times))
    )

    colnames(cumulative_rates) <- transitions$transition

    number_of_states <- max(transitions$from, transitions$to)

    # Same format as one Jump Forest prediction:
    # one cumulative-rate matrix for each requested time.
    lapply(seq_along(times), function(i) {
        cumulative_matrix <- matrix(0, nrow = number_of_states, ncol = number_of_states)

        cumulative_matrix[
            cbind(transitions$from, transitions$to)
        ] <- cumulative_rates[i, ]

        diag(cumulative_matrix) <- -rowSums(cumulative_matrix)
        cumulative_matrix
    })
}

# Time-averaged oracle curve errors provide one metric shared by all three
# methods. With equal state weights, occupation MISE is proportional to the
# integrated Brier-score regret relative to the true probabilities.
step_matrix_at <- function(times, values, evaluation_times) {
    values <- as.matrix(values)
    if (!is.numeric(times) || length(times) != nrow(values) || is.unsorted(times)) {
        stop("times must be sorted and match the rows of values")
    }
    index <- findInterval(evaluation_times, times)
    if (any(index == 0L)) {
        stop("the prediction grid must begin no later than the evaluation grid")
    }
    values[index, , drop = FALSE]
}

curve_error_table <- function(truth_times, truth_values, prediction_times, prediction_values, model, component_names, profile_names) {
    if (length(truth_values) != length(prediction_values) ||
        length(profile_names) != length(truth_values)) {
        stop("truth, predictions and profile_names must have matching panels")
    }
    horizon <- max(truth_times) - min(truth_times)
    if (!is.finite(horizon) || horizon <= 0) {
        stop("truth_times must cover a positive finite interval")
    }

    do.call(rbind, lapply(seq_along(truth_values), function(i) {
        panel_times <- if (is.list(prediction_times)) {
            prediction_times[[i]]
        } else {
            prediction_times
        }
        truth <- as.matrix(truth_values[[i]])
        prediction <- step_matrix_at(
            panel_times, prediction_values[[i]], truth_times
        )
        if (!identical(dim(prediction), dim(truth)) ||
            length(component_names) != ncol(truth)) {
            stop("truth and prediction matrices must have matching dimensions")
        }

        squared_error <- (prediction - truth)^2
        increments <- sweep(
            (squared_error[-1L, , drop = FALSE] +
             squared_error[-nrow(squared_error), , drop = FALSE]) / 2,
            1L, diff(truth_times), "*"
        )
        mise <- colSums(increments) / horizon
        data.frame(
            model = model, profile = profile_names[i], component = component_names,
            MISE = mise, RMSE = sqrt(mise), row.names = NULL
        )
    }))
}

summarise_curve_errors <- function(errors) {
    summary <- stats::aggregate(MISE ~ model, data = errors, FUN = mean)
    summary$RMSE <- sqrt(summary$MISE)
    summary[order(summary$MISE), ]
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

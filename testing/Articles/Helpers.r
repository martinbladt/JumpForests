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

# Generic multi-state prediction and model-comparison helpers
#--------------------------------------------------------------------------------

# Plot one predicted cumulative transition rate for all covariate profiles.
plot_transition_rate <- function(predictions, event_times, new_data, from, to,
                                 colour_by, linetype_by, max_time = NULL,
                                 xlab = "Time", save = FALSE,
                                 plot_directory = "Plots",
                                 width = 10, height = 6, dpi = 300) {
    if (!is.data.frame(new_data)) {
        stop("new_data must be a data frame")
    }
    if (!all(c(colour_by, linetype_by) %in% names(new_data))) {
        stop("colour_by and linetype_by must name columns in new_data")
    }
    if (!is.numeric(event_times) || length(event_times) == 0L ||
        any(!is.finite(event_times)) || is.unsorted(event_times, strictly = TRUE)) {
        stop("event_times must be a non-empty, strictly increasing numeric vector")
    }
    if (!is.list(predictions) || length(predictions) != nrow(new_data)) {
        stop("predictions must contain one prediction for each row of new_data")
    }
    if (!is.null(max_time) &&
        (!is.numeric(max_time) || length(max_time) != 1L ||
         !is.finite(max_time) || max_time <= 0)) {
        stop("max_time must be NULL or one positive finite number")
    }
    if (!is.logical(save) || length(save) != 1L || is.na(save)) {
        stop("save must be TRUE or FALSE")
    }

    valid_state <- function(state) {
        is.numeric(state) && length(state) == 1L && is.finite(state) &&
            state >= 1 && state == as.integer(state)
    }
    if (!valid_state(from) || !valid_state(to) || from == to) {
        stop("from and to must be distinct positive integers")
    }
    from <- as.integer(from)
    to <- as.integer(to)

    add_zero <- event_times[1L] > 0
    plot_data <- do.call(rbind, lapply(seq_along(predictions), function(i) {
        predicted_matrices <- predictions[[i]]
        if (length(predicted_matrices) != length(event_times)) {
            stop("each prediction must contain one matrix per event time")
        }

        transition_rate <- vapply(predicted_matrices, function(rate_matrix) {
            if (!is.matrix(rate_matrix) ||
                from > nrow(rate_matrix) || to > ncol(rate_matrix)) {
                stop("from or to is outside the predicted transition matrix")
            }
            rate_matrix[from, to]
        }, numeric(1))

        data.frame(
            time = if (add_zero) c(0, event_times) else event_times,
            transition_rate = if (add_zero) c(0, transition_rate) else transition_rate,
            profile = i,
            colour_value = as.character(new_data[[colour_by]][i]),
            linetype_value = as.character(new_data[[linetype_by]][i])
        )
    }))

    plot_data$colour_value <- factor(
        plot_data$colour_value,
        levels = unique(as.character(new_data[[colour_by]]))
    )
    plot_data$linetype_value <- factor(
        plot_data$linetype_value,
        levels = unique(as.character(new_data[[linetype_by]]))
    )

    p <- ggplot(
        plot_data,
        aes(
            x = time,
            y = transition_rate,
            colour = colour_value,
            linetype = linetype_value,
            group = profile
        )
    ) +
        geom_step(linewidth = 0.9, direction = "hv") +
        labs(
            title = paste0("Predicted transition rate: ", from, " → ", to),
            x = xlab,
            y = "Cumulative transition rate",
            colour = colour_by,
            linetype = linetype_by
        ) +
        guides(
            colour = guide_legend(order = 1),
            linetype = guide_legend(order = 2)
        ) +
        theme_classic() +
        theme(
            legend.position = "bottom",
            plot.title = element_text(hjust = 0.5)
        )

    if (!is.null(max_time)) {
        p <- p + coord_cartesian(xlim = c(0, max_time))
    }

    if (save) {
        dir.create(plot_directory, recursive = TRUE, showWarnings = FALSE)
        ggsave(
            filename = file.path(
                plot_directory,
                paste0(
                    "transition_rate_", from, "_", to, "_",
                    colour_by, "_", linetype_by, ".png"
                )
            ),
            plot = p,
            width = width,
            height = height,
            units = "in",
            dpi = dpi
        )
    }
    p
}

# Validate the common trajectory format used by the multi-state helpers.
validate_jump_paths <- function(paths) {
    if (!is.list(paths) || length(paths) == 0L) {
        stop("paths must be a non-empty list")
    }

    for (i in seq_along(paths)) {
        path <- paths[[i]]
        if (!is.list(path) || !all(c("states", "times") %in% names(path)) ||
            length(path$states) != length(path$times) ||
            length(path$states) < 2L ||
            !is.integer(path$states) || !is.numeric(path$times) ||
            any(!is.finite(path$states)) || any(!is.finite(path$times)) ||
            any(path$states < 1) ||
            path$times[1L] != 0 || any(diff(path$times) <= 0)) {
            stop(
                "invalid jump path at observation ", i,
                "; states must be integer and times must start at zero and increase"
            )
        }
    }
    invisible(TRUE)
}

# Convert trajectories to counting-process rows for one cause-specific
# transition. Competing transitions and a repeated final state have event = 0;
# intervals beginning after time zero retain their delayed-entry start time.
paths_to_transition_cox_data <- function(paths, features, from, to) {
    validate_jump_paths(paths)
    if (!is.data.frame(features) || nrow(features) != length(paths)) {
        stop("features must be a data frame with one row per path")
    }
    if (length(from) != 1L || length(to) != 1L ||
        !is.numeric(from) || !is.numeric(to) ||
        !is.finite(from) || !is.finite(to) ||
        from < 1 || to < 1 || from != as.integer(from) ||
        to != as.integer(to) || from == to) {
        stop("from and to must be distinct positive integers")
    }
    from <- as.integer(from)
    to <- as.integer(to)

    rows <- Map(function(path, id) {
        interval <- seq_len(length(path$states) - 1L)
        interval <- interval[path$states[interval] == from]
        if (length(interval) == 0L) {
            return(NULL)
        }

        cbind(
            data.frame(
                id = rep.int(id, length(interval)),
                tstart = path$times[interval],
                tstop = path$times[interval + 1L],
                event = as.integer(path$states[interval + 1L] == to)
            ),
            features[rep.int(id, length(interval)), , drop = FALSE]
        )
    }, paths, seq_along(paths))

    result <- do.call(rbind, rows)
    if (is.null(result) || nrow(result) == 0L) {
        stop("no risk intervals were found for transition ", from, " -> ", to)
    }
    rownames(result) <- NULL
    result
}

# A common representation for transition-specific Cox hazards. A transition
# entry may reuse a fitted model and add fixed values to its prediction data.
new_multistate_cox_model <- function(transitions, number_of_states, label) {
    if (!is.list(transitions) || length(transitions) == 0L ||
        !is.numeric(number_of_states) || length(number_of_states) != 1L ||
        !is.finite(number_of_states) || number_of_states < 2L ||
        number_of_states != as.integer(number_of_states)) {
        stop("invalid transition-specific Cox model specification")
    }
    number_of_states <- as.integer(number_of_states)
    transitions <- lapply(transitions, function(transition) {
        if (is.list(transition) && is.null(transition$fixed_values)) {
            transition$fixed_values <- list()
        }
        transition
    })
    valid_state_index <- function(value) {
        is.numeric(value) && length(value) == 1L && is.finite(value) &&
            value == as.integer(value) && value >= 1L &&
            value <= number_of_states
    }

    transition_keys <- vapply(transitions, function(transition) {
        if (!is.list(transition) ||
            !all(c("from", "to", "fit") %in% names(transition)) ||
            !inherits(transition$fit, "coxph")) {
            stop("each transition must contain from, to and a coxph fit")
        }
        if (!valid_state_index(transition$from) ||
            !valid_state_index(transition$to) ||
            transition$from == transition$to) {
            stop("a transition has an invalid state index")
        }
        paste(as.integer(transition$from), as.integer(transition$to), sep = ":")
    }, character(1))
    if (anyDuplicated(transition_keys)) {
        stop("each directed transition must occur only once")
    }

    structure(
        list(
            label = label,
            number_of_states = number_of_states,
            transitions = transitions
        ),
        class = c("multistate_cox_model", "list")
    )
}

cox_term_summary <- function(fit, term, confidence_level = 0.95) {
    coefficient_table <- summary(fit)$coefficients
    term_index <- match(term, rownames(coefficient_table))
    if (is.na(term_index)) {
        stop("term was not found in the fitted Cox model: ", term)
    }
    if (!is.numeric(confidence_level) || length(confidence_level) != 1L ||
        confidence_level <= 0 || confidence_level >= 1) {
        stop("confidence_level must lie strictly between zero and one")
    }

    estimate <- coefficient_table[term_index, "coef"]
    standard_error <- coefficient_table[term_index, "se(coef)"]
    critical_value <- stats::qnorm(1 - (1 - confidence_level) / 2)
    data.frame(
        term = term,
        estimate = estimate,
        standard_error = standard_error,
        hazard_ratio = exp(estimate),
        lower_confidence_limit = exp(estimate - critical_value * standard_error),
        upper_confidence_limit = exp(estimate + critical_value * standard_error),
        p_value = coefficient_table[term_index, "Pr(>|z|)"],
        row.names = NULL
    )
}

normalise_occupation_probabilities <- function(probabilities, label = "prediction", tolerance = 1e-7) {
    probabilities <- as.matrix(probabilities)
    storage.mode(probabilities) <- "double"
    if (nrow(probabilities) == 0L || ncol(probabilities) < 2L ||
        any(!is.finite(probabilities)) ||
        any(probabilities < -tolerance) ||
        any(probabilities > 1 + tolerance)) {
        stop(label, " contains invalid occupation probabilities")
    }

    probabilities[probabilities < 0] <- 0
    probabilities[probabilities > 1] <- 1
    probability_totals <- rowSums(probabilities)
    if (any(probability_totals <= 0) ||
        any(abs(probability_totals - 1) > tolerance)) {
        stop(label, " contains rows whose probabilities do not sum to one")
    }
    sweep(probabilities, 1L, probability_totals, "/")
}

baseline_cumulative_hazard_at <- function(fit, evaluation_times, baseline = NULL) {
    if (is.null(baseline)) {
        # basehaz() emits an irrelevant warning for interaction models about a
        # mean-covariate curve; centered = FALSE requests the zero-covariate
        # baseline explicitly.
        baseline <- suppressWarnings(survival::basehaz(fit, centered = FALSE))
    }
    if ("strata" %in% names(baseline)) {
        stop("stratified Cox baselines are not supported by this predictor")
    }
    index <- findInterval(evaluation_times, baseline$time)
    result <- numeric(length(evaluation_times))
    result[index > 0L] <- baseline$hazard[index[index > 0L]]
    result
}

# Convert any model created by new_multistate_cox_model() to one time-by-state
# occupation-probability matrix per individual. The exponential stype = 2
# update is the stable default. The direct Aalen-Johansen update is available
# and can be selected by callers that require a direct product integral.
predict_multistate_cox <- function(model, new_data, evaluation_times, initial, stype = c("exponential", "aalen-johansen")) {
    stype <- match.arg(stype)
    if (!inherits(model, "multistate_cox_model") || !is.data.frame(new_data)) {
        stop("model must be a multistate_cox_model and new_data a data frame")
    }
    if (!is.numeric(evaluation_times) || length(evaluation_times) == 0L ||
        any(!is.finite(evaluation_times)) || evaluation_times[1L] != 0 ||
        any(diff(evaluation_times) <= 0)) {
        stop("evaluation_times must be strictly increasing and start at zero")
    }
    number_of_states <- model$number_of_states
    if (!is.numeric(initial) || length(initial) != number_of_states ||
        any(!is.finite(initial)) || any(initial < 0) ||
        abs(sum(initial) - 1) > 1e-10) {
        stop("initial must be a probability vector with one value per state")
    }

    number_of_observations <- nrow(new_data)
    number_of_times <- length(evaluation_times)
    number_of_transitions <- length(model$transitions)
    risk_multipliers <- matrix(0, number_of_observations, number_of_transitions)
    baseline_tables <- vector("list", number_of_transitions)
    product_times <- evaluation_times

    for (transition_index in seq_along(model$transitions)) {
        transition <- model$transitions[[transition_index]]
        prediction_data <- new_data
        for (variable in names(transition$fixed_values)) {
            prediction_data[[variable]] <- transition$fixed_values[[variable]]
        }
        linear_predictor <- stats::predict(
            transition$fit, newdata = prediction_data,
            type = "lp", reference = "zero"
        )
        risk_multipliers[, transition_index] <- exp(as.numeric(linear_predictor))
        baseline_tables[[transition_index]] <- suppressWarnings(
            survival::basehaz(transition$fit, centered = FALSE)
        )
        if ("strata" %in% names(baseline_tables[[transition_index]])) {
            stop("stratified Cox baselines are not supported by this predictor")
        }
        baseline_event_times <- baseline_tables[[transition_index]]$time
        product_times <- sort(unique(c(
            product_times,
            baseline_event_times[baseline_event_times <= tail(evaluation_times, 1L)]
        )))
    }

    # A product integral must update at every baseline-hazard jump. This union
    # makes predictions correct even when the requested output grid is sparse.
    number_of_product_times <- length(product_times)
    hazard_increments <- matrix(
        0, number_of_product_times, number_of_transitions
    )
    for (transition_index in seq_along(model$transitions)) {
        cumulative_hazard <- baseline_cumulative_hazard_at(
            model$transitions[[transition_index]]$fit,
            product_times,
            baseline = baseline_tables[[transition_index]]
        )
        hazard_increments[, transition_index] <- c(
            cumulative_hazard[1L], diff(cumulative_hazard)
        )
    }
    if (any(!is.finite(risk_multipliers)) ||
        any(!is.finite(hazard_increments)) || any(hazard_increments < -1e-12)) {
        stop("the Cox model produced an invalid transition hazard")
    }

    current <- matrix(
        rep(initial, each = number_of_observations),
        nrow = number_of_observations, ncol = number_of_states
    )
    probability_array <- array(
        NA_real_, dim = c(number_of_observations, number_of_times, number_of_states)
    )
    output_time_index <- match(product_times, evaluation_times)
    transition_keys <- vapply(model$transitions, function(transition) {
        paste(transition$from, transition$to, sep = ":")
    }, character(1))
    illness_death_columns <- match(c("1:2", "1:3", "2:3"), transition_keys)
    fast_illness_death <- number_of_states == 3L &&
        length(model$transitions) == 3L && !anyNA(illness_death_columns)

    for (time_index in seq_len(number_of_product_times)) {
        individual_hazards <- sweep(
            risk_multipliers, 2L,
            hazard_increments[time_index, ], "*"
        )

        if (stype == "aalen-johansen") {
            updated <- current
            # All jumps use the probabilities immediately before this time.
            for (transition_index in seq_along(model$transitions)) {
                transition <- model$transitions[[transition_index]]
                jump <- current[, transition$from] *
                    individual_hazards[, transition_index]
                updated[, transition$from] <- updated[, transition$from] - jump
                updated[, transition$to] <- updated[, transition$to] + jump
            }
            current <- updated
        } else if (fast_illness_death) {
            # Closed form of exp(dA) for the acyclic 1 -> 2, 1 -> 3, 2 -> 3
            # system. This avoids millions of small matrix exponentials in the
            # illness-death prediction loop.
            hazard_12 <- individual_hazards[, illness_death_columns[1L]]
            hazard_13 <- individual_hazards[, illness_death_columns[2L]]
            hazard_23 <- individual_hazards[, illness_death_columns[3L]]
            exit_1 <- hazard_12 + hazard_13
            probability_11 <- exp(-exit_1)
            probability_22 <- exp(-hazard_23)
            difference <- exit_1 - hazard_23
            close_rates <- abs(difference) <=
                sqrt(.Machine$double.eps) * pmax(1, abs(exit_1), abs(hazard_23))
            probability_12 <- numeric(number_of_observations)
            probability_12[close_rates] <-
                hazard_12[close_rates] * exp(-exit_1[close_rates])
            probability_12[!close_rates] <- hazard_12[!close_rates] *
                (exp(-hazard_23[!close_rates]) - exp(-exit_1[!close_rates])) /
                difference[!close_rates]
            # The following bounds only remove round-off at the two boundaries.
            probability_12 <- pmax(
                0, pmin(1 - probability_11, probability_12)
            )
            probability_13 <- 1 - probability_11 - probability_12
            probability_23 <- 1 - probability_22

            updated <- current
            updated[, 1L] <- current[, 1L] * probability_11
            updated[, 2L] <- current[, 1L] * probability_12 +
                current[, 2L] * probability_22
            updated[, 3L] <- current[, 1L] * probability_13 +
                current[, 2L] * probability_23 + current[, 3L]
            current <- updated
        } else {
            if (!requireNamespace("Matrix", quietly = TRUE)) {
                stop("Matrix is required for exponential prediction of this transition system")
            }
            updated <- current
            for (observation in seq_len(number_of_observations)) {
                generator_increment <- matrix(
                    0, nrow = number_of_states, ncol = number_of_states
                )
                for (transition_index in seq_along(model$transitions)) {
                    transition <- model$transitions[[transition_index]]
                    generator_increment[transition$from, transition$to] <-
                        generator_increment[transition$from, transition$to] +
                        individual_hazards[observation, transition_index]
                }
                diag(generator_increment) <- -rowSums(generator_increment)
                updated[observation, ] <- as.numeric(
                    current[observation, ] %*%
                        as.matrix(Matrix::expm(generator_increment))
                )
            }
            current <- updated
        }
        if (!is.na(output_time_index[time_index])) {
            probability_array[, output_time_index[time_index], ] <- current
        }
    }

    state_names <- paste("State", seq_len(number_of_states))
    values <- lapply(seq_len(number_of_observations), function(i) {
        probabilities <- matrix(
            probability_array[i, , , drop = FALSE],
            nrow = number_of_times, ncol = number_of_states,
            dimnames = list(NULL, state_names)
        )
        normalise_occupation_probabilities(
            probabilities, paste(model$label, "observation", i)
        )
    })
    list(times = evaluation_times, values = values)
}

stratified_multistate_folds <- function(paths, number_of_folds = 5L, seed = 2026) {
    validate_jump_paths(paths)
    if (!is.numeric(number_of_folds) || length(number_of_folds) != 1L ||
        !is.finite(number_of_folds) ||
        number_of_folds != as.integer(number_of_folds) ||
        number_of_folds < 2L || number_of_folds > length(paths)) {
        stop("number_of_folds must be between 2 and the number of paths")
    }
    number_of_folds <- as.integer(number_of_folds)

    # Balance the observed state-sequence patterns across folds.
    path_type <- vapply(paths, function(path) {
        paste(path$states, collapse = "-")
    }, character(1))
    set.seed(seed)
    folds <- integer(length(paths))
    for (stratum in unique(path_type)) {
        indices <- sample(which(path_type == stratum))
        folds[indices] <- rep(seq_len(number_of_folds), length.out = length(indices))
    }
    if (any(tabulate(folds, nbins = number_of_folds) == 0L)) {
        stop("fold construction produced an empty validation fold")
    }
    folds
}

validate_fold_assignments <- function(folds, number_of_observations) {
    if (!is.numeric(folds) || length(folds) != number_of_observations ||
        any(!is.finite(folds)) || any(folds < 1) ||
        any(folds != as.integer(folds))) {
        stop("folds must contain one positive integer per observation")
    }
    folds <- as.integer(folds)
    fold_levels <- sort(unique(folds))
    if (!identical(fold_levels, seq_len(max(folds))) || length(fold_levels) < 2L) {
        stop("folds must use consecutive values beginning at one")
    }
    folds
}

crossfit_multistate_cox <- function(paths, features, fit_function, evaluation_times, number_of_folds = 5L, seed = 2026, 
                                    initial, verbose = TRUE, prediction_stype = "exponential", folds = NULL, ...) {
    if (!is.function(fit_function) || !is.data.frame(features) ||
        nrow(features) != length(paths)) {
        stop("fit_function must be a function and features must match paths")
    }
    if (is.null(folds)) {
        folds <- stratified_multistate_folds(paths, number_of_folds, seed)
    } else {
        folds <- validate_fold_assignments(folds, length(paths))
        number_of_folds <- max(folds)
    }
    predictions <- vector("list", length(paths))

    for (fold in seq_len(number_of_folds)) {
        validation_rows <- which(folds == fold)
        training_rows <- which(folds != fold)
        fitted <- fit_function(
            paths[training_rows], features[training_rows, , drop = FALSE], ...
        )
        fold_predictions <- predict_multistate_cox(
            fitted,
            new_data = features[validation_rows, , drop = FALSE],
            evaluation_times = evaluation_times,
            initial = initial,
            stype = prediction_stype
        )
        predictions[validation_rows] <- fold_predictions$values
        if (verbose) {
            message("Finished Cox cross-fitting fold ", fold, " of ", number_of_folds)
        }
        rm(fitted, fold_predictions)
        gc(verbose = FALSE)
    }

    structure(
        list(times = evaluation_times, values = predictions, folds = folds),
        class = c("occupation_prediction_set", "list")
    )
}

# Apply an initial distribution and product integral to a set of cumulative
# Nelson-Aalen matrix predictions. This works for fitted-forest OOB predictions
# as well as predictions made by a cross-fitted forest.
nelson_aalen_to_occupation_predictions <- function(predictions, prediction_times, initial, label = "forest") {
    if (!is.list(predictions) || length(predictions) == 0L ||
        !is.numeric(prediction_times) || length(prediction_times) == 0L) {
        stop("invalid Nelson-Aalen predictions")
    }
    number_of_observations <- length(predictions)
    number_of_states <- nrow(predictions[[1L]][[1L]])
    initial_values <- if (is.list(initial)) {
        initial
    } else {
        rep(list(initial), number_of_observations)
    }
    if (length(initial_values) != number_of_observations ||
        any(vapply(initial_values, length, integer(1)) != number_of_states)) {
        stop("initial must be one vector or one vector per observation")
    }
    valid_initial <- vapply(initial_values, function(value) {
        is.numeric(value) && all(is.finite(value)) && all(value >= 0) &&
            abs(sum(value) - 1) <= 1e-10
    }, logical(1))
    if (!all(valid_initial)) {
        stop("every initial distribution must be a probability vector")
    }

    state_names <- paste("State", seq_len(number_of_states))
    lapply(seq_len(number_of_observations), function(i) {
        if (length(predictions[[i]]) != length(prediction_times)) {
            stop("a forest prediction does not match its event-time grid")
        }
        probabilities <- do.call(
            rbind, occprob_from_data(predictions[[i]], initial_values[[i]])
        )
        colnames(probabilities) <- state_names
        normalise_occupation_probabilities(
            probabilities, paste(label, "observation", i)
        )
    })
}

# Turn the forest's cumulative Nelson-Aalen matrices into the same prediction
# representation used by predict_multistate_cox().
as_jumpforest_occupation_predictions <- function(forest, oob = TRUE, initial = NULL) {
    prediction_name <- if (oob) "oob.predictions" else "predictions"
    initial_name <- if (oob) "oob.init" else "init"
    predictions <- forest[[prediction_name]]
    prediction_times <- forest$unique.event.times
    if (is.null(predictions) || is.null(prediction_times)) {
        stop("the forest does not contain saved ", prediction_name)
    }

    if (is.null(initial)) {
        initial <- forest[[initial_name]]
        if (is.null(initial)) {
            stop("supply initial because the forest has no saved initial distributions")
        }
    }
    occupation_probabilities <- nelson_aalen_to_occupation_predictions(
        predictions, prediction_times, initial, label = "JumpForest"
    )

    structure(
        list(times = prediction_times, values = occupation_probabilities),
        class = c("occupation_prediction_set", "list")
    )
}

# Matched K-fold predictions for a JumpForest. Supplying the same `folds` to
# this helper and crossfit_multistate_cox() gives every model exactly the same
# training and validation subjects.
crossfit_jumpforest <- function(paths, features, initial, number_of_folds = 5L, folds = NULL, seed = 2026, verbose = TRUE, formula = MM ~ ., ...) {
    validate_jump_paths(paths)
    if (!is.data.frame(features) || nrow(features) != length(paths)) {
        stop("features must be a data frame with one row per path")
    }
    if (is.null(folds)) {
        folds <- stratified_multistate_folds(paths, number_of_folds, seed)
    } else {
        folds <- validate_fold_assignments(folds, length(paths))
        number_of_folds <- max(folds)
    }
    forest_arguments <- list(...)
    reserved_arguments <- c(
        "formula", "data", "feature_data", "seed", "save_predictions"
    )
    if (any(names(forest_arguments) %in% reserved_arguments)) {
        stop("do not pass reserved fitting arguments through ...")
    }

    prediction_values <- vector("list", length(paths))
    prediction_times <- vector("list", length(paths))
    # The JumpForests C++ interface expects this exact element order and type.
    forest_paths <- lapply(paths, function(path) {
        list(times = as.numeric(path$times), states = as.integer(path$states))
    })
    for (fold in seq_len(number_of_folds)) {
        validation_rows <- which(folds == fold)
        training_rows <- which(folds != fold)
        fit_arguments <- c(
            list(
                formula = formula,
                data = forest_paths[training_rows],
                feature_data = features[training_rows, , drop = FALSE]
            ),
            forest_arguments,
            list(seed = seed + fold - 1L, save_predictions = FALSE)
        )
        fitted <- do.call(jfforest, fit_arguments)
        fold_nelson_aalen <- jfforest.predict(
            fitted, features[validation_rows, , drop = FALSE]
        )
        fold_occupation <- nelson_aalen_to_occupation_predictions(
            fold_nelson_aalen,
            fitted$unique.event.times,
            initial,
            label = paste("JumpForest fold", fold)
        )
        prediction_values[validation_rows] <- fold_occupation
        prediction_times[validation_rows] <- rep(
            list(fitted$unique.event.times), length(validation_rows)
        )
        if (verbose) {
            message("Finished JumpForest cross-fitting fold ", fold,
                    " of ", number_of_folds)
        }
        rm(fitted, fold_nelson_aalen, fold_occupation)
        gc(verbose = FALSE)
    }

    structure(
        list(times = prediction_times, values = prediction_values, folds = folds),
        class = c("occupation_prediction_set", "list")
    )
}

# Marginal reverse Kaplan-Meier estimate for the censoring distribution. At a
# tied time, terminal events are removed before censorings, matching the usual
# competing-event convention used by JumpForests.
reverse_km_censoring <- function(paths) {
    validate_jump_paths(paths)
    endpoint_times <- vapply(paths, function(path) tail(path$times, 1L), numeric(1))
    censored <- vapply(paths, function(path) {
        number_of_states <- length(path$states)
        path$states[number_of_states] == path$states[number_of_states - 1L]
    }, logical(1))

    unique_times <- sort(unique(endpoint_times))
    survival <- numeric(length(unique_times))
    number_at_risk <- length(paths)
    previous_survival <- 1
    for (i in seq_along(unique_times)) {
        at_time <- endpoint_times == unique_times[i]
        number_of_events <- sum(at_time & !censored)
        number_censored <- sum(at_time & censored)
        denominator <- number_at_risk - number_of_events
        if (denominator > 0L) {
            previous_survival <- previous_survival *
                (1 - number_censored / denominator)
        }
        survival[i] <- previous_survival
        number_at_risk <- number_at_risk - number_of_events - number_censored
    }

    structure(
        list(
            times = unique_times,
            survival = survival,
            endpoint_times = endpoint_times,
            censored = censored
        ),
        class = c("reverse_km_censoring", "list")
    )
}

censoring_survival_at <- function(censoring_model, times, left_limit = FALSE) {
    index <- findInterval(times, censoring_model$times)
    if (left_limit) {
        exact_match <- index > 0L
        exact_match[exact_match] <-
            censoring_model$times[index[exact_match]] == times[exact_match]
        index[exact_match] <- index[exact_match] - 1L
    }
    result <- rep(1, length(times))
    result[index > 0L] <- censoring_model$survival[index[index > 0L]]
    result
}

# The state at a transition time is the post-transition state. At and after a
# censoring endpoint it is unknown (NA); a terminal state remains known.
observed_states_at <- function(paths, evaluation_times) {
    validate_jump_paths(paths)
    number_of_states <- max(unlist(lapply(paths, `[[`, "states")))
    observed <- matrix(
        NA_integer_, nrow = length(paths), ncol = length(evaluation_times)
    )

    for (i in seq_along(paths)) {
        path <- paths[[i]]
        previous_states <- head(path$states, -1L)
        next_states <- tail(path$states, -1L)
        genuine_transition <- next_states != previous_states
        known_times <- c(path$times[1L], tail(path$times, -1L)[genuine_transition])
        known_states <- c(path$states[1L], next_states[genuine_transition])
        index <- findInterval(evaluation_times, known_times)
        if (any(index == 0L)) {
            stop("evaluation_times cannot precede the initial path time")
        }
        observed[i, ] <- as.integer(known_states[index])

        if (tail(path$states, 1L) == tail(previous_states, 1L)) {
            observed[i, evaluation_times >= tail(path$times, 1L)] <- NA_integer_
        }
    }
    colnames(observed) <- as.character(evaluation_times)
    attr(observed, "number_of_states") <- number_of_states
    observed
}

multistate_ipcw <- function(paths, evaluation_times, minimum_censoring_survival = 0.05, max_time = NULL) {
    # This default uses a marginal G(t), so its statistical interpretation
    # assumes censoring is independent without further covariate adjustment.
    if (!is.numeric(evaluation_times) || length(evaluation_times) == 0L ||
        any(!is.finite(evaluation_times)) || evaluation_times[1L] != 0 ||
        any(diff(evaluation_times) <= 0)) {
        stop("evaluation_times must be strictly increasing and start at zero")
    }
    if (!is.numeric(minimum_censoring_survival) ||
        length(minimum_censoring_survival) != 1L ||
        minimum_censoring_survival <= 0 || minimum_censoring_survival > 1) {
        stop("minimum_censoring_survival must lie in (0, 1]")
    }
    if (!is.null(max_time) &&
        (!is.numeric(max_time) || length(max_time) != 1L ||
         !is.finite(max_time) || max_time <= 0)) {
        stop("max_time must be NULL or a positive finite number")
    }

    censoring_model <- reverse_km_censoring(paths)
    if (!is.null(max_time)) {
        evaluation_times <- evaluation_times[evaluation_times <= max_time]
    }
    censoring_survival <- censoring_survival_at(
        censoring_model, evaluation_times
    )
    keep <- is.finite(censoring_survival) &
        censoring_survival >= minimum_censoring_survival
    evaluation_times <- evaluation_times[keep]
    censoring_survival <- censoring_survival[keep]
    if (length(evaluation_times) < 2L) {
        stop("fewer than two evaluation times remain after IPCW truncation")
    }

    observed_states <- observed_states_at(paths, evaluation_times)
    weights <- matrix(
        0, nrow = length(paths), ncol = length(evaluation_times)
    )
    number_of_observations <- length(paths)
    for (i in seq_along(paths)) {
        endpoint <- censoring_model$endpoint_times[i]
        before_endpoint <- evaluation_times < endpoint
        weights[i, before_endpoint] <-
            1 / (number_of_observations * censoring_survival[before_endpoint])

        if (!censoring_model$censored[i]) {
            after_terminal_event <- evaluation_times >= endpoint
            censoring_before_event <- censoring_survival_at(
                censoring_model, endpoint, left_limit = TRUE
            )
            if (any(after_terminal_event) &&
                (!is.finite(censoring_before_event) || censoring_before_event <= 0)) {
                stop("the censoring survival is zero before a terminal event")
            }
            weights[i, after_terminal_event] <-
                1 / (number_of_observations * censoring_before_event)
        }
    }

    list(
        times = evaluation_times,
        weights = weights,
        observed_states = observed_states,
        censoring_survival = censoring_survival,
        censoring_model = censoring_model
    )
}

step_occupation_probabilities_at <- function(times, values, evaluation_times) {
    values <- as.matrix(values)
    if (!is.numeric(times) || length(times) != nrow(values) ||
        any(!is.finite(times)) || any(diff(times) <= 0)) {
        stop("prediction times must be strictly increasing and match values")
    }
    index <- findInterval(evaluation_times, times)
    if (any(index == 0L)) {
        stop("a prediction grid begins after the evaluation grid")
    }
    values[index, , drop = FALSE]
}

normalised_trapezoid <- function(times, values) {
    horizon <- tail(times, 1L) - times[1L]
    if (length(times) < 2L || !is.finite(horizon) || horizon <= 0) {
        stop("times must cover a positive finite interval")
    }
    sum(
        (head(values, -1L) + tail(values, -1L)) * diff(times) / 2
    ) / horizon
}

# Generic empirical Brier, KL and spherical score curves for any named set of
# occupation predictions. Weights follow JumpForests' defaults: 1/S for Brier
# and spherical, and 1 for KL. If state_weights is supplied, it is used for all
# three metrics, which also matches the package interface.
multistate_score_curves <- function(prediction_sets, paths, evaluation_times, state_weights = NULL, minimum_censoring_survival = 0.05, max_time = NULL) {
    if (!is.list(prediction_sets) || length(prediction_sets) == 0L ||
        is.null(names(prediction_sets)) || any(names(prediction_sets) == "") ||
        anyDuplicated(names(prediction_sets))) {
        stop("prediction_sets must be a named non-empty list")
    }

    ipcw <- multistate_ipcw(
        paths, evaluation_times,
        minimum_censoring_survival = minimum_censoring_survival,
        max_time = max_time
    )
    score_times <- ipcw$times
    observed_states <- ipcw$observed_states
    number_of_states <- attr(observed_states, "number_of_states")
    state_names <- paste("State", seq_len(number_of_states))

    if (is.null(state_weights)) {
        metric_weights <- list(
            brier = rep(1 / number_of_states, number_of_states),
            kl = rep(1, number_of_states),
            spherical = rep(1 / number_of_states, number_of_states)
        )
    } else {
        if (!is.numeric(state_weights) || length(state_weights) != number_of_states ||
            any(!is.finite(state_weights)) || any(state_weights < 0) ||
            !any(state_weights > 0)) {
            stop("state_weights must be a non-negative vector with one value per state")
        }
        metric_weights <- list(
            brier = state_weights,
            kl = state_weights,
            spherical = state_weights
        )
    }

    curve_rows <- list()
    integrated_rows <- list()
    curve_row <- 0L
    integrated_row <- 0L

    for (model_name in names(prediction_sets)) {
        prediction_set <- prediction_sets[[model_name]]
        if (!is.list(prediction_set) ||
            !all(c("times", "values") %in% names(prediction_set)) ||
            length(prediction_set$values) != length(paths)) {
            stop(model_name, " must contain one prediction matrix per path")
        }
        if (is.list(prediction_set$times) &&
            length(prediction_set$times) != length(paths)) {
            stop(model_name, " must contain one prediction-time grid per path")
        }
        scores <- list(
            brier = matrix(0, length(score_times), number_of_states),
            kl = matrix(0, length(score_times), number_of_states),
            spherical = matrix(0, length(score_times), number_of_states)
        )

        for (i in seq_along(paths)) {
            individual_times <- if (is.list(prediction_set$times)) {
                prediction_set$times[[i]]
            } else {
                prediction_set$times
            }
            probabilities <- step_occupation_probabilities_at(
                individual_times, prediction_set$values[[i]], score_times
            )
            probabilities <- normalise_occupation_probabilities(
                probabilities, paste(model_name, "observation", i)
            )
            if (ncol(probabilities) != number_of_states) {
                stop(model_name, " predicts the wrong number of states")
            }

            individual_weights <- ipcw$weights[i, ]
            individual_states <- observed_states[i, ]
            known <- which(individual_weights > 0 & !is.na(individual_states))
            if (length(known) == 0L) {
                next
            }

            indicators <- matrix(0, length(score_times), number_of_states)
            indicators[cbind(known, individual_states[known])] <- 1
            for (state in seq_len(number_of_states)) {
                scores$brier[, state] <- scores$brier[, state] +
                    individual_weights * metric_weights$brier[state] *
                    (indicators[, state] - probabilities[, state])^2
            }

            observed_probabilities <- probabilities[
                cbind(known, individual_states[known])
            ]
            probabilities_for_log <- pmin(
                1 - 1e-15, pmax(1e-15, observed_probabilities)
            )
            kl_contribution <- -individual_weights[known] *
                metric_weights$kl[individual_states[known]] *
                log(probabilities_for_log)
            scores$kl[cbind(known, individual_states[known])] <-
                scores$kl[cbind(known, individual_states[known])] +
                kl_contribution

            spherical_denominator <- sqrt(rowSums(sweep(
                probabilities^2, 2L, metric_weights$spherical, "*"
            )))
            spherical_reward <- rep(0, length(known))
            positive_denominator <- spherical_denominator[known] > 0
            spherical_reward[positive_denominator] <-
                metric_weights$spherical[
                    individual_states[known][positive_denominator]
                ] * observed_probabilities[positive_denominator] /
                spherical_denominator[known][positive_denominator]
            spherical_contribution <- individual_weights[known] *
                (1 - spherical_reward)
            scores$spherical[cbind(known, individual_states[known])] <-
                scores$spherical[cbind(known, individual_states[known])] +
                spherical_contribution
        }

        for (metric in names(scores)) {
            colnames(scores[[metric]]) <- state_names
            values <- cbind(scores[[metric]], Total = rowSums(scores[[metric]]))
            components <- colnames(values)
            curve_row <- curve_row + 1L
            curve_rows[[curve_row]] <- data.frame(
                model = model_name,
                metric = metric,
                component = rep(components, each = length(score_times)),
                time = rep(score_times, times = length(components)),
                error = as.vector(values),
                stringsAsFactors = FALSE
            )
            integrated_row <- integrated_row + 1L
            integrated_rows[[integrated_row]] <- data.frame(
                model = model_name,
                metric = metric,
                component = components,
                integrated_error = vapply(
                    seq_along(components),
                    function(component) {
                        normalised_trapezoid(score_times, values[, component])
                    },
                    numeric(1)
                ),
                evaluation_start = score_times[1L],
                evaluation_end = tail(score_times, 1L),
                stringsAsFactors = FALSE
            )
        }
    }

    structure(
        list(
            curves = do.call(rbind, curve_rows),
            integrated = do.call(rbind, integrated_rows),
            evaluation_times = score_times,
            censoring_survival = ipcw$censoring_survival,
            state_weights = metric_weights,
            integration_horizon = range(score_times),
            minimum_censoring_survival = minimum_censoring_survival
        ),
        class = c("multistate_score_curves", "list")
    )
}

plot_multistate_error_curves <- function(score_result,
                                         metrics = c("brier", "kl", "spherical"),
                                         xlab = "Time", save = FALSE,
                                         plot_directory = "Plots",
                                         filename_prefix = "error_curve",
                                         width = 10, height = 7, dpi = 300) {
    if (!inherits(score_result, "multistate_score_curves")) {
        stop("score_result must be returned by multistate_score_curves()")
    }
    available_metrics <- unique(score_result$curves$metric)
    if (any(!metrics %in% available_metrics)) {
        stop("unknown metric: ", paste(setdiff(metrics, available_metrics), collapse = ", "))
    }
    if (!is.logical(save) || length(save) != 1L || is.na(save)) {
        stop("save must be TRUE or FALSE")
    }

    model_levels <- unique(score_result$curves$model)
    component_levels <- unique(score_result$curves$component)
    colours <- setNames(
        grDevices::hcl.colors(length(model_levels), palette = "Dark 3"),
        model_levels
    )
    line_types <- setNames(
        rep(c("solid", "dashed", "dotdash", "longdash", "twodash"),
            length.out = length(model_levels)),
        model_levels
    )

    plots <- setNames(vector("list", length(metrics)), metrics)
    for (metric in metrics) {
        plot_data <- score_result$curves[
            score_result$curves$metric == metric, , drop = FALSE
        ]
        plot_data$model <- factor(plot_data$model, levels = model_levels)
        plot_data$component <- factor(
            plot_data$component, levels = component_levels
        )
        metric_label <- switch(
            metric,
            brier = "Brier",
            kl = "KL",
            spherical = "Spherical"
        )

        p <- ggplot(
            plot_data,
            aes(x = time, y = error, colour = model, linetype = model)
        ) +
            geom_step(direction = "hv", linewidth = 0.8) +
            facet_wrap(~component, ncol = 2, scales = "free_y") +
            scale_colour_manual(values = colours) +
            scale_linetype_manual(values = line_types) +
            labs(
                title = paste(metric_label, "IPCW error curves"),
                x = xlab,
                y = paste(metric_label, "error"),
                colour = NULL,
                linetype = NULL
            ) +
            theme_bw() +
            theme(
                legend.position = "bottom",
                plot.title = element_text(hjust = 0.5)
            )
        plots[[metric]] <- p

        if (save) {
            dir.create(plot_directory, recursive = TRUE, showWarnings = FALSE)
            ggsave(
                filename = file.path(
                    plot_directory,
                    paste0(filename_prefix, "_", metric, ".png")
                ),
                plot = p, width = width, height = height,
                units = "in", dpi = dpi
            )
        }
    }
    plots
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

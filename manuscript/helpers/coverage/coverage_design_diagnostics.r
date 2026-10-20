# Deterministic CTMC calculations for the coverage-study design report.
# Source this file, then call run_coverage_design_diagnostics(output_path).
# No datasets are simulated and no forests are fitted.
# All manuscript entry points and helpers use manuscript/ as the working directory.
coverage_design_source_directory <- normalizePath("helpers/coverage", mustWork = TRUE)

run_coverage_design_diagnostics <- function(
    output_path, parameters = NULL, n = c(500L, 1000L, 2000L),
    risk_times = c(0, 0.5, 1, 2, 3.3, 3.5, 4, 4.5, 5, 5.25, 5.5, 6, 8),
    state2_start = 0, end_time = 4) {
    defaults <- new.env(parent = baseenv())
    sys.source(file.path(coverage_design_source_directory, "coverage_dgp.r"),
               envir = defaults)
    initial_distribution <- defaults$coverage_initial_distribution
    parameters <- utils::modifyList(defaults$non_proportional_hazards_parameters,
                                    if (is.null(parameters)) list() else parameters)
    if (any(!is.finite(unlist(parameters))) || parameters$censor_lower < 0 ||
        parameters$censor_upper <= parameters$censor_lower ||
        state2_start < 0 || end_time <= state2_start ||
        end_time >= parameters$censor_upper ||
        any(!is.finite(n) | n < 1 | n != floor(n)) ||
        any(!is.finite(risk_times) | risk_times < 0)) {
        stop("Invalid DGP parameters, sample sizes, or diagnostic time interval")
    }
    profiles <- expand.grid(X1 = 0:1, X2 = 1:4, X3 = 0:1)
    risk_times <- sort(unique(risk_times))
    switches <- sort(unique(unlist(parameters[c("s1", "s2", "s3", "s4")],
                                    use.names = FALSE)))
    if (any(switches < 0)) stop("Switch times must be nonnegative")

    # exp(Q * dt) applied to a row probability vector by uniformization:
    # sum_k Pois(k; q*dt) p (I + Q/q)^k. The Poisson tail is below 1e-14.
    advance <- function(p, Q, dt) {
        q <- max(-diag(Q))
        if (dt == 0 || q == 0) return(p)
        P <- diag(3L) + Q / q
        lambda <- q * dt
        term <- p
        result <- stats::dpois(0L, lambda) * term
        last <- max(1L, stats::qpois(1 - 1e-14, lambda) + 2L)
        for (k in seq_len(last)) {
            term <- drop(term %*% P)
            result <- result + stats::dpois(k, lambda) * term
        }
        result
    }
    generator <- function(x, time) {
        r12 <- if (x$X1 == 1L) {
            parameters$a1 + parameters$b1 * (time >= parameters$s1) *
                c(-1, 0, 1, 4)[x$X2]
        } else parameters$a2 + parameters$b2 * (time >= parameters$s2)
        r23 <- if (x$X3 == 1L) {
            parameters$a3 + parameters$b3 * (time >= parameters$s3)
        } else parameters$a4 + parameters$b4 * (time >= parameters$s4)
        r13 <- parameters$rate_13
        r21 <- parameters$rate_21
        if (any(c(r12, r13, r21, r23) < 0)) stop("Negative transition intensity")
        matrix(c(-r12 - r13, r12, r13,
                 r21, -r21 - r23, r23,
                 0, 0, 0), nrow = 3L, byrow = TRUE)
    }
    occupation <- function(x, time) {
        if (time == 0) return(initial_distribution)
        knots <- sort(unique(c(0, switches[switches > 0 & switches < time], time)))
        p <- initial_distribution
        for (k in seq_len(length(knots) - 1L)) {
            p <- advance(p, generator(x, knots[k]), knots[k + 1L] - knots[k])
        }
        p
    }
    censor_survival <- function(time) {
        pmax(0, pmin(1, (parameters$censor_upper - time) /
                           (parameters$censor_upper - parameters$censor_lower)))
    }
    probability_errors <- numeric()
    risks <- do.call(rbind, lapply(seq_len(nrow(profiles)), function(i) {
        do.call(rbind, lapply(risk_times, function(time) {
            p <- occupation(profiles[i, ], time)
            probability_errors <<- c(probability_errors, abs(sum(p) - 1))
            G <- censor_survival(time)
            data.frame(profiles[i, ], t = time, p1 = p[1L], p2 = p[2L],
                       y1 = G * p[1L], y2 = G * p[2L])
        }))
    }))
    risks$censor_survival <- censor_survival(risks$t)
    # Preserve the schema of the report's original risk table.
    risks$expected_risk1_n500_signal <- 500 / 16 * risks$y1
    risks$expected_risk2_n500_signal <- 500 / 16 * risks$y2
    risks$expected_risk1_n500_noise <- 500 / 96 * risks$y1
    risks$expected_risk2_n500_noise <- 500 / 96 * risks$y2

    # Integrate separately at switches, including for user-supplied parameters.
    integrate_piecewise <- function(f, lower, upper) {
        knots <- sort(unique(c(lower, switches[switches > lower & switches < upper],
                               parameters$censor_lower[
                                   parameters$censor_lower > lower &
                                       parameters$censor_lower < upper], upper)))
        sum(vapply(seq_len(length(knots) - 1L), function(k) {
            stats::integrate(f, knots[k], knots[k + 1L], rel.tol = 1e-10)$value
        }, numeric(1)))
    }
    censor_probability <- integrate_piecewise(function(times) {
        vapply(times, function(time) {
            mean(vapply(seq_len(nrow(profiles)), function(i) {
                sum(occupation(profiles[i, ], time)[1:2])
            }, numeric(1))) / (parameters$censor_upper - parameters$censor_lower)
        }, numeric(1))
    }, parameters$censor_lower, parameters$censor_upper)
    per_subject_returns <- vapply(seq_len(nrow(profiles)), function(i) {
        integrate_piecewise(function(times) {
            vapply(times, function(time) {
                parameters$rate_21 * occupation(profiles[i, ], time)[2L] *
                    censor_survival(time)
            }, numeric(1))
        }, state2_start, end_time)
    }, numeric(1))
    returns <- do.call(rbind, lapply(n, function(size) {
        do.call(rbind, lapply(c(16L, 96L), function(number_profiles) {
            data.frame(n = size,
                       arm = if (number_profiles == 16L) "without_noise" else "with_noise",
                       number_profiles = number_profiles, profiles,
                       interval_start = state2_start, interval_end = end_time,
                       expected_events_per_subject = per_subject_returns,
                       expected_stratum_events = size / number_profiles * per_subject_returns)
        }))
    }))
    bounds <- do.call(rbind, lapply(split(returns, list(returns$n, returns$arm)), function(z) {
        data.frame(n = z$n[1L], arm = z$arm[1L], number_profiles = z$number_profiles[1L],
                   interval_start = state2_start, interval_end = end_time,
                   min_expected_events = min(z$expected_stratum_events),
                   mean_expected_events = mean(z$expected_stratum_events),
                   max_expected_events = max(z$expected_stratum_events),
                   mean_profile_coverage_upper_bound = if (parameters$rate_21 > 0) {
                       mean(pmin(1, z$expected_stratum_events))
                   } else NA_real_)
    }))
    summary <- data.frame(censoring_probability = censor_probability,
                          initial_probability_1 = initial_distribution[1L],
                          initial_probability_2 = initial_distribution[2L],
                          initial_probability_3 = initial_distribution[3L],
                          maximum_probability_sum_error = max(probability_errors),
                          state2_start = state2_start, end_time = end_time,
                          true_return_increment = parameters$rate_21 * (end_time - state2_start),
                          min_returns_per_subject = min(per_subject_returns),
                          mean_returns_per_subject = mean(per_subject_returns),
                          max_returns_per_subject = max(per_subject_returns))
    dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
    tables <- list(coverage_expected_risk = risks, coverage_expected_returns = returns,
                   coverage_return_coverage_bounds = bounds, coverage_design_summary = summary)
    for (name in names(tables)) {
        utils::write.table(tables[[name]], file.path(output_path, paste0(name, ".tsv")),
                           sep = "\t", row.names = FALSE, quote = FALSE)
    }
    invisible(tables)
}

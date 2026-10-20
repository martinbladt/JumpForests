# Inference helpers for the non-proportional-hazards coverage study.
# This file has no package dependencies and does not fit models or run studies.
#
# The coverage study uses standard leaf Nelson--Aalen variance estimates,
# averaged over the fitted trees and multiplied by the WOR forest factor.
# Mixed leaves are retained. Consistency is justified under the paper's
# eventual-purity assumptions; finite-sample pooling bias is not corrected.

coverage_leaf_variance_factor <- function(n, m, B) {
  values <- c(n, m, B)
  if (length(values) != 3L || any(!is.finite(values)) ||
      any(values < 1 | values != floor(values)) || m > n) {
    stop("Require integer n >= m >= 1 and B >= 1")
  }
  m / n + (1 - m / n) / B
}

.coverage_check_level <- function(level) {
  if (length(level) != 1L || !is.finite(level) || level <= 0 || level >= 1) {
    stop("level must be a finite number strictly between zero and one.")
  }
}

.coverage_check_monotone <- function(x, name, tolerance = 1e-10) {
  if (!is.numeric(x) || any(is.infinite(x))) {
    stop(name, " must be numeric and contain no infinite values.")
  }
  increments <- diff(c(0, x))
  scale <- max(1, abs(x), na.rm = TRUE)
  if (any(increments < -tolerance * scale, na.rm = TRUE)) {
    stop(name, " must be nonnegative and nondecreasing.")
  }
  invisible(NULL)
}

# For a single leaf with untied events, a jump is 1/Y and its square is 1/Y^2.
# This helper also retains the former forest-jump calculation for historical
# comparisons/tests. The coverage driver now squares jumps BEFORE averaging.
# For a forest, squaring the AVERAGED jump includes all cross-tree products:
# sum_u (Delta A_F(u))^2 = B^(-2) sum_{b,c,u} Delta A_b(u) Delta A_c(u).
# Shared observations therefore contribute to the covariance automatically.
# Do not divide the result by B or multiply it by another forest correction.
coverage_squared_jumps <- function(A, tolerance = 1e-10) {
  if (!is.matrix(A) || !is.numeric(A) || any(dim(A) == 0L)) {
    stop("A must be a nonempty numeric matrix with times in rows.")
  }
  if (length(tolerance) != 1L || !is.finite(tolerance) || tolerance < 0) {
    stop("tolerance must be one finite nonnegative number.")
  }
  Q <- A
  for (j in seq_len(ncol(A))) {
    .coverage_check_monotone(A[, j], paste0("A[, ", j, "]"), tolerance)
    increments <- diff(c(0, A[, j]))
    # Suppress only negative roundoff; retain NA and all positive increments.
    increments[increments < 0 & !is.na(increments)] <- 0
    Q[, j] <- cumsum(increments^2)
  }
  Q
}

# Distribution of sup_{0 <= t <= 1} |W(t)| for standard Brownian motion.
# At small a use the heat-equation eigenfunction expansion; at larger a use
# the reflection expansion to avoid cancellation in a CDF close to one.
.coverage_brownian_sup_cdf <- function(a) {
  if (a <= 0) return(0)
  if (is.infinite(a)) return(1)
  total <- 0
  for (k in 0:10000) {
    odd <- 2 * k + 1
    term <- if (a < 1) {
      4 / pi * (-1)^k / odd * exp(-odd^2 * pi^2 / (8 * a^2))
    } else {
      4 * (-1)^k * pnorm(odd * a, lower.tail = FALSE)
    }
    total <- total + term
    if (abs(term) < 1e-15) break
    if (k == 10000) stop("Brownian supremum series did not converge.")
  }
  answer <- if (a < 1) total else 1 - total
  min(1, max(0, answer))
}

coverage_brownian_sup_quantile <- function(level = 0.95) {
  .coverage_check_level(level)
  # The reflection/union bound gives P(sup |W| > a) <= 4 P(W(1) > a).
  upper <- max(1, qnorm((1 - level) / 4, lower.tail = FALSE))
  uniroot(function(a) .coverage_brownian_sup_cdf(a) - level,
          interval = c(0, upper * (1 + 1e-10)), tol = 1e-12)$root
}

# Construct intervals on [start, end]. A, Q and truth refer to the ORIGINAL
# cumulative curves on the supplied full grid. All three curves are anchored
# at their right-continuous values at start. Thus a positive start targets
# Lambda(t) - Lambda(start), not the full cumulative rate from time zero.
# Missing values remain missing; a zero-event curve has zero estimated variance
# and is retained explicitly instead of being dropped from coverage summaries.
#
# Simultaneous bands use the exact CONTINUOUS Brownian supremum quantile.
# For a continuous nondecreasing limiting variance V,
# sup_t |W(V(t))| =d sqrt(V(end)) sup_{0<=s<=1}|W(s)|.
# On a finite jump grid this continuous envelope is conservative relative to
# simulating only the grid points, and it has the required asymptotic limit.
coverage_interval_curve <- function(times, A, Q, truth, start, end,
                                    level = 0.95, sup_critical = NULL,
                                    clip_lower = FALSE) {
  .coverage_check_level(level)
  if (!is.numeric(times) || !length(times) || any(!is.finite(times)) ||
      any(diff(times) <= 0)) {
    stop("times must be finite and strictly increasing.")
  }
  if (length(start) != 1L || length(end) != 1L || !is.finite(start) ||
      !is.finite(end) || start >= end) {
    stop("start and end must be finite scalars with start < end.")
  }
  if (!(start %in% times) || !(end %in% times)) {
    stop("The full time grid must contain start and end exactly.")
  }
  if (length(A) != length(times) || length(Q) != length(times) ||
      length(truth) != length(times)) {
    stop("A, Q and truth must have one value per time.")
  }
  .coverage_check_monotone(A, "A")
  .coverage_check_monotone(Q, "Q")
  .coverage_check_monotone(truth, "truth")
  if (length(clip_lower) != 1L || is.na(clip_lower) || !is.logical(clip_lower)) {
    stop("clip_lower must be TRUE or FALSE.")
  }
  if (is.null(sup_critical)) {
    sup_critical <- coverage_brownian_sup_quantile(level)
  }
  if (length(sup_critical) != 1L || !is.finite(sup_critical) || sup_critical <= 0) {
    stop("sup_critical must be one finite positive number.")
  }

  anchor <- match(start, times)
  keep <- which(times >= start & times <= end)
  estimate <- A[keep] - A[anchor]
  target <- truth[keep] - truth[anchor]
  variance <- pmax(0, Q[keep] - Q[anchor])
  se <- sqrt(variance)
  pointwise_half_width <- qnorm((1 + level) / 2) * se
  simultaneous_half_width <- sup_critical * sqrt(tail(variance, 1))
  pointwise_lower <- estimate - pointwise_half_width
  simultaneous_lower <- estimate - simultaneous_half_width
  if (clip_lower) {
    pointwise_lower <- pmax(0, pointwise_lower)
    simultaneous_lower <- pmax(0, simultaneous_lower)
  }
  pointwise_upper <- estimate + pointwise_half_width
  simultaneous_upper <- estimate + simultaneous_half_width

  data.frame(
    time = times[keep], estimate = estimate, true = target,
    bias = estimate - target, variance = variance, se = se,
    pointwise_lower = pointwise_lower, pointwise_upper = pointwise_upper,
    pointwise_width = pointwise_upper - pointwise_lower,
    pointwise_covered = target >= pointwise_lower & target <= pointwise_upper,
    simultaneous_lower = simultaneous_lower,
    simultaneous_upper = simultaneous_upper,
    simultaneous_width = simultaneous_upper - simultaneous_lower,
    simultaneous_covered = target >= simultaneous_lower &
      target <= simultaneous_upper,
    positive_variance = variance > 0,
    is_pointwise_anchor = times[keep] == start,
    row.names = NULL
  )
}

# Exact path-coverage check when the true cumulative rate is continuous and
# nondecreasing, and estimates/bounds are right-continuous step functions.
# The grid MUST contain every estimator/bound jump and both interval endpoints.
# Between consecutive grid times, checking the true curve at BOTH endpoints is
# sufficient: the bounds are constant and the true curve is monotone. Looking
# only at the post-jump estimates can otherwise miss a failure just before a
# jump. For unknown values return NA, rather than silently discarding them.
coverage_continuum_covered <- function(times, estimate, truth, lower, upper) {
  if (!is.numeric(times) || !length(times) || any(!is.finite(times)) ||
      any(diff(times) <= 0)) {
    stop("times must be finite and strictly increasing.")
  }
  values <- list(estimate = estimate, truth = truth, lower = lower, upper = upper)
  if (any(vapply(values, length, integer(1)) != length(times)) ||
      any(!vapply(values, is.numeric, logical(1)))) {
    stop("estimate, truth, lower and upper must be numeric with one value per time.")
  }
  if (any(vapply(values, function(x) any(is.infinite(x)), logical(1)))) {
    stop("Curves must not contain infinite values.")
  }
  if (any(vapply(values, anyNA, logical(1)))) return(NA)
  if (any(diff(truth) < -1e-10 * max(1, abs(truth)))) {
    stop("truth must be nondecreasing for the continuum coverage check.")
  }
  if (any(lower > upper)) stop("lower must not exceed upper.")
  right_covered <- truth >= lower & truth <= upper
  if (length(times) == 1L) return(all(right_covered))
  left_covered <- truth[-1L] >= head(lower, -1L) &
    truth[-1L] <= head(upper, -1L)
  all(right_covered) && all(left_covered)
}

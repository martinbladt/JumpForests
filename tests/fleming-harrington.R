# Self-contained integration and validation tests for the multi-state
# Fleming--Harrington splitting rule. Run against an installed JumpForests
# package with:
# Rscript tests/fleming-harrington.R

library(JumpForests)

assert_error <- function(expression, pattern) {
  message <- tryCatch(
    {
      force(expression)
      NULL
    },
    error = function(condition) conditionMessage(condition)
  )
  if (is.null(message)) {
    stop("Expected an error containing: ", pattern)
  }
  if (!grepl(pattern, message, fixed = TRUE)) {
    stop("Expected error containing '", pattern, "', got: ", message)
  }
}

tree_table <- function(tree) {
  JumpForests:::getTreeTable(tree$Tree)
}

same_tree <- function(first, second) {
  first_table <- tree_table(first)
  second_table <- tree_table(second)
  identical(first_table$left.daughters, second_table$left.daughters) &&
    identical(first_table$feature.IDs, second_table$feature.IDs) &&
    identical(first_table$thresholds, second_table$thresholds)
}

make_three_state_data <- function(n = 48L) {
  paths <- lapply(seq_len(n), function(i) {
    switch(
      as.character(i %% 5L),
      "0" = list(times = c(0, 0.7 + i / 100, 2.8 + i / 100), states = c(1L, 2L, 3L)),
      "1" = list(times = c(0, 1.2 + i / 100), states = c(1L, 3L)),
      "2" = list(times = c(0, 0.9 + i / 100, 3.1 + i / 100), states = c(2L, 1L, 3L)),
      "3" = list(times = c(0, 1.7 + i / 100), states = c(2L, 3L)),
      "4" = list(times = c(0, 1.0 + i / 100, 2.2 + i / 100), states = c(1L, 2L, 2L))
    )
  })
  features <- data.frame(
    x = seq_len(n),
    z = sin(seq_len(n)),
    group = factor(rep(letters[1:4], length.out = n))
  )
  list(paths = paths, features = features)
}

manual_fh_root_threshold <- function(paths, feature, min_node_size, a, b) {
  num_obs <- length(paths)
  num_states <- max(unlist(lapply(paths, function(path) path$states)))
  transitions <- do.call(rbind, lapply(seq_along(paths), function(observation) {
    path <- paths[[observation]]
    positions <- which(path$states[-1] != path$states[-length(path$states)]) + 1L
    if (length(positions) == 0L) {
      return(NULL)
    }
    data.frame(
      observation = observation,
      time = path$times[positions],
      from = path$states[positions - 1L],
      to = path$states[positions]
    )
  }))
  event_times <- c(0, sort(unique(transitions$time)))
  num_times <- length(event_times)

  compute_counts <- function(indices) {
    jumps <- array(0, dim = c(num_times, num_states, num_states))
    at_risk <- matrix(0, nrow = num_times, ncol = num_states)
    for (observation in indices) {
      at_risk[1, paths[[observation]]$states[1]] <-
        at_risk[1, paths[[observation]]$states[1]] + 1
    }
    selected <- transitions$observation %in% indices
    for (row in which(selected)) {
      time_id <- match(transitions$time[row], event_times)
      jumps[time_id, transitions$from[row], transitions$to[row]] <-
        jumps[time_id, transitions$from[row], transitions$to[row]] + 1
    }
    for (time_id in 2:num_times) {
      at_risk[time_id, ] <- at_risk[time_id - 1L, ]
      for (from in seq_len(num_states)) {
        for (to in seq_len(num_states)) {
          count <- jumps[time_id - 1L, from, to]
          at_risk[time_id, from] <- at_risk[time_id, from] - count
          at_risk[time_id, to] <- at_risk[time_id, to] + count
        }
      }
    }
    list(jumps = jumps, at_risk = at_risk)
  }

  parent <- compute_counts(seq_len(num_obs))
  occupation <- matrix(0, nrow = num_times, ncol = num_states)
  occupation[1, ] <- parent$at_risk[1, ] / num_obs
  for (time_id in 2:num_times) {
    previous <- occupation[time_id - 1L, ]
    occupation[time_id, ] <- previous
    for (from in seq_len(num_states)) {
      for (to in seq_len(num_states)) {
        count <- parent$jumps[time_id, from, to]
        if (count > 0) {
          change <- previous[from] * count / parent$at_risk[time_id, from]
          occupation[time_id, from] <- occupation[time_id, from] - change
          occupation[time_id, to] <- occupation[time_id, to] + change
        }
      }
    }
  }

  valid_jumps <- unique(transitions[c("from", "to")])
  valid_jumps <- valid_jumps[order(valid_jumps$from, valid_jumps$to), , drop = FALSE]
  thresholds <- sort(unique(feature))
  scores <- rep(-1, length(thresholds))
  for (threshold_id in seq_along(thresholds)) {
    daughter_indices <- which(feature > thresholds[threshold_id])
    if (length(daughter_indices) < min_node_size ||
        num_obs - length(daughter_indices) < min_node_size) {
      next
    }
    daughter <- compute_counts(daughter_indices)
    statistic <- 0
    for (jump_id in seq_len(nrow(valid_jumps))) {
      from <- valid_jumps$from[jump_id]
      to <- valid_jumps$to[jump_id]
      sum_num <- 0
      sum_den <- 0
      for (time_id in which(parent$jumps[, from, to] > 0)) {
        d <- parent$jumps[time_id, from, to]
        d1 <- daughter$jumps[time_id, from, to]
        risk <- parent$at_risk[time_id, from]
        risk1 <- daughter$at_risk[time_id, from]
        if (risk < 2 || risk1 < 1) {
          next
        }
        fraction <- risk1 / risk
        weight <- occupation[time_id, from]^a[from] *
          (1 - occupation[time_id, from])^b[from]
        sum_num <- sum_num + weight * (d1 - d * fraction)
        sum_den <- sum_den + weight^2 * d * fraction * (1 - fraction) *
          (risk - d) / (risk - 1)
      }
      if (sum_den > 0) {
        statistic <- statistic + sum_num / sqrt(sum_den)
      }
    }
    statistic <- abs(statistic)
    if (statistic > 0) {
      scores[threshold_id] <- statistic
    }
  }
  thresholds[which.max(scores)]
}

three_state <- make_three_state_data()
paths <- three_state$paths
features <- three_state$features

# NULL and numeric(0) both resolve to all-one defaults, and explicit defaults
# produce the same fitted tree.
default_tree <- jftree(
  MM ~ x + z, paths, features,
  splitrule = "flemingharrington", min_node_size = 5, nsplits = 30, seed = 100
)
empty_tree <- jftree(
  MM ~ x + z, paths, features,
  splitrule = "flemingharrington", min_node_size = 5, nsplits = 30, seed = 100,
  fh_weights_a = numeric(0), fh_weights_b = numeric(0)
)
explicit_default_tree <- jftree(
  MM ~ x + z, paths, features,
  splitrule = "flemingharrington", min_node_size = 5, nsplits = 30, seed = 100,
  fh_weights_a = rep(1, 3), fh_weights_b = rep(1, 3)
)
stopifnot(
  identical(default_tree$fh.weights.a, rep(1, 3)),
  identical(default_tree$fh.weights.b, rep(1, 3)),
  same_tree(default_tree, empty_tree),
  same_tree(default_tree, explicit_default_tree)
)

# Custom fractional/zero exponents, one omitted vector, integer inputs, honest
# fitting, event-grid thinning, and categorical splitting all work.
custom_tree <- jftree(
  MM ~ x + group, paths, features,
  splitrule = "flemingharrington", min_node_size = 5, nsplits = 30, seed = 101,
  fh_weights_a = c(0, 0.5, 2), fh_weights_b = c(1, 0, 3)
)
one_default_tree <- jftree(
  MM ~ x, paths, features,
  splitrule = "flemingharrington", min_node_size = 5, nsplits = 30, seed = 102,
  fh_weights_a = 0:2
)
honest_tree <- jftree(
  MM ~ x + z, paths, features,
  splitrule = "flemingharrington", min_node_size = 4, nsplits = 20,
  honest = TRUE, num_event_times = 12, seed = 103,
  fh_weights_a = c(0, 1, 2), fh_weights_b = c(2, 1, 0)
)
categorical_tree <- jftree(
  MM ~ group, paths, features,
  splitrule = "flemingharrington", min_node_size = 5, nsplits = 20, seed = 104,
  fh_weights_a = c(0.25, 0.75, 1.25), fh_weights_b = c(1.25, 0.75, 0.25)
)
stopifnot(
  identical(custom_tree$fh.weights.a, c(0, 0.5, 2)),
  identical(custom_tree$fh.weights.b, c(1, 0, 3)),
  identical(one_default_tree$fh.weights.a, c(0, 1, 2)),
  identical(one_default_tree$fh.weights.b, rep(1, 3)),
  custom_tree$num.nodes > 1,
  honest_tree$num.nodes >= 1,
  categorical_tree$num.nodes > 1,
  all(is.finite(unlist(custom_tree$predictions))),
  all(is.finite(unlist(honest_tree$predictions))),
  all(is.finite(unlist(categorical_tree$predictions)))
)

# Independently reconstruct all pooled jumps, risk sets, occupation
# probabilities, and candidate scores for a root node with several transition
# types. The selected C++ threshold must equal the direct formula evaluation.
formula_paths <- lapply(seq_len(32L), function(i) {
  switch(
    as.character(i %% 4L),
    "0" = list(times = c(0, 0.8 + i / 100, 3.0 + i / 100), states = c(1L, 2L, 3L)),
    "1" = list(times = c(0, 1.5 + i / 100), states = c(1L, 3L)),
    "2" = list(times = c(0, 0.7 + i / 100, 2.5 + i / 100), states = c(2L, 1L, 3L)),
    "3" = list(times = c(0, 2.0 + i / 100), states = c(2L, 3L))
  )
})
formula_feature <- (11 * seq_len(32L)) %% 37
formula_features <- data.frame(x = formula_feature)
formula_a <- c(0.25, 1.5, 4)
formula_b <- c(2, 0.5, 3)
expected_threshold <- manual_fh_root_threshold(
  formula_paths, formula_feature, min_node_size = 5,
  a = formula_a, b = formula_b
)
formula_tree <- jftree(
  MM ~ x, formula_paths, formula_features,
  splitrule = "flemingharrington", min_node_size = 5, nsplits = 100, seed = 150,
  fh_weights_a = formula_a, fh_weights_b = formula_b
)
actual_threshold <- tree_table(formula_tree)$thresholds[[1]][1]
stopifnot(identical(actual_threshold, expected_threshold))

# With one possible transition and a=b=0, every FH weight is one. The FH
# statistic is |Z| while log-rank is Z^2, so their split rankings and fitted
# trees must agree. This also exercises 0^0 at occupation probabilities 0/1.
n_two_state <- 48L
two_state_paths <- lapply(seq_len(n_two_state), function(i) {
  if (i %% 6L == 0L) {
    list(times = c(0, 1.4 + i / 100), states = c(1L, 1L))
  } else {
    list(times = c(0, 0.5 + ((7L * i) %% n_two_state) / 20), states = c(1L, 2L))
  }
})
two_state_features <- data.frame(
  x = seq_len(n_two_state),
  z = cos(seq_len(n_two_state))
)
zero_fh_tree <- jftree(
  MM ~ x + z, two_state_paths, two_state_features,
  splitrule = "flemingharrington", min_node_size = 5, nsplits = 100, seed = 200,
  fh_weights_a = c(0, 0), fh_weights_b = c(0, 0)
)
logrank_tree <- jftree(
  MM ~ x + z, two_state_paths, two_state_features,
  splitrule = "logrank", min_node_size = 5, nsplits = 100, seed = 200
)
stopifnot(same_tree(zero_fh_tree, logrank_tree))

# State 2 has no outgoing transition in this data. Changing only its exponents
# must therefore leave every split unchanged, demonstrating source-state
# indexing rather than transition or destination-state indexing.
unused_state_low <- jftree(
  MM ~ x + z, two_state_paths, two_state_features,
  splitrule = "flemingharrington", min_node_size = 5, nsplits = 100, seed = 201,
  fh_weights_a = c(0.5, 0), fh_weights_b = c(1, 0)
)
unused_state_high <- jftree(
  MM ~ x + z, two_state_paths, two_state_features,
  splitrule = "flemingharrington", min_node_size = 5, nsplits = 100, seed = 201,
  fh_weights_a = c(0.5, 1e6), fh_weights_b = c(1, 1e6)
)
stopifnot(same_tree(unused_state_low, unused_state_high))

# Very large but finite exponents used to underflow every raw pow() weight to
# zero. Log-scale normalisation must retain an informative finite statistic.
extreme_tree <- jftree(
  MM ~ x + z, two_state_paths, two_state_features,
  splitrule = "flemingharrington", min_node_size = 5, nsplits = 100, seed = 202,
  fh_weights_a = c(.Machine$double.xmax, 0), fh_weights_b = c(0, 0)
)
stopifnot(
  extreme_tree$num.nodes > 1,
  all(is.finite(unlist(extreme_tree$predictions)))
)

# Forest plumbing: custom values reach all trees, results are independent of
# worker count, and honest/double-bootstrap/lazy-prediction fitting succeeds.
forest_one_worker <- jfforest(
  MM ~ x + group, paths, features,
  splitrule = "flemingharrington", ntrees = 4, min_node_size = 5, nsplits = 20,
  seed = 300, nworkers = 1,
  fh_weights_a = c(0, 0.5, 2), fh_weights_b = c(1, 0, 3)
)
forest_two_workers <- jfforest(
  MM ~ x + group, paths, features,
  splitrule = "flemingharrington", ntrees = 4, min_node_size = 5, nsplits = 20,
  seed = 300, nworkers = 2,
  fh_weights_a = c(0, 0.5, 2), fh_weights_b = c(1, 0, 3)
)
lazy_honest_forest <- jfforest(
  MM ~ x + z, paths, features,
  splitrule = "flemingharrington", ntrees = 3, min_node_size = 3, nsplits = 15,
  honest = TRUE, double_bootstrap = TRUE, sample_rate = 0.8,
  save_predictions = FALSE, seed = 301, nworkers = 2,
  fh_weights_a = c(0, 1, 2), fh_weights_b = c(2, 1, 0)
)
lazy_predictions <- jfforest.predict(
  lazy_honest_forest, compute_initial = TRUE, compute_censoring = TRUE
)
stopifnot(
  identical(forest_one_worker$fh.weights.a, c(0, 0.5, 2)),
  identical(forest_one_worker$fh.weights.b, c(1, 0, 3)),
  isTRUE(all.equal(forest_one_worker$predictions, forest_two_workers$predictions, tolerance = 0)),
  all(is.finite(unlist(lazy_predictions$predictions))),
  all(is.finite(unlist(lazy_predictions$predictions.init))),
  all(is.finite(lazy_predictions$censoring))
)

# Validation: length, type, finiteness, sign, and split-rule relevance.
invalid_values <- list(
  list(value = c(-1, 0, 0), index = 1L),
  list(value = c(0, NA_real_, 0), index = 2L),
  list(value = c(0, NaN, 0), index = 2L),
  list(value = c(0, Inf, 0), index = 2L),
  list(value = c(0, -Inf, 0), index = 2L)
)
for (invalid in invalid_values) {
  assert_error(
    jftree(
      MM ~ x, paths, features,
      splitrule = "flemingharrington", min_node_size = 5,
      fh_weights_a = invalid$value
    ),
    paste0("fh_weights_a[", invalid$index, "] must be finite and non-negative")
  )
  assert_error(
    jftree(
      MM ~ x, paths, features,
      splitrule = "flemingharrington", min_node_size = 5,
      fh_weights_b = invalid$value
    ),
    paste0("fh_weights_b[", invalid$index, "] must be finite and non-negative")
  )
}

assert_error(
  jftree(
    MM ~ x, paths, features,
    splitrule = "flemingharrington", min_node_size = 5,
    fh_weights_a = c(0, 1)
  ),
  "fh_weights_a must have length 3 (one exponent per state); got 2"
)
assert_error(
  jfforest(
    MM ~ x, paths, features,
    splitrule = "flemingharrington", ntrees = 2, min_node_size = 5,
    fh_weights_b = rep(1, 4)
  ),
  "fh_weights_b must have length 3 (one exponent per state); got 4"
)
assert_error(
  jftree(
    MM ~ x, paths, features,
    splitrule = "flemingharrington", min_node_size = 5,
    fh_weights_a = c("0", "1", "2")
  ),
  "fh_weights_a must be a numeric vector"
)
assert_error(
  jftree(
    MM ~ x, paths, features,
    splitrule = "flemingharrington", min_node_size = 5,
    fh_weights_a = matrix(c(0, 1, 2), ncol = 1)
  ),
  "fh_weights_a must be a numeric vector"
)
assert_error(
  jftree(
    MM ~ x, paths, features,
    splitrule = "logrank", min_node_size = 5,
    fh_weights_a = rep(1, 3)
  ),
  "fh_weights_a and fh_weights_b may only be supplied"
)
assert_error(
  jfforest(
    MM ~ x, paths, features,
    splitrule = "petoprentice", ntrees = 2, min_node_size = 5,
    fh_weights_b = rep(1, 3)
  ),
  "fh_weights_a and fh_weights_b may only be supplied"
)
ordinary_data <- data.frame(y = seq_len(12), x = seq_len(12))
assert_error(
  jftree(y ~ x, ordinary_data, fh_weights_a = 1),
  "fh_weights_a and fh_weights_b are only valid for multi-state models"
)
assert_error(
  jfforest(y ~ x, ordinary_data, ntrees = 2, fh_weights_b = 1),
  "fh_weights_a and fh_weights_b are only valid for multi-state models"
)

cat("All Fleming--Harrington tests passed.\n")

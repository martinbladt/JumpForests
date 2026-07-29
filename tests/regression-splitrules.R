# Standalone integration, formula, and validation tests for regression
# splitting rules. Run against an installed JumpForests package with:
# Rscript tests/regression-splitrules.R

library(JumpForests)

assert_error <- function(expression, pattern = NULL) {
  message <- tryCatch(
    {
      force(expression)
      NULL
    },
    error = function(condition) conditionMessage(condition)
  )
  if (is.null(message)) {
    stop("Expected an error", if (!is.null(pattern)) paste0(" containing: ", pattern))
  }
  if (!is.null(pattern) && !grepl(pattern, message, fixed = TRUE)) {
    stop("Expected error containing '", pattern, "', got: ", message)
  }
  invisible(message)
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

root_threshold <- function(tree) {
  table <- tree_table(tree)
  if (length(table$left.daughters) == 0L || table$left.daughters[1] == 0L) {
    stop("Expected the fitted tree to split at its root")
  }
  table$thresholds[[1]]
}

fit_rule_tree <- function(data, splitrule, splitrule_par = NULL, ...) {
  arguments <- list(
    formula = y ~ x,
    data = data,
    splitrule = splitrule,
    mtry = 1,
    min_node_size = 7,
    nsplits = 100,
    seed = 719
  )
  if (!is.null(splitrule_par)) {
    arguments$splitrule_par <- splitrule_par
  }
  overrides <- list(...)
  if (length(overrides) > 0L) {
    arguments[names(overrides)] <- overrides
  }
  do.call(jftree, arguments)
}

# Independent unit-deviance and Huber implementations. These intentionally
# operate on complete response vectors rather than the sufficient-statistic
# formulae used by the C++ split sweep.
x_log_ratio <- function(x, reference) {
  result <- numeric(length(x))
  positive <- x > 0
  if (any(positive)) {
    if (reference <= 0) {
      return(rep(Inf, length(x)))
    }
    result[positive] <- x[positive] * log(x[positive] / reference)
  }
  result
}

huber_center <- function(y, delta) {
  # The minimizer solves sum(clamp(y_i - center, -delta, delta)) = 0.
  lower <- min(y) - delta
  upper <- max(y) + delta
  for (iteration in seq_len(120L)) {
    middle <- lower + (upper - lower) / 2
    score <- sum(pmax(-delta, pmin(delta, y - middle)))
    if (score > 0) {
      lower <- middle
    } else {
      upper <- middle
    }
  }
  lower + (upper - lower) / 2
}

node_loss <- function(y, splitrule, splitrule_par = NULL) {
  mu <- mean(y)
  if (splitrule == "mse" || splitrule == "variance") {
    return(sum((y - mu)^2))
  }
  if (splitrule == "mae") {
    return(sum(abs(y - median(y))))
  }
  if (splitrule == "binomial") {
    return(2 * sum(
      x_log_ratio(y, mu) +
        x_log_ratio(1 - y, 1 - mu)
    ))
  }
  if (splitrule == "negativebinomial") {
    size <- splitrule_par
    return(2 * sum(
      x_log_ratio(y, mu) -
        (y + size) * log((y + size) / (mu + size))
    ))
  }
  if (splitrule == "poisson") {
    return(2 * sum(x_log_ratio(y, mu) - (y - mu)))
  }
  if (splitrule == "gamma") {
    return(2 * sum(-log(y) + log(mu) + (y - mu) / mu))
  }
  if (splitrule == "inversegaussian") {
    return(sum((y / mu - 1)^2 / y))
  }
  if (splitrule == "tweedie") {
    power <- splitrule_par
    if (power == 0) {
      return(sum((y - mu)^2))
    }
    if (power == 1) {
      return(2 * sum(x_log_ratio(y, mu) - (y - mu)))
    }
    if (power == 2) {
      return(2 * sum(-log(y) + log(mu) + (y - mu) / mu))
    }
    if (power == 3) {
      return(sum((y / mu - 1)^2 / y))
    }
    # For xi < 0, the Tweedie mean parameter is non-negative. The node
    # optimum is the sample mean when positive and the boundary zero otherwise.
    if (power < 0) {
      mu <- max(mu, 0)
    }
    return(2 * sum(
      pmax(y, 0)^(2 - power) / ((1 - power) * (2 - power)) -
        y * mu^(1 - power) / (1 - power) +
        mu^(2 - power) / (2 - power)
    ))
  }
  if (splitrule == "huber") {
    center <- huber_center(y, splitrule_par)
    residual <- abs(y - center)
    return(sum(ifelse(
      residual <= splitrule_par,
      residual^2,
      splitrule_par * (2 * residual - splitrule_par)
    )))
  }
  stop("Unknown oracle split rule: ", splitrule)
}

manual_root_split <- function(y, x, min_node_size, splitrule,
                              splitrule_par = NULL) {
  candidates <- sort(unique(x[!is.na(x)]))
  scores <- rep(-Inf, length(candidates))
  parent_loss <- node_loss(y, splitrule, splitrule_par)
  for (candidate_id in seq_along(candidates)) {
    goes_left <- !is.na(x) & x <= candidates[candidate_id]
    if (sum(goes_left) < min_node_size ||
        sum(!goes_left) < min_node_size) {
      next
    }
    scores[candidate_id] <- parent_loss -
      node_loss(y[goes_left], splitrule, splitrule_par) -
      node_loss(y[!goes_left], splitrule, splitrule_par)
  }
  if (!any(is.finite(scores))) {
    stop("The oracle fixture has no valid root split")
  }
  best_id <- which.max(scores)
  finite_scores <- sort(scores[is.finite(scores)], decreasing = TRUE)
  gap <- if (length(finite_scores) > 1L) {
    finite_scores[1] - finite_scores[2]
  } else {
    Inf
  }
  if (gap <= 1e-10) {
    stop("The oracle fixture does not have a unique best root split")
  }
  list(
    threshold = candidates[best_id],
    score = scores[best_id],
    gap = gap,
    scores = setNames(scores, candidates)
  )
}

manual_categorical_root_split <- function(y, x, min_node_size, splitrule,
                                          splitrule_par = NULL) {
  values <- sort(unique(x[!is.na(x)]))
  has_missing_values <- anyNA(x)
  if (length(values) == 0L ||
      (length(values) < 2L && !has_missing_values)) {
    stop("The categorical oracle has no possible split")
  }
  # Complementary subsets are equivalent only when every observation belongs
  # to an observed category. Missing values are fixed in the right daughter,
  # so both orientations must be considered when they are present.
  num_masks <- if (has_missing_values) {
    2^length(values) - 1L
  } else {
    2^(length(values) - 1L) - 1L
  }
  masks <- seq_len(num_masks)
  scores <- rep(-Inf, length(masks))
  subsets <- vector("list", length(masks))
  parent_loss <- node_loss(y, splitrule, splitrule_par)
  for (mask_id in seq_along(masks)) {
    selected <- bitwAnd(
      masks[mask_id],
      bitwShiftL(1L, seq_along(values) - 1L)
    ) != 0L
    subsets[[mask_id]] <- values[selected]
    goes_left <- !is.na(x) & x %in% subsets[[mask_id]]
    if (sum(goes_left) < min_node_size ||
        sum(!goes_left) < min_node_size) {
      next
    }
    scores[mask_id] <- parent_loss -
      node_loss(y[goes_left], splitrule, splitrule_par) -
      node_loss(y[!goes_left], splitrule, splitrule_par)
  }
  best_id <- which.max(scores)
  finite_scores <- sort(scores[is.finite(scores)], decreasing = TRUE)
  if (length(finite_scores) < 1L) {
    stop("The categorical oracle fixture has no valid split")
  }
  if (length(finite_scores) > 1L &&
      finite_scores[1] - finite_scores[2] <= 1e-10) {
    stop("The categorical oracle fixture has no unique best split")
  }
  list(subset = subsets[[best_id]], score = scores[best_id])
}

# Each fixture has a unique optimum under its requested rule and a different
# MSE optimum. This checks every string-to-small-integer mapping indirectly:
# an unimplemented switch arm or fallback to MSE selects the wrong threshold.
root_cases <- list(
  binomial = list(
    y = c(
      0.25, 0.98, 1, 0.9, 0.25, 0.5, 0.1, 0.5, 0.9, 0.25,
      0.9, 0, 0.75, 0.25, 0.1, 0, 0.1, 0.5, 0.1, 0
    ),
    par = NULL,
    threshold = 13
  ),
  negativebinomial = list(
    y = c(2, 3, 2, 6, 6, 3, 4, 3, 1, 0, 2, 1, 4, 1, 1, 3, 5, 4, 1, 6),
    par = 2,
    threshold = 8
  ),
  poisson = list(
    y = c(8, 5, 6, 4, 8, 3, 7, 5, 3, 4, 6, 6, 2, 3, 6, 7, 3, 6, 1, 3),
    par = NULL,
    threshold = 8
  ),
  gamma = list(
    y = c(
      0.356, 18.785, 1.063, 0.01, 0.282, 0.336, 0.036, 37.438, 3.414, 0.077,
      4.283, 4.226, 1.444, 0.525, 0.265, 0.21, 1.28, 0.222, 2.543, 4.045
    ),
    par = NULL,
    threshold = 9
  ),
  inversegaussian = list(
    y = c(
      0.154, 450.822, 0.38, 0.248, 3.162, 0.028, 14.044, 3.783, 10.71, 1.425,
      0.68, 0.049, 0.191, 0.533, 1.021, 0.731, 0.009, 1.294, 0.231, 12.255
    ),
    par = NULL,
    threshold = 9
  ),
  tweedie = list(
    y = c(
      0.045, 0.149, 0.206, 2.733, 1.051, 0.467, 0.006, 1.638, 7.126, 2.606,
      0.22, 5.064, 0.102, 1.105, 14.638, 0.27, 2.413, 0.444, 0.193, 19.1
    ),
    par = 1.5,
    threshold = 8
  ),
  huber = list(
    y = c(
      -1.442, 0.285, 1.045, 0.729, -0.06, -0.278, -0.148, -0.545, -0.236,
      0.531, 0.948, -2.355, 0.09, 2.124, 0.968, -0.233, 20.243, -24.948,
      -0.606, -19.548
    ),
    par = 1.2,
    threshold = 11
  )
)

root_trees <- vector("list", length(root_cases))
names(root_trees) <- names(root_cases)
for (splitrule in names(root_cases)) {
  current <- root_cases[[splitrule]]
  x <- seq_along(current$y)
  expected <- manual_root_split(
    current$y, x, min_node_size = 7,
    splitrule = splitrule, splitrule_par = current$par
  )
  mse_expected <- manual_root_split(
    current$y, x, min_node_size = 7, splitrule = "mse"
  )
  stopifnot(
    isTRUE(all.equal(expected$threshold, current$threshold, tolerance = 0)),
    expected$threshold != mse_expected$threshold,
    expected$score > 0
  )

  tree <- fit_rule_tree(
    data.frame(y = current$y, x = x),
    splitrule = splitrule,
    splitrule_par = current$par
  )
  actual <- root_threshold(tree)
  goes_left <- x <= expected$threshold
  expected_predictions <- ifelse(
    goes_left,
    mean(current$y[goes_left]),
    mean(current$y[!goes_left])
  )
  stopifnot(
    tree$num.nodes == 3,
    length(actual) == 1L,
    isTRUE(all.equal(actual, expected$threshold, tolerance = 0)),
    isTRUE(all.equal(tree$predictions, expected_predictions, tolerance = 1e-12)),
    identical(tree$splitrule, splitrule),
    all(is.finite(tree$predictions))
  )
  if (is.null(current$par)) {
    stopifnot(is.null(tree$splitrule.par))
  } else {
    stopifnot(
      isTRUE(all.equal(tree$splitrule.par, current$par, tolerance = 0))
    )
  }
  root_trees[[splitrule]] <- tree
}

# Parameter values must reach the criterion, not merely fitted-object metadata.
# Each listed value has a distinct independently computed optimum.
parameter_cases <- list(
  negativebinomial = list(
    y = c(2, 0, 1, 5, 1, 4, 4, 1, 3, 4, 5, 0, 1, 6, 4, 2, 1, 4, 3, 1),
    values = c(0.1, 100),
    thresholds = c(8, 13)
  ),
  tweedie = list(
    y = c(
      0.181, 1.713, 1.125, 0.533, 0.798, 0.284, 0.209, 5.32, 0.143, 0.398,
      0.987, 6.194, 36.944, 3.115, 0.357, 17.997, 13.95, 0.411, 2.415, 32.972
    ),
    values = c(-1, 1.5, 3),
    thresholds = c(12, 11, 7)
  ),
  huber = list(
    y = c(
      -0.679, 0.425, -1.647, -0.737, 0.181, -0.905, -1.114, 0.699, -0.609,
      0.922, 1.001, -0.151, -0.352, 0.701, -3.258, -0.498, 3.979, 23.793,
      14.695, -7.646
    ),
    values = c(0.1, 3),
    thresholds = c(7, 9)
  )
)

for (splitrule in names(parameter_cases)) {
  current <- parameter_cases[[splitrule]]
  x <- seq_along(current$y)
  actual_thresholds <- numeric(length(current$values))
  for (parameter_id in seq_along(current$values)) {
    parameter <- current$values[parameter_id]
    expected <- manual_root_split(
      current$y, x, 7, splitrule, parameter
    )
    tree <- fit_rule_tree(
      data.frame(y = current$y, x = x),
      splitrule = splitrule,
      splitrule_par = parameter,
      seed = 720 + parameter_id
    )
    actual_thresholds[parameter_id] <- root_threshold(tree)
    stopifnot(
      isTRUE(all.equal(
        expected$threshold,
        current$thresholds[parameter_id],
        tolerance = 0
      )),
      isTRUE(all.equal(
        actual_thresholds[parameter_id],
        expected$threshold,
        tolerance = 0
      )),
      isTRUE(all.equal(tree$splitrule.par, parameter, tolerance = 0))
    )
  }
  stopifnot(length(unique(actual_thresholds)) == length(actual_thresholds))
}

# Boundary identities of the exponential-dispersion deviances, plus the
# existing variance alias and the all-quadratic Huber regime.
identity_data <- data.frame(
  y = parameter_cases$negativebinomial$y + 1,
  x = seq_along(parameter_cases$negativebinomial$y)
)
identity_arguments <- list(
  formula = y ~ x,
  data = identity_data,
  mtry = 1,
  min_node_size = 3,
  nsplits = 100,
  seed = 811
)
identity_mse <- do.call(jftree, c(identity_arguments, list(splitrule = "mse")))
identity_variance <- do.call(
  jftree, c(identity_arguments, list(splitrule = "variance"))
)
identity_tweedie_zero <- do.call(
  jftree,
  c(identity_arguments, list(splitrule = "tweedie", splitrule_par = 0))
)
identity_poisson <- do.call(
  jftree, c(identity_arguments, list(splitrule = "poisson"))
)
identity_tweedie_one <- do.call(
  jftree,
  c(identity_arguments, list(splitrule = "tweedie", splitrule_par = 1))
)
identity_gamma <- do.call(
  jftree, c(identity_arguments, list(splitrule = "gamma"))
)
identity_tweedie_two <- do.call(
  jftree,
  c(identity_arguments, list(splitrule = "tweedie", splitrule_par = 2))
)
identity_inverse_gaussian <- do.call(
  jftree, c(identity_arguments, list(splitrule = "inversegaussian"))
)
identity_tweedie_three <- do.call(
  jftree,
  c(identity_arguments, list(splitrule = "tweedie", splitrule_par = 3))
)
identity_huber <- do.call(
  jftree,
  c(
    identity_arguments,
    list(
      splitrule = "huber",
      splitrule_par = diff(range(identity_data$y)) + 1
    )
  )
)
signed_identity_data <- data.frame(
  y = root_cases$huber$y,
  x = seq_along(root_cases$huber$y)
)
signed_identity_mse <- jftree(
  y ~ x,
  signed_identity_data,
  splitrule = "mse",
  min_node_size = 3,
  nsplits = 100,
  seed = 812
)
signed_identity_tweedie_zero <- jftree(
  y ~ x,
  signed_identity_data,
  splitrule = "tweedie",
  splitrule_par = 0,
  min_node_size = 3,
  nsplits = 100,
  seed = 812
)
stopifnot(
  same_tree(identity_mse, identity_variance),
  same_tree(identity_mse, identity_tweedie_zero),
  same_tree(identity_poisson, identity_tweedie_one),
  same_tree(identity_gamma, identity_tweedie_two),
  same_tree(identity_inverse_gaussian, identity_tweedie_three),
  same_tree(identity_mse, identity_huber),
  same_tree(signed_identity_mse, signed_identity_tweedie_zero)
)

# Powers adjacent to the Tweedie boundary cases must approach their limits
# without cancellation changing the split ordering.
tweedie_poisson_limit_data <- data.frame(
  y = c(14, 15, 11, 9, 7, 10, 11, 8, 15, 12,
        7, 14, 11, 6, 9, 16, 12, 4, 12, 4),
  x = c(1, 8, 6, 2, 10, 1, 4, 8, 10, 1,
        8, 9, 8, 9, 1, 3, 9, 3, 3, 9)
)
tweedie_limit_arguments <- list(
  formula = y ~ x,
  data = tweedie_poisson_limit_data,
  mtry = 1,
  min_node_size = 4,
  nsplits = 100,
  seed = 7
)
poisson_limit_tree <- do.call(
  jftree,
  c(tweedie_limit_arguments, list(splitrule = "poisson"))
)
tweedie_above_one_tree <- do.call(
  jftree,
  c(
    tweedie_limit_arguments,
    list(
      splitrule = "tweedie",
      splitrule_par = 1 + .Machine$double.eps
    )
  )
)
tweedie_zero_limit_data <- data.frame(
  y = c(
    1.00000000302896, 1.00000000638205, 1.00000000283243,
    0.999999999279545, 1.00000000774107, 1.00000000126497,
    0.999999995555445, 1.00000000338067, 1.00000000208382,
    0.999999993631503, 1.0000000096889, 0.999999990527957,
    1.00000000431478, 0.999999990098986, 0.999999993971436,
    1.00000000863122, 1.00000000025811, 1.00000000422199,
    1.0000000091304, 1.00000000637512
  ),
  x = seq_len(20)
)
tweedie_zero_limit_mse <- jftree(
  y ~ x,
  tweedie_zero_limit_data,
  splitrule = "mse",
  mtry = 1,
  min_node_size = 7,
  nsplits = 100,
  seed = 8
)
tweedie_below_zero_tree <- jftree(
  y ~ x,
  tweedie_zero_limit_data,
  splitrule = "tweedie",
  splitrule_par = -.Machine$double.xmin,
  mtry = 1,
  min_node_size = 7,
  nsplits = 100,
  seed = 8
)
stopifnot(
  identical(root_threshold(poisson_limit_tree), 8),
  identical(root_threshold(tweedie_above_one_tree), 8),
  identical(root_threshold(tweedie_zero_limit_mse), 11),
  identical(root_threshold(tweedie_below_zero_tree), 11)
)

# The negative-power interior decomposition must avoid subtracting powers of
# nearly equal, large node means.
negative_power_baseline <- 1e15
negative_power_data <- data.frame(
  y = negative_power_baseline +
    c(-17, 18, -20, 13, 2, -7, -3, 12, 0, 0,
      -11, -14, -12, -6, 0, 16, 20, 4, 16, 16),
  x = seq_len(20)
)
negative_power_tree <- jftree(
  y ~ x,
  negative_power_data,
  splitrule = "tweedie",
  splitrule_par = -1,
  mtry = 1,
  min_node_size = 7,
  nsplits = 100,
  seed = 9
)
stopifnot(identical(root_threshold(negative_power_tree), 13))

# Tweedie scores are homogeneous in the response. Normalising away the common
# parent scale preserves ordering for large negative powers.
fit_homogeneous_tweedie <- function(scale) {
  jftree(
    y ~ x,
    data.frame(
      y = root_cases$tweedie$y * scale,
      x = seq_along(root_cases$tweedie$y)
    ),
    splitrule = "tweedie",
    splitrule_par = -18,
    mtry = 1,
    min_node_size = 7,
    nsplits = 100,
    seed = 10
  )
}
stopifnot(all(vapply(
  c(1, 1e-250, 1e-300),
  function(scale) identical(root_threshold(fit_homogeneous_tweedie(scale)), 13),
  logical(1)
)))

# Exact categorical splitting with missing values. Factor levels are encoded
# 1,...,4 and missing values always join the right daughter.
categorical_data <- data.frame(
  y = c(
    7, 6, 8, 7, 9,
    0, 1, 0, 2, 1,
    8, 7, 9, 6, 8,
    1, 0, 2, 1, 0
  ),
  group = factor(rep(letters[1:4], each = 5))
)
categorical_data$group[c(2, 18)] <- NA
encoded_group <- as.integer(categorical_data$group)
expected_category <- manual_categorical_root_split(
  categorical_data$y, encoded_group, min_node_size = 7,
  splitrule = "poisson"
)
categorical_tree <- jftree(
  y ~ group,
  categorical_data,
  splitrule = "poisson",
  mtry = 1,
  min_node_size = 7,
  nsplits = 100,
  seed = 901
)
actual_category <- sort(root_threshold(categorical_tree))
goes_left_category <- !is.na(encoded_group) &
  encoded_group %in% expected_category$subset
left_mean <- mean(categorical_data$y[goes_left_category])
right_mean <- mean(categorical_data$y[!goes_left_category])
new_groups <- data.frame(
  group = factor(c("a", "b", "c", "d", NA), levels = letters[1:4])
)
categorical_predictions <- jftree.predict(categorical_tree, new_groups)
new_group_ids <- c(1:4, NA_integer_)
expected_categorical_predictions <- ifelse(
  !is.na(new_group_ids) & new_group_ids %in% expected_category$subset,
  left_mean,
  right_mean
)
stopifnot(
  categorical_tree$num.nodes == 3,
  isTRUE(all.equal(
    actual_category, sort(expected_category$subset), tolerance = 0
  )),
  isTRUE(all.equal(
    as.numeric(categorical_predictions),
    expected_categorical_predictions,
    tolerance = 1e-12
  ))
)

# Missing values fixed in the right daughter break categorical complement
# symmetry. The best split here requires the orientation that was historically
# pruned (B left, A and missing right).
asymmetric_missing_data <- data.frame(
  y = c(rep(0, 3), rep(10, 3), rep(0, 3)),
  x = factor(c(rep("A", 3), rep("B", 3), rep(NA, 3)),
             levels = c("A", "B"))
)
asymmetric_encoded <- as.integer(asymmetric_missing_data$x)
asymmetric_expected <- manual_categorical_root_split(
  asymmetric_missing_data$y,
  asymmetric_encoded,
  min_node_size = 3,
  splitrule = "poisson"
)
asymmetric_tree <- jftree(
  y ~ x,
  asymmetric_missing_data,
  splitrule = "poisson",
  mtry = 1,
  min_node_size = 3,
  nsplits = 100,
  seed = 903
)
stopifnot(
  identical(asymmetric_expected$subset, 2L),
  identical(root_threshold(asymmetric_tree), 2),
  isTRUE(all.equal(
    as.numeric(jftree.predict(
      asymmetric_tree,
      data.frame(x = factor(c("A", "B", NA), levels = c("A", "B")))
    )),
    c(0, 10, 0),
    tolerance = 1e-12
  ))
)

# Continuous missing values use the same right-daughter convention in split
# scoring, training routing, and prediction.
missing_data <- data.frame(
  y = root_cases$inversegaussian$y,
  x = seq_along(root_cases$inversegaussian$y)
)
missing_data$x[c(2, 17)] <- NA
expected_missing <- manual_root_split(
  missing_data$y, missing_data$x, 7, "inversegaussian"
)
missing_tree <- fit_rule_tree(missing_data, "inversegaussian", seed = 902)
missing_threshold <- root_threshold(missing_tree)
goes_left_missing <- !is.na(missing_data$x) &
  missing_data$x <= expected_missing$threshold
missing_left_mean <- mean(missing_data$y[goes_left_missing])
missing_right_mean <- mean(missing_data$y[!goes_left_missing])
missing_predictions <- jftree.predict(
  missing_tree,
  data.frame(x = c(
    min(missing_data$x, na.rm = TRUE),
    expected_missing$threshold,
    expected_missing$threshold + 0.5,
    NA_real_
  ))
)
stopifnot(
  missing_tree$num.nodes == 3,
  isTRUE(all.equal(missing_threshold, expected_missing$threshold, tolerance = 0)),
  isTRUE(all.equal(
    as.numeric(missing_predictions),
    c(
      missing_left_mean, missing_left_mean,
      missing_right_mean, missing_right_mean
    ),
    tolerance = 1e-12
  ))
)

# A single observed value can still split from missing values. Exercise both
# continuous and categorical paths, which share the fixed-right convention.
single_value_continuous_data <- data.frame(
  y = c(rep(10, 4), rep(0, 4)),
  x = c(rep(1, 4), rep(NA_real_, 4))
)
single_value_continuous_tree <- jftree(
  y ~ x,
  single_value_continuous_data,
  splitrule = "poisson",
  mtry = 1,
  min_node_size = 4,
  nsplits = 100,
  seed = 904
)
single_value_categorical_data <- data.frame(
  y = single_value_continuous_data$y,
  x = factor(c(rep("A", 4), rep(NA, 4)), levels = "A")
)
single_value_categorical_tree <- jftree(
  y ~ x,
  single_value_categorical_data,
  splitrule = "poisson",
  mtry = 1,
  min_node_size = 4,
  nsplits = 100,
  seed = 905
)
stopifnot(
  identical(root_threshold(single_value_continuous_tree), 1),
  identical(root_threshold(single_value_categorical_tree), 1),
  isTRUE(all.equal(
    as.numeric(jftree.predict(
      single_value_continuous_tree,
      data.frame(x = c(1, NA_real_))
    )),
    c(10, 0),
    tolerance = 1e-12
  )),
  isTRUE(all.equal(
    as.numeric(jftree.predict(
      single_value_categorical_tree,
      data.frame(x = factor(c("A", NA), levels = "A"))
    )),
    c(10, 0),
    tolerance = 1e-12
  ))
)

# Honest forests exercise tree and forest parameter plumbing, continuous and
# categorical features, bootstrap paths, and worker-count determinism for all
# seven new integer-coded rules.
forest_fits <- vector("list", length(root_cases))
names(forest_fits) <- names(root_cases)
for (rule_id in seq_along(root_cases)) {
  splitrule <- names(root_cases)[rule_id]
  current <- root_cases[[splitrule]]
  forest_data <- data.frame(
    y = current$y,
    x = (7 * seq_along(current$y)) %% 23,
    group = factor(rep(letters[1:4], length.out = length(current$y)))
  )
  forest_data$x[c(4, 15)] <- NA
  forest_data$group[c(6, 19)] <- NA
  arguments <- list(
    formula = y ~ x + group,
    data = forest_data,
    splitrule = splitrule,
    splitrule_par = current$par,
    mtry = 2,
    min_node_size = 2,
    nsplits = 8,
    ntrees = 3,
    honest = TRUE,
    double_bootstrap = TRUE,
    sample_rate = 0.8,
    seed = 1000 + rule_id
  )
  if (is.null(current$par)) {
    arguments$splitrule_par <- NULL
  }
  one_worker <- do.call(jfforest, c(arguments, list(nworkers = 1)))
  two_workers <- do.call(jfforest, c(arguments, list(nworkers = 2)))
  stopifnot(
    identical(one_worker$splitrule, splitrule),
    identical(one_worker$honest, TRUE),
    all(is.finite(one_worker$predictions)),
    isTRUE(all.equal(
      one_worker$predictions, two_workers$predictions, tolerance = 0
    )),
    isTRUE(all.equal(
      one_worker$oob.predictions, two_workers$oob.predictions, tolerance = 0
    )),
    isTRUE(all.equal(
      one_worker$avg.num.nodes, two_workers$avg.num.nodes, tolerance = 0
    ))
  )
  if (is.null(current$par)) {
    stopifnot(is.null(one_worker$splitrule.par))
  } else {
    stopifnot(
      isTRUE(all.equal(one_worker$splitrule.par, current$par, tolerance = 0)),
      isTRUE(all.equal(two_workers$splitrule.par, current$par, tolerance = 0))
    )
  }
  forest_fits[[splitrule]] <- one_worker
}

# With-replacement sampling creates duplicate observation IDs. Robust response
# ordering must preserve those multiplicities and remain worker deterministic.
swr_huber_arguments <- list(
  formula = y ~ x,
  data = data.frame(
    y = root_cases$huber$y,
    x = factor(rep(letters[1:4], length.out = length(root_cases$huber$y)))
  ),
  splitrule = "huber",
  splitrule_par = root_cases$huber$par,
  mtry = 1,
  min_node_size = 3,
  nsplits = 100,
  ntrees = 12,
  honest = TRUE,
  swr = TRUE,
  sample_rate = 1,
  seed = 1099,
  save_predictions = TRUE
)
swr_huber_one_worker <- do.call(
  jfforest,
  c(swr_huber_arguments, list(nworkers = 1))
)
swr_huber_two_workers <- do.call(
  jfforest,
  c(swr_huber_arguments, list(nworkers = 2))
)
stopifnot(
  isTRUE(all.equal(
    swr_huber_one_worker$predictions,
    swr_huber_two_workers$predictions,
    tolerance = 0
  )),
  isTRUE(all.equal(
    swr_huber_one_worker$oob.predictions,
    swr_huber_two_workers$oob.predictions,
    tolerance = 0
  )),
  all(is.finite(swr_huber_one_worker$predictions))
)

# Degenerate boundary means must not create NaN split scores or predictions.
zero_data <- data.frame(y = rep(0, 16), x = seq_len(16))
one_data <- data.frame(y = rep(1, 16), x = seq_len(16))
zero_binomial <- jftree(
  y ~ x, zero_data, splitrule = "binomial",
  min_node_size = 4, nsplits = 100, seed = 1101
)
one_binomial <- jftree(
  y ~ x, one_data, splitrule = "binomial",
  min_node_size = 4, nsplits = 100, seed = 1102
)
zero_poisson <- jftree(
  y ~ x, zero_data, splitrule = "poisson",
  min_node_size = 4, nsplits = 100, seed = 1103
)
zero_negative_binomial <- jftree(
  y ~ x, zero_data, splitrule = "negativebinomial", splitrule_par = 0.01,
  min_node_size = 4, nsplits = 100, seed = 1104
)
zero_tweedie <- jftree(
  y ~ x, zero_data, splitrule = "tweedie", splitrule_par = 1.5,
  min_node_size = 4, nsplits = 100, seed = 1105
)
stopifnot(
  all(zero_binomial$predictions == 0),
  all(one_binomial$predictions == 1),
  all(zero_poisson$predictions == 0),
  all(zero_negative_binomial$predictions == 0),
  all(zero_tweedie$predictions == 0),
  all(is.finite(c(
    zero_binomial$predictions,
    one_binomial$predictions,
    zero_poisson$predictions,
    zero_negative_binomial$predictions,
    zero_tweedie$predictions
  )))
)

# Success and failure masses are accumulated separately. A single fractional
# deficit near one remains visible even when it is below the ULP of sum(y).
rare_failure_data <- data.frame(
  y = {
    values <- rep(1, 4096)
    values[4096] <- 1 - 2^-53
    values
  },
  x = rep(1:4, each = 1024)
)
rare_failure_tree <- jftree(
  y ~ x,
  rare_failure_data,
  splitrule = "binomial",
  mtry = 1,
  min_node_size = 1024,
  nsplits = 100,
  seed = 1105
)
stopifnot(identical(root_threshold(rare_failure_tree), 3))

# Daughter sums retain their low component through mean differencing. This
# keeps a one-ULP Bernoulli perturbation visible even after aggregating 8,192
# observations per feature value.
sub_ulp_binomial_data <- data.frame(
  y = {
    values <- rep(0.5, 32768)
    values[32768] <- 0.5 + 2^-53
    values
  },
  x = rep(1:4, each = 8192)
)
sub_ulp_binomial_tree <- jftree(
  y ~ x,
  sub_ulp_binomial_data,
  splitrule = "binomial",
  mtry = 1,
  min_node_size = 8192,
  nsplits = 100,
  seed = 1105
)
stopifnot(identical(root_threshold(sub_ulp_binomial_tree), 3))

# Scores are evaluated on a scale that avoids subtracting huge node losses.
# Exercise values spanning 500 orders of magnitude and a finite Huber outlier
# without demanding a finite MSE diagnostic (squaring such residuals may
# legitimately overflow).
extreme_positive_data <- data.frame(
  y = rep(c(
    1e-250, 1e-100, 1, 1e100, 1e250, 1e50, 1e-50, 10
  ), 2),
  x = seq_len(16)
)
extreme_fits <- lapply(
  c("gamma", "inversegaussian"),
  function(splitrule) {
    jftree(
      y ~ x,
      extreme_positive_data,
      splitrule = splitrule,
      min_node_size = 4,
      nsplits = 100,
      seed = 1106
    )
  }
)
extreme_fits[[3]] <- jftree(
  y ~ x,
  extreme_positive_data,
  splitrule = "tweedie",
  splitrule_par = 3,
  min_node_size = 4,
  nsplits = 100,
  seed = 1107
)
extreme_huber <- jftree(
  y ~ x,
  data.frame(y = c(rep(0, 15), 1e300), x = seq_len(16)),
  splitrule = "huber",
  splitrule_par = 1,
  min_node_size = 4,
  nsplits = 100,
  seed = 1108
)
extreme_gamma_expected <- manual_root_split(
  extreme_positive_data$y,
  extreme_positive_data$x,
  4,
  "gamma"
)
extreme_inverse_gaussian_scores <- vapply(
  extreme_positive_data$x,
  function(threshold) {
    goes_left <- extreme_positive_data$x <= threshold
    if (sum(goes_left) < 4 || sum(!goes_left) < 4) {
      return(-Inf)
    }
    parent_mean <- mean(extreme_positive_data$y)
    left_mean <- mean(extreme_positive_data$y[goes_left])
    right_mean <- mean(extreme_positive_data$y[!goes_left])
    sum(goes_left) * (left_mean / parent_mean - 1)^2 / left_mean +
      sum(!goes_left) * (right_mean / parent_mean - 1)^2 / right_mean
  },
  numeric(1)
)
extreme_inverse_gaussian_expected <-
  extreme_positive_data$x[which.max(extreme_inverse_gaussian_scores)]
stopifnot(
  all(vapply(
    extreme_fits,
    function(tree) all(is.finite(tree$predictions)),
    logical(1)
  )),
  all(is.finite(extreme_huber$predictions)),
  isTRUE(all.equal(
    root_threshold(extreme_fits[[1]]),
    extreme_gamma_expected$threshold,
    tolerance = 0
  )),
  isTRUE(all.equal(
    root_threshold(extreme_fits[[2]]),
    extreme_inverse_gaussian_expected,
    tolerance = 0
  )),
  isTRUE(all.equal(
    root_threshold(extreme_fits[[3]]),
    extreme_inverse_gaussian_expected,
    tolerance = 0
  ))
)

# Direct loss differences keep the robust split ordering finite when a single
# response is far beyond the scale of every other observation.
robust_outlier_data <- data.frame(
  y = c(
    -17, 18, -20, 13, 2, -7, -3, 12, 0, 0,
    -11, -14, -12, -6, 0, 16, 20, 4, 16, 1e300
  ),
  x = seq_len(20)
)
robust_outlier_roots <- c(
  mae = root_threshold(jftree(
    y ~ x, robust_outlier_data, splitrule = "mae",
    mtry = 1, min_node_size = 7, nsplits = 100, seed = 1108
  )),
  huber = root_threshold(jftree(
    y ~ x, robust_outlier_data, splitrule = "huber", splitrule_par = 1,
    mtry = 1, min_node_size = 7, nsplits = 100, seed = 1108
  ))
)
stopifnot(all(robust_outlier_roots == 13))

# Huge positive and negative Tweedie powers are compared in the log domain.
tweedie_large_power_data <- data.frame(
  y = c(rep(2, 13), rep(1, 7)),
  x = seq_len(20)
)
tweedie_large_power_roots <- vapply(
  c(-1e5, 1e5),
  function(power) {
    root_threshold(jftree(
      y ~ x,
      tweedie_large_power_data,
      splitrule = "tweedie",
      splitrule_par = power,
      mtry = 1,
      min_node_size = 7,
      nsplits = 100,
      seed = 1108
    ))
  },
  numeric(1)
)
stopifnot(all(tweedie_large_power_roots == 13))

# Exact zero daughter deviances must remain zero in the log-domain fallback.
# The small non-zero gains at thresholds 3 and 4 are otherwise dominated by a
# rounding artefact at threshold 2.
tweedie_zero_gain_data <- data.frame(
  y = 2^52 + c(-1, 1, 2, 0, -1, -1),
  x = seq_len(6)
)
tweedie_zero_gain_roots <- vapply(
  c(1 + 2^-20, 10, -10, 1e5, 1000002),
  function(power) {
    root_threshold(jftree(
      y ~ x,
      tweedie_zero_gain_data,
      splitrule = "tweedie",
      splitrule_par = power,
      mtry = 1,
      min_node_size = 2,
      nsplits = 100,
      seed = 1108
    ))
  },
  numeric(1)
)
stopifnot(all(tweedie_zero_gain_roots == 4))

# At enormous finite powers, candidate ordering depends on a count coefficient
# that cannot be represented beside a scalar log exponent. The structured
# power-sum comparator retains that coefficient for both signs of xi.
tweedie_huge_positive_data <- data.frame(
  y = c(rep(1, 8), rep(2, 8)),
  x = seq_len(16)
)
tweedie_huge_negative_data <- data.frame(
  y = rev(tweedie_huge_positive_data$y),
  x = seq_len(16)
)
tweedie_huge_power_roots <- c(
  vapply(
    c(1e19, 1e308),
    function(power) {
      root_threshold(jftree(
        y ~ x,
        tweedie_huge_positive_data,
        splitrule = "tweedie",
        splitrule_par = power,
        mtry = 1,
        min_node_size = 4,
        nsplits = 100,
        seed = 1108
      ))
    },
    numeric(1)
  ),
  vapply(
    c(-1e19, -1e308),
    function(power) {
      root_threshold(jftree(
        y ~ x,
        tweedie_huge_negative_data,
        splitrule = "tweedie",
        splitrule_par = power,
        mtry = 1,
        min_node_size = 4,
        nsplits = 100,
        seed = 1108
      ))
    },
    numeric(1)
  )
)
stopifnot(all(tweedie_huge_power_roots == 8))

# The comparator anchor must be an attainable daughter mean rather than the
# raw response minimum; all feasible daughter means exceed one here.
tweedie_feasible_anchor_data <- data.frame(
  y = c(rep(3, 8), rep(2, 7), 1),
  x = seq_len(16)
)
tweedie_feasible_anchor_tree <- jftree(
  y ~ x,
  tweedie_feasible_anchor_data,
  splitrule = "tweedie",
  splitrule_par = 1e308,
  mtry = 1,
  min_node_size = 4,
  nsplits = 100,
  seed = 1108
)
stopifnot(identical(root_threshold(tweedie_feasible_anchor_tree), 12))

tweedie_exponent_data <- data.frame(
  y = c(rep(1e250, 7), 10^seq(-100, -250, length.out = 13)),
  x = seq_len(20)
)
tweedie_exponent_tree <- jftree(
  y ~ x,
  tweedie_exponent_data,
  splitrule = "tweedie",
  splitrule_par = 32,
  mtry = 1,
  min_node_size = 7,
  nsplits = 100,
  seed = 1108
)
stopifnot(identical(root_threshold(tweedie_exponent_tree), 13))

# Finite responses remain finite after node aggregation, even when a double
# sum would overflow. This also exercises every regression criterion family.
large_constant_data <- data.frame(y = rep(1e308, 10), x = seq_len(10))
large_constant_rules <- list(
  mse = NULL,
  mae = NULL,
  negativebinomial = 1e-30,
  poisson = NULL,
  gamma = NULL,
  inversegaussian = NULL,
  tweedie = 3,
  huber = 1e300
)
large_constant_fits <- lapply(
  names(large_constant_rules),
  function(splitrule) {
    arguments <- list(
      formula = y ~ x,
      data = large_constant_data,
      splitrule = splitrule,
      mtry = 1,
      min_node_size = 3,
      nsplits = 100,
      seed = 1109
    )
    parameter <- large_constant_rules[[splitrule]]
    if (!is.null(parameter)) {
      arguments$splitrule_par <- parameter
    }
    do.call(jftree, arguments)
  }
)
stopifnot(all(vapply(
  large_constant_fits,
  function(tree) {
    all(is.finite(tree$predictions)) &&
      all(tree$predictions == 1e308) &&
      isTRUE(all.equal(tree$mse.error, 0, tolerance = 0))
  },
  logical(1)
)))
large_constant_forest <- jfforest(
  y ~ x,
  data.frame(y = rep(1e308, 20), x = seq_len(20)),
  splitrule = "gamma",
  mtry = 1,
  min_node_size = 3,
  nsplits = 10,
  ntrees = 3,
  honest = FALSE,
  swr = FALSE,
  sample_rate = 0.8,
  seed = 1110,
  nworkers = 1
)
stopifnot(
  all(is.finite(large_constant_forest$predictions)),
  all(large_constant_forest$predictions == 1e308),
  all(is.na(large_constant_forest$oob.predictions) |
      is.finite(large_constant_forest$oob.predictions)),
  all(large_constant_forest$oob.predictions[
    !is.na(large_constant_forest$oob.predictions)
  ] == 1e308),
  isTRUE(all.equal(
    as.numeric(jfforest.predict(
      large_constant_forest,
      data.frame(x = c(1, 10, 20))
    )),
    rep(1e308, 3),
    tolerance = 0
  ))
)

# Multiplying both responses and a robust threshold by the same finite scale
# must preserve the split. Long-double candidate scores prevent underflow, and
# long-double median averaging prevents overflow.
robust_scale_y <- c(
  1.89789037, -1.32641363, 0.72195406, -1.39343063, 1.01189040,
  0.00592238, 1.33165530, -0.13505167, -0.65578325, -0.50390441,
  -1.32934747, 0.21177399, 1.20608492, -1.79831695, -0.55138493,
  -1.68167518, -0.34856673, 0.53170397, 0.76632381, -0.50015792
)
fit_scaled_huber <- function(scale) {
  jftree(
    y ~ x,
    data.frame(y = robust_scale_y * scale, x = seq_along(robust_scale_y)),
    splitrule = "huber",
    splitrule_par = 0.5 * scale,
    mtry = 1,
    min_node_size = 3,
    nsplits = 100,
    seed = 1111
  )
}
mae_scale_y <- c(
  1.272350, 1.408143, 1.066266, 1.596842, 1.550723,
  1.083540, 1.424513, 1.056670, 1.273541, 1.433631,
  1.199982, 1.385786, 1.021102, 1.691264, 1.071830,
  1.551049, 1.116609, 1.574879, 1.091799, 1.048989
)
fit_scaled_mae <- function(scale) {
  jftree(
    y ~ x,
    data.frame(y = mae_scale_y * scale, x = seq_along(mae_scale_y)),
    splitrule = "mae",
    mtry = 1,
    min_node_size = 3,
    nsplits = 100,
    seed = 1112
  )
}
stopifnot(
  identical(root_threshold(fit_scaled_huber(1)), 7),
  identical(root_threshold(fit_scaled_huber(1e-170)), 7),
  identical(root_threshold(fit_scaled_mae(1)), 12),
  identical(root_threshold(fit_scaled_mae(1e308)), 12)
)
quadratic_huber_data <- data.frame(
  y = c(-17, 18, -20, 13, 2, -7, -3, 12, 0, 0,
        -11, -14, -12, -6, 0, 16, 20, 4, 16, 16),
  x = seq_len(20)
)
quadratic_huber_tree <- jftree(
  y ~ x,
  quadratic_huber_data,
  splitrule = "huber",
  splitrule_par = 1e308,
  mtry = 1,
  min_node_size = 7,
  nsplits = 100,
  seed = 1112
)
stopifnot(identical(root_threshold(quadratic_huber_tree), 13))

# Far-ratio Gamma values have finite, ordered scores rather than a set of
# saturated infinities.
far_ratio_gamma_data <- data.frame(
  y = c(rep(1, 8), rep(1e100, 8)),
  x = seq_len(16)
)
far_ratio_gamma_tree <- jftree(
  y ~ x,
  far_ratio_gamma_data,
  splitrule = "gamma",
  mtry = 1,
  min_node_size = 4,
  nsplits = 100,
  seed = 1113
)
stopifnot(identical(root_threshold(far_ratio_gamma_tree), 8))

# Daughter sums are accumulated independently; a small daughter must not be
# recovered by subtracting a huge right sum from the parent.
imbalanced_sum_data <- data.frame(
  y = c(rep(1, 8), rep(1e19, 8)),
  x = seq_len(16)
)
imbalanced_rules <- list(
  gamma = NULL,
  inversegaussian = NULL,
  tweedie_gamma = 2,
  tweedie_inverse_gaussian = 3
)
imbalanced_trees <- lapply(
  names(imbalanced_rules),
  function(label) {
    splitrule <- if (startsWith(label, "tweedie")) "tweedie" else label
    arguments <- list(
      formula = y ~ x,
      data = imbalanced_sum_data,
      splitrule = splitrule,
      mtry = 1,
      min_node_size = 4,
      nsplits = 100,
      seed = 1114
    )
    parameter <- imbalanced_rules[[label]]
    if (!is.null(parameter)) {
      arguments$splitrule_par <- parameter
    }
    do.call(jftree, arguments)
  }
)
imbalanced_categorical_tree <- jftree(
  y ~ x,
  data.frame(
    y = imbalanced_sum_data$y,
    x = factor(rep(c("small", "large"), each = 8),
               levels = c("small", "large"))
  ),
  splitrule = "gamma",
  mtry = 1,
  min_node_size = 4,
  nsplits = 100,
  seed = 1115
)
stopifnot(
  all(vapply(
    imbalanced_trees,
    function(tree) identical(root_threshold(tree), 8),
    logical(1)
  )),
  identical(root_threshold(imbalanced_categorical_tree), 1)
)

# Compensated independent daughter sums also preserve cancellation between
# opposite-sign extremes for the existing MSE rule.
signed_cancellation_data <- data.frame(
  y = c(
    69, -36, -69, 87, -51, -31, 68, 87, 47, 84,
    57, 71, 17, 31, 44, -9, -51, 1e308, 56, -1e308
  ),
  x = seq_len(20)
)
signed_cancellation_tree <- jftree(
  y ~ x,
  signed_cancellation_data,
  splitrule = "mse",
  mtry = 1,
  min_node_size = 7,
  nsplits = 100,
  seed = 1115
)
stopifnot(identical(root_threshold(signed_cancellation_tree), 7))

# Preserve a one-unit perturbation after summing thousands of values at 2^52.
# The exact optimum is the split immediately before the perturbed fourth group.
sub_ulp_group_size <- 4096
sub_ulp_data <- data.frame(
  y = {
    values <- rep(2^52, 4 * sub_ulp_group_size)
    values[length(values)] <- 2^52 + 1
    values
  },
  x = rep(1:4, each = sub_ulp_group_size)
)
sub_ulp_rules <- list(
  mse = NULL,
  poisson = NULL,
  gamma = NULL,
  inversegaussian = NULL,
  tweedie = 3,
  huber = 2
)
sub_ulp_roots <- vapply(
  names(sub_ulp_rules),
  function(splitrule) {
    arguments <- list(
      formula = y ~ x,
      data = sub_ulp_data,
      splitrule = splitrule,
      mtry = 1,
      min_node_size = sub_ulp_group_size,
      nsplits = 100,
      seed = 1115
    )
    if (!is.null(sub_ulp_rules[[splitrule]])) {
      arguments$splitrule_par <- sub_ulp_rules[[splitrule]]
    }
    root_threshold(do.call(jftree, arguments))
  },
  numeric(1)
)
stopifnot(all(sub_ulp_roots == 3))

# A compensated sum must remain a two-component value until daughter means are
# differenced; collapsing it first loses the final +1 in this balanced fixture.
pair_ceiling_group_size <- 8192
pair_ceiling_data <- data.frame(
  y = {
    values <- rep(
      rep(c(0, 2^52), each = pair_ceiling_group_size / 2),
      4
    )
    values[length(values)] <- 2^52 + 1
    values
  },
  x = rep(1:4, each = pair_ceiling_group_size)
)
pair_ceiling_mse <- jftree(
  y ~ x,
  pair_ceiling_data,
  splitrule = "mse",
  mtry = 1,
  min_node_size = pair_ceiling_group_size,
  nsplits = 100,
  seed = 1115
)
pair_ceiling_huber <- jftree(
  y ~ x,
  pair_ceiling_data,
  splitrule = "huber",
  splitrule_par = 2^52 + 1,
  mtry = 1,
  min_node_size = pair_ceiling_group_size,
  nsplits = 100,
  seed = 1115
)
stopifnot(
  identical(root_threshold(pair_ceiling_mse), 3),
  identical(root_threshold(pair_ceiling_huber), 3)
)

# Extreme Tweedie comparisons likewise retain each daughter's low sum
# component across candidates, rather than comparing rounded absolute means.
tweedie_compensated_data <- data.frame(
  y = rep(
    rep(c(1, 2^52), each = pair_ceiling_group_size / 2),
    4
  ),
  x = rep(1:4, each = pair_ceiling_group_size)
)
tweedie_compensated_positive <- tweedie_compensated_data
tweedie_compensated_positive$y[nrow(tweedie_compensated_positive)] <-
  2^52 + 1
tweedie_compensated_negative <- tweedie_compensated_data
tweedie_compensated_negative$y[nrow(tweedie_compensated_negative)] <-
  2^52 - 1
tweedie_compensated_positive_tree <- jftree(
  y ~ x,
  tweedie_compensated_positive,
  splitrule = "tweedie",
  splitrule_par = 1e308,
  mtry = 1,
  min_node_size = pair_ceiling_group_size,
  nsplits = 100,
  seed = 1115
)
tweedie_compensated_negative_tree <- jftree(
  y ~ x,
  tweedie_compensated_negative,
  splitrule = "tweedie",
  splitrule_par = -1e308,
  mtry = 1,
  min_node_size = pair_ceiling_group_size,
  nsplits = 100,
  seed = 1115
)
stopifnot(
  identical(root_threshold(tweedie_compensated_positive_tree), 3),
  identical(root_threshold(tweedie_compensated_negative_tree), 3)
)

# Actual daughter means are retained separately from their precise difference.
# Reconstructing the small mean as parent plus a nearly opposite difference
# would round it to zero and break logarithmic or reciprocal criteria.
far_ratio_data <- data.frame(
  y = c(rep(1, 8), rep(1e100, 8)),
  x = seq_len(16)
)
far_ratio_rules <- list(
  gamma = NULL,
  inversegaussian = NULL,
  tweedie_gamma = 2,
  tweedie_inverse_gaussian = 3,
  tweedie_large_power = 32
)
far_ratio_roots <- vapply(
  names(far_ratio_rules),
  function(label) {
    arguments <- list(
      formula = y ~ x,
      data = far_ratio_data,
      splitrule = if (startsWith(label, "tweedie")) "tweedie" else label,
      mtry = 1,
      min_node_size = 8,
      nsplits = 100,
      seed = 1115
    )
    if (!is.null(far_ratio_rules[[label]])) {
      arguments$splitrule_par <- far_ratio_rules[[label]]
    }
    root_threshold(do.call(jftree, arguments))
  },
  numeric(1)
)
stopifnot(all(far_ratio_roots == 8))

# Xi < 0 permits real-valued responses, including negative observations. This
# also exercises the non-negative boundary optimum for a daughter mean.
negative_tweedie_data <- data.frame(
  y = c(-5, -4, -3, -2, -1, 0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 16),
  x = seq_len(16)
)
negative_tweedie_expected <- manual_root_split(
  negative_tweedie_data$y, negative_tweedie_data$x, 6, "tweedie", -1
)
negative_tweedie_tree <- jftree(
  y ~ x,
  negative_tweedie_data,
  splitrule = "tweedie",
  splitrule_par = -1,
  min_node_size = 6,
  nsplits = 100,
  seed = 1110
)
stopifnot(
  isTRUE(all.equal(
    root_threshold(negative_tweedie_tree),
    negative_tweedie_expected$threshold,
    tolerance = 0
  )),
  all(is.finite(negative_tweedie_tree$predictions))
)
constrained_homogeneous_roots <- vapply(
  c(1, 1e-250, 1e250),
  function(scale) {
    tree <- jftree(
      y ~ x,
      data.frame(
        y = negative_tweedie_data$y * scale,
        x = negative_tweedie_data$x
      ),
      splitrule = "tweedie",
      splitrule_par = -18,
      mtry = 1,
      min_node_size = 6,
      nsplits = 100,
      seed = 1116
    )
    root_threshold(tree)
  },
  numeric(1)
)
stopifnot(all(constrained_homogeneous_roots == 10))

# Parameter validation: requiredness, scalar numeric shape, finiteness, rule
# relevance, sign/range constraints, and both tree/forest entry points.
valid_parameter_data <- list(
  negativebinomial = data.frame(
    y = root_cases$negativebinomial$y,
    x = seq_along(root_cases$negativebinomial$y)
  ),
  tweedie = data.frame(
    y = root_cases$tweedie$y,
    x = seq_along(root_cases$tweedie$y)
  ),
  huber = data.frame(
    y = root_cases$huber$y,
    x = seq_along(root_cases$huber$y)
  )
)
for (splitrule in c("tweedie", "huber")) {
  current_data <- valid_parameter_data[[splitrule]]
  assert_error(
    jftree(y ~ x, current_data, splitrule = splitrule, min_node_size = 7),
    "splitrule_par"
  )
}

# Negative binomial retains k = 1 as its backwards-compatible default while
# accepting splitrule_par to make k explicit.
negative_binomial_default_expected <- manual_root_split(
  valid_parameter_data$negativebinomial$y,
  valid_parameter_data$negativebinomial$x,
  7,
  "negativebinomial",
  1
)
negative_binomial_default <- jftree(
  y ~ x,
  valid_parameter_data$negativebinomial,
  splitrule = "negativebinomial",
  min_node_size = 7,
  nsplits = 100,
  seed = 1201
)
negative_binomial_default_forest <- jfforest(
  y ~ x,
  valid_parameter_data$negativebinomial,
  splitrule = "negativebinomial",
  min_node_size = 3,
  nsplits = 10,
  ntrees = 2,
  seed = 1202,
  nworkers = 1
)
stopifnot(
  isTRUE(all.equal(
    root_threshold(negative_binomial_default),
    negative_binomial_default_expected$threshold,
    tolerance = 0
  )),
  isTRUE(all.equal(
    negative_binomial_default$splitrule.par, 1, tolerance = 0
  )),
  isTRUE(all.equal(
    negative_binomial_default_forest$splitrule.par, 1, tolerance = 0
  )),
  all(is.finite(negative_binomial_default_forest$predictions))
)

# The negative-binomial criterion is O(k) as k approaches zero. Its signal
# must be retained instead of cancelling to an all-zero set of gains.
tiny_k_data <- data.frame(
  y = c(10, 19, 18, 5, 20, 6, 12, 16, 6, 10,
        15, 19, 18, 2, 12, 5, 17, 13, 4, 19),
  x = seq_len(20)
)
tiny_k_trees <- lapply(
  c(1e-30, 1e-8),
  function(size) {
    jftree(
      y ~ x,
      tiny_k_data,
      splitrule = "negativebinomial",
      splitrule_par = size,
      mtry = 1,
      min_node_size = 3,
      nsplits = 100,
      seed = 1203
    )
  }
)
stopifnot(all(vapply(
  tiny_k_trees,
  function(tree) identical(root_threshold(tree), 13),
  logical(1)
)))
large_k_data <- data.frame(
  y = c(45, 13, 8, 31, 39, 42, 36, 48, 38, 22,
        48, 38, 23, 7, 22, 19, 3, 0, 10, 29),
  x = seq_len(20)
)
large_k_poisson <- jftree(
  y ~ x,
  large_k_data,
  splitrule = "poisson",
  mtry = 1,
  min_node_size = 7,
  nsplits = 100,
  seed = 1204
)
large_k_negative_binomial <- jftree(
  y ~ x,
  large_k_data,
  splitrule = "negativebinomial",
  splitrule_par = 1e308,
  mtry = 1,
  min_node_size = 7,
  nsplits = 100,
  seed = 1204
)
stopifnot(
  identical(root_threshold(large_k_poisson), 13),
  identical(root_threshold(large_k_negative_binomial), 13)
)

# The finite-k correction remains decision-relevant even when k is much larger
# than every observed count; this fixture differs from the Poisson-limit tie.
exact_large_k_data <- data.frame(
  y = c(
    1547740299198662, 1547740299198662, 0,
    1904519401602676, 500000000000000, 500000000000000
  ),
  x = seq_len(6)
)
exact_large_k_tree <- jftree(
  y ~ x,
  exact_large_k_data,
  splitrule = "negativebinomial",
  splitrule_par = 1e25,
  mtry = 1,
  min_node_size = 2,
  nsplits = 100,
  seed = 1204
)
stopifnot(identical(root_threshold(exact_large_k_tree), 4))

assert_error(
  jfforest(
    y ~ x, valid_parameter_data$huber,
    splitrule = "huber", ntrees = 2, min_node_size = 3
  ),
  "splitrule_par"
)

invalid_parameter_shapes <- list(
  numeric(0),
  c(1, 2),
  "1",
  matrix(1, nrow = 1)
)
for (value in invalid_parameter_shapes) {
  assert_error(
    jftree(
      y ~ x, valid_parameter_data$huber,
      splitrule = "huber", splitrule_par = value,
      min_node_size = 7
    ),
    "splitrule_par"
  )
}
for (value in c(NA_real_, NaN, Inf, -Inf)) {
  assert_error(
    jftree(
      y ~ x, valid_parameter_data$huber,
      splitrule = "huber", splitrule_par = value,
      min_node_size = 7
    ),
    "finite"
  )
}
for (value in c(0, -1)) {
  assert_error(
    jftree(
      y ~ x, valid_parameter_data$negativebinomial,
      splitrule = "negativebinomial", splitrule_par = value,
      min_node_size = 7
    ),
    "strictly positive"
  )
  assert_error(
    jftree(
      y ~ x, valid_parameter_data$huber,
      splitrule = "huber", splitrule_par = value,
      min_node_size = 7
    ),
    "strictly positive"
  )
}
assert_error(
  jftree(
    y ~ x, valid_parameter_data$tweedie,
    splitrule = "tweedie", splitrule_par = 0.5,
    min_node_size = 7
  ),
  "xi <= 0 or xi >= 1"
)

nonparameter_rules <- c(
  "mse", "variance", "mae", "binomial", "poisson", "gamma",
  "inversegaussian"
)
for (splitrule in nonparameter_rules) {
  current_data <- if (splitrule == "binomial") {
    data.frame(y = root_cases$binomial$y, x = seq_len(20))
  } else if (splitrule == "poisson") {
    data.frame(y = root_cases$poisson$y, x = seq_len(20))
  } else if (splitrule %in% c("gamma", "inversegaussian")) {
    data.frame(y = root_cases$gamma$y, x = seq_len(20))
  } else {
    data.frame(y = root_cases$huber$y, x = seq_len(20))
  }
  assert_error(
    jftree(
      y ~ x, current_data,
      splitrule = splitrule, splitrule_par = 1,
      min_node_size = 7
    ),
    "may only be supplied"
  )
}

classification_data <- data.frame(
  y = factor(rep(c("a", "b"), each = 8)),
  x = seq_len(16)
)
assert_error(
  jftree(
    y ~ x, classification_data,
    splitrule = "gini", splitrule_par = 1
  ),
  "only valid for regression models"
)
assert_error(
  jftree(
    y ~ x, data.frame(y = seq_len(16), x = seq_len(16)),
    splitrule = "not-a-rule"
  ),
  "Invalid regression splitrule"
)

# Response-support and finiteness checks. The valid side of every Tweedie
# boundary is covered above by the identity and parameter-effect fits.
assert_error(
  jftree(
    y ~ x, data.frame(y = c(rep(0.5, 15), 1.01), x = seq_len(16)),
    splitrule = "binomial"
  ),
  "in [0, 1]"
)
for (splitrule in c("negativebinomial", "poisson")) {
  parameter <- if (splitrule == "negativebinomial") 2 else NULL
  arguments <- list(
    formula = y ~ x,
    data = data.frame(y = c(0:14, 1.5), x = seq_len(16)),
    splitrule = splitrule,
    min_node_size = 4
  )
  if (!is.null(parameter)) {
    arguments$splitrule_par <- parameter
  }
  assert_error(do.call(jftree, arguments), "non-negative integer")
}
for (splitrule in c("gamma", "inversegaussian")) {
  assert_error(
    jftree(
      y ~ x,
      data.frame(y = c(rep(1, 15), 0), x = seq_len(16)),
      splitrule = splitrule
    ),
    "strictly positive"
  )
}
assert_error(
  jftree(
    y ~ x,
    data.frame(y = c(0:14, 1.5), x = seq_len(16)),
    splitrule = "tweedie", splitrule_par = 1
  ),
  "non-negative integer"
)
assert_error(
  jftree(
    y ~ x,
    data.frame(y = c(rep(1, 15), -0.1), x = seq_len(16)),
    splitrule = "tweedie", splitrule_par = 1.5
  ),
  "non-negative responses"
)
for (power in c(2, 3)) {
  assert_error(
    jftree(
      y ~ x,
      data.frame(y = c(rep(1, 15), 0), x = seq_len(16)),
      splitrule = "tweedie", splitrule_par = power
    ),
    "strictly positive"
  )
}
for (bad_response in list(NA_real_, Inf, -Inf, NaN)) {
  assert_error(
    jftree(
      y ~ x,
      data.frame(y = c(rep(1, 15), bad_response), x = seq_len(16)),
      splitrule = "huber", splitrule_par = 1
    ),
    "finite"
  )
}

cat("All regression splitting-rule tests passed.\n")

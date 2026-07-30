library(JumpForests)

assert_error <- function(expression, pattern) {
  message <- tryCatch(
    {
      force(expression)
      NULL
    },
    error = function(condition) conditionMessage(condition)
  )
  if (is.null(message) || !grepl(pattern, message, fixed = TRUE)) {
    stop("Expected an error containing: ", pattern)
  }
}

# Every observation starts in state 1 and moves to state 2 or 3 at time
# one. The signal produces pure terminal nodes, so perturbing it changes a
# correct one-hot prediction into either the same or the other one-hot
# prediction.
#
# With c_j = 1/3, the spherical-loss increase for a wrong prediction is
# 1/sqrt(3), while the Brier-loss increase is 2/3. Their VIMPs must
# therefore have the exact ratio sqrt(3)/2, independently of which rows
# are perturbed.
num_obs <- 120L
end_states <- rep(c(2L, 3L), each = num_obs / 2)
jump_data <- lapply(seq_len(num_obs), function(i) {
  list(times = c(0, 1), states = c(1L, end_states[i]))
})
features <- data.frame(
  signal = factor(end_states),
  constant = 0
)

fit_forest <- function(nworkers, save_predictions) {
  jfforest(
    MM ~ .,
    data = jump_data,
    feature_data = features,
    ntrees = 30,
    mtry = 2,
    min_node_size = 5,
    nsplits = 10,
    seed = 42,
    nworkers = nworkers,
    save_predictions = save_predictions
  )
}

lazy_forest <- fit_forest(nworkers = 1, save_predictions = FALSE)
parallel_forest <- fit_forest(nworkers = 2, save_predictions = TRUE)

for (method in c("permute", "random")) {
  brier <- jfforest.vimp(
    lazy_forest,
    feature = "signal",
    seed = 123,
    method = method,
    loss = "brier"
  )
  spherical <- jfforest.vimp(
    lazy_forest,
    feature = "signal",
    seed = 123,
    method = method,
    loss = "spherical"
  )
  spherical_repeated <- jfforest.vimp(
    lazy_forest,
    feature = "signal",
    seed = 123,
    method = method,
    loss = "spherical"
  )
  spherical_parallel <- jfforest.vimp(
    parallel_forest,
    feature = "signal",
    seed = 123,
    method = method,
    loss = "spherical"
  )
  unused_feature <- jfforest.vimp(
    lazy_forest,
    feature = "constant",
    seed = 123,
    method = method,
    loss = "spherical"
  )

  stopifnot(
    is.finite(spherical),
    spherical > 0,
    isTRUE(all.equal(spherical / brier, sqrt(3) / 2, tolerance = 1e-12)),
    isTRUE(all.equal(spherical, spherical_repeated, tolerance = 1e-12)),
    isTRUE(all.equal(spherical, spherical_parallel, tolerance = 1e-12)),
    identical(unused_feature, 0)
  )
}

# The existing multi-state default remains Brier, including after the
# spherical baseline has occupied the VIMP cache.
default_vimp <- jfforest.vimp(
  lazy_forest,
  feature = "signal",
  seed = 901,
  method = "permute",
  loss = "default"
)
brier_vimp <- jfforest.vimp(
  lazy_forest,
  feature = "signal",
  seed = 901,
  method = "permute",
  loss = "brier"
)
stopifnot(isTRUE(all.equal(default_vimp, brier_vimp, tolerance = 1e-12)))

# Computing all features returns a finite spherical VIMP and preserves the
# exact zero for a feature that no tree can split on.
all_vimp <- jfforest.vimp(
  lazy_forest,
  seed = 456,
  method = "permute",
  loss = "spherical"
)
stopifnot(
  is.finite(all_vimp$vimp$signal),
  all_vimp$vimp$signal > 0,
  identical(all_vimp$vimp$constant, 0)
)

# Exercise IPCW before and after subject endpoints on non-pure predictions,
# multiple event times, multiple starting states, and censored paths.
censored_jump_data <- lapply(seq_len(60L), function(i) {
  switch(
    as.character(i %% 5L),
    "0" = list(times = c(0, 0.7 + i / 100, 2.8 + i / 100), states = c(1L, 2L, 3L)),
    "1" = list(times = c(0, 1.2 + i / 100), states = c(1L, 3L)),
    "2" = list(times = c(0, 0.9 + i / 100, 3.1 + i / 100), states = c(2L, 1L, 3L)),
    "3" = list(times = c(0, 1.7 + i / 100), states = c(2L, 3L)),
    "4" = list(times = c(0, 1.0 + i / 100, 2.2 + i / 100), states = c(1L, 2L, 2L))
  )
})
censored_features <- data.frame(
  signal = seq_len(60L) %% 5L,
  noise = sin(seq_len(60L))
)
censored_forest <- jfforest(
  MM ~ .,
  data = censored_jump_data,
  feature_data = censored_features,
  ntrees = 12,
  min_node_size = 5,
  nsplits = 20,
  seed = 410,
  nworkers = 2,
  save_predictions = FALSE
)
for (method in c("permute", "random")) {
  censored_vimp <- jfforest.vimp(
    censored_forest,
    seed = 812,
    method = method,
    loss = "spherical"
  )
  stopifnot(
    length(censored_vimp$vimp) == ncol(censored_features),
    all(is.finite(unlist(censored_vimp$vimp)))
  )
}

assert_error(
  jfforest.vimp(
    lazy_forest,
    feature = "signal",
    seed = 123,
    loss = "sphere"
  ),
  "'spherical'"
)

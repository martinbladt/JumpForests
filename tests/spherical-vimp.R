library(JumpForests)

feature_vimp <- function(...) {
  result <- jfforest.vimp(...)
  stopifnot(inherits(result, "jf_vimp"), identical(names(result), "vimp"),
            length(result$vimp) == 1L)
  result$vimp[[1L]]
}

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
  brier <- feature_vimp(
    lazy_forest,
    feature = "signal",
    seed = 123,
    method = method,
    loss = "brier"
  )
  spherical <- feature_vimp(
    lazy_forest,
    feature = "signal",
    seed = 123,
    method = method,
    loss = "spherical"
  )
  spherical_repeated <- feature_vimp(
    lazy_forest,
    feature = "signal",
    seed = 123,
    method = method,
    loss = "spherical"
  )
  spherical_parallel <- feature_vimp(
    parallel_forest,
    feature = "signal",
    seed = 123,
    method = method,
    loss = "spherical"
  )
  unused_feature <- feature_vimp(
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
default_vimp <- feature_vimp(
  lazy_forest,
  feature = "signal",
  seed = 901,
  method = "permute",
  loss = "default"
)
brier_vimp <- feature_vimp(
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

# Check the public VIMP result and print contract across all model types.
local({
  num_obs <- 40L
  features <- data.frame(x = seq_len(num_obs), group = rep(c(0, 1), num_obs / 2))
  cases <- list(
    regression = list(
      formula = y ~ x + group,
      data = data.frame(y = sin(features$x / 4), features)
    ),
    classification = list(
      formula = y ~ x + group,
      data = data.frame(y = factor(rep(c("a", "b"), num_obs / 2)), features)
    ),
    survival = list(
      formula = Surv(time, status) ~ x + group,
      data = data.frame(time = seq(0.5, 20, length.out = num_obs),
                        status = rep(c(1, 1, 0, 1), num_obs / 4), features)
    ),
    multistate = list(
      formula = MM ~ x + group,
      data = lapply(seq_len(num_obs), function(i) {
        list(times = c(0, 0.5 + (i %% 5), 6 + (i %% 3)),
             states = c(1L, 2L, 3L))
      }),
      feature_data = features,
      save_predictions = FALSE
    )
  )

  for (arguments in cases) {
    fit <- do.call(jfforest, c(arguments,
      list(ntrees = 10, nworkers = 1, min_node_size = 5, max_depth = 2, seed = 17)))
    original_names <- names(fit)
    original_classes <- class(fit)
    for (method in c("permute", "random")) {
      all_values <- jfforest.vimp(fit, seed = 23, method = method)
      one_value <- jfforest.vimp(fit, feature = "x", seed = 23, method = method)
      for (result in list(all_values, one_value)) {
        stopifnot(
          inherits(result, "jf_vimp"),
          identical(names(result), "vimp"),
          identical(names(attributes(result)), c("names", "class")),
          is.list(result$vimp),
          all(vapply(result$vimp, function(value) is.numeric(value) &&
            length(value) == 1L && is.finite(value), logical(1)))
        )
        output <- capture.output(returned <- withVisible(print(result)))
        stopifnot(
          identical(returned$value, result), !returned$visible,
          any(grepl("VIMP", output, fixed = TRUE)),
          any(grepl("x", output, fixed = TRUE)),
          any(grepl("$vimp", output, fixed = TRUE)),
          length(output) < 10L
        )
      }
      stopifnot(
        identical(names(all_values$vimp), names(features)),
        identical(names(one_value$vimp), "x"),
        identical(one_value$vimp$x, one_value$vimp[["x"]]),
        identical(names(fit), original_names),
        identical(class(fit), original_classes),
        is.null(fit[["vimp", exact = TRUE]])
      )
    }
  }

  # Large results print a bounded preview without discarding computed values.
  large_result <- all_values
  large_result$vimp <- as.list(seq_len(100L))
  names(large_result$vimp) <- paste0("feature_", seq_len(100L))
  output <- capture.output(print(large_result))
  stopifnot(
    length(large_result$vimp) == 100L,
    any(grepl("feature_20", output, fixed = TRUE)),
    !any(grepl("feature_21", output, fixed = TRUE)),
    any(grepl("Showing 20 of 100", output, fixed = TRUE)),
    length(output) < 30L,
    any(grepl("feature_100", capture.output(print(large_result, n = Inf)), fixed = TRUE))
  )
  for (invalid_n in list(NA_real_, -1, 1.5, "all", c(1, 2))) {
    error <- tryCatch(print(large_result, n = invalid_n), error = identity)
    stopifnot(inherits(error, "error"),
              grepl("whole number or Inf", conditionMessage(error), fixed = TRUE))
  }

  message("VIMP-only return values and printing tests passed")
})

# A survival model is a two-state model with an absorbing death state.
# Match their one-split trees and OOB samples, then compare both probability
# losses. Mixed leaves give nonzero baseline losses, so this checks that the
# baseline AND both perturbation methods honour the selected loss.
local({
  group <- rep(c(0, 1), each = 100L)
  outcome <- rep(seq_len(10L), 20L)
  time <- ifelse(group == 0, ifelse(outcome <= 7L, 1, 2),
                 ifelse(outcome <= 2L, 1, 2))
  status <- as.numeric(outcome != 10L)
  time[status == 0] <- 3
  survival_data <- data.frame(time = time, status = status,
                              signal = group, constant = 0)
  jump_data <- lapply(seq_along(time), function(i) {
    list(times = c(0, time[i]),
         states = c(1L, if (status[i] == 1) 2L else 1L))
  })
  features <- survival_data[c("signal", "constant")]
  fit_survival <- function(nworkers, save_predictions) {
    jfforest(Surv(time, status) ~ signal + constant, data = survival_data,
             ntrees = 20, mtry = 2, min_node_size = 10, max_depth = 1,
             seed = 42, nworkers = nworkers, save_predictions = save_predictions)
  }
  survival_lazy <- fit_survival(1, FALSE)
  survival_saved <- fit_survival(2, TRUE)
  multistate <- jfforest(MM ~ signal + constant, data = jump_data,
                         feature_data = features, ntrees = 20, mtry = 2,
                         min_node_size = 10, max_depth = 1, seed = 42,
                         nworkers = 1, save_predictions = FALSE)

  for (method in c("permute", "random")) {
    losses <- list()
    # Evaluate Brier first to exercise switching between loss caches.
    for (loss in c("brier", "kl")) {
      value <- feature_vimp(survival_lazy, feature = "signal", seed = 123,
                            method = method, loss = loss)
      reference <- feature_vimp(multistate, feature = "signal", seed = 123,
                                method = method, loss = loss)
      saved <- feature_vimp(survival_saved, feature = "signal", seed = 123,
                            method = method, loss = loss)
      repeated <- feature_vimp(survival_lazy, feature = "signal", seed = 123,
                               method = method, loss = loss)
      stopifnot(
        is.finite(value), value > 0,
        isTRUE(all.equal(value, reference, tolerance = 1e-10)),
        isTRUE(all.equal(value, saved, tolerance = 1e-12)),
        identical(value, repeated)
      )
      losses[[loss]] <- value
    }
    stopifnot(!isTRUE(all.equal(losses$kl, losses$brier, tolerance = 1e-6)))
    all_kl <- jfforest.vimp(survival_lazy, seed = 321, method = method, loss = "kl")
    stopifnot(is.finite(all_kl$vimp$signal), all_kl$vimp$signal > 0,
              identical(all_kl$vimp$constant, 0))
    brier_again <- feature_vimp(survival_lazy, feature = "signal", seed = 123,
                                method = method, loss = "brier")
    stopifnot(identical(brier_again, losses$brier))
  }
  message("Survival KL and Brier VIMP agree with equivalent two-state losses")
})

library(JumpForests)

manual_spherical_error <- function(predictions, init, jump_data, event_times, state_weights) {
  num_obs <- length(jump_data)
  pointwise <- matrix(0, nrow = num_obs, ncol = length(event_times))

  for (i in seq_len(num_obs)) {
    occupation <- JumpForests:::occupation_prob(init[[i]], predictions[[i]])
    transitions <- jump_data[[i]]$times[-1]
    transition_states <- jump_data[[i]]$states[-1]

    for (t in seq_along(event_times)) {
      past_transitions <- which(transitions <= event_times[t])
      observed_state <- if (length(past_transitions) == 0) {
        jump_data[[i]]$states[1]
      } else {
        transition_states[max(past_transitions)]
      }
      probability <- as.numeric(occupation[[t]])
      denominator <- sqrt(sum(state_weights * probability^2))
      reward <- if (denominator > 0) {
        state_weights[observed_state] * probability[observed_state] / denominator
      } else {
        0
      }
      pointwise[i, t] <- 1 - reward
    }
  }

  score <- colMeans(pointwise)
  integrated <- sum((head(score, -1) + tail(score, -1)) * diff(event_times) / 2)
  list(
    integrated = integrated,
    normalised = integrated / tail(event_times, 1),
    pointwise = pointwise
  )
}

num_obs <- 60L
end_states <- rep(c(2L, 3L), length.out = num_obs)
transition_times <- rep(1:5, length.out = num_obs)
jump_data <- lapply(seq_len(num_obs), function(i) {
  list(times = c(0, transition_times[i]), states = c(1L, end_states[i]))
})
features <- data.frame(x = seq_len(num_obs))
state_weights <- c(0.2, 0.3, 0.5)

tree <- jftree(
  MM ~ x,
  data = jump_data,
  feature_data = features,
  min_node_size = 100,
  nsplits = 1,
  seed = 7,
  state_weights = state_weights
)
tree_manual <- manual_spherical_error(
  tree$predictions,
  tree$init,
  jump_data,
  tree$unique.event.times,
  state_weights
)
tree_external <- jftree.error(
  tree,
  new_data = features,
  jump_data = jump_data,
  state_weights = state_weights
)

stopifnot(
  isTRUE(all.equal(tree$is, tree_manual$integrated, tolerance = 1e-12)),
  isTRUE(all.equal(tree$is.normalised, tree_manual$normalised, tolerance = 1e-12)),
  isTRUE(all.equal(tree_external$IS.error, tree_manual$integrated, tolerance = 1e-12)),
  isTRUE(all.equal(tree_external$normalised.IS.error, tree_manual$normalised, tolerance = 1e-12)),
  all(tree_manual$pointwise >= 0),
  all(tree_manual$pointwise <= 1),
  tree$is.normalised > 0,
  tree$is.normalised <= 1
)

forest <- jfforest(
  MM ~ x,
  data = jump_data,
  feature_data = features,
  ntrees = 1,
  min_node_size = 100,
  nsplits = 1,
  seed = 7,
  nworkers = 1,
  save_predictions = TRUE,
  state_weights = state_weights
)
forest_predictions <- jfforest.predict(forest, new_data = features, compute_initial = TRUE)
forest_manual <- manual_spherical_error(
  forest_predictions$predictions,
  forest_predictions$init,
  jump_data,
  forest$unique.event.times,
  state_weights
)
forest_external <- jfforest.error(
  forest,
  new_data = features,
  jump_data = jump_data,
  state_weights = state_weights
)

stopifnot(
  isTRUE(all.equal(forest_external$IS.error, forest_manual$integrated, tolerance = 1e-12)),
  isTRUE(all.equal(forest_external$normalised.IS.error, forest_manual$normalised, tolerance = 1e-12)),
  forest$is.normalised > 0,
  forest_external$normalised.IS.error > 0,
  forest_external$normalised.IS.error <= 1
)

# The lazy OOB-error path used when predictions are not saved must also retain
# explicitly supplied spherical weights.
lazy_forest <- jfforest(
  MM ~ x,
  data = jump_data,
  feature_data = features,
  ntrees = 10,
  min_node_size = 100,
  nsplits = 1,
  seed = 7,
  nworkers = 2,
  save_predictions = FALSE,
  state_weights = state_weights
)
lazy_error <- jfforest.error(lazy_forest, state_weights = state_weights)
stopifnot(
  is.finite(lazy_error$IS.error),
  is.finite(lazy_error$normalised.IS.error),
  lazy_error$normalised.IS.error > 0
)

# Empty state_weights use equal spherical weights.
tree_default <- jftree(
  MM ~ x,
  data = jump_data,
  feature_data = features,
  min_node_size = 100,
  nsplits = 1,
  seed = 7
)
default_manual <- manual_spherical_error(
  tree_default$predictions,
  tree_default$init,
  jump_data,
  tree_default$unique.event.times,
  rep(1 / 3, 3)
)
stopifnot(
  isTRUE(all.equal(tree_default$is.normalised, default_manual$normalised, tolerance = 1e-12))
)

# Non-negative weights may make the denominator zero. This is treated as zero
# reward (loss one), rather than producing NaN.
zero_denominator_error <- jftree.error(
  tree,
  new_data = features,
  jump_data = jump_data,
  state_weights = c(0, 0, 1)
)
stopifnot(
  is.finite(zero_denominator_error$IS.error),
  is.finite(zero_denominator_error$normalised.IS.error),
  zero_denominator_error$normalised.IS.error >= 0,
  zero_denominator_error$normalised.IS.error <= 1
)

message("Spherical multi-state error tests passed")

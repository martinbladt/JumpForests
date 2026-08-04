library(JumpForests)

integrate_survival_score <- function(score, event_times) {
  score[1] * event_times[1] / 2 +
    sum((head(score, -1) + tail(score, -1)) * diff(event_times) / 2)
}

integrate_multistate_score <- function(score, event_times, num_states) {
  score_by_time <- colSums(matrix(score, nrow = num_states))
  sum((head(score_by_time, -1) + tail(score_by_time, -1)) *
        diff(event_times) / 2)
}

check_survival_scores <- function(object) {
  brier <- unlist(object$bs, use.names = FALSE)
  kl <- unlist(object$kl, use.names = FALSE)
  stopifnot(
    is.list(object$bs),
    is.list(object$kl),
    all(vapply(object$bs, is.double, logical(1))),
    all(vapply(object$kl, is.double, logical(1))),
    all(lengths(object$bs) == 1L),
    all(lengths(object$kl) == 1L),
    is.null(object$spherical),
    length(object$bs) == length(object$unique.event.times),
    length(object$kl) == length(object$unique.event.times),
    isTRUE(all.equal(
      integrate_survival_score(brier, object$unique.event.times),
      object$ibs,
      tolerance = 1e-12
    )),
    isTRUE(all.equal(
      integrate_survival_score(kl, object$unique.event.times),
      object$ikl,
      tolerance = 1e-12
    ))
  )
}

survival_data <- data.frame(
  time = seq(0.5, 20, length.out = 40),
  status = rep(c(1, 1, 0, 1), 10),
  x = rep(c(0, 1), 20)
)

survival_tree <- jftree(
  Surv(time, status) ~ x,
  data = survival_data,
  min_node_size = 5,
  seed = 17
)
survival_forest <- jfforest(
  Surv(time, status) ~ x,
  data = survival_data,
  ntrees = 20,
  min_node_size = 5,
  seed = 17,
  nworkers = 1,
  save_predictions = TRUE
)

check_survival_scores(survival_tree)
check_survival_scores(survival_forest)

num_obs <- 60L
multistate_data <- lapply(seq_len(num_obs), function(i) {
  list(
    times = c(0, 0.5 + (i %% 5), 6 + (i %% 3)),
    states = c(1L, 2L, 3L)
  )
})
multistate_features <- data.frame(x = seq_len(num_obs))

multistate_tree <- jftree(
  MM ~ x,
  data = multistate_data,
  feature_data = multistate_features,
  min_node_size = 10,
  seed = 19
)
multistate_forest <- jfforest(
  MM ~ x,
  data = multistate_data,
  feature_data = multistate_features,
  ntrees = 20,
  min_node_size = 10,
  seed = 19,
  nworkers = 1,
  save_predictions = TRUE
)

check_multistate_scores <- function(object) {
  num_states <- length(object$init[[1]])
  num_event_times <- length(object$unique.event.times)
  brier <- unlist(object$bs, use.names = FALSE)
  kl <- unlist(object$kl, use.names = FALSE)
  spherical <- unlist(object$spherical, use.names = FALSE)
  stopifnot(
    is.list(object$bs),
    is.list(object$kl),
    is.list(object$spherical),
    all(vapply(object$bs, is.double, logical(1))),
    all(vapply(object$kl, is.double, logical(1))),
    all(vapply(object$spherical, is.double, logical(1))),
    length(object$bs) == num_event_times,
    length(object$kl) == num_event_times,
    length(object$spherical) == num_event_times,
    all(lengths(object$bs) == num_states),
    all(lengths(object$kl) == num_states),
    all(lengths(object$spherical) == num_states),
    isTRUE(all.equal(
      integrate_multistate_score(
        brier, object$unique.event.times, num_states
      ),
      object$ibs,
      tolerance = 1e-12
    )),
    isTRUE(all.equal(
      integrate_multistate_score(
        kl, object$unique.event.times, num_states
      ),
      object$ikl,
      tolerance = 1e-12
    )),
    isTRUE(all.equal(
      integrate_multistate_score(
        spherical, object$unique.event.times, num_states
      ),
      object$is,
      tolerance = 1e-12
    ))
  )
}

check_multistate_scores(multistate_tree)
check_multistate_scores(multistate_forest)

message("Survival and multi-state score-vector tests passed")

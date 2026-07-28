# Regression tests for state weights supplied while fitting a multi-state
# forest. Run against an installed JumpForests package with:
# Rscript tests/state-weights.R

library(JumpForests)

make_three_state_data <- function(n = 48L) {
  paths <- lapply(seq_len(n), function(i) {
    switch(
      as.character(i %% 5L),
      "0" = list(
        times = c(0, 0.7 + i / 100, 2.8 + i / 100),
        states = c(1L, 2L, 3L)
      ),
      "1" = list(
        times = c(0, 1.2 + i / 100),
        states = c(1L, 3L)
      ),
      "2" = list(
        times = c(0, 0.9 + i / 100, 3.1 + i / 100),
        states = c(2L, 1L, 3L)
      ),
      "3" = list(
        times = c(0, 1.7 + i / 100),
        states = c(2L, 3L)
      ),
      "4" = list(
        times = c(0, 1.0 + i / 100, 2.2 + i / 100),
        states = c(1L, 2L, 2L)
      )
    )
  })
  features <- data.frame(
    x = seq_len(n),
    z = sin(seq_len(n))
  )
  list(paths = paths, features = features)
}

three_state <- make_three_state_data()
state_weights <- c(0.25, 2, 0.5)

weighted_forest <- jfforest(
  MM ~ x + z,
  data = three_state$paths,
  feature_data = three_state$features,
  ntrees = 8,
  min_node_size = 5,
  nsplits = 20,
  seed = 410,
  nworkers = 1,
  state_weights = state_weights
)

fit_error <- jfforest.error(weighted_forest)
recomputed_error <- jfforest.error(
  weighted_forest,
  state_weights = state_weights
)
default_error <- jfforest.error(
  weighted_forest,
  state_weights = numeric(0)
)

stopifnot(
  isTRUE(all.equal(fit_error, recomputed_error, tolerance = 1e-12)),
  !isTRUE(all.equal(fit_error, default_error, tolerance = 1e-12))
)


# Standalone integration and validation tests for the maximum-depth limit.
# Run against an installed JumpForests package with:
# Rscript tests/max-depth.R

library(JumpForests)

n <- 32L
regression_data <- data.frame(
  x = seq_len(n),
  y = c(rep(0, n / 2), rep(1, n / 2))
)
classification_data <- transform(regression_data, y = factor(y))
survival_data <- data.frame(
  x = seq_len(n),
  time = seq_len(n) + 0.5,
  status = rep(c(0, 1), length.out = n)
)
paths <- lapply(seq_len(n), function(i) {
  list(times = c(0, 1 + i / 100), states = c(1L, 2L))
})
features <- data.frame(x = seq_len(n))

trees <- list(
  regression = jftree(
    y ~ x, regression_data, mtry = 1, min_node_size = 1,
    nsplits = 100, max_depth = 0, seed = 1
  ),
  classification = jftree(
    y ~ x, classification_data, mtry = 1, min_node_size = 1,
    nsplits = 100, max_depth = 0, seed = 1
  ),
  survival = jftree(
    Surv(time, status) ~ x, survival_data, mtry = 1, min_node_size = 1,
    nsplits = 100, max_depth = 0, seed = 1
  ),
  multistate = jftree(
    MM ~ x, paths, feature_data = features, mtry = 1,
    min_node_size = 1, nsplits = 100, max_depth = 0, seed = 1
  )
)

forest_arguments <- list(
  mtry = 1,
  min_node_size = 1,
  nsplits = 100,
  ntrees = 3,
  max_depth = 0,
  seed = 1,
  nworkers = 1
)
forests <- list(
  regression = do.call(
    jfforest,
    c(list(formula = y ~ x, data = regression_data), forest_arguments)
  ),
  classification = do.call(
    jfforest,
    c(list(formula = y ~ x, data = classification_data), forest_arguments)
  ),
  survival = do.call(
    jfforest,
    c(list(formula = Surv(time, status) ~ x, data = survival_data), forest_arguments)
  ),
  multistate = do.call(
    jfforest,
    c(
      list(formula = MM ~ x, data = paths, feature_data = features),
      forest_arguments
    )
  )
)

stopifnot(
  all(vapply(trees, function(tree) tree$max.depth == 0, logical(1))),
  all(vapply(trees, function(tree) tree$tree.depth == 0, logical(1))),
  all(vapply(forests, function(forest) forest$max.depth == 0, logical(1))),
  all(vapply(forests, function(forest) forest$avg.tree.depth == 0, logical(1)))
)

depth_one_tree <- jftree(
  y ~ x, regression_data, mtry = 1, min_node_size = 1,
  nsplits = 100, max_depth = 1, seed = 1
)
stopifnot(depth_one_tree$tree.depth == 1)

unlimited_tree <- jftree(
  y ~ x, regression_data, mtry = 1, min_node_size = 1,
  nsplits = 100, seed = 2
)
unlimited_forest <- jfforest(
  y ~ x, regression_data, mtry = 1, min_node_size = 1,
  ntrees = 2, seed = 2, nworkers = 1
)
stopifnot(
  is.infinite(unlimited_tree$max.depth),
  is.infinite(unlimited_forest$max.depth)
)

invalid_depths <- list(-1, 1.5, NA_real_, NaN, "2", numeric())
for (max_depth in invalid_depths) {
  result <- try(
    jftree(y ~ x, regression_data, max_depth = max_depth),
    silent = TRUE
  )
  stopifnot(inherits(result, "try-error"))
}

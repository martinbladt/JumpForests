#nolint start: line_length_linter

#' Fit a JumpForests tree
#'
#' Fits a single regression, classification, survival, or multi-state tree.
#'
#' @param formula Model formula: `y ~ x1 + x2` for regression or classification,
#'   `Surv(time, status) ~ x1 + x2` for survival (`status = 1` for an event,
#'   `0` for censoring), or `MM ~ x1 + x2` for multi-state models. A numeric
#'   response selects regression; a factor, character, or logical response
#'   selects classification. Convert numeric class labels to a factor to fit
#'   a classification tree. Numeric features are treated as continuous;
#'   factor, character, and logical features are treated as categorical.
#' @param data Training data. A data.frame including both responses and features for regression,
#'   classification and survival. For multi-state models, a list of lists (one for each
#'   observation), each containing two vectors, `"times"` (observed jump times) and
#'   `"states"` (the state occupied at the corresponding time). The states must be encoded
#'   as integers. Repetition of the final state indicates censoring.
#' @param feature_data Feature data as a data.frame. Only required for multi-state models.
#' @param splitrule Splitting rule. `NULL` uses the model-specific defaults
#'   below. For regression, choose `"mse"` (default),
#'   `"variance"` (an alias for `"mse"`), `"mae"`, `"binomial"`,
#'   `"negativebinomial"`, `"poisson"`, `"gamma"`, `"inversegaussian"`,
#'   `"tweedie"`, or `"huber"`. Binomial responses must lie in `[0, 1]`;
#'   negative-binomial and Poisson responses must be non-negative integers;
#'   Gamma and inverse-Gaussian responses must be positive. Tweedie support
#'   follows its power: real for \eqn{\xi \le 0}, non-negative counts at
#'   \eqn{\xi = 1}, non-negative for \eqn{1 < \xi < 2}, and positive for
#'   \eqn{\xi \ge 2}. For classification, choose `"gini"` (default), `"entropy"`,
#'   `"misc"`, `"twoing"` or `"hellinger"`. For survival, choose `"logrank"`
#'   (default), `"approxlogrank"` or `"conserve"`. For multi-state models,
#'   choose `"logrank"` (default), `"gehan"`, `"taroneware"`, `"approxlogrank"`,
#'   `"conserve"`, `"petoprentice"` or `"flemingharrington"`. Choosing `"flemingharrington"`
#'   allows for specifying the weights, see the options `"fh_weights_a"` and
#'   `"fh_weights_b"`.
#' @param splitrule_par Numeric scalar for parameterised regression rules:
#'   negative-binomial size \eqn{k > 0} (default 1), Tweedie power
#'   \eqn{\xi \le 0} or \eqn{\xi \ge 1}, or Huber threshold
#'   \eqn{\delta > 0}. It is required for Tweedie and Huber and must be `NULL`
#'   for every other rule and model type. The default is `NULL`.
#' @param mtry Number of randomly selected candidate features at each split.
#'   `NULL` uses `ceiling(sqrt(p))`, where `p` is the number of features.
#' @param min_node_size Minimum number of observations required in each
#'   daughter node for a split. `NULL` uses 5 for regression, 1 for
#'   classification, 15 for survival, and 20 for multi-state models.
#' @param max_depth Maximum depth of the tree, where the root has depth zero.
#'   `NULL` (the default) imposes no depth limit.
#' @param nsplits Number of randomly sampled candidate thresholds per
#'   continuous feature, or candidate partitions per categorical feature.
#'   Defaults to 10; fewer candidates are used if fewer distinct splits exist.
#' @param honest Whether to split the observations into approximately equal
#'   sets for choosing splits and estimating leaf predictions. Defaults to
#'   `FALSE`, which uses the same observations for both tasks.
#' @param seed Random seed. `NULL` draws a seed using R's current random-number
#'   state; supply a seed for reproducible fitting.
#' @param num_event_times Maximum number of times in the prediction grid for
#'   survival and multi-state models. The default, 0, retains the full grid.
#'   A positive value coarsens the grid when necessary; the resulting times
#'   are stored in `unique.event.times`.
#' @param state_weights Numeric vector of finite non-negative weights, one
#'   per state in the order `1, ..., num_states`, for multi-state error
#'   calculations. Supplied weights are used directly for all scores and are
#'   not normalised. `NULL` uses `1 / num_states` per state for Brier and
#'   spherical scores and 1 per state for Kullback-Leibler loss. These weights
#'   affect error calculations, not splitting or prediction.
#' @param fh_weights_a Optional finite non-negative Fleming--Harrington
#'   a-exponents, one for each state `1, ..., num_states`. `NULL` uses one for
#'   every state. Only valid for multi-state trees with
#'   `splitrule = "flemingharrington"`. A transition from state `s` is weighted
#'   by \eqn{p_s(t)^{a_s}(1-p_s(t))^{b_s}}, where \eqn{p_s(t)} is the
#'   estimated occupation probability of state `s` in the parent node.
#' @param fh_weights_b Optional finite non-negative Fleming--Harrington
#'   b-exponents, with the same ordering and default as `fh_weights_a`.
#'
#' @return A fitted tree object as a list with class `jf_tree`. Use `print()`
#'   for a summary and `predict()` for predictions. Multi-state prediction
#'   lists print compact summaries. Training predictions are stored in
#'   `predictions`. For survival and multi-state trees, `censoring` contains
#'   leaf Kaplan-Meier estimates of the censoring survival probability, with
#'   one row per observation on `unique.event.times`. The estimates on the
#'   full censoring-time grid are in `censoring.full`, with times in
#'   `censoring.times`.
#' @md
#' @export
#'
# the main function for fitting trees (feature_data is only relevant for multi-state trees in which case data is a list and not a data.frame)
jftree <- function(formula, data, feature_data = NULL, splitrule = NULL, mtry = NULL, min_node_size = NULL, max_depth = NULL, nsplits = 10,
                   honest = FALSE, seed = NULL, num_event_times = 0, state_weights = NULL, fh_weights_a = NULL, fh_weights_b = NULL, splitrule_par = NULL) {
  lhs <- as.character(formula[[2]])
  splitrule_par <- normalize_splitrule_par(splitrule_par)
  max_depth <- normalize_max_depth(max_depth)
  if (lhs[1] != "MM" && (!is.null(fh_weights_a) || !is.null(fh_weights_b))) {
    stop("fh_weights_a and fh_weights_b are only valid for multi-state models")
  }
  if (lhs[1] %in% c("MM", "Surv") && length(splitrule_par) > 0) {
    stop("splitrule_par is only valid for regression models")
  }
  # if seed is not set, generate a random one
  if (is.null(seed)) {
    seed <- runif(n = 1, min = 1, max = 10^6)
  }

  # if left hand side is of length 1, classification, regression or multi-state
  if (length(lhs) == 1 && lhs[1] != "MM") {
    # preprocess the entire dataset
    processed_data <- preprocess_data(data)
    response_index <- which(names(data) %in% as.character(formula[[2]])[1]) - 1
    if (formula[[3]] == ".") {
      covariates <- names(data)[-(response_index + 1)]
    } else {
      covariates <- attr(terms(formula), "term.labels")
    }
    feature_indices <- which(names(data) %in% covariates) - 1

    # by default, the number of variables tried at each split is the
    # square root of the number of covariates
    if (is.null(mtry)) {
      mtry <- ceiling(sqrt(length(covariates)))
    }

    # if the response is categorical, classification, otherwise regression
    if (processed_data$categorical[response_index + 1]) {
      if (length(splitrule_par) > 0) {
        stop("splitrule_par is only valid for regression models")
      }
      # for classification, the default minimal node size is 1
      if (is.null(min_node_size)) {
        min_node_size <- 1
      }
      # set default splitting rule
      if (is.null(splitrule)) {
        splitrule <- "gini"
      }
      result <- JFCppTree(2, processed_data$data, mtry, min_node_size, max_depth, nsplits, splitrule, honest,
                          response_index, feature_indices, processed_data$categorical,
                          processed_data$unique_values, seed, 0, splitrule_par)
      result$categorical.levels <- processed_data$categorical_levels
      result$class.levels <- processed_data$categorical_levels[[response_index + 1]]
      if (is.null(result$class.levels)) {
        result$class.levels <- paste0("class.", seq_len(result$num.classes))
      }
      colnames(result$predictions.prob) <- result$class.levels
      names(result$misc.error) <- result$class.levels
      dimnames(result$confusion) <- list(
        observations = result$class.levels,
        predicted = result$class.levels
      )
      class(result) <- c("jf_tree", "list")
      return(result)
    } else {
      # for regression, the default minimal node size is 5
      if (is.null(min_node_size)) {
        min_node_size <- 5
      }
      # set default splitting rule (mean squared error for regression)
      if (is.null(splitrule)) {
        splitrule <- "mse"
      }

      result <- JFCppTree(1, processed_data$data, mtry, min_node_size, max_depth, nsplits, splitrule, honest,
                          response_index, feature_indices, processed_data$categorical,
                          processed_data$unique_values, seed, 0, splitrule_par)
      result$categorical.levels <- processed_data$categorical_levels
      class(result) <- c("jf_tree", "list")
      return(result)
    }
    
  }
  # if left hand side is "Surv(time, status)", survival
  else if (lhs[1] == "Surv") {
    # preprocess the entire dataset
    processed_data <- preprocess_data(data)
    # ensure that time is the first argument
    response_indices <- which(names(data) %in% as.character(formula[[2]])[2:3])
    if (length(unique(data[, response_indices[1]])) == 2) {
      temp <- response_indices
      response_indices <- c(temp[2], temp[1])
    }
    response_indices <- response_indices - 1  # for C++
    
    # formula determines the type of tree, the features and the response. if the
    # user supplies ~ ., all covariates are used
    if (formula[[3]] == ".") {
      covariates <- names(data)[-(response_indices + 1)]
    } else {
      covariates <- attr(terms(formula), "term.labels")
    }
    feature_indices <- which(names(data) %in% covariates) - 1

    # by default, the number of variables tried at each split is the
    # square root of the number of covariates
    if (is.null(mtry)) {
      mtry <- ceiling(sqrt(length(covariates)))
    }
    # for survival, the default minimal node size is 15
    if (is.null(min_node_size)) {
      min_node_size <- 15
    }
    # set default splitting rule (log-rank for survival)
    if (is.null(splitrule)) {
      splitrule <- "logrank"
    }

    result <- JFCppTree(3, processed_data$data, mtry, min_node_size, max_depth, nsplits, splitrule, honest,
                        response_indices, feature_indices, processed_data$categorical,
                        processed_data$unique_values, seed, num_event_times, splitrule_par)
    result$categorical.levels <- processed_data$categorical_levels
    class(result) <- c("jf_tree", "list")
    return(result)
  }
  # if the left hand side is "MM", multi-state
  else if (lhs[1] == "MM") {
    if (formula[[3]] == ".") {
      covariates <- names(feature_data)
    } else {
      covariates <- attr(terms(formula), "term.labels")
    }
    feature_indices <- which(names(feature_data) %in% covariates) - 1

    # by default, the number of variables tried at each split is the
    # square root of the number of covariates
    if (is.null(mtry)) {
      mtry <- ceiling(sqrt(length(covariates)))
    }
    # for multi-states, the default minimal node size is 20
    if (is.null(min_node_size)) {
      min_node_size <- 20
    }
    # set default splitting rule (log-rank for multi-states so far)
    if (is.null(splitrule)) {
      splitrule <- "logrank"
    }

    # for multi-state trees, we need to process the feature data
    if (is.null(feature_data)) {
      stop("For multi-state trees, features have to be provided in feature_data")
    } else {
      processed_data <- preprocess_data(feature_data)
    }

    # determine number of states and max_response_length
    max_response_length <- max(sapply(data, function(e) length(e$states)))
    num_states <- length(unique(unlist(lapply(data, '[[', "states"))))
    if (is.null(state_weights)) {
      state_weights <- numeric(0)
    }
    fh_weights_a <- normalize_fh_weights(fh_weights_a, "fh_weights_a")
    fh_weights_b <- normalize_fh_weights(fh_weights_b, "fh_weights_b")

    # data here is jump data, a list of lists, each containing a vector 'times' and a vector 'states'
    result <- JFCppTreeMultistate(data, max_response_length, num_states, processed_data$data,
                          mtry, min_node_size, max_depth, nsplits, splitrule, honest, feature_indices,
                          processed_data$categorical, processed_data$unique_values, seed,
                          state_weights, num_event_times, fh_weights_a, fh_weights_b)
    result$categorical.levels <- processed_data$categorical_levels
    result$num.states <- num_states
    result$predictions <- as_jf_multistate_predictions(
      result$predictions, result$unique.event.times, num_states, model_type = "tree")
    class(result) <- c("jf_multistate_tree", "jf_tree", "list")
    return(result)

  } else {
    stop("Type of tree not recognised from the formula.")
  }
}

#' Predict from a fitted tree
#'
#' @param tree_list A fitted tree object from [jftree()].
#' @param new_data Optional data.frame containing all fitted feature names,
#'   with the same variable types as the training data. Response columns are
#'   not needed; extra columns are ignored. Non-missing categorical values
#'   must occur in the training levels. Missing feature values are sent to the
#'   right daughter at each split on that feature. `NULL` returns
#'   training predictions.
#' @param compute_censoring Whether to return censoring survival probabilities
#'   estimated by leaf Kaplan-Meier estimators. Defaults to `FALSE`; only
#'   valid for survival and multi-state trees. For survival trees this flag
#'   applies when `new_data` is supplied; for multi-state trees it also applies
#'   to training predictions.
#' @param compute_initial Should predicted initial distributions also be returned?
#'   Defaults to `FALSE`; only valid for multi-state trees.
#'
#' @return Regression: A numeric vector with one predicted response per training
#'   observation. With `new_data`, a numeric matrix with one column and one
#'   row per observation.
#'
#'   Classification: A numeric vector of predicted class codes for training
#'   observations. With `new_data`, a numeric matrix whose first column,
#'   `prediction`, contains the class codes and whose remaining columns contain
#'   class probabilities, named by `tree_list$class.levels`. Class codes index
#'   these class levels. Training class probabilities are stored separately in
#'   `tree_list$predictions.prob`.
#'
#'   Survival: A matrix of estimated cumulative hazards (Nelson-Aalen
#'   estimators), with one row per observation and one column per time in
#'   `tree_list$unique.event.times`. With `new_data` and
#'   `compute_censoring = TRUE`, a list containing this matrix in `predictions`
#'   and censoring Kaplan-Meier estimates on the same grid in `censoring`.
#'   The list also contains `censoring.full`, a matrix of censoring estimates
#'   on the censoring-time grid given by `censoring.times`. Without `new_data`,
#'   the returned object is the cumulative hazard matrix; stored training
#'   censoring estimates can be accessed through `tree_list$censoring` and
#'   `tree_list$censoring.full`.
#'
#'   Multi-state: Predictions are a list of individuals when both optional
#'   estimates are disabled; access one individual with `result[[i]]`.
#'   Otherwise, `result$predictions` contains the Nelson-Aalen estimators.
#'   The fields `init` and `censoring` contain the optional estimates when
#'   requested. Censoring estimates are Kaplan-Meier censoring survival
#'   probabilities, with one row per individual and one column per time in
#'   `tree_list$unique.event.times`. When requested, `censoring.full` and
#'   `censoring.times` also provide the full censoring-time grid.
#'   Multi-state results print compact summaries.
#' @md
#' @export
#'
# the main function for predicting with trees
jftree.predict <- function(tree_list, new_data = NULL, compute_censoring = FALSE, compute_initial = FALSE) {
  if (tree_list$tree.type == "Multi-state") {
    num_states <- tree_list$num.states
    if (is.null(num_states)) {
      num_states <- length(tree_list$init[[1]])
    }
  }
  # if data is not supplied, return predictions based on training data
  if (is.null(new_data)) {
    if (tree_list$tree.type == "Multi-state") {
      predictions <- as_jf_multistate_predictions(
        tree_list$predictions, tree_list$unique.event.times, num_states, model_type = "tree")
      if (!compute_initial && !compute_censoring) {
        return(predictions)
      }
      result <- list(predictions = predictions)
      if (compute_initial) {
        result$init <- tree_list$init
      }
      if (compute_censoring) {
        result$censoring <- tree_list$censoring
        result$censoring.full <- tree_list$censoring.full
        result$censoring.times <- tree_list$censoring.times
      }
      return(as_jf_multistate_prediction_result(result, model_type = "tree"))
    } else {
      return(tree_list$predictions)
    }
  }

  # when new data is supplied
  if (inherits(new_data, "data.frame")) {
    # if new data is supplied, start by preprocessing the data and
    # extracting relevant columns
    covariates <- tree_list$feature.names
    if (ncol(new_data) > 1) {
      new_data <- new_data[, covariates, drop = FALSE]  # ensures the columns have the same order as the original dataset
    }
    feature_indices <- which(names(new_data) %in% covariates) - 1
    processed_data <- preprocess_data(new_data, tree_list$categorical.levels)

    # may need to be adapted when more types of trees are implemented
    if (tree_list$tree.type != "Multi-state") {
      if (compute_censoring) {
        return(JFCppTreePredictCensoring(tree_list, processed_data$data, feature_indices,
                            processed_data$categorical, processed_data$unique_values))
      } else {
        predictions <- JFCppTreePredict(tree_list, processed_data$data, feature_indices,
                            processed_data$categorical, processed_data$unique_values)
        if (tree_list$tree.type == "Classification") {
          colnames(predictions) <- c("prediction", tree_list$class.levels)
        }
        return(predictions)
      }
    } else {
      result <- JFCppTreePredictMultistate(tree_list, processed_data$data, feature_indices, processed_data$categorical,
             processed_data$unique_values, compute_initial, compute_censoring)
      if (compute_initial || compute_censoring) {
        result$predictions <- as_jf_multistate_predictions(
          result$predictions, tree_list$unique.event.times, num_states, model_type = "tree")
        return(as_jf_multistate_prediction_result(result, model_type = "tree"))
      }
      return(as_jf_multistate_predictions(result, tree_list$unique.event.times, num_states, model_type = "tree"))
    }
  } else {
    cat("Error: If new_data is supplied, it must be a data.frame with the same names as the original dataset \n")
  }
}

#' Compute tree prediction error
#'
#' @param tree_list A fitted tree object from [jftree()].
#' @param new_data Optional evaluation data.frame. For regression,
#'   classification, and survival, include all fitted feature and response
#'   columns. For multi-state models, include the features here and supply
#'   responses in `jump_data`. Variable types and categorical levels must
#'   match the training data; responses must be fully observed apart from
#'   the model's censoring representation. `NULL` returns stored training
#'   errors.
#' @param jump_data Multi-state response paths in the same format as
#'   `jftree()`'s `data`, with one path per row of `new_data` in the same order.
#'   Required only for evaluating multi-state models on new data; defaults
#'   to `NULL`.
#' @param state_weights Multi-state error weights as described in [jftree()].
#'   With new data, `NULL` uses the loss-specific defaults, regardless of
#'   weights used during fitting. Without new data, stored errors are returned
#'   and this argument is ignored.
#'
#' @details Without `new_data`, errors are the stored training errors. With
#'   `new_data`, predictions and errors are computed on those observations.
#'   Survival and multi-state probability scores account for censoring using
#'   inverse censoring probability weights estimated by the fitted tree.
#'
#' @return A named list of error metrics. Smaller losses are better; larger
#'   `R2` is better.
#'
#'   * Regression: `mse.error` is mean squared error. `R2` is
#'     \eqn{1 - \mathrm{MSE}/\mathrm{Var}(y)}{1 - MSE / Var(y)}, using the empirical variance
#'     with divisor `n` on the evaluated responses; it can be negative.
#'   * Classification: `BS.error` is the squared class-probability error
#'     averaged over observations and classes. `normalised.BS.error` divides
#'     this by the score for uniform class probabilities, so 1 is the uniform
#'     baseline. `misclassification.error.total` is the overall proportion
#'     of wrong labels; `misclassification.error` gives this proportion within
#'     each observed class (or `NA` if that class is absent).
#'     `confusion.matrix` contains counts, with observed classes in rows and
#'     predicted classes in columns.
#'   * Survival: `C.error` is `1 - C-index`, so smaller values mean better
#'     concordance. `IBS.error` is the integrated Brier score for survival
#'     probabilities; `IKL.error` is the integrated Kullback-Leibler
#'     (logarithmic) loss. Their time-normalised counterparts are
#'     `normalised.IBS.error` and `normalised.IKL.error`.
#'   * Multi-state: `IBS.error`, `IKL.error`, and `IS.error` are the integrated
#'     Brier, Kullback-Leibler, and spherical losses for state occupation
#'     probabilities, with state weights as specified above. The corresponding
#'     fields are `normalised.IBS.error`, `normalised.IKL.error`, and
#'     `normalised.IS.error`.
#'
#'   Integrated scores use trapezoidal integration on the fitted event-time
#'   grid. Their `normalised.*` versions divide by the grid's final time,
#'   giving an average loss over time; normalisation
#'   does not guarantee a value between zero and one. This differs from the
#'   uniform-probability normalisation used for classification Brier scores.
#' @md
#' @export
#'
jftree.error <- function(tree_list, new_data = NULL, jump_data = NULL, state_weights = NULL) {
  # if data is not supplied, return the error based on training data
  if (is.null(new_data)) {
    if (tree_list$tree.type == "Regression") {
      return(list("mse.error" = tree_list$mse.error, "R2" = tree_list$R2))
    }
    if (tree_list$tree.type == "Classification") {
      return(list("BS.error" = tree_list$bs, "normalised.BS.error" = tree_list$bs.normalised,
                  "misclassification.error.total" = tree_list$misc.error.total , "misclassification.error" = tree_list$misc.error,
                  "confusion.matrix" = tree_list$confusion))
    }
    if (tree_list$tree.type == "Survival") {
      return(list("C.error" = tree_list$C.error, "IBS.error" = tree_list$ibs, "normalised.IBS.error" = tree_list$ibs.normalised,
                  "IKL.error" = tree_list$ikl, "normalised.IKL.error" = tree_list$ikl.normalised))
    }
    if (tree_list$tree.type == "Multi-state") {
      return(list("IBS.error" = tree_list$ibs, "normalised.IBS.error" = tree_list$ibs.normalised,
                  "IKL.error" = tree_list$ikl, "normalised.IKL.error" = tree_list$ikl.normalised,
                  "IS.error" = tree_list$is, "normalised.IS.error" = tree_list$is.normalised))
    }
  }

  # we need both jump data and feature data, so we need to check if the user has supplied a list of jump data and a data.frame of feature data
  covariates <- tree_list$feature.names
  if (tree_list$tree.type == "Multi-state") {
    if (is.null(jump_data) || !is.data.frame(new_data)) {
      stop("For multi-state trees, jump_data and feature_data must be supplied.")
    }

    missing_columns <- setdiff(covariates, names(new_data))
    if (length(missing_columns) > 0) {
      stop("new_data is missing column(s): ", paste(missing_columns, collapse = ", "))
    }

    # ensures the response and features have the same order as the original dataset
    new_data <- new_data[, covariates, drop = FALSE]
    feature_indices <- which(names(new_data) %in% covariates) - 1
    processed_data <- preprocess_data(new_data, tree_list$categorical.levels)

    # determine response dimensions
    max_response_length <- max(sapply(jump_data, function(e) length(e$states)))
    num_states <- length(tree_list$init[[1]])
    if (is.null(state_weights)) {
      state_weights <- numeric(0)
    }

    return(JFCppTreeErrorMultistate(tree_list, max_response_length, num_states, jump_data, processed_data$data, feature_indices,
                      processed_data$categorical, processed_data$unique_values, state_weights))
  }

  # if new_data is supplied, compute predictions and error from scratch
  response <- tree_list$response.names
  missing_columns <- setdiff(c(response, covariates), names(new_data))
  if (length(missing_columns) > 0) {
    stop("new_data is missing column(s): ", paste(missing_columns, collapse = ", "))
  }

  # ensures the response and features have the same order as the original dataset
  new_data <- new_data[, c(response, covariates), drop = FALSE]
  response_indices <- which(names(new_data) %in% response) - 1
  feature_indices <- which(names(new_data) %in% covariates) - 1
  processed_data <- preprocess_data(new_data, tree_list$categorical.levels)
  result <- JFCppTreeError(tree_list, processed_data$data, feature_indices,
                           processed_data$categorical, processed_data$unique_values, response_indices)
  if (tree_list$tree.type == "Classification") {
    names(result$misclassification.error) <- tree_list$class.levels
    dimnames(result$confusion.matrix) <- list(
      observations = tree_list$class.levels,
      predicted = tree_list$class.levels
    )
  }
  return(result)
}

#' Fit a JumpForests forest
#'
#' Fits a random forest for regression, classification, survival, or multi-state data.
#'
#' @param formula Model formula: `y ~ x1 + x2` for regression or classification,
#'   `Surv(time, status) ~ x1 + x2` for survival (`status = 1` for an event,
#'   `0` for censoring), or `MM ~ x1 + x2` for multi-state models. A numeric
#'   response selects regression; a factor, character, or logical response
#'   selects classification. Convert numeric class labels to a factor to fit
#'   a classification forest. Numeric features are treated as continuous;
#'   factor, character, and logical features are treated as categorical.
#' @param data Training data. A data.frame including both responses and features for regression,
#'   classification and survival. For multi-state models, a list of lists (one for each
#'   observation), each containing two vectors, `"times"` (observed jump times) and
#'   `"states"` (the state occupied at the corresponding time). The states must be encoded
#'   as integers. Repetition of the final state indicates censoring.
#' @param feature_data Feature data as a data.frame. Only required for multi-state models.
#' @param splitrule Splitting rule. `NULL` uses the model-specific defaults
#'   below. For regression, choose `"mse"` (default),
#'   `"variance"` (an alias for `"mse"`), `"mae"`, `"binomial"`,
#'   `"negativebinomial"`, `"poisson"`, `"gamma"`, `"inversegaussian"`,
#'   `"tweedie"`, or `"huber"`. Binomial responses must lie in `[0, 1]`;
#'   negative-binomial and Poisson responses must be non-negative integers;
#'   Gamma and inverse-Gaussian responses must be positive. Tweedie support
#'   follows its power: real for \eqn{\xi \le 0}, non-negative counts at
#'   \eqn{\xi = 1}, non-negative for \eqn{1 < \xi < 2}, and positive for
#'   \eqn{\xi \ge 2}. For classification, choose `"gini"` (default), `"entropy"`,
#'   `"misc"`, `"twoing"` or `"hellinger"`. For survival, choose `"logrank"`
#'   (default), `"approxlogrank"` or `"conserve"`. For multi-state models,
#'   choose `"logrank"` (default), `"gehan"`, `"taroneware"`, `"approxlogrank"`,
#'   `"conserve"`, `"petoprentice"` or `"flemingharrington"`. Choosing `"flemingharrington"`
#'   allows for specifying the weights, see the options `"fh_weights_a"` and
#'   `"fh_weights_b"`.
#' @param splitrule_par Numeric scalar for parameterised regression rules:
#'   negative-binomial size \eqn{k > 0} (default 1), Tweedie power
#'   \eqn{\xi \le 0} or \eqn{\xi \ge 1}, or Huber threshold
#'   \eqn{\delta > 0}. It is required for Tweedie and Huber and must be `NULL`
#'   for every other rule and model type. The default is `NULL`.
#' @param mtry Number of randomly selected candidate features at each split.
#'   `NULL` uses `ceiling(sqrt(p))`, where `p` is the number of features.
#' @param min_node_size Minimum number of observations required in each
#'   daughter node for a split. `NULL` uses 5 for regression, 1 for
#'   classification, 15 for survival, and 20 for multi-state models.
#' @param max_depth Maximum depth allowed for each tree, where the root has
#'   depth zero. `NULL` (the default) imposes no depth limit.
#' @param nsplits Number of randomly sampled candidate thresholds per
#'   continuous feature, or candidate partitions per categorical feature.
#'   Defaults to 10; fewer candidates are used if fewer distinct splits exist.
#' @param ntrees Number of trees. `NULL` uses 1000 for regression and
#'   classification, and 500 for survival and multi-state models.
#' @param honest Whether to use separate sets for choosing splits and
#'   estimating leaf predictions, each approximately half of the tree's
#'   sample. Defaults to `FALSE`, which uses the same observations for both
#'   tasks. See `double_bootstrap` for how these sets are sampled.
#' @param swr Whether to sample with replacement. Defaults to `FALSE`.
#' @param sample_rate Fraction of observations sampled for each tree, in
#'   `(0, 1]`. `NULL` uses 0.7 without replacement (`swr = FALSE`) and 1 with
#'   replacement (`swr = TRUE`). With `honest = TRUE` and
#'   `double_bootstrap = TRUE`, the rate applies to each half separately.
#' @param double_bootstrap Only relevant when `honest = TRUE`. If `TRUE`,
#'   first partition the training observations into disjoint halves, then
#'   sample separately within each half for splitting and leaf estimation.
#'   If `FALSE` (the default), sample first and then partition the sample
#'   into halves. With replacement, the latter can put copies of the same
#'   observation in both halves.
#' @param seed Random seed. `NULL` draws a seed using R's current random-number
#'   state; supply a seed for reproducible fitting.
#' @param nworkers Number of worker threads. The default, 0, uses all
#'   hardware threads reported by the system.
#' @param save_predictions Whether to compute and store ordinary training
#'   predictions, OOB predictions, and errors during fitting for survival and
#'   multi-state forests. Defaults to `TRUE`; `FALSE` defers these calculations
#'   until needed. Regression and classification currently always store
#'   predictions and errors, regardless of this flag.
#' @param num_event_times Maximum number of times in the prediction grid for
#'   survival and multi-state models. The default, 0, retains the full grid.
#'   A positive value coarsens the grid when necessary; the resulting times
#'   are stored in `unique.event.times`.
#' @param state_weights Numeric vector of finite non-negative weights, one
#'   per state in the order `1, ..., num_states`, for multi-state error
#'   calculations. Supplied weights are used directly for all scores and are
#'   not normalised. `NULL` uses `1 / num_states` per state for Brier and
#'   spherical scores and 1 per state for Kullback-Leibler loss. These weights
#'   affect error calculations, not splitting or prediction.
#' @param fh_weights_a Optional finite non-negative Fleming--Harrington
#'   a-exponents, one for each state `1, ..., num_states`. `NULL` uses one for
#'   every state. Only valid for multi-state forests with
#'   `splitrule = "flemingharrington"`. A transition from state `s` is weighted
#'   by \eqn{p_s(t)^{a_s}(1-p_s(t))^{b_s}}, where \eqn{p_s(t)} is the
#'   estimated occupation probability of state `s` in the parent node.
#' @param fh_weights_b Optional finite non-negative Fleming--Harrington
#'   b-exponents, with the same ordering and default as `fh_weights_a`.
#'
#' @return A fitted forest object as a list with class `jf_forest`. Use
#'   `print()` for a summary and `predict()` for predictions. When saved,
#'   `predictions` contains ordinary training predictions using all trees,
#'   and `oob.predictions` contains OOB predictions. Classification also
#'   stores the corresponding probability matrices in `predictions.prob`
#'   and `oob.predictions.prob`. For survival and multi-state forests,
#'   `censoring.oob` contains OOB estimates of the censoring survival
#'   probability obtained by averaging leaf Kaplan-Meier estimators, with
#'   one row per observation and one column per time in `unique.event.times`.
#'   OOB estimates require OOB trees for the observation. With
#'   `save_predictions = FALSE`, survival and multi-state
#'   prediction and error fields are initially `NULL`.
#' @md
#' @export
#'
# the main function for fitting forests (feature_data is only relevant for multi-state trees in which case data is a list and not a data.frame)
jfforest <- function(formula, data, feature_data = NULL, splitrule = NULL, mtry = NULL, min_node_size = NULL, max_depth = NULL, nsplits = 10,
                     ntrees = NULL, honest = FALSE, swr = FALSE, sample_rate = NULL, double_bootstrap = FALSE, 
                     seed = NULL, nworkers = 0, save_predictions = TRUE, num_event_times = 0, state_weights = NULL,
                     fh_weights_a = NULL, fh_weights_b = NULL, splitrule_par = NULL) {
  lhs <- as.character(formula[[2]])
  splitrule_par <- normalize_splitrule_par(splitrule_par)
  max_depth <- normalize_max_depth(max_depth)
  if (lhs[1] != "MM" && (!is.null(fh_weights_a) || !is.null(fh_weights_b))) {
    stop("fh_weights_a and fh_weights_b are only valid for multi-state models")
  }
  if (lhs[1] %in% c("MM", "Surv") && length(splitrule_par) > 0) {
    stop("splitrule_par is only valid for regression models")
  }

  # if seed is not set, generate a random one
  if (is.null(seed)) {
    seed <- runif(n = 1, min = 1, max = 10^6)
  }

  if (is.null(sample_rate)) {
    if (swr) {
      sample_rate <- 1.0
    } else {
      sample_rate <- 0.7
    }
  }

  # check that sample_rate is in (0, 1]
  if (sample_rate > 1 || sample_rate <= 0) {
    if (swr) {
      cat("sample_rate should be in (0, 1], setting sample_rate to default for sampling with replacement (1)")
      sample_rate <- 1.0
    } else {
      cat("sample_rate should be in (0, 1], setting sample_rate to default for sampling without replacement (0.7)")
      sample_rate <- 0.7
    }
  }

  # if left hand side is of length 1, classification, regression or multi-state
  if (length(lhs) == 1 && lhs[1] != "MM") {
    # preprocess the entire dataset
    processed_data <- preprocess_data(data)
    response_index <- which(names(data) %in% as.character(formula[[2]])[1]) - 1
    # formula determines the type of tree, the features and the response. if the
    # user supplies ~ ., all covariates are used
    if (formula[[3]] == ".") {
      covariates <- names(data)[-(response_index + 1)]
    } else {
      covariates <- attr(terms(formula), "term.labels")
    }
    feature_indices <- which(names(data) %in% covariates) - 1

    # by default, the number of variables tried at each split is the
    # square root of the number of covariates
    if (is.null(mtry)) {
      mtry <- ceiling(sqrt(length(covariates)))
    }

    # if the response is categorical, classification, otherwise regression
    if (processed_data$categorical[response_index + 1]) {
      if (length(splitrule_par) > 0) {
        stop("splitrule_par is only valid for regression models")
      }
      # for classification, the default minimal node size is 1
      if (is.null(min_node_size)) {
        min_node_size <- 1
      }
      # for classification, the default number of trees is 1000
      if (is.null(ntrees)) {
        ntrees <- 1000
      }
      # set default splitting rule
      if (is.null(splitrule)) {
        splitrule <- "gini"
      }
      result <- JFCppForest(2, processed_data$data, mtry, min_node_size, max_depth, nsplits, splitrule, ntrees, honest, swr,
                            sample_rate, double_bootstrap, response_index, feature_indices, processed_data$categorical,
                            processed_data$unique_values, seed, nworkers, save_predictions, 0, splitrule_par)
      result$categorical.levels <- processed_data$categorical_levels
      result$class.levels <- processed_data$categorical_levels[[response_index + 1]]
      if (is.null(result$class.levels)) {
        result$class.levels <- paste0("class.", seq_len(result$num.classes))
      }
      colnames(result$predictions.prob) <- result$class.levels
      names(result$misc.error) <- result$class.levels
      dimnames(result$confusion) <- list(
        observations = result$class.levels,
        predicted = result$class.levels
      )
      class(result) <- c("jf_forest", "list")
      return(result)
    } else {
      # for regression, the default minimal node size is 5
      if (is.null(min_node_size)) {
        min_node_size <- 5
      }
      # for regression, the default number of trees is 1000
      if (is.null(ntrees)) {
        ntrees <- 1000
      }
      # for regression, the default splitting rule is mean squared error
      if (is.null(splitrule)) {
        splitrule <- "mse"
      }
      result <- JFCppForest(1, processed_data$data, mtry, min_node_size, max_depth, nsplits, splitrule, ntrees, honest, swr,
                            sample_rate, double_bootstrap, response_index, feature_indices, processed_data$categorical,
                            processed_data$unique_values, seed, nworkers, save_predictions, 0, splitrule_par)
      result$categorical.levels <- processed_data$categorical_levels
      class(result) <- c("jf_forest", "list")
      return(result)
    }

  }
  # if left hand side is "Surv(time, status)", survival
  else if (lhs[1] == "Surv") {
    processed_data <- preprocess_data(data)
    # time is always assumed to be the first argument
    response_indices <- which(names(data) %in% as.character(formula[[2]])[2:3])
    if (length(unique(data[, response_indices[1]])) == 2) {
      temp <- response_indices
      response_indices <- c(temp[2], temp[1])
    }
    response_indices <- response_indices - 1

    # formula determines the type of tree, the features and the response. if the
    # user supplies ~ ., all covariates are used
    if (formula[[3]] == ".") {
      covariates <- names(data)[-(response_indices + 1)]
    } else {
      covariates <- attr(terms(formula), "term.labels")
    }
    feature_indices <- which(names(data) %in% covariates) - 1

    # by default, the number of variables tried at each split is the
    # square root of the number of covariates
    if (is.null(mtry)) {
      mtry <- ceiling(sqrt(length(covariates)))
    }
    # for survival, the default minimal node size is 15
    if (is.null(min_node_size)) {
      min_node_size <- 15
    }
    # for survival, the default number of trees is 500
    if (is.null(ntrees)) {
      ntrees <- 500
    }
    # for survival, the default splitting rule is log-rank
    if (is.null(splitrule)) {
      splitrule <- "logrank"
    }

    result <- JFCppForest(3, processed_data$data, mtry, min_node_size, max_depth, nsplits, splitrule, ntrees, honest, swr,
                          sample_rate, double_bootstrap, response_indices, feature_indices, processed_data$categorical,
                          processed_data$unique_values, seed, nworkers, save_predictions, num_event_times,
                          splitrule_par)
    result$categorical.levels <- processed_data$categorical_levels
    class(result) <- c("jf_forest", "list")
    return(result)
  }
  # if the left hand side is "MM(...)", multi-state
  else if (lhs[1] == "MM") {
    if (formula[[3]] == ".") {
      covariates <- names(feature_data)
    } else {
      covariates <- attr(terms(formula), "term.labels")
    }
    feature_indices <- which(names(feature_data) %in% covariates) - 1

    # by default, the number of variables tried at each split is the
    # square root of the number of covariates
    if (is.null(mtry)) {
      mtry <- ceiling(sqrt(length(covariates)))
    }
    # for multi-states, the default minimal node size is 20
    if (is.null(min_node_size)) {
      min_node_size <- 20
    }
    # use the same default forest size as survival forests
    if (is.null(ntrees)) {
      ntrees <- 500
    }
    # set default splitting rule (log-rank for multi-states so far)
    if (is.null(splitrule)) {
      splitrule <- "logrank"
    }

    # for multi-state trees, we need to process the feature data
    if (is.null(feature_data)) {
      stop("For multi-state trees, features have to be provided in feature_data")
    } else {
      processed_data <- preprocess_data(feature_data)
    }

    # determine number of states and max_response_length
    max_response_length <- max(sapply(data, function(e) length(e$states)))
    num_states <- length(unique(unlist(lapply(data, '[[', "states"))))
    if (is.null(state_weights)) {
      state_weights <- numeric(0)
    }
    fh_weights_a <- normalize_fh_weights(fh_weights_a, "fh_weights_a")
    fh_weights_b <- normalize_fh_weights(fh_weights_b, "fh_weights_b")

    result <- JFCppForestMultistate(data, max_response_length, num_states, processed_data$data, mtry, min_node_size,
                            max_depth, nsplits, splitrule, ntrees, honest, swr, sample_rate, double_bootstrap, feature_indices,
                            processed_data$categorical, processed_data$unique_values, seed, nworkers, save_predictions, 
                            state_weights, num_event_times, fh_weights_a, fh_weights_b)
    result$categorical.levels <- processed_data$categorical_levels
    result$predictions <- as_jf_multistate_predictions(result$predictions, result$unique.event.times, result$num.states)
    result$oob.predictions <- as_jf_multistate_predictions(result$oob.predictions, result$unique.event.times, result$num.states)
    class(result) <- c("jf_multistate_forest", "jf_forest", "list")
    return(result)

  } else {
    stop("Type of tree not recognised from the formula.")
  }
}

normalize_fh_weights <- function(weights, argument_name) {
  if (is.null(weights)) {
    return(numeric(0))
  }
  if (!is.numeric(weights) || !is.null(dim(weights))) {
    stop(argument_name, " must be a numeric vector")
  }
  as.numeric(weights)
}

normalize_splitrule_par <- function(splitrule_par) {
  if (is.null(splitrule_par)) {
    return(numeric(0))
  }
  if (!is.numeric(splitrule_par) || !is.null(dim(splitrule_par)) ||
      length(splitrule_par) != 1) {
    stop("splitrule_par must be a single numeric value")
  }
  as.numeric(splitrule_par)
}

normalize_max_depth <- function(max_depth) {
  if (is.null(max_depth)) {
    return(Inf)
  }
  if (!is.numeric(max_depth) || !is.null(dim(max_depth)) ||
      length(max_depth) != 1L || is.na(max_depth) || max_depth < 0 ||
      (!is.infinite(max_depth) && max_depth != floor(max_depth))) {
    stop("max_depth must be a non-negative whole number or Inf")
  }
  as.numeric(max_depth)
}

#' Predict from a fitted forest
#'
#' @param forest_list A fitted forest object from [jfforest()].
#' @param new_data Optional data.frame containing all fitted feature names,
#'   with the same variable types as the training data. Response columns are
#'   not needed; extra columns are ignored. Non-missing categorical values
#'   must occur in the training levels. Missing feature values are sent to the
#'   right daughter at each split on that feature. `NULL` returns ordinary
#'   training predictions using all trees. Stored OOB predictions are in
#'   `forest_list$oob.predictions`.
#' @param compute_censoring Should the censoring survival distribution also be
#'   estimated? For multi-state forests, returns out-of-bag estimates for
#'   training data and ordinary estimates for `new_data`. For survival forests,
#'   this applies when `new_data` is supplied. Defaults to `FALSE`; only
#'   valid for survival and multi-state forests. Estimates are averages of
#'   leaf Kaplan-Meier censoring survival probabilities.
#' @param compute_initial Should predicted initial distributions also be returned?
#'   For multi-state forests with deferred predictions, this also controls
#'   whether they are computed during the prediction call. Defaults to
#'   `FALSE`; only valid for multi-state forests.
#'
#' @details With `new_data = NULL`, returned predictions (and multi-state
#'   initial distributions when requested) use all trees. OOB predictions
#'   are stored separately in `forest_list$oob.predictions`; training
#'   censoring estimates use OOB trees. With `new_data`, all returned
#'   estimates use all trees. Training censoring estimates require OOB trees
#'   for the observation. With deferred predictions, this call
#'   computes the requested estimates; use the returned object to access them.
#'
#' @return Regression: A numeric vector with one predicted response per training
#'   observation. With `new_data`, a numeric matrix with one column and one
#'   row per observation.
#'
#'   Classification: A numeric vector of predicted class codes for training
#'   observations. With `new_data`, a numeric matrix whose first column,
#'   `prediction`, contains the class codes and whose remaining columns contain
#'   class probabilities, named by `forest_list$class.levels`. Class codes index
#'   these class levels. Training class probabilities are stored separately in
#'   `forest_list$predictions.prob`.
#'
#'   Survival: A matrix of estimated cumulative hazards (Nelson-Aalen
#'   estimators), with one row per observation and one column per time in
#'   `forest_list$unique.event.times`. With `new_data` and
#'   `compute_censoring = TRUE`, a list containing this matrix in `predictions`
#'   and censoring Kaplan-Meier estimates on the same grid in `censoring`.
#'   Without `new_data`, the returned object is the cumulative hazard matrix;
#'   stored out-of-bag censoring estimates, when available, can be accessed
#'   through `forest_list$censoring.oob`.
#'
#'   Multi-state: Predictions are a list of subjects when both optional
#'   estimates are disabled; access one subject with `result[[subject]]`.
#'   Otherwise, `result$predictions` contains the subject predictions. With
#'   `compute_initial = TRUE`, the result also has `init`. With
#'   `compute_censoring = TRUE`, it has `censoring`, a subject-by-event-time
#'   matrix. Training censoring estimates use out-of-bag trees; new-data
#'   estimates use all trees. The names `initial` for new-data initial
#'   distributions and `censoring.oob` for training censoring estimates are
#'   retained as aliases.
#' @md
#' @export
#'
# the main function for predicting with forests
jfforest.predict <- function(forest_list, new_data = NULL, compute_censoring = FALSE, compute_initial = FALSE) {
  # if new data is not supplied and predictions are already computed, return predictions based on training data
  if (is.null(new_data) & !is.null(forest_list$predictions)) {
    if (forest_list$tree.type != "Multi-state") {
      return(forest_list$predictions)
    } else {
      predictions <- as_jf_multistate_predictions(forest_list$predictions, forest_list$unique.event.times, forest_list$num.states)
      if (!compute_initial && !compute_censoring) {
        return(predictions)
      }
      result <- list("predictions" = predictions)
      if (compute_initial) {
        result$init <- forest_list$init
      }
      if (compute_censoring) {
        result$censoring.oob <- forest_list$censoring.oob
      }
      return(as_jf_multistate_prediction_result(result))
    }
  }
  # if new data is not supplied and predictions are not computed, compute the predictions based on the training data (in-bag and oob)
  if (is.null(new_data) & is.null(forest_list$predictions)) {
    cat("Computing predictions...\n")
    forest_list <- JFCppForestPredictTraining(forest_list, compute_initial, compute_censoring) # computes predictions from scratch
    if (forest_list$tree.type != "Multi-state") {
      return(forest_list$predictions)
    } else {
      predictions <- as_jf_multistate_predictions(forest_list$predictions, forest_list$unique.event.times, forest_list$num.states)
      if (!compute_initial && !compute_censoring) {
        return(predictions)
      }
      result <- list("predictions" = predictions)
      if (compute_initial) {
        result$init <- forest_list$init
      }
      if (compute_censoring) {
        result$censoring.oob <- forest_list$censoring.oob
      }
      return(as_jf_multistate_prediction_result(result))
    }
  }

  # when new data is supplied
  if (inherits(new_data, "data.frame")) {
    # if new data is supplied, start by preprocessing the data and
    # extracting relevant columns
    covariates <- forest_list$feature.names
    if (ncol(new_data) > 1) {
      new_data <- new_data[, covariates, drop = FALSE]  # ensures the columns have the same order as the original dataset
    }
    feature_indices <- which(names(new_data) %in% covariates) - 1
    processed_data <- preprocess_data(new_data, forest_list$categorical.levels)
    if (forest_list$tree.type != "Multi-state") {
      if (compute_censoring) {
        return(JFCppForestPredictCensoring(forest_list, processed_data$data, feature_indices,
                            processed_data$categorical, processed_data$unique_values))
      } else {
        predictions <- JFCppForestPredict(forest_list, processed_data$data, feature_indices,
                            processed_data$categorical, processed_data$unique_values)
        if (forest_list$tree.type == "Classification") {
          colnames(predictions) <- c("prediction", forest_list$class.levels)
        }
        return(predictions)
      }
    } else {
      result <- JFCppForestPredictMultistate(forest_list, processed_data$data, feature_indices,
                            processed_data$categorical, processed_data$unique_values, compute_initial, compute_censoring)
      if (compute_initial || compute_censoring) {
        result$predictions <- as_jf_multistate_predictions(result$predictions, forest_list$unique.event.times, forest_list$num.states)
        return(as_jf_multistate_prediction_result(result))
      }
      return(as_jf_multistate_predictions(result, forest_list$unique.event.times, forest_list$num.states))
    }
  } else {
    cat("Error: If new_data is supplied, it must be a data.frame with the same names as the original dataset \n")
  }
}

#' Compute forest prediction error
#'
#' @param forest_list A fitted forest object from [jfforest()].
#' @param new_data Optional evaluation data.frame. For regression,
#'   classification, and survival, include all fitted feature and response
#'   columns. For multi-state models, include the features here and supply
#'   responses in `jump_data`. Variable types and categorical levels must
#'   match the training data; responses must be fully observed apart from
#'   the model's censoring representation. `NULL` requests OOB errors.
#' @param jump_data Multi-state response paths in the same format as
#'   `jfforest()`'s `data`, with one path per row of `new_data` in the same order.
#'   Required only for evaluating multi-state models on new data; defaults
#'   to `NULL`.
#' @param state_weights Multi-state error weights as described in [jfforest()].
#'   Without new data, `NULL` returns stored OOB errors when available,
#'   retaining the weights used during fitting. Supplying weights recomputes
#'   the OOB errors with those weights. With new data, or when OOB errors must
#'   first be computed, `NULL` uses the loss-specific defaults.
#'
#' @details Without `new_data`, this function returns OOB errors, computing
#'   deferred survival or multi-state errors when needed. With `new_data`,
#'   predictions use all trees and errors are evaluated on those observations.
#'   Survival and multi-state probability scores account for censoring using
#'   inverse censoring probability weights. Training errors use OOB censoring
#'   estimates; new-data errors use censoring estimates from all trees.
#'
#' @return A named list of error metrics. Smaller losses are better; larger
#'   `R2` is better.
#'
#'   * Regression: `mse.error` for OOB evaluation, or `error.mse` with new
#'     data, is mean squared error. `R2` is
#'     \eqn{1 - \mathrm{MSE}/\mathrm{Var}(y)}{1 - MSE / Var(y)}, using the empirical variance
#'     with divisor `n` on the evaluated responses; it can be negative.
#'   * Classification: `BS.error` is the squared class-probability error
#'     averaged over observations and classes. `normalised.BS.error` divides
#'     this by the score for uniform class probabilities, so 1 is the uniform
#'     baseline. `misclassification.error.total` is the overall proportion
#'     of wrong labels; `misclassification.error` gives this proportion within
#'     each observed class (or `NA` if that class is absent).
#'     `confusion.matrix` contains counts, with observed classes in rows and
#'     predicted classes in columns.
#'   * Survival: `C.error` is `1 - C-index`, so smaller values mean better
#'     concordance. `IBS.error` is the integrated Brier score for survival
#'     probabilities; `KL.error` for OOB evaluation, or `IKL.error` with new
#'     data, is the integrated Kullback-Leibler (logarithmic) loss. The
#'     time-normalised fields are `normalised.IBS.error` and
#'     `normalised.KL.error` (OOB) or `normalised.IKL.error` (new data).
#'   * Multi-state: `IBS.error`, `IKL.error`, and `IS.error` are the integrated
#'     Brier, Kullback-Leibler, and spherical losses for state occupation
#'     probabilities, with state weights as specified above. The corresponding
#'     fields are `normalised.IBS.error`, `normalised.IKL.error`, and
#'     `normalised.IS.error`.
#'
#'   Integrated scores use trapezoidal integration on the fitted event-time
#'   grid. Their `normalised.*` versions divide by the grid's final time,
#'   giving an average loss over time; normalisation
#'   does not guarantee a value between zero and one. This differs from the
#'   uniform-probability normalisation used for classification Brier scores.
#' @md
#' @export
#'
jfforest.error <- function(forest_list, new_data = NULL, jump_data = NULL, state_weights = NULL) {
  # if data is not supplied and error metrics are already computed, return the error based on OOB data
  if (is.null(new_data)) {
    if (forest_list$tree.type == "Regression") {
      return(list("mse.error" = forest_list$mse.error, "R2" = forest_list$R2))
    }
    if (forest_list$tree.type == "Classification") {
      return(list("BS.error" = forest_list$bs, "normalised.BS.error" = forest_list$bs.normalised,
                  "misclassification.error.total" = forest_list$misc.error.total , "misclassification.error" = forest_list$misc.error,
                  "confusion.matrix" = forest_list$confusion))
    }
    if (forest_list$tree.type == "Survival") {
      # if the errors are already computed and saved, simply return them, otherwise compute them from scratch
      if (!is.null(forest_list$C.error)) {
        return(list("C.error" = forest_list$C.error, "IBS.error" = forest_list$ibs, "normalised.IBS.error" = forest_list$ibs.normalised,
                    "KL.error" = forest_list$ikl, "normalised.KL.error" = forest_list$ikl.normalised))
      } else {
        JFCppForestPredictTraining(forest_list)                  # compute predictions from scratch
        result <- JFCppForestErrorSurvivalExternal(forest_list)  # use just computed predictions to compute errors
        return(result)
      }
    }
    if (forest_list$tree.type == "Multi-state") {
      if (!is.null(forest_list$ibs) && is.null(state_weights)) {
        return(list("IBS.error" = forest_list$ibs,
                    "normalised.IBS.error" = forest_list$ibs.normalised,
                    "IKL.error" = forest_list$ikl,
                    "normalised.IKL.error" = forest_list$ikl.normalised,
                    "IS.error" = forest_list$is,
                    "normalised.IS.error" = forest_list$is.normalised))
      } else {
        if (is.null(state_weights)) {
          state_weights <- numeric(0)
        }
        return(JFCppForestErrorMultistateExternal(forest_list, state_weights))
      }
    }
  }

  # if new_data is supplied, compute predictions and error from scratch
  covariates <- forest_list$feature.names
  if (forest_list$tree.type == "Multi-state") {
    if (is.null(jump_data) || !is.data.frame(new_data)) {
      stop("For multi-state forests, jump_data and feature data must be supplied.")
    }
    if (length(jump_data) != nrow(new_data)) {
      stop("jump_data and new_data must contain the same number of observations.")
    }

    missing_columns <- setdiff(covariates, names(new_data))
    if (length(missing_columns) > 0) {
      stop("new_data is missing column(s): ", paste(missing_columns, collapse = ", "))
    }

    new_data <- new_data[, covariates, drop = FALSE]
    feature_indices <- which(names(new_data) %in% covariates) - 1
    processed_data <- preprocess_data(new_data, forest_list$categorical.levels)
    max_response_length <- max(sapply(jump_data, function(e) length(e$states)))
    num_states <- forest_list$num.states
    if (is.null(num_states)) {
      num_states <- length(unique(unlist(lapply(jump_data, `[[`, "states"))))
    }
    if (is.null(state_weights)) {
      state_weights <- numeric(0)
    }

    return(JFCppForestErrorMultistate(
      forest_list, max_response_length, num_states, jump_data, processed_data$data,
      feature_indices, processed_data$categorical, processed_data$unique_values, state_weights
    ))
  }

  response <- forest_list$response.names
  missing_columns <- setdiff(c(response, covariates), names(new_data))
  if (length(missing_columns) > 0) {
    stop("new_data is missing column(s): ", paste(missing_columns, collapse = ", "))
  }

  # ensures the response and features have the same order as the original dataset
  new_data <- new_data[, c(response, covariates), drop = FALSE]
  response_indices <- which(names(new_data) %in% response) - 1
  feature_indices <- which(names(new_data) %in% covariates) - 1
  processed_data <- preprocess_data(new_data, forest_list$categorical.levels)
  result <- JFCppForestError(forest_list, processed_data$data, feature_indices,
                             processed_data$categorical, processed_data$unique_values, response_indices)
  if (forest_list$tree.type == "Classification") {
    names(result$misclassification.error) <- forest_list$class.levels
    dimnames(result$confusion.matrix) <- list(
      observations = forest_list$class.levels,
      predicted = forest_list$class.levels
    )
  }
  return(result)
}

#' Variable importance for a fitted forest
#'
#' @param forest_list A fitted forest object from [jfforest()].
#' @param feature Name of one fitted feature, or `NULL` (the default) to
#'   compute importance for every feature.
#' @param seed Random seed for the perturbations. `NULL` draws a seed using
#'   R's current random-number state; supply a seed for reproducible results.
#' @param method Importance method. `"permute"` (the default) shuffles the
#'   selected feature among each tree's OOB observations. `"random"` sends
#'   observations to a random daughter whenever that feature is used for a
#'   split, with probabilities proportional to the daughter node sizes.
#' @param loss Loss function for importance: `"mse"` for regression,
#'   `"misc"` or `"brier"` for classification, and `"concordance"`, `"brier"`,
#'   or `"kl"` for survival; and `"brier"`, `"kl"`, or `"spherical"` for
#'   multi-state forests. `"default"` (the default) selects `"mse"`, `"misc"`,
#'   `"concordance"`, or `"brier"` for regression, classification, survival,
#'   or multi-state forests, respectively.
#'
#' @details Importance compares each tree's OOB loss before and after
#'   perturbing the selected feature, without refitting the tree. Positive
#'   values mean that perturbing the feature worsens predictions; values near
#'   zero indicate little change, and negative values mean that perturbation
#'   improves predictions on the OOB observations. Values are signed loss
#'   differences, not percentages, and depend on the selected loss and method.
#'
#'   Tree contributions are averaged over trees with a defined OOB loss.
#'   For survival with `method = "random"` and `loss = "concordance"`, the
#'   average is weighted by each tree's OOB observation count. Concordance
#'   importance uses the decrease in C-index, equivalently the increase in
#'   `1 - C-index`.
#'
#'   Regression uses mean squared error, and classification uses
#'   misclassification rate or the normalised Brier scale described in
#'   [jfforest.error()]. Survival and multi-state probability scores use
#'   censoring weights and time-normalised integrated losses. Multi-state
#'   scores evaluate state occupation probabilities derived from leaf
#'   Nelson-Aalen estimators and initial distributions; their state weights
#'   are always `1 / num_states` for Brier and spherical scores and 1 for
#'   Kullback-Leibler loss, regardless of `state_weights` used during fitting.
#'   Censoring weights from the unperturbed leaf assignments are retained
#'   when evaluating perturbed predictions.
#'
#'   The forest must have OOB observations; sampling without replacement at
#'   `sample_rate = 1` leaves none. Survival concordance also needs comparable
#'   survival pairs, and probability scores need observed event times.
#'   Required leaf estimates are computed when needed, so fitting with
#'   `save_predictions = FALSE` does not prevent VIMP calculation.
#'   For an all-feature call, separate seeds are generated for each feature;
#'   a single-feature call with the same seed need not give the identical value.
#'
#' @return A VIMP-only list with class `jf_vimp`, for both single-feature and
#'   all-feature calls. Its `vimp` field is a named list of numeric importance
#'   values. Use `result$vimp` for all computed values or
#'   `result$vimp[["feature_name"]]` for one feature. Printing displays the
#'   first 20 values; use `print(result, n = Inf)` to display all values.
#' @md
#' @export
#'
jfforest.vimp <- function(forest_list, feature = NULL, seed = NULL, method = "permute", loss = "default") {
  if (is.null(seed)) {
    seed <- runif(n = 1, min = 1, max = 10^6)
  }
  # if no feature is supplied, compute VIMP for all features in the forest
  if (is.null(feature)) {
    values <- JFCppForestVIMP(forest_list, seed, method, loss)
  } else {
    value <- JFCppForestVIMPFeature(forest_list, feature, seed, method, loss)
    values <- list(value)
    names(values) <- feature
  }
  as_jf_vimp(values)
}

#' Print information about a fitted JumpForests tree
#'
#' Displays summary information about a fitted tree, including tree type,
#' data info, and training/prediction errors. Can optionally print full tree details.
#'
#' @param tree_list A fitted tree object returned by \code{\link{jftree}}.
#' @param full If `TRUE`, also prints the full tree structure with all split
#'   thresholds and feature IDs. Default is `FALSE`.
#'
#' @return Invisible NULL. Prints information to console as a side effect.
#' @export
#'
print_tree <- function(tree_list, full = FALSE) {
  # print type of tree
  if (tree_list$tree.type == "Regression") {
    cat("Type of tree: Regression\n")
  }
  if (tree_list$tree.type == "Classification") {
    cat("Type of tree: Classification\n")
  }
  if (tree_list$tree.type == "Survival") {
    cat("Type of tree: Survival\n")
  }
  if (tree_list$tree.type == "Multi-state") {
    cat("Type of tree: Multi-state\n")
  }

  # print basic data info
  cat("Number of observations:", tree_list$num.obs, "\n")
  cat("Number of features:", tree_list$num.features, "\n")

  if (tree_list$tree.type == "Regression") {
    cat("Training error (MSE):",tree_list$mse.error, "\n")
    cat("Training error (R^2):",tree_list$R2, "\n")
  }
  if (tree_list$tree.type == "Classification") {
    cat("Training error (overall misclassification):",tree_list$misc.error.total, "\n")
    cat("Training error (Brier score):", tree_list$bs, "\n")
    if (!is.null(tree_list$bs.normalised)) {
      cat("Training error (normalised Brier score):", tree_list$bs.normalised, "\n")
    }
    cat("Training error (class-wise misclassification):\n")
    print(tree_list$misc.error)
    cat("Confusion matrix:\n")
    confusion <- tree_list$confusion
    if (is.null(dimnames(confusion)) && !is.null(tree_list$class.levels)) {
      dimnames(confusion) <- list(
        observations = tree_list$class.levels,
        predicted = tree_list$class.levels
      )
    }
    print(confusion)
  }
  if (tree_list$tree.type == "Survival") {
    cat("Number of deaths:", tree_list$num.deaths, "\n")
    if (length(tree_list$unique.event.times) <= 20) {
      cat("Unique event times:", tree_list$unique.event.times, "\n")
    }
    cat("Training error (C-index):",tree_list$C.error, "\n")
    cat("Training error (IBS):", tree_list$ibs, "\n")
    cat("Training error (normalised IBS):", tree_list$ibs.normalised, "\n")
    cat("Training error (IKL):", tree_list$ikl, "\n")
    cat("Training error (normalised IKL):", tree_list$ikl.normalised, "\n")
  }
  if (tree_list$tree.type == "Multi-state") {
    if (length(tree_list$unique.event.times) <= 20) {
      cat("Unique event times:", tree_list$unique.event.times, "\n")
    }
    cat("Training error (IBS):", tree_list$ibs, "\n")
    cat("Training error (normalised IBS):", tree_list$ibs.normalised, "\n")
    cat("Training error (IKL):", tree_list$ikl, "\n")
    cat("Training error (normalised IKL):", tree_list$ikl.normalised, "\n")
    cat("Training error (IS):", tree_list$is, "\n")
    cat("Training error (normalised IS):", tree_list$is.normalised, "\n")
  }

  # print hyperparameters
  cat("Minimal node size:", tree_list$min.node.size, "\n")
  cat("Maximum tree depth:", tree_list$max.depth, "\n")
  cat("Number of selected features in each split:", tree_list$mtry, "\n")
  cat("Number of possible splits considered for each feature:", tree_list$nsplits, "\n")
  cat("Splitting rule:", tree_list$splitrule, "\n")
  if (!is.null(tree_list$splitrule.par)) {
    cat("Splitting-rule parameter:", tree_list$splitrule.par, "\n")
  }
  if (tree_list$honest) {
    cat("Honest: Yes \n")
  } else {
    cat("Honest: No \n")
  }

  # print info about the tree itself
  cat("Number of nodes:", tree_list$num.nodes, "\n")
  cat("Number of terminal nodes:", tree_list$num.terminal.nodes, "\n")
  cat("Tree depth:", tree_list$tree.depth, "\n")

  if (full) {
    full_info <- getTreeTable(tree_list$Tree)
    # determine categorical status
    categorical_features <- full_info$categorical
    categorical <- rep(0, tree_list$num.nodes)
    for (i in 1:tree_list$num.nodes) {
      categorical[i] <- categorical_features[full_info$feature.IDs[i] + 1]
    }
    table <- data.frame(NodeID = 0:(tree_list$num.nodes-1),
                        LeftDaughter = full_info$left.daughters,
                        RightDaughter = ifelse(full_info$left.daughters == 0, 0, full_info$left.daughters + 1),
                        FeatureID = full_info$feature.IDs,
                        Thresholds = sapply(full_info$thresholds, function(x) {paste0("{", paste(x, collapse = ", "), "}")}),
                        Categorical = categorical,
                        NodeDepth = full_info$depths)
    print(table)
  }
}

#' Print information about a fitted JumpForests forest
#'
#' Displays summary information about a fitted random forest, including forest type,
#' data info, and out-of-bag errors (only available if predictions are computed and
#' saved in the forest, i.e. if save_predictions = `TRUE`)
#'
#' @param forest_list A fitted forest object returned by \code{\link{jfforest}}.
#'
#' @return Invisible NULL. Prints information to console as a side effect.
#' @export
#'
print_forest <- function(forest_list) {
  # print type of forest
  if (forest_list$tree.type == "Regression") {
    cat("Type of tree: Regression\n")
  }
  if (forest_list$tree.type == "Classification") {
    cat("Type of tree: Classification\n")
  }
  if (forest_list$tree.type == "Survival") {
    cat("Type of tree: Survival\n")
  }
  if (forest_list$tree.type == "Multi-state") {
    cat("Type of tree: Multi-state\n")
  }

  # print basic data info
  cat("Number of observations:", forest_list$num.obs, "\n")
  cat("Number of features:", forest_list$num.features, "\n")

  if (forest_list$tree.type == "Regression") {
    cat("OOB error (MSE):", forest_list$mse.error, "\n")
    cat("OOB error (R2:)", forest_list$R2, "\n")
  }
  if (forest_list$tree.type == "Classification") {
    cat("OOB error (overall misclassification)", forest_list$misc.error.total, "\n")
    cat("OOB error (Brier score)", forest_list$bs, "\n")
    if (!is.null(forest_list$bs.normalised)) {
      cat("OOB error (normalised Brier score):", forest_list$bs.normalised, "\n")
    }
    cat("OOB error (class-wise misclassification):\n")
    print(forest_list$misc.error)
    confusion <- forest_list$confusion
    if (is.null(dimnames(confusion)) && !is.null(forest_list$class.levels)) {
      dimnames(confusion) <- list(
        observations = forest_list$class.levels,
        predicted = forest_list$class.levels
      )
    }
    print(confusion)
  }
  if (forest_list$tree.type == "Survival") {
    cat("Number of deaths:", forest_list$num.deaths, "\n")
    if (length(forest_list$unique.event.times) <= 20) {
      cat("Unique event times:", forest_list$unique.event.times, "\n")
    }
    if (!is.null(forest_list$C.error)) {
      cat("OOB error (C-index):", forest_list$C.error, "\n")
    }
    if (!is.null(forest_list$ibs)) {
      cat("OOB error (IBS):", forest_list$ibs, "\n")
      cat("OOB error (normalised IBS):", forest_list$ibs.normalised, "\n")
      cat("OOB error (IKL):", forest_list$ikl, "\n")
      cat("OOB error (normalised IKL):", forest_list$ikl.normalised, "\n")
    }
  }
  if (forest_list$tree.type == "Multi-state") {
    if (length(forest_list$unique.event.times) <= 20) {
      cat("Unique event times:", forest_list$unique.event.times, "\n")
    }
    if (!is.null(forest_list$ibs)) {
      cat("OOB error (IBS):", forest_list$ibs, "\n")
      cat("OOB error (normalised IBS):", forest_list$ibs.normalised, "\n")
      cat("OOB error (IKL):", forest_list$ikl, "\n")
      cat("OOB error (normalised IKL):", forest_list$ikl.normalised, "\n")
      cat("OOB error (IS):", forest_list$is, "\n")
      cat("OOB error (normalised IS):", forest_list$is.normalised, "\n")
    }
  }

  # print hyperparameters
  if (forest_list$sampling.type) {
    cat("Subsampling scheme: With replacement \n")
  } else {
    cat("Subsampling scheme: Without replacement \n")
  }
  cat("Minimal node size:", forest_list$min.node.size, "\n")
  cat("Maximum depth allowed:", forest_list$max.depth, "\n")
  cat("Number of selected features in each split:", forest_list$mtry, "\n")
  cat("Number of possible splits considered for each feature:", forest_list$nsplits, "\n")
  cat("Splitting rule:", forest_list$splitrule, "\n")
  if (!is.null(forest_list$splitrule.par)) {
    cat("Splitting-rule parameter:", forest_list$splitrule.par, "\n")
  }
  if (forest_list$honest) {
    cat("Honest: Yes \n")
    cat("Resample size used to grow trees:", floor(forest_list$subsample.size/2), "(approx.)", "\n")
    if (forest_list$double.bootstrap) {
      cat("Double bootstrap: Yes", "\n")
    } else {
      cat("Double bootstrap: No", "\n")
    }
  } else {
    cat("Honest: No \n")
    cat("Resample size used to grow trees:", forest_list$subsample.size, "\n")
  }

  # print info about the forest itself
  cat("Number of trees:", forest_list$num.trees,"\n")
  cat("Average number of nodes:", forest_list$avg.num.nodes, "\n")
  cat("Average number of terminal nodes:", forest_list$avg.num.terminal.nodes, "\n")
  cat("Average tree depth:", forest_list$avg.tree.depth, "\n")
}

# prepares the DataFrame data to be processed by the C++ struct Data
# it is up to the user to make sure that categorical variables are supplied
# either as a Factor, chr or as Booleans, while continuous variables should
# be numeric
preprocess_data <- function(data, categorical_levels = NULL) {
  categorical <- rep(0, ncol(data))
  unique_values <- rep(0, ncol(data))
  new_categorical_levels <- vector("list", ncol(data))
  names(new_categorical_levels) <- names(data)

  for (i in 1:ncol(data)) {
    current_levels <- categorical_levels[[names(data)[i]]]
    has_training_levels <- !is.null(current_levels)
    if (has_training_levels || inherits(data[, i], "character") || inherits(data[, i], "factor")) {
      if (is.null(current_levels) && inherits(data[, i], "factor")) {
        current_levels <- levels(data[, i])
      } else if (is.null(current_levels)) {
        current_levels <- sort(unique(data[, i]))
      }
      encoded <- match(as.character(data[, i]), current_levels)
      unknown <- is.na(encoded) & !is.na(data[, i])
      if (any(unknown)) {
        stop("New categorical value(s) in ", names(data)[i], " not seen in the training data.")
      }
      data[, i] <- encoded
      categorical[i] <- 1
      unique_values[i] <- length(current_levels)
      new_categorical_levels[[i]] <- current_levels
    } else if (inherits(data[, i], "logical")) {
      current_levels <- c("FALSE", "TRUE")
      encoded <- match(as.character(data[, i]), current_levels)
      unknown <- is.na(encoded) & !is.na(data[, i])
      if (any(unknown)) {
        stop("New categorical value(s) in ", names(data)[i], " not seen in the training data.")
      }
      data[, i] <- encoded
      categorical[i] <- 1
      unique_values[i] <- length(current_levels)
      new_categorical_levels[[i]] <- current_levels
    }
  }
  list(data = data, unique_values = unique_values, categorical = categorical,
       categorical_levels = new_categorical_levels)
}

# function for computing the Kaplan-Meier estimator from a Nelson-Aalen estimator
# na is the Nelson-Aalen estimator, a vector
km <- function(na) {
  res <- rep(1, length(na))
  for (i in 2:length(na)) {
    res[i] <- res[i - 1] * (1 - (na[i] - na[i - 1]))
  }
  res
}

# function for computing the Aalen-Johansen estimator from a Nelson-Aalen estimator
# na is the Nelson-Aalen estimator, a list of matrices
# a0 is the initial value for the AJ estimator (assumed to be the identity if a0 is not supplied)
aj <- function(na, a0 = NULL) {
  res <- list()
  if(is.null(a0)) {
    a0 = diag(dim(na[[1]])[1])
  }
  res[[1]] <- a0
  for (i in seq_along(na)[-1L]) {
    Delta <- na[[i]] - na[[i - 1]]
    res[[i]] <- res[[i - 1]] + as.vector(res[[i - 1]] %*% Delta) - res[[i - 1]] * rowSums((Delta))
  }
  res
}

#' Compute occupation probabilities from initial distribution and Nelson-Aalen estimator
#'
#' Computes state occupation probabilities by combining the initial state distribution
#' with the Aalen-Johansen estimator derived from the Nelson-Aalen estimator.
#'
#' @param init A finite numeric vector of initial state probabilities, one per
#'   state. Probabilities must be between zero and one and sum to one.
#' @param na The Nelson-Aalen estimator for one individual: a non-empty list of
#'   equally sized square numeric cumulative intensity matrices in time order.
#'   The first matrix must be zero. Diagonal entries must equal the negative
#'   sum of the off-diagonal entries in their row.
#'
#' @details Pass `predictions[[i]]` when both optional prediction flags are
#'   disabled, or `result$predictions[[i]]` otherwise. With
#'   `compute_initial = TRUE`, use `result$init[[i]]` for `init`.
#'   Invalid inputs or intensity increments that cannot produce valid
#'   probabilities raise an informative error.
#'
#' @return A list of occupation probability matrices.
#' @export
#'
occupation_prob <- function(init, na) {
  if (missing(init)) {
    jf_probability_error("occupation_prob", "`init` is missing. Supply an initial distribution, such as c(1, 0, 0), or result$init[[i]].")
  }
  if (missing(na)) {
    jf_probability_error("occupation_prob", "`na` is missing. Supply the Nelson-Aalen estimator for one individual.")
  }
  num_states <- validate_jf_nelson_aalen(na, "occupation_prob")
  validate_jf_initial_distribution(init, num_states, "occupation_prob")
  lapply(checked_jf_aalen_johansen(na, "occupation_prob"), function(z) init %*% z)
}

#' Compute transition probabilities via the Aalen-Johansen estimator from a
#' Nelson-Aalen estimator by directly applying the product integral. Only valid
#' for Markov models.
#'
#' @param na The Nelson-Aalen estimator for one individual: a non-empty list of
#'   equally sized square numeric cumulative intensity matrices in time order.
#'   The first matrix must be zero. Diagonal entries must equal the negative
#'   sum of the off-diagonal entries in their row.
#'
#' @details Pass `predictions[[i]]` when both optional prediction flags are
#'   disabled, or `result$predictions[[i]]` otherwise. Invalid inputs or
#'   intensity increments that cannot produce valid probabilities raise an
#'   informative error.
#'
#' @return A list of Aalen-Johansen estimator matrices.
#' @export
transition_prob <- function(na) {
  if (missing(na)) {
    jf_probability_error("transition_prob", "`na` is missing. Supply the Nelson-Aalen estimator for one individual.")
  }
  validate_jf_nelson_aalen(na, "transition_prob")
  checked_jf_aalen_johansen(na, "transition_prob")
}

jf_probability_error <- function(caller, message) {
  stop(paste0(caller, "(): ", message), call. = FALSE)
}

validate_jf_nelson_aalen <- function(na, caller) {
  individual_hint <- "Supply one individual's estimator: predictions[[i]] or result$predictions[[i]]."
  if (is.null(na)) {
    jf_probability_error(caller, paste("`na` is NULL.", individual_hint))
  }
  if (inherits(na, "jf_multistate_prediction_result")) {
    jf_probability_error(caller, "`na` is a complete prediction result. Use result$predictions[[i]] for one individual.")
  }
  if (inherits(na, "jf_multistate_forest") || inherits(na, "jf_multistate_tree")) {
    jf_probability_error(caller, "`na` is a fitted model. Use fitted_model$predictions[[i]] for one individual.")
  }
  if (inherits(na, "jf_multistate_predictions")) {
    jf_probability_error(caller, "`na` contains predictions for all individuals. Use predictions[[i]] for one individual.")
  }
  if (!is.list(na) || is.data.frame(na) || length(na) == 0L) {
    jf_probability_error(caller, paste("`na` must be a non-empty list of square numeric matrices, one per event time.", individual_hint))
  }

  tolerance <- sqrt(.Machine$double.eps)
  num_states <- NULL
  previous <- NULL
  for (i in seq_along(na)) {
    current <- na[[i]]
    label <- paste0("`na[[", i, "]]`")
    if (!is.matrix(current) || !is.numeric(current) || is.complex(current)) {
      jf_probability_error(caller, paste(label, "must be a square numeric matrix.", individual_hint))
    }
    if (nrow(current) == 0L || nrow(current) != ncol(current)) {
      jf_probability_error(caller, paste(label, "must be a non-empty square matrix."))
    }
    if (is.null(num_states)) {
      num_states <- nrow(current)
      off_diagonal <- row(current) != col(current)
    } else if (nrow(current) != num_states) {
      jf_probability_error(caller, paste0(label, " must have ", num_states, " rows and columns, matching na[[1]]."))
    }
    if (any(!is.finite(current))) {
      jf_probability_error(caller, paste(label, "contains NA, NaN or infinite values. All intensities must be finite."))
    }
    if (any(current[off_diagonal] < -tolerance)) {
      jf_probability_error(caller, paste(label, "has negative off-diagonal intensities. Cumulative transition intensities must be non-negative."))
    }
    row_totals <- rowSums(current)
    if (any(!is.finite(row_totals)) ||
        any(abs(row_totals) > tolerance * pmax(1, rowSums(abs(current))))) {
      jf_probability_error(caller, paste(label, "has an invalid diagonal. Each diagonal entry must equal the negative sum of the other entries in its row."))
    }
    if (i == 1L) {
      if (any(abs(current) > tolerance)) {
        jf_probability_error(caller, "`na[[1]]` must be a zero matrix at time zero. Prepend a zero matrix to an estimator that starts at its first event.")
      }
    } else {
      increment <- current - previous
      if (any(!is.finite(increment))) {
        jf_probability_error(caller, paste("The intensity increment at", label, "is not finite. Check the cumulative intensity values."))
      }
      if (any(increment[off_diagonal] < -tolerance)) {
        jf_probability_error(caller, paste(label, "has decreasing cumulative intensities. Check that the matrices are in event-time order."))
      }
      diag(increment) <- 0
      if (any(rowSums(increment) > 1 + tolerance)) {
        jf_probability_error(caller, paste(label, "has a total outgoing intensity increment greater than one. These increments cannot produce valid transition probabilities."))
      }
    }
    previous <- current
  }
  num_states
}

validate_jf_initial_distribution <- function(init, num_states, caller) {
  if (is.null(init)) {
    jf_probability_error(caller, "`init` is NULL. Supply an initial distribution or predict with compute_initial = TRUE and use result$init[[i]].")
  }
  if (!is.numeric(init) || is.complex(init) || !is.null(dim(init))) {
    jf_probability_error(caller, "`init` must be a numeric vector with one probability per state, such as c(1, 0, 0).")
  }
  if (length(init) != num_states) {
    jf_probability_error(caller, paste0("`init` must contain ", num_states,
      " probabilities, matching the number of states in `na`; received ", length(init), "."))
  }
  if (any(!is.finite(init))) {
    jf_probability_error(caller, "`init` contains NA, NaN or infinite values. All probabilities must be finite.")
  }
  if (any(init < 0 | init > 1)) {
    jf_probability_error(caller, "Every probability in `init` must be between zero and one.")
  }
  if (abs(sum(init) - 1) > sqrt(.Machine$double.eps)) {
    jf_probability_error(caller, "The probabilities in `init` must sum to one.")
  }
  invisible(NULL)
}

checked_jf_aalen_johansen <- function(na, caller) {
  probabilities <- tryCatch(aj(na), error = function(error) {
    jf_probability_error(caller, paste("Could not compute the Aalen-Johansen estimator:", conditionMessage(error)))
  })
  tolerance <- sqrt(.Machine$double.eps)
  for (i in seq_along(probabilities)) {
    current <- probabilities[[i]]
    if (any(!is.finite(current)) || any(current < -tolerance | current > 1 + tolerance) ||
        any(abs(rowSums(current) - 1) > tolerance)) {
      jf_probability_error(caller, paste0("`na` does not produce valid transition probabilities at time index ", i,
        ". Check the cumulative intensity matrices and their increments."))
    }
  }
  probabilities
}

# function for testing the methods in Data.cpp
test_data_functions <- function(data, response_indices, feature_indices) {
  processed_data <- preprocess_data(data)
  response_indices <- response_indices - 1
  feature_indices <- feature_indices - 1
  testData(processed_data$data, response_indices, feature_indices,
           processed_data$categorical, processed_data$unique_values)
}

test_data_functions_multistate <- function(jump_data, feature_data, feature_indices) {
  processed_data <- preprocess_data(feature_data)
  feature_indices <- feature_indices - 1
  max_response_length <- max(sapply(jump_data, function(e) length(e$states)))
  num_states <- length(unique(unlist(lapply(jump_data, '[[', "states"))))
  testDataMultistate(jump_data, max_response_length, num_states, processed_data$data,
             feature_indices, processed_data$categorical, processed_data$unique_values)
}

# rough function to fit a survival tree on the data
# later we should provide a formula (Times, Indicators) ~ <covariates>
# to make the function more user friendly and the indices used for
# choosing the subset of the data should be given as a c (bootstrap,
# w/wo replacement etc.)
fit_survival_tree <- function(data, mtry, min_node_size, nsplits,
                              response_indices, feature_indices,
                              subset_indices) {
  stop("fit_survival_tree() is no longer exposed. Use jftree(..., splitrule = 'logrank') instead.")
}

test_unique_event_times <- function(unique_event_times, num_event_times) {
  testUniqueEventTimesThinning(unique_event_times, num_event_times)
}

# Add a print class to the existing list of subject-by-time Nelson--Aalen matrices.
# The nested lists and matrices retain their original indexing and values.
as_jf_multistate_predictions <- function(predictions, event_times, num_states, model_type = "forest") {
  if (is.null(predictions)) {
    return(NULL)
  }
  structure(predictions, class = c("jf_multistate_predictions", "list"),
            event_times = event_times, num_states = num_states, model_type = model_type)
}

as_jf_multistate_prediction_result <- function(result, model_type = "forest") {
  # Use the same explicit field names for training and new data. Retain the
  # existing names as aliases, sharing the underlying lists and matrices.
  initial <- result[["initial", exact = TRUE]]
  if (is.null(initial)) {
    initial <- result[["predictions.init", exact = TRUE]]
  }
  if (!is.null(initial)) {
    result$init <- initial
  }
  censoring_oob <- result[["censoring.oob", exact = TRUE]]
  if (!is.null(censoring_oob)) {
    result$censoring <- censoring_oob
  }
  class(result) <- c("jf_multistate_prediction_result", "list")
  attr(result, "model_type") <- model_type
  result
}

print_jf_multistate_prediction_summary <- function(x) {
  cat("  Number of observations:", length(x), "\n")
  cat("  Event times:", length(attr(x, "event_times")), "\n")
  cat("  States:", as.integer(attr(x, "num_states")), "\n")
}

jf_multistate_prediction_output <- function(compute_initial, compute_censoring, model_type = "forest") {
  prediction_function <- if (identical(model_type, "tree")) "jftree.predict" else "jfforest.predict"
  if (compute_initial && compute_censoring) {
    return(paste(
      "Output: A list containing a list of Nelson-Aalen estimators, a list of initial distributions and a matrix of the censoring Kaplan-Meier estimators.",
      "To access the Nelson-Aalen estimator for individual i, use $predictions[[i]], for the initial distribution, use $init[[i]] and for the censoring, $censoring[i,]."
    ))
  }
  if (compute_initial) {
    return(paste(
      "Output: A list containing a list of Nelson-Aalen estimators and the estimated initial distribution.",
      "To obtain the former for individual i, use $predictions[[i]], for the latter, $init[[i]].",
      paste0("To also obtain censoring estimates, add compute_censoring = TRUE in ", prediction_function, ".")
    ))
  }
  if (compute_censoring) {
    return(paste(
      "Output: A list containing a list of Nelson-Aalen estimators and the censoring Kaplan-Meier estimators, the latter a matrix with a row for each individual.",
      "To obtain the Nelson-Aalen estimator for individual i, use $predictions[[i]], for the censoring, $censoring[i,].",
      paste0("To also obtain the initial distribution, add compute_initial = TRUE in ", prediction_function, ".")
    ))
  }
  paste(
    "Output: A list of Nelson-Aalen estimators for every individual.",
    "To access the Nelson--Aalen estimator for individual i, use [[i]].",
    paste0("Add compute_initial = TRUE in ", prediction_function,
      " to also obtain initial distribution estimates and compute_censoring = TRUE in ",
      prediction_function, " to obtain censoring estimates.")
  )
}

#' @export
print.jf_multistate_predictions <- function(x, ...) {
  cat("Multi-state Nelson--Aalen predictions\n")
  print_jf_multistate_prediction_summary(x)
  cat("  ", jf_multistate_prediction_output(FALSE, FALSE, attr(x, "model_type")), "\n", sep = "")
  invisible(x)
}

#' @export
print.jf_multistate_prediction_result <- function(x, ...) {
  model_type <- attr(x, "model_type")
  if (is.null(model_type)) {
    model_type <- "forest"
  }
  cat("Multi-state", model_type, "prediction result\n")
  print_jf_multistate_prediction_summary(x$predictions)
  compute_initial <- !is.null(x[["init", exact = TRUE]]) ||
    !is.null(x[["initial", exact = TRUE]])
  compute_censoring <- !is.null(x[["censoring", exact = TRUE]]) ||
    !is.null(x[["censoring.oob", exact = TRUE]])
  cat("  ", jf_multistate_prediction_output(compute_initial, compute_censoring, model_type), "\n", sep = "")
  invisible(x)
}

#' Print a fitted JumpForests forest
#'
#' @param x A fitted forest returned by [jfforest()].
#' @param ... Additional arguments, currently unused.
#' @return The fitted forest, invisibly. Prints the summary from [print_forest()].
#' @export
print.jf_forest <- function(x, ...) {
  print_forest(x)
  invisible(x)
}

#' @rdname print.jf_forest
#' @export
print.jf_multistate_forest <- print.jf_forest

#' Print a fitted JumpForests tree
#'
#' @param x A fitted tree returned by [jftree()].
#' @param full If `TRUE`, also print the full tree structure.
#' @param ... Additional arguments, currently unused.
#' @return The fitted tree, invisibly. Prints the summary from [print_tree()].
#' @export
print.jf_tree <- function(x, full = FALSE, ...) {
  print_tree(x, full = full)
  invisible(x)
}

#' @rdname print.jf_tree
#' @export
print.jf_multistate_tree <- print.jf_tree

#' Predict from a fitted JumpForests forest
#'
#' @param object A fitted forest returned by [jfforest()].
#' @param newdata Optional data frame for prediction. If omitted or `NULL`,
#'   return ordinary training-data predictions using all trees. Feature names,
#'   types, and categorical levels must meet the `new_data` requirements in
#'   [jfforest.predict()]; responses are not required.
#' @param ... Arguments passed to [jfforest.predict()], including
#'   `compute_initial` and `compute_censoring`. The existing `new_data`
#'   spelling is also accepted when `newdata` is omitted.
#' @inherit jfforest.predict return
#' @md
#' @export
#' @importFrom stats predict
predict.jf_forest <- function(object, newdata = NULL, ...) {
  if (missing(newdata)) {
    return(jfforest.predict(object, ...))
  }
  jfforest.predict(object, new_data = newdata, ...)
}

#' @rdname predict.jf_forest
#' @export
predict.jf_multistate_forest <- predict.jf_forest

#' Predict from a fitted JumpForests tree
#'
#' @param object A fitted tree returned by [jftree()].
#' @param newdata Optional data frame for prediction. If omitted or `NULL`,
#'   return training-data predictions. Feature names, types, and categorical
#'   levels must meet the `new_data` requirements in [jftree.predict()];
#'   responses are not required.
#' @param ... Arguments passed to [jftree.predict()], including
#'   `compute_initial` and `compute_censoring`. The existing `new_data`
#'   spelling is also accepted when `newdata` is omitted.
#' @inherit jftree.predict return
#' @md
#' @export
predict.jf_tree <- function(object, newdata = NULL, ...) {
  if (missing(newdata)) {
    return(jftree.predict(object, ...))
  }
  jftree.predict(object, new_data = newdata, ...)
}

#' @rdname predict.jf_tree
#' @export
predict.jf_multistate_tree <- predict.jf_tree

as_jf_vimp <- function(values) {
  structure(list(vimp = values), class = c("jf_vimp", "list"))
}

#' Print forest variable importance
#'
#' @param x A VIMP result returned by [jfforest.vimp()].
#' @param n Maximum number of feature values to display. Defaults to 20.
#'   Use `Inf` to display all computed values.
#' @param ... Additional arguments passed to [print.data.frame()].
#' @return The VIMP result, invisibly. Prints a table of importance values
#'   and instructions for accessing them.
#' @export
print.jf_vimp <- function(x, n = 20L, ...) {
  if (!is.numeric(n) || is.complex(n) || length(n) != 1L || is.na(n) ||
      n < 0 || (!is.infinite(n) && n != floor(n))) {
    stop("`n` must be a non-negative whole number or Inf.", call. = FALSE)
  }
  values <- x$vimp
  num_features <- length(values)
  num_displayed <- min(n, num_features)
  displayed <- seq_len(num_displayed)
  cat("Forest variable importance\n")
  cat("  Number of features:", num_features, "\n")
  if (num_displayed > 0L) {
    table <- data.frame(
      Feature = names(values)[displayed],
      VIMP = vapply(values[displayed], as.numeric, numeric(1)),
      row.names = NULL
    )
    print(table, row.names = FALSE, ...)
  }
  if (num_displayed < num_features) {
    cat("  Showing", num_displayed, "of", num_features,
        "values. Use print(result, n = Inf) to display all computed values.\n")
  }
  cat(paste0(
    "  Output: A list containing the computed variable importance values. ",
    "To access all values, use $vimp; for feature x, use $vimp[[\"x\"]] ",
    "or $vimp$x. Use unlist(result$vimp) to obtain a named numeric vector.\n"
  ))
  invisible(x)
}

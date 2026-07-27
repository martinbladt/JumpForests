#nolint start: line_length_linter

#' Fit a JumpForests tree
#'
#' Fits a single regression, survival, or multi-state tree.
#'
#' @param formula Model formula.
#' @param data Training data.
#' @param feature_data Optional feature data for multi-state models.
#' @param splitrule Splitting rule. For regression, choose "mse" (default), "variance" (alias for "mse"), or "mae".
#' @param mtry Number of candidate features at each split.
#' @param min_node_size Minimal node size.
#' @param nsplits Number of split points per feature.
#' @param honest Whether to use honest splitting.
#' @param seed Optional random seed.
#' @param num_event_times Maximum number of event times to use (only relevant for survival and multi-states)
#' @param state_weights Optional vector of weights for each state to be used in error computations (only relevant for multi-state trees)
#' @param fh_weights_a Optional finite non-negative Fleming--Harrington
#'   a-exponents, one for each state `1, ..., num_states`. `NULL` uses one for
#'   every state. Only valid for multi-state trees with
#'   `splitrule = "flemingharrington"`.
#' @param fh_weights_b Optional finite non-negative Fleming--Harrington
#'   b-exponents, with the same ordering and default as `fh_weights_a`.
#'
#' @return A fitted tree object as a list.
#' @export
#'
# the main function for fitting trees (feature_data is only relevant for multi-state trees in which case data is a list and not a data.frame)
jftree <- function(formula, data, feature_data = NULL, splitrule = NULL, mtry = NULL, min_node_size = NULL, nsplits = 10,
                   honest = FALSE, seed = NULL, num_event_times = 0, state_weights = NULL, fh_weights_a = NULL, fh_weights_b = NULL) {
  lhs <- as.character(formula[[2]])
  if (lhs[1] != "MM" && (!is.null(fh_weights_a) || !is.null(fh_weights_b))) {
    stop("fh_weights_a and fh_weights_b are only valid for multi-state models")
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
      # for classification, the default minimal node size is 1
      if (is.null(min_node_size)) {
        min_node_size <- 1
      }
      # set default splitting rule
      if (is.null(splitrule)) {
        splitrule <- "gini"
      }
      result <- JFCppTree(2, processed_data$data, mtry, min_node_size, nsplits, splitrule, honest,
                          response_index, feature_indices, processed_data$categorical,
                          processed_data$unique_values, seed)
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

      result <- JFCppTree(1, processed_data$data, mtry, min_node_size, nsplits, splitrule, honest,
                          response_index, feature_indices, processed_data$categorical,
                          processed_data$unique_values, seed)
      result$categorical.levels <- processed_data$categorical_levels
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

    result <- JFCppTree(3, processed_data$data, mtry, min_node_size, nsplits, splitrule, honest,
                        response_indices, feature_indices, processed_data$categorical,
                        processed_data$unique_values, seed, num_event_times)
    result$categorical.levels <- processed_data$categorical_levels
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
                          mtry, min_node_size, nsplits, splitrule, honest, feature_indices,
                          processed_data$categorical, processed_data$unique_values, seed,
                          state_weights, num_event_times, fh_weights_a, fh_weights_b)
    result$categorical.levels <- processed_data$categorical_levels
    return(result)

  } else {
    stop("Type of tree not recognised from the formula.")
  }
}

#' Predict from a fitted tree
#'
#' @param tree_list A fitted tree object from [jftree()].
#' @param new_data Optional new data.
#' @param compute_censoring Should the censoring distribution be estimated? (only valid for survival and multi-state trees)
#' @param compute_initial Should the initial distribution also be estimated? (only valid for multi-state trees)
#'
#' @return Predictions.
#' @export
#'
# the main function for predicting with trees
jftree.predict <- function(tree_list, new_data = NULL, compute_censoring = FALSE, compute_initial = FALSE) {
  # if data is not supplied, return predictions based on training data
  if (is.null(new_data)) {
    if (tree_list$tree.type == "Multi-state") {
      return(list("predictions" = tree_list$predictions, "init" = tree_list$init))
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
      return(JFCppTreePredictMultistate(tree_list, processed_data$data, feature_indices, processed_data$categorical,
             processed_data$unique_values, compute_initial, compute_censoring))
    }
  } else {
    cat("Error: If new_data is supplied, it must be a data.frame with the same names as the original dataset \n")
  }
}

#' Compute tree prediction error
#'
#' @param tree_list A fitted tree object from [jftree()].
#' @param new_data Optional evaluation data.
#' @param jump_data Jump data (only needed for multistate trees)
#' @param state_weights Optional vector of weights for each state to be used in error computations (only relevant for multi-state trees)
#'
#' @return Error metrics as a list.
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
                  "IKL.error" = tree_list$ikl, "normalised.IKL.error" = tree_list$ikl.normalised))
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
#' Fits a random forest for regression, survival, or multi-state data.
#'
#' @param formula Model formula.
#' @param data Training data.
#' @param feature_data Optional feature data for multi-state models.
#' @param splitrule Splitting rule. For regression, choose "mse" (default), "variance" (alias for "mse"), or "mae".
#' @param mtry Number of candidate features at each split.
#' @param min_node_size Minimal node size.
#' @param nsplits Number of split points per feature.
#' @param ntrees Number of trees.
#' @param honest Whether to use honest splitting.
#' @param swr Whether to sample with replacement.
#' @param sample_rate Sampling rate.
#' @param double_bootstrap Whether to use double bootstrap.
#' @param seed Optional random seed.
#' @param nworkers Number of worker threads.
#' @param save_predictions Logical, whether to compute and store in-sample/OOB predictions at fit time for multi-state forests.
#' @param num_event_times Maximum number of event times to use (only relevant for survival and multi-states)
#' @param fh_weights_a Optional finite non-negative Fleming--Harrington
#'   a-exponents, one for each state `1, ..., num_states`. `NULL` uses one for
#'   every state. Only valid for multi-state forests with
#'   `splitrule = "flemingharrington"`.
#' @param fh_weights_b Optional finite non-negative Fleming--Harrington
#'   b-exponents, with the same ordering and default as `fh_weights_a`.
#'
#' @return A fitted forest object as a list.
#' @export
#'
# the main function for fitting forests (feature_data is only relevant for multi-state trees in which case data is a list and not a data.frame)
jfforest <- function(formula, data, feature_data = NULL, splitrule = NULL, mtry = NULL, min_node_size = NULL, nsplits = 10,
                     ntrees = NULL, honest = FALSE, swr = FALSE, sample_rate = NULL, double_bootstrap = FALSE, 
                     seed = NULL, nworkers = 0, save_predictions = TRUE, num_event_times = 0, fh_weights_a = NULL, fh_weights_b = NULL) {
  lhs <- as.character(formula[[2]])
  if (lhs[1] != "MM" && (!is.null(fh_weights_a) || !is.null(fh_weights_b))) {
    stop("fh_weights_a and fh_weights_b are only valid for multi-state models")
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
      result <- JFCppForest(2, processed_data$data, mtry, min_node_size, nsplits, splitrule, ntrees, honest, swr,
                            sample_rate, double_bootstrap, response_index, feature_indices, processed_data$categorical,
                            processed_data$unique_values, seed, nworkers, save_predictions)
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
      result <- JFCppForest(1, processed_data$data, mtry, min_node_size, nsplits, splitrule, ntrees, honest, swr,
                            sample_rate, double_bootstrap, response_index, feature_indices, processed_data$categorical,
                            processed_data$unique_values, seed, nworkers, save_predictions)
      result$categorical.levels <- processed_data$categorical_levels
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

    result <- JFCppForest(3, processed_data$data, mtry, min_node_size, nsplits, splitrule, ntrees, honest, swr,
                          sample_rate, double_bootstrap, response_indices, feature_indices, processed_data$categorical,
                          processed_data$unique_values, seed, nworkers, save_predictions, num_event_times)
    result$categorical.levels <- processed_data$categorical_levels
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
    fh_weights_a <- normalize_fh_weights(fh_weights_a, "fh_weights_a")
    fh_weights_b <- normalize_fh_weights(fh_weights_b, "fh_weights_b")

    result <- JFCppForestMultistate(data, max_response_length, num_states, processed_data$data, mtry, min_node_size,
                            nsplits, splitrule, ntrees, honest, swr, sample_rate, double_bootstrap, feature_indices, 
                            processed_data$categorical, processed_data$unique_values, seed, nworkers, save_predictions, 
                            num_event_times, fh_weights_a, fh_weights_b)
    result$categorical.levels <- processed_data$categorical_levels
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

#' Predict from a fitted forest
#'
#' @param forest_list A fitted forest object from [jfforest()].
#' @param new_data Optional new data.
#' @param compute_censoring Should the censoring distribution also be estimated (only valid for survival and multi-state trees)
#' @param compute_initial Should the initial distribution also be estimated? (only valid for multi-state trees)
#'
#' @return Predictions.
#' @export
#'
# the main function for predicting with forests
jfforest.predict <- function(forest_list, new_data = NULL, compute_censoring = FALSE, compute_initial = FALSE) {
  # if new data is not supplied and predictions are already computed, return predictions based on training data
  if (is.null(new_data) & !is.null(forest_list$predictions)) {
    if (forest_list$tree.type != "Multi-state") {
      return(forest_list$predictions)
    } else {
      return(list("predictions" = forest_list$predictions, "init" = forest_list$init))
    }
  }
  # if new data is not supplied and predictions are not computed, compute the predictions based on the training data (in-bag and oob)
  if (is.null(new_data) & is.null(forest_list$predictions)) {
    cat("Computing predictions...")
    JFCppForestPredictTraining(forest_list) # computes predictions from scratch
    if (forest_list$tree.type != "Multi-state") {
      return(forest_list$predictions)
    } else {
      return(list("predictions" = forest_list$predictions, "init" = forest_list$init))
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
      return(JFCppForestPredictMultistate(forest_list, processed_data$data, feature_indices,
                            processed_data$categorical, processed_data$unique_values, compute_initial))
    }
  } else {
    cat("Error: If new_data is supplied, it must be a data.frame with the same names as the original dataset \n")
  }
}

#' Compute forest prediction error
#'
#' @param forest_list A fitted forest object from [jfforest()].
#' @param new_data Optional evaluation data.
#' @param jump_data Jump data (only needed for multi-state forests).
#' @param state_weights Optional weights for each state in multi-state error calculations.
#'
#' @return Error metrics as a list.
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
                    "normalised.IKL.error" = forest_list$ikl.normalised))
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
#' @param feature Optional feature name.
#' @param seed Optional random seed.
#' @param method Importance method, `"permute"` or `"random"`.
#' @param loss Loss function for importance: `"mse"` for regression,
#'   `"misc"` or `"brier"` for classification, and `"concordance"`, `"brier"`,
#'   or `"kl"` for survival; and `"brier"` or `"kl"` for multi-state
#'   forests. `"default"` selects the forest-type default.
#'
#' @return Variable importance values.
#' @export
#'
jfforest.vimp <- function(forest_list, feature = NULL, seed = NULL, method = "permute", loss = "default") {
  if (is.null(seed)) {
    seed <- runif(n = 1, min = 1, max = 10^6)
  }
  # if no feature is supplied, compute VIMP for all features in the forest
  if (is.null(feature)) {
    JFCppForestVIMP(forest_list, seed, method, loss)
    #forest_list$vimp <- unlist(forest_list$vimp)
  } else {
    JFCppForestVIMPFeature(forest_list, feature, seed, method, loss)
  }
}

# prints all information about the tree. if full = TRUE, also print the full
# tree (all thresholds etc.)
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
  }

  # print hyperparameters
  cat("Minimal node size:", tree_list$min.node.size, "\n")
  cat("Number of selected features in each split:", tree_list$mtry, "\n")
  cat("Number of possible splits considered for each feature:", tree_list$nsplits, "\n")
  cat("Splitting rule:", tree_list$splitrule, "\n")
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
    }
  }

  # print hyperparameters
  if (forest_list$sampling.type) {
    cat("Subsampling scheme: With replacement \n")
  } else {
    cat("Subsampling scheme: Without replacement \n")
  }
  cat("Minimal node size:", forest_list$min.node.size, "\n")
  cat("Number of selected features in each split:", forest_list$mtry, "\n")
  cat("Number of possible splits considered for each feature:", forest_list$nsplits, "\n")
  cat("Splitting rule:", forest_list$splitrule, "\n")
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
  for (i in 2:length(na)) {
    Delta <- na[[i]] - na[[i - 1]]
    res[[i]] <- res[[i - 1]] + as.vector(res[[i - 1]] %*% Delta) - res[[i - 1]] * rowSums((Delta))
  }
  res
}

# function for computing occupation probabilities
# init is a vector of initial probabilities, a vector
# na is the Nelson-Aalen estimator, a list of matrices
occupation_prob <- function(init, na) {
  lapply(aj(na), function(z) init %*% z)
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

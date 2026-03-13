#nolint start: line_length_linter

#' Fit a JumpForests tree
#'
#' Fits a single regression, survival, or multi-state tree.
#'
#' @param formula Model formula.
#' @param data Training data.
#' @param feature_data Optional feature data for multi-state models.
#' @param splitrule Splitting rule.
#' @param mtry Number of candidate features at each split.
#' @param min_node_size Minimal node size.
#' @param nsplits Number of split points per feature.
#' @param honest Whether to use honest splitting.
#' @param seed Optional random seed.
#' @param num_event_times Maximum number of event times to use (only relevant for survival and multi-states)
#'
#' @return A fitted tree object as a list.
#' @export
#'
# the main function for fitting trees (feature_data is only relevant for multi-state trees in which case data is a list and not a data.frame)
jftree <- function(formula, data, feature_data = NULL, splitrule = NULL, mtry = NULL, min_node_size = NULL, nsplits = 10,
                   honest = FALSE, seed = NULL, num_event_times = 0) {
  lhs <- as.character(formula[[2]])
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
    if (processed_data$categorical[response_index]) {
      # for classification, the default minimal node size is 1
      if (is.null(min_node_size)) {
        min_node_size <- 1
      }
      # set default splitting rule (TODO)
      if (is.null(splitrule)) {
        # TODO
      }
      JFCppTree(2, processed_data$data, mtry, min_node_size, nsplits, splitrule, honest,
                response_index, feature_indices, processed_data$categorical,
                processed_data$unique_values, seed)
    } else {
      # for regression, the default minimal node size is 5
      if (is.null(min_node_size)) {
        min_node_size <- 5
      }
      # set default splitting rule (mean squared error for regression)
      if (is.null(splitrule)) {
        splitrule <- "mse"
      }

      JFCppTree(1, processed_data$data, mtry, min_node_size, nsplits, splitrule, honest,
                response_index, feature_indices, processed_data$categorical,
                processed_data$unique_values, seed)
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

    JFCppTree(3, processed_data$data, mtry, min_node_size, nsplits, splitrule, honest,
              response_indices, feature_indices, processed_data$categorical,
              processed_data$unique_values, seed, num_event_times)
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

    cat("About to call JFCppTreeMM from R", "\n")
    # data here is jump data, a list of lists, each containing a vector 'times' and a vector 'states'
    JFCppTreeMM(data, max_response_length, num_states, processed_data$data,
                mtry, min_node_size, nsplits, splitrule, honest, feature_indices,
                processed_data$categorical, processed_data$unique_values, seed)

  } else {
    stop("Type of tree not recognised from the formula.")
  }
}

#' Predict from a fitted tree
#'
#' @param tree_list A fitted tree object from [jftree()].
#' @param new_data Optional new data.
#' @param compute_initial Should the initial distribution also be estimated? (only valid for multi-state trees)
#'
#' @return Predictions.
#' @export
#'
# the main function for predicting with trees
jftree.predict <- function(tree_list, new_data = NULL, compute_initial = FALSE) {
  # if data is not supplied, return predictions based on training data
  if (is.null(new_data)) {
    if (tree_list$tree.type != "Multi-state") {
      return(tree_list$predictions)
    } else {
      return(list("predictions" = tree_list$predictions, "init" = tree_list$init))
    }
  }
  # if new data is supplied, start by preprocessing the data and
  # extracting relevant columns
  covariates <- tree_list$feature.names
  if (ncol(new_data) > 1) {
      new_data <- new_data[, covariates]  # ensures the columns have the same order as the original dataset
    }
  feature_indices <- which(names(new_data) %in% covariates) - 1
  processed_data <- preprocess_data(new_data)

  # may need to be adapted when more types of trees are implemented
  if (tree_list$tree.type != "Multi-state") {
    return(JFCppTreePredict(tree_list, processed_data$data, feature_indices,
                          processed_data$categorical, processed_data$unique_values))
  } else {
    return(JFCppTreePredictMM(tree_list, processed_data$data, feature_indices,
                          processed_data$categorical, processed_data$unique_values, compute_initial))
  }
}

#' Compute tree prediction error
#'
#' @param tree_list A fitted tree object from [jftree()].
#' @param new_data Optional evaluation data.
#'
#' @return Error metrics as a list.
#' @export
#'
jftree.error <- function(tree_list, new_data = NULL) {
  # if data is not supplied, return the error based on training data
  if (is.null(new_data)) {
    if (tree_list$tree.type == "Regression") {
      return(list("mse.error" = tree_list$mse.error, "R2.error" = tree_list$R2.error))
    }
    if (tree_list$tree.type == "Classification") {

    }
    if (tree_list$tree.type == "Survival") {
      return(list("C.error" = tree_list$error))
    }
    if (tree_list$tree.type == "Multi-state") {

    }
  }

  # if new_data is supplied, compute predictions and error from scratch
  covariates <- tree_list$feature.names
  response <- tree_list$response.names
  response_indices <- which(names(new_data) %in% response) - 1

  # ensures the columns have the same order as the original dataset
  current <- names(new_data)
  current[match(covariates, current)] <- covariates
  new_data <- new_data[current]

  feature_indices <- which(names(new_data) %in% covariates) - 1
  processed_data <- preprocess_data(new_data)
  return(JFCppTreeError(tree_list, processed_data$data, feature_indices,
                          processed_data$categorical, processed_data$unique_values, response_indices))
}

#' Fit a JumpForests forest
#'
#' Fits a random forest for regression, survival, or multi-state data.
#'
#' @param formula Model formula.
#' @param data Training data.
#' @param feature_data Optional feature data for multi-state models.
#' @param splitrule Splitting rule.
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
#'
#' @return A fitted forest object as a list.
#' @export
#'
# the main function for fitting forests (feature_data is only relevant for multi-state trees in which case data is a list and not a data.frame)
jfforest <- function(formula, data, feature_data = NULL, splitrule = NULL, mtry = NULL, min_node_size = NULL, nsplits = 10,
                     ntrees = NULL, honest = FALSE, swr = FALSE, sample_rate = NULL, double_bootstrap = FALSE, 
                     seed = NULL, nworkers = 0, save_predictions = TRUE, num_event_times = 0) {
  lhs <- as.character(formula[[2]])

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
    response_indices <- which(names(data) %in% as.character(formula[[2]])[1]) - 1
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
    JFCppForest(1, processed_data$data, mtry, min_node_size, nsplits, splitrule, ntrees, honest, swr,
              sample_rate, response_indices, feature_indices, processed_data$categorical,
              processed_data$unique_values, seed, nworkers)
  }
  # if left hand side is "Surv(time, status)", survival
  else if (lhs[1] == "Surv") {
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

    JFCppForest(3, processed_data$data, mtry, min_node_size, nsplits, splitrule, ntrees, honest, swr,
              sample_rate, response_indices, feature_indices, processed_data$categorical,
              processed_data$unique_values, seed, nworkers, num_event_times)
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

    JFCppForestMM(data, max_response_length, num_states, processed_data$data, mtry, min_node_size,
                  nsplits, splitrule, ntrees, honest, swr, sample_rate, feature_indices, processed_data$categorical,
                  processed_data$unique, seed, nworkers, save_predictions, num_event_times);

  } else {
    stop("Type of tree not recognised from the formula.")
  }
}

#' Predict from a fitted forest
#'
#' @param forest_list A fitted forest object from [jfforest()].
#' @param new_data Optional new data.
#' @param compute_initial Should the initial distribution also be estimated? (only valid for multi-state trees)
#'
#' @return Predictions.
#' @export
#'
# the main function for predicting with forests
jfforest.predict <- function(forest_list, new_data = NULL, compute_initial = FALSE) {
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
      new_data <- new_data[, covariates]  # ensures the columns have the same order as the original dataset
    }
    feature_indices <- which(names(new_data) %in% covariates) - 1
    processed_data <- preprocess_data(new_data)
    if (forest_list$tree.type != "Multi-state") {
      return(JFCppForestPredict(forest_list, processed_data$data, feature_indices,
                            processed_data$categorical, processed_data$unique_values))
    } else {
      return(JFCppForestPredictMM(forest_list, processed_data$data, feature_indices,
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
#'
#' @return Error metrics as a list.
#' @export
#'
jfforest.error <- function(forest_list, new_data = NULL) {
  # if data is not supplied, return the error based on training data
  if (is.null(new_data)) {
    if (forest_list$tree.type == "Regression") {
      return(list("mse.error" = forest_list$mse.error, "R2.error" = forest_list$R2.error))
    }
    if (forest_list$tree.type == "Classification") {

    }
    if (forest_list$tree.type == "Survival") {
      return(list("C.error" = forest_list$oob.error))
    }
    if (forest_list$tree.type == "Multi-state") {

    }
  }

  # if new_data is supplied, compute predictions and error from scratch
  covariates <- forest_list$feature.names
  response <- forest_list$response.names
  response_indices <- which(names(new_data) %in% response) - 1

  # ensures the columns have the same order as the original dataset
  current <- names(new_data)
  current[match(covariates, current)] <- covariates
  new_data <- new_data[current]

  feature_indices <- which(names(new_data) %in% covariates) - 1
  processed_data <- preprocess_data(new_data)
  return(JFCppForestError(forest_list, processed_data$data, feature_indices,
                          processed_data$categorical, processed_data$unique_values, response_indices))
}

#' Variable importance for a fitted forest
#'
#' @param forest_list A fitted forest object from [jfforest()].
#' @param feature Optional feature name.
#' @param seed Optional random seed.
#' @param method Importance method, `"permute"` or `"random"`.
#'
#' @return Variable importance values.
#' @export
#'
jfforest.vimp <- function(forest_list, feature = NULL, seed = NULL, method = "permute") {
  if (is.null(seed)) {
    seed <- runif(n = 1, min = 1, max = 10^6)
  }
  # if no feature is supplied, compute VIMP for all features in the forest
  if (is.null(feature)) {
    JFCppForestVIMP(forest_list, seed, method)
    #forest_list$vimp <- unlist(forest_list$vimp)
  } else {
    JFCppForestVIMPFeature(forest_list, feature, seed, method)
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
    cat("Training error (R^2):",tree_list$R2.error, "\n")
  }
  if (tree_list$tree.type == "Classification") {
    # TODO
  }
  if (tree_list$tree.type == "Survival") {
    cat("Number of deaths:", tree_list$num.deaths, "\n")
    if (length(tree_list$unique.event.times) <= 20) {
      cat("Unique event times:", tree_list$unique.event.times, "\n")
    }
    cat("Training error:",tree_list$error, "\n")
  }
  if (tree_list$tree.type == "Multi-state") {
    if (length(tree_list$unique.event.times) <= 20) {
      cat("Unique event times:", tree_list$unique.event.times, "\n")
    }
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
    # TODO
  }
  if (forest_list$tree.type == "Classification") {
    # TODO
  }
  if (forest_list$tree.type == "Survival") {
    cat("Number of deaths:", forest_list$num.deaths, "\n")
    if (length(forest_list$unique.event.times) <= 20) {
      cat("Unique event times:", forest_list$unique.event.times, "\n")
    }
    cat("OOB error:", forest_list$oob.error, "\n")
  }
  if (forest_list$tree.type == "Multi-state") {
    if (length(forest_list$unique.event.times) <= 20) {
      cat("Unique event times:", forest_list$unique.event.times, "\n")
    }
  }

  # print hyperparameters
  if (forest_list$sampling.type) {
    cat("Subsampling scheme: With replacement \n")
  } else {
    cat("Subsampling scheme: Without replacement \n")
  }
  cat("Resample size used to grow trees:", forest_list$subsample.size, "\n")
  cat("Minimal node size:", forest_list$min.node.size, "\n")
  cat("Number of selected features in each split:", forest_list$mtry, "\n")
  cat("Number of possible splits considered for each feature:", forest_list$nsplits, "\n")
  cat("Splitting rule:", forest_list$splitrule, "\n")
  if (forest_list$honest) {
    cat("Honest: Yes \n")
  } else {
    cat("Honest: No \n")
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
preprocess_data <- function(data) {
  categorical <- rep(0, ncol(data))
  unique_values <- rep(0, ncol(data))

  for (i in 1:ncol(data)) {
    if (inherits(data[, i], "character")) {
      data[, i] <- as.numeric(as.factor(data[, i]))
      categorical[i] <- 1
      unique_values[i] <- length(unique(data[, i]))
    } else if (inherits(data[, i], "factor") || inherits(data[, i], "logical")) {
      data[, i] <- as.numeric(data[, i])
      categorical[i] <- 1
      unique_values[i] <- length(unique(data[, i]))
    }
  }
  list(data = data, unique_values = unique_values, categorical = categorical)
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
           processed_data$categorical, processed_data$unique)
}

test_data_functions_mm <- function(jump_data, feature_data, feature_indices) {
  processed_data <- preprocess_data(feature_data)
  feature_indices <- feature_indices - 1
  max_response_length <- max(sapply(jump_data, function(e) length(e$states)))
  num_states <- length(unique(unlist(lapply(jump_data, '[[', "states"))))
  testDataMM(jump_data, max_response_length, num_states, processed_data$data, 
             feature_indices, processed_data$categorical, processed_data$unique)
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

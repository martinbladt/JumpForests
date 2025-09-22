# the main function for fitting trees
jftree <- function(formula, data, splitrule = NULL, mtry = NULL, min_node_size = NULL, nsplits = 10, honest = FALSE, seed = NULL) {
  # preprocess the entire dataset
  processed_data <- preprocess_data(data)
  lhs <- as.character(formula[[2]])

  # if seed is not set, generate a random one
  if (is.null(seed)) {
    seed <- runif(n = 1, min = 1, max = 10^6)
  }

  # if left hand side is of length 1, classification or regression
  if (length(lhs) == 1) {
    response_indices <- which(names(data) %in% as.character(formula[[2]])[1]) - 1
    # TODO
  }
  # if left hand side is "Surv(time, status)", survival
  else if (lhs[1] == "Surv") {
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
              processed_data$unique_values, seed)
  }
  # if the left hand side is "MM(...)", multi-state
  else if (lhs[1] == "MM") {
    # TODO
  } else {
    stop("Type of tree not recognised from the formula.")
  }
}

# the main function for predicting with trees
jftree.predict <- function(tree_list, new_data = NULL) {
  # if data is not supplied, return predictions based on training data
  if (is.null(new_data)) {
    return(tree_list$predictions)
  }
  # if new data is supplied, start by preprocessing the data and
  # extracting relevant columns
  covariates <- tree_list$feature.names
  new_data <- new_data[, covariates]  # ensures the columns have the same order as the original dataset
  feature_indices <- which(names(new_data) %in% covariates) - 1
  processed_data <- preprocess_data(new_data)
  return(JFCppTreePredict(tree_list, processed_data$data, feature_indices,
                          processed_data$categorical, processed_data$unique_values))
}

jftree.error <- function(tree_list, new_data = NULL) {
  # if data is not supplied, return the error based on training data
  if (is.null(new_data)) {
    return(tree_list$error)
  }
  # if new data is supplied, compute predictions from scratch
  predictions <- jftree.predict(tree_list, new_data)
  response_indices <- which(names(new_data) %in% tree_list$response.names)
  times <- new_data[, response_indices[1]]
  ind <- new_data[, response_indices[2]]
  # fix this to not just be survival!
  return(JFCppErrorSurvival(predictions, times, ind))
}

# the main function for fitting forests
jfforest <- function(formula, data, splitrule = NULL, mtry = NULL, min_node_size = NULL, nsplits = 10,
                     ntrees = NULL, honest = FALSE, swr = FALSE, sample_rate = NULL, double_bootstrap = FALSE, 
                     seed = NULL, nworkers = 0) {
  # preprocess the entire dataset
  processed_data <- preprocess_data(data)
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

  # if left hand side is of length 1, classification or regression
  if (length(lhs) == 1) {
    response_indices <- which(names(data) %in% as.character(formula[[2]])[1]) - 1
    # TODO
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
              processed_data$unique_values, seed, nworkers)
  }
  # if the left hand side is "MM(...)", multi-state
  else if (lhs[1] == "MM") {
    # TODO
  } else {
    stop("Type of tree not recognised from the formula.")
  }
}

# the main function for predicting with forests
jfforest.predict <- function(forest_list, new_data = NULL) {
  # if data is not supplied, return predictions based on data
  if (is.null(new_data)) {
    return(forest_list$predictions)
  }
  if (class(new_data) == "data.frame") {
    # if new data is supplied, start by preprocessing the data and
    # extracting relevant columns
    covariates <- forest_list$feature.names
    if (ncol(new_data) > 1) {
      new_data <- new_data[, covariates]  # ensures the columns have the same order as the original dataset
    }
    feature_indices <- which(names(new_data) %in% covariates) - 1
    processed_data <- preprocess_data(new_data)
    return(JFCppForestPredict(forest_list, processed_data$data, feature_indices,
                            processed_data$categorical, processed_data$unique_values))
  } else {
    cat("Error: If new_data is supplied, it must be a data.frame with the same names as the original dataset \n")
  }
}

jfforest.error <- function(forest_list, new_data = NULL) {
  # if data is not supplied, return the error based on training data
  if (is.null(new_data)) {
    return(forest_list$oob.error)
  }
  # if new data is supplied, compute predictions from scratch
  predictions <- jfforest.predict(forest_list, new_data)
  response_indices <- which(names(new_data) %in% forest_list$response.names)
  times <- new_data[, response_indices[1]]
  ind <- new_data[, response_indices[2]]
  # change this to not just be survival!
  return(JFCppErrorSurvival(predictions, times, ind))
}

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
    # TODO
  }
  if (tree_list$tree.type == "Classification") {
    # TODO
  }
  if (tree_list$tree.type == "Survival") {
    cat("Type of tree: Survival\n")
  }
  if (tree_list$tree.type == "Multi-state") {
    # TODO
  }

  # print basic data info
  cat("Number of observations:", tree_list$num.obs, "\n")
  cat("Number of features:", tree_list$num.features, "\n")

  if (tree_list$tree.type == "Regression") {
    # TODO
  }
  if (tree_list$tree.type == "Classification") {
    # TODO
  }
  if (tree_list$tree.type == "Survival") {
    cat("Number of deaths:", tree_list$num.deaths, "\n")
    if (length(tree_list$unique.event.times) <= 20) {
      cat("Unique event times:", tree_list$unique.event.times, "\n")
    }
  }
  if (tree_list$tree.type == "Multi-state") {
    # TODO
  }

  # print hyperparameters
  cat("Minimal node size:", tree_list$min.node.size, "\n")
  cat("Number of selected features in each split:", tree_list$mtry, "\n")
  cat("Number of possible splits considered in each node:", tree_list$nsplits, "\n")
  cat("Splitting rule:", tree_list$splitrule, "\n")
  if (tree_list$honest) {
    cat("Honest: Yes \n")
  } else {
    cat("Honest: No \n")
  }

  # print info about the tree itself
  cat("Training error:",tree_list$error, "\n")
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
    # TODO
  }
  if (forest_list$tree.type == "Classification") {
    # TODO
  }
  if (forest_list$tree.type == "Survival") {
    cat("Type of tree: Survival\n")
  }
  if (forest_list$tree.type == "Multi-state") {
    # TODO
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
  }
  if (forest_list$tree.type == "Multi-state") {
    # TODO
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
  cat("Number of possible splits considered in each node:", forest_list$nsplits, "\n")
  cat("Splitting rule:", forest_list$splitrule, "\n")
  if (forest_list$honest) {
    cat("Honest: Yes \n")
  } else {
    cat("Honest: No \n")
  }

  # print info about the forest itself
  cat("OOB error:", forest_list$oob.error, "\n")
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
    if (class(data[, i]) == "character") {
      data[, i] <- as.numeric(as.factor(data[, i]))
      categorical[i] <- 1
      unique_values[i] <- length(unique(data[, i]))
    } else if (class(data[, i]) == "factor" || class(data[, i]) == "logical") {
      data[, i] <- as.numeric(data[, i])
      categorical[i] <- 1
      unique_values[i] <- length(unique(data[, i]))
    }
  }
  list(data = data, unique_values = unique_values, categorical = categorical)
}

# function for testing the methods in Data.cpp
test_data_functions <- function(data, response_indices,
                                feature_indices) {
  processed_data <- preprocess_data(data)
  response_indices <- response_indices - 1
  feature_indices <- feature_indices - 1
  testData(processed_data$data, response_indices, feature_indices,
           processed_data$categorical, processed_data$unique)
}

# rough function to fit a survival tree on the data
# later we should provide a formula (Times, Indicators) ~ <covariates>
# to make the function more user friendly and the indices used for
# choosing the subset of the data should be given as a c (bootstrap,
# w/wo replacement etc.)
fit_survival_tree <- function(data, mtry, min_node_size, nsplits,
                              response_indices, feature_indices,
                              subset_indices) {

  processed_data <- preprocess_data(data)
  response_indices <- response_indices - 1
  feature_indices <- feature_indices - 1
  subset_indices <- subset_indices - 1
  
  # for debugging
  cat("Categorical:", processed_data$categorical, "\n")
  cat("Unique values:", processed_data$unique_values, "\n")
  cat("Response indices:", response_indices, "\n")
  cat("Feature indices:", feature_indices, "\n")
  cat("Subset indices", subset_indices, "\n")
  fitSurvivalTree(processed_data$data, mtry, min_node_size, nsplits, response_indices,
                  feature_indices, processed_data$categorical, processed_data$unique_values,
                  subset_indices)
}

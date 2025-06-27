# the main function for fitting trees
jftree <- function(formula, data, mtry, min_node_size, nsplits, seed = NULL) {
  # preprocess the entire dataset
  processed_data <- preprocess_data(data)

  # formula determines the type of tree, the features and the response
  lhs <- as.character(formula[[2]])
  covariates <- attr(terms(formula), "term.labels")
  feature_indices <- which(names(data) %in% covariates) - 1

  # if seed is not set, generate a random one
  if (is.null(seed)) {
    seed = runif(n = 1, min = 1, max = 10^6)
  }

  # if left hand side is of length 1, classification or regression
  if (length(lhs) == 1) {
    response_indices <- which(names(data) %in% as.character(formula[[2]])[1]) - 1
    # TODO
  }
  # if left hand side is "Surv(time, status)", survival
  else if (lhs[1] == "Surv") {
    # time is always assumed to be the first argument
    response_indices <- which(names(data) %in% as.character(formula[[2]])[2:3]) - 1
    JFCppTree(3, processed_data$data, mtry, min_node_size, nsplits,
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
jftree.predict <- function(tree_list, oob = FALSE, new_data = NULL) {
  # if data is not supplied, return predictions based on data
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

# the main function for fitting forests
jfforest <- function(formula, data, mtry, min_node_size, nsplits, ntrees, seed = NULL) {
  # preprocess the entire dataset
  processed_data <- preprocess_data(data)

  # formula determines the type of tree, the features and the response
  lhs <- as.character(formula[[2]])
  covariates <- attr(terms(formula), "term.labels")
  feature_indices <- which(names(data) %in% covariates) - 1

  # if seed is not set, generate a random one
  if (is.null(seed)) {
    seed = runif(n = 1, min = 1, max = 10^6)
  }

  # if left hand side is of length 1, classification or regression
  if (length(lhs) == 1) {
    response_indices <- which(names(data) %in% as.character(formula[[2]])[1]) - 1
    # TODO
  }
  # if left hand side is "Surv(time, status)", survival
  else if (lhs[1] == "Surv") {
    # time is always assumed to be the first argument
    response_indices <- which(names(data) %in% as.character(formula[[2]])[2:3]) - 1
    JFCppForest(3, processed_data$data, mtry, min_node_size, nsplits, ntrees,
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

# the main function for predicting with forests
jfforest.predict <- function(forest_list, new_data = NULL) {
  # if data is not supplied, return predictions based on data
  if (is.null(new_data)) {
    return(forest_list$predictions)
  }
  # if new data is supplied, start by preprocessing the data and
  # extracting relevant columns
  covariates <- forest_list$feature.names
  new_data <- new_data[, covariates]  # ensures the columns have the same order as the original dataset
  feature_indices <- which(names(new_data) %in% covariates) - 1
  processed_data <- preprocess_data(new_data)
  return(JFCppForestPredict(forest_list, processed_data$data, feature_indices,
                          processed_data$categorical, processed_data$unique_values))
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
    cat("Unique event times:", tree_list$unique.event.times, "\n")
  }
  if (tree_list$tree.type == "Multi-state") {
    # TODO
  }

  # print hyperparameters
  cat("Minimal node size:", tree_list$min.node.size, "\n")
  cat("Number of selected features in each split", tree_list$mtry, "\n")
  cat("Number of possible splits considered in each node", tree_list$nsplits, "\n")

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

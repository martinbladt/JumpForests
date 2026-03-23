#nolint start: line_length_linter
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)

set.seed(2026)
n <- 500
X1 <- rbinom(n, 3, 0.2)
X2 <- rnorm(n, 3, 2)
X3 <- rnorm(n)
X4 <- rnorm(n)

test_data <- data.frame(X1 = X1, X2 = X2, X3 = X3, X4 = X4, Y = 2 * X1 + 3 * X2 - 2 * X3^2)
new_data <- data.frame(X2 = rnorm(n, 3, 2), X1 = rbinom(n, 3, 0.2), Y = 2 * X1 + 3 * X2 - 2 * X3^2, X3 = rnorm(n), X4 = rnorm(n))

test_forest <- jfforest(Y ~ ., data = test_data, honest = FALSE, min_node_size = 5)
test_forest
print_forest(test_forest)   # needs to print error (also fix the subsample size with honesty)
mean((test_data$Y - mean(test_data$Y))^2)   # 41.74698

jfforest.predict(test_forest)
jfforest.predict(test_forest, new_data = test_data)
jfforest.predict(test_forest, new_data = new_data)

jfforest.error(test_forest)
jfforest.error(test_forest, new_data = test_data)
jfforest.error(test_forest, new_data = new_data)

test_forest_SRC <- rfsrc(Y ~ ., data = test_data, ntree = 1000)
test_forest_SRC

test_forest_ranger <- ranger(Y ~ ., data = test_data, num.trees = 1000)
test_forest_ranger

test_forest$oob.predictions[1:20]
test_forest_SRC$predicted.oob[1:20]
abs(test_forest$oob.predictions[1:20] - test_forest_SRC$predicted.oob[1:20])

length(test_forest$oob.predictions[test_forest$oob.predictions > 0])
length(test_forest_SRC$predicted.oob[test_forest_SRC$predicted.oob > 0])

min(test_data$Y)                    # -8.781968 (for n = 1000)
min(test_forest$oob.predictions)    # -8.225581
min(test_forest_SRC$predicted.oob)  # 1.86989

# and too many negative predicted values

# current issues:
# - poor fitting (the error scales as the data grows?? But for n = 50, 100, the errors and predictions are comparable)
#   no explanation at the moment... but it is not because the trees are too deep, the number of terminal nodes is actually
#   smaller than for SRC, the fitting itself is somehow problematic, even if the code is as good as identical to ranger
#   which works perfectly fine
# - speed?       (much slower than randomForestSRC)
# - honesty gives NaN values for error in regression
# - display correct subsample size when honest = TRUE
# - display error for regression forest in print_forest
# - should choose a convention on predicted values (NumericMatrix/NumericVector)

test_tree <- jftree(Y ~ ., data = test_data, min_node_size = 100)
test_tree
print_tree(test_tree)

jftree.predict(test_tree)
jftree.predict(test_tree, new_data = test_data)
jftree.predict(test_tree, new_data = new_data)

jftree.error(test_tree)
jftree.error(test_tree, new_data = test_data)
jftree.error(test_tree, new_data = new_data)

#nolint_end
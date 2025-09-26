setwd("/home/themathlad/Documents/GitHub/JumpForests/testing/")
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)

set.seed(2025)
n <- 1000
X1 <- rbinom(n, 3, 0.2)
X2 <- rnorm(n, 3, 2)

test_data <- data.frame(X1 = X1, X2 = X2, X3 = rnorm(n), Y = 2 * X1 + 3 * X2)
new_data <- data.frame(X2 = rnorm(n, 3, 2), X1 = rbinom(n, 3, 0.2), Y = 2 * X1 + 3 * X2, X3 = rnorm(n))

test_forest <- jfforest(Y ~ ., data = test_data, honest = FALSE, min_node_size = 20)
print_forest(test_forest)   # needs to print error (also fix the subsample size with honesty)

jfforest.predict(test_forest)
jfforest.predict(test_forest, new_data = test_data)
jfforest.predict(test_forest, new_data = new_data)

jfforest.error(test_forest)
jfforest.error(test_forest, new_data = test_data)
jfforest.error(test_forest, new_data = new_data)

test_forest_SRC <- rfsrc(Y ~ ., data = test_data, ntree = 1000)
test_forest_SRC

test_forest$oob.predictions[1:10]
test_forest_SRC$predicted.oob[1:10]

length(test_forest$oob.predictions[test_forest$oob.predictions > 0])
length(test_forest_SRC$predicted.oob[test_forest_SRC$predicted.oob > 0])
# my implementation fits very poorly, too complex trees apparently
# and too many negative predicted values

# current issues:
# - poor fitting
# - speed?
# - honesty gives NaN values for error in regression
# - display correct subsample size when honest = TRUE
# - display error for regression forest in print_forest
# - should choose a convention on predicted values (NumericMatrix/NumericVector)

test_tree <- jftree(Y ~ ., data = test_data)
print_tree(test_tree)

jftree.predict(test_tree)
jftree.predict(test_tree, new_data = test_data)
jftree.predict(test_tree, new_data = new_data)

jftree.error(test_tree)
jftree.error(test_tree, new_data = test_data)
jftree.error(test_tree, new_data = new_data)

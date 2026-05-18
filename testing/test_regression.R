#nolint start: line_length_linter
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)

set.seed(2026)
n <- 2000
X1 <- rbinom(n, 3, 0.2)
X2 <- rnorm(n, 3, 2)
X3 <- rnorm(n)
X4 <- rnorm(n)

test_data <- data.frame(X1 = X1, X2 = X2, X3 = X3, X4 = X4, Y = 2 * X1 + 3 * X2 - 2 * X3^2)
new_data <- data.frame(X2 = rnorm(n, 3, 2), X1 = rbinom(n, 3, 0.2), Y = 2 * X1 + 3 * X2 - 2 * X3^2, X3 = rnorm(n), X4 = rnorm(n))

test_forest <- jfforest(Y ~ ., data = test_data, honest = TRUE, min_node_size = 5)
test_forest
print_forest(test_forest)   # needs to print error (also fix the subsample size with honesty)
mean((test_data$Y - mean(test_data$Y))^2)   # 44.95195

jfforest.predict(test_forest)
jfforest.predict(test_forest, new_data = test_data)
jfforest.predict(test_forest, new_data = new_data)

jfforest.error(test_forest)
# without honesty: $mse.error = 1.381681, R2.error = 0.9692632
# with honesty: $mse.error = 2.123524, R2.error = 0.9527601
jfforest.error(test_forest, new_data = test_data)
jfforest.error(test_forest, new_data = new_data)

test_forest_SRC <- rfsrc(Y ~ ., data = test_data, ntree = 1000)
# (OOB) Requested performance error = 1.11704754, (OOB) R squared = 0.97516261
test_forest_SRC
predict.rfsrc(test_forest_SRC, newdata = new_data)

# now completely comparable

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
# - display correct subsample size when honest = TRUE
# - display error for regression forest in print_forest

test_tree <- jftree(Y ~ ., data = test_data, min_node_size = 100)
test_tree
print_tree(test_tree)

jftree.predict(test_tree)
jftree.predict(test_tree, new_data = test_data)
jftree.predict(test_tree, new_data = new_data)

jftree.error(test_tree)
jftree.error(test_tree, new_data = test_data)
jftree.error(test_tree, new_data = new_data)

# testing on different datasets
#-------------------------------------------------------------------------------------------------
data("BostonHousing", package = "mlbench")
head(BostonHousing)
set.seed(2026)
housing_forest <- jfforest(medv ~ ., data = BostonHousing, honest = FALSE, min_node_size = 5)
housing_forest_SRC <- rfsrc(medv ~ ., data = BostonHousing, min_node_size = 5)

print_forest(housing_forest)    
jfforest.error(housing_forest)  # MSE = 13.02692, R^2 = 0.8456884
housing_forest_SRC              # MSE = 11.56478888, R^2 = 0.86327891

# comment: with a bit of tuning, the OOB error of our forest can be made at least as small as the error for SRC

data("CO2", package = "datasets")
head(CO2)

CO2_forest <- jfforest(uptake ~ ., data = CO2, min_node_size = 3)
CO2_forest_SRC <- rfsrc(uptake ~ ., data = CO2, min_node_size = 5)

print_forest(CO2_forest)
CO2_forest_SRC

#nolint_end
#nolint start: line_length_linter
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)

# plan for Monday 25 - Friday 29
# - debug!
# - finish implementing the classification tree and forest in JumpForest.cpp + JumpForest.h
# - implement the R interface for classification including the print_forest (OOB misclassification and Brier score, also confusion matrix)
# - test and compare VIMP for classification
# - simulation study for regression



# wine
#-------------------------------------------------------------------------------------------------

data(wine, package = "randomForestSRC")
head(wine)
wine$quality <- as.factor(wine$quality)

wine_tree <- jftree(quality ~ ., data = wine, min_node_size = 5)
print_tree(wine_tree)
wine_tree

wine_forest <- jfforest(quality ~ ., data = wine)
wine_forest

wine_forest_SRC <- rfsrc(quality ~., data = wine)
wine_forest_SRC

# iris
#-------------------------------------------------------------------------------------------------

iris_tree <- jftree(Species ~ ., data = iris, min_node_size = 5)
iris_tree
print_tree(iris_tree)

iris_forest <- jfforest(Species ~ ., data = iris)
print_forest(iris_forest)

iris_forest_SRC <- rfsrc(Species ~ ., data = iris)
iris_forest_SRC

# very much comparable

# should probably test more datasets


data("Zoo", package = "mlbench")
data("Satellite", package = "mlbench")
data("Shuttle", package = "mlbench")
data("Sonar", package = "mlbench")
data("Glass", package = "mlbench")

#nolint end

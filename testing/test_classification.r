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
print_forest(wine_forest)

wine_forest_SRC <- rfsrc(quality ~., data = wine)
wine_forest_SRC

# iris
#-------------------------------------------------------------------------------------------------

iris_tree <- jftree(Species ~ ., data = iris, min_node_size = 5)
iris_tree
print_tree(iris_tree)

iris_forest <- jfforest(Species ~ ., data = iris, splitrule = "gini")
print_forest(iris_forest)

iris_forest_SRC <- rfsrc(Species ~ ., data = iris)
iris_forest_SRC

# very much comparable for gini, try other splitting rules (gini, entropy, misc, twoing or hellinger)
iris_forest <- jfforest(Species ~ ., data = iris, splitrule = "entropy")
print_forest(iris_forest)

iris_forest_SRC <- rfsrc(Species ~ ., data = iris, splitrule = "entropy")
iris_forest_SRC

# should probably test more datasets
data("Zoo", package = "mlbench")
data("Satellite", package = "mlbench")
data("Shuttle", package = "mlbench")
data("Sonar", package = "mlbench")
data("Glass", package = "mlbench")

set.seed(2026)
zoo_forest <- jfforest(type ~ ., data = Zoo, splitrule = "gini")
print_forest(zoo_forest)
zoo_forest_src <- rfsrc(type ~ ., data = Zoo)
zoo_forest_src

# almost perfectly aligned, and all splitting rules make sense

# Sonar is highdimensional (60 variables, 208 observations)
Sonar_forest <- jfforest(Class ~ ., data = Sonar, splitrule = "gini")
print_forest(Sonar_forest)
Sonar_forest_src <- rfsrc(Class ~ ., data = Sonar)
Sonar_forest_src

# concurs and all splitting rules make sense

Glass_forest <- jfforest(Type ~ ., data = Glass, splitrule = "gini")
print_forest(Glass_forest)
Glass_forest_src <- rfsrc(Type ~ ., data = Glass, splitrule = "gini")
Glass_forest_src

# concurs nicely and all splitting rules make sense

# finally, Satellite (big dataset, )
Satellite_forest <- jfforest(classes ~ ., data = Satellite, splitrule = "gini", ntrees = 500)
print_forest(Satellite_forest)
Satellite_forest_src <- rfsrc(classes ~ ., data = Satellite, splitrule = "gini", ntrees = 500)
Satellite_forest_src

# also concurs (only tried gini here)

# VIMP
#-------------------------------------------------------------------------------------------------




#nolint end

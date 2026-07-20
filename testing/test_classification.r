#nolint start: line_length_linter
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)


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

devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)

set.seed(2026)
iris_forest <- jfforest(Species ~ ., data = iris)
iris_forest_SRC <- rfsrc(Species ~ ., data = iris)
iris_forest_ranger <- ranger(Species ~ ., data = iris, importance = "permutation")

# misclassification loss
vimp_permute <- unlist(jfforest.vimp(iris_forest, method = "permute")$vimp)
vimp_random <- unlist(jfforest.vimp(iris_forest, method = "random")$vimp)

vimp.rfsrc(iris_forest_SRC, importance = "permute", block.size = 1)$importance
importance(iris_forest_ranger)
vimp_permute

vimp.rfsrc(iris_forest_SRC, importance = "random", block.size = 1)$importance
vimp_random

# Brier score loss
vimp_permute <- unlist(jfforest.vimp(iris_forest, method = "permute", loss = "brier")$vimp)
vimp_random <- unlist(jfforest.vimp(iris_forest, method = "random", loss = "brier")$vimp)

vimp.rfsrc(iris_forest_SRC, importance = "permute", block.size = 1, perf.type = "brier")$importance
vimp_permute

vimp.rfsrc(iris_forest_SRC, importance = "random", block.size = 1, perf.type = "brier")$importance
vimp_random

# seems to concur quite well for iris

data("Zoo", package = "mlbench")
data("Sonar", package = "mlbench")
data("Glass", package = "mlbench")

set.seed(2026)
zoo_forest <- jfforest(type ~ ., data = Zoo)
zoo_forest_src <- rfsrc(type ~ ., data = Zoo)
zoo_forest_ranger <- ranger(type ~ ., data = Zoo, importance = "permutation")

# misclassification loss
vimp_permute <- unlist(jfforest.vimp(zoo_forest, method = "permute")$vimp)
vimp_random <- unlist(jfforest.vimp(zoo_forest, method = "random")$vimp)

vimp.rfsrc(zoo_forest_src, importance = "permute", block.size = 1)$importance[,1]
importance(zoo_forest_ranger)
vimp_permute

vimp.rfsrc(zoo_forest_src, importance = "random", block.size = 1)$importance[,1]
vimp_random

# Brier loss
vimp_permute <- unlist(jfforest.vimp(zoo_forest, method = "permute", loss = "brier")$vimp)
vimp_random <- unlist(jfforest.vimp(zoo_forest, method = "random", loss = "brier")$vimp)

vimp.rfsrc(zoo_forest_src, importance = "permute", block.size = 1, perf.type = "brier")$importance[,1]
vimp_permute

vimp.rfsrc(zoo_forest_src, importance = "random", block.size = 1)$importance[,1]
vimp_random

# concurs very nicely

set.seed(2026)
sonar_forest <- jfforest(Class ~ ., data = Sonar)
sonar_forest_src <- rfsrc(Class ~ ., data = Sonar)
sonar_forest_ranger <- ranger(Class ~ ., data = Sonar, importance = "permutation")

vimp_permute <- unlist(jfforest.vimp(sonar_forest, method = "permute")$vimp)
vimp_random <- unlist(jfforest.vimp(sonar_forest, method = "random")$vimp)

vimp.rfsrc(sonar_forest_src, importance = "permute", block.size = 1)$importance[,1]
importance(sonar_forest_ranger)
vimp_permute

vimp.rfsrc(sonar_forest_src, importance = "random", block.size = 1)$importance[,1]
vimp_random

# hard to conclude anything (too many variables, all with small values)

set.seed(2026)
glass_forest <- jfforest(Type ~ ., data = Glass)
glass_forest_src <- rfsrc(Type ~ ., data = Glass)
glass_forest_ranger <- ranger(Type ~ ., data = Glass, importance = "permutation")

# misclassification error
vimp_permute <- unlist(jfforest.vimp(glass_forest, method = "permute")$vimp)
vimp_random <- unlist(jfforest.vimp(glass_forest, method = "random")$vimp)

vimp.rfsrc(glass_forest_src, importance = "permute", block.size = 1)$importance[,1]
importance(glass_forest_ranger)
vimp_permute

vimp.rfsrc(glass_forest_src, importance = "random", block.size = 1)$importance[,1]
vimp_random

# Brier error
vimp_permute <- unlist(jfforest.vimp(glass_forest, method = "permute", loss = "brier")$vimp)
vimp_random <- unlist(jfforest.vimp(glass_forest, method = "random", loss = "brier")$vimp)

vimp.rfsrc(glass_forest_src, importance = "permute", block.size = 1, perf.type = "brier")$importance[,1]
vimp_permute

vimp.rfsrc(glass_forest_src, importance = "random", block.size = 1, perf.type = "brier")$importance[,1]
vimp_random

# concurs nicely

# conclusion: the misclassification and Brier based VIMP values seem to be correctly implemented

#nolint end


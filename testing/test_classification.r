#nolint start: line_length_linter
devtools::load_all()
library(Rcpp)
library(randomForestSRC)
library(ranger)

# plan for Monday 25
# - debug!
# - finish implementing the classification tree and forest in JumpForest.cpp + JumpForest.h
# - implement the R interface for classification including the print_forest (OOB misclassification and Brier score, also confusion matrix)
# - test and compare VIMP for classification

data(wine, package = "randomForestSRC")
head(wine)

typeof(wine$quality)
class(wine$quality)
is.factor(wine$quality)
wine$quality <- as.factor(wine$quality)
typeof(wine$quality)
class(wine$quality)
is.factor(wine$quality)

# to transform the 'factor' vector to a double vector for use in C++:
as.numeric(wine$quality)

wine_forest_SRC <- rfsrc(quality ~., data = wine)

wine_forest_SRC

#nolint end
#setwd("testing")
devtools::load_all()
library(Rcpp)

# for testing the data interaction between C++ and R
test_data <- data.frame(Categorical1 = as.factor(c(1, 2, 1, 1, 2, 1, 2, 1, 2)), # nolint: line_length_linter.
                        Time = c(0.13, 0.65, 0.15, 2.1, 0.415, 1.1, 2, 0.65, 2.1), # nolint: line_length_linter.
                        Numerical = c(0.41, 1.31, 0.78, 0.56, 0.61, 0.42, 0.13, 0.34, 0.15), # nolint: line_length_linter.
                        Death = c(0, 1, 1, 1, 0, 1, 0, 1, 1),
                        Categorical2 = c("Yes", "No", "No", "Maybe", "Yes", "No", "Maybe", "No", "Yes")) # nolint: line_length_linter.
new_data <- data.frame(Numerical = c(0.57, 0.13, 0.156, 0.81, 1.2),
                       Categorical1 = as.factor(c(1, 1, 2, 1, 1)),
                       Categorical2 = c("Yes", "Maybe", "Maybe", "No", "Yes"))
#test_data2 <- data.frame(Time = c(0.13, 0.65, 0.15, 2.1, 0.415, 1.1, 2, 0.65, 2.1), # nolint: line_length_linter.
#                         Numerical1 = c(0.41, 1.31, 0.78, 0.56, 0.61, 0.42, 0.13, 0.34, 0.15), # nolint: line_length_linter.
#                         Numerical2 = c(0.271, 0.81, 0.1, 2.1, 0.145, 0.82, 0.91, 0.41, 1.34), #nolint: line_length_linter
#                         Death = c(0, 1, 1, 1, 0, 1, 0, 1, 1))

# functions for testing the transcription to a C++ data object
#data_testing(test_data, c(2, 4), c(1, 3, 5))
#data_testing(new_data, c(), c(1, 2, 3))

# function to fit a survival tree (outdated)
#fit_survival_tree(test_data, 1, 2, 2, c(2, 4), c(1, 3, 5), 1:9)

# proper function to fit a survival tree
fitted_tree <- jftree(Surv(Time, Death) ~ Categorical1 + Numerical + Categorical2,
                      data = test_data, mtry = 1, min_node_size = 2, nsplits = 2, seed = 1)

# change full to TRUE to get a full table of the tree
# (a daughter being 0 means that the node is a leaf)
print_tree(fitted_tree, full = TRUE)

jftree.predict(fitted_tree)
# predict on the dataset, should equal the output from above line
jftree.predict(fitted_tree, test_data)
# predict on new dataset
jftree.predict(fitted_tree, new_data)

fitted_forest <- jfforest(Surv(Time, Death) ~ Categorical1 + Numerical + Categorical2,
                          data = test_data, mtry = 1, min_node_size = 2, nsplits = 2, ntrees = 1000, seed = 2025)
# preditions on the dataset
jfforest.predict(fitted_forest)
# prediction on new dataset
jfforest.predict(fitted_forest, new_data)

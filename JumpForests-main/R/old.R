# all functions below are strictly for testing purposes and do not
# go into the final library
#----------------------------------------------------------------------------

list_testing <- function() {
  testList()
}

data_testing <- function(data, response_indices, feature_indices) {
  processed_list <- preprocess_data(data)
  response_indices <- response_indices - 1
  feature_indices <- feature_indices - 1

  df_test(processed_list$data, response_indices, feature_indices,
          processed_list$categorical, processed_list$unique_values)
}

# simple test function for sampling without replacement using C++
sampling_test <- function(k) {
  silly_sampler(k)
}

# simple test function that picks out a column from the data and returns
# as a vector
test_function <- function(data, column_index) {
  extract_column(data, column_index)
}

grow_tree <- function(data = NULL, type = NULL) {
}

# just for testing

#' Square a number using C++
#' 
#' @param x A numeric value
#' @return The square of x
#' @export
#' @importFrom Rcpp JumpForests.cpp
square <- function(x) {
  square_cpp(x)
}

#' Multiply matrix by 2
#'
#' @param mat A numeric matrix.
#' @return The input matrix multiplied by 2.
#' @export
my_times_two <- function(mat) {
  timesTwoMatrix(mat)
}
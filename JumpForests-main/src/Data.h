// file for transferring and processing the data into a C++ format
#ifndef DATA_H
#define DATA_H

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_set>
#include <Rcpp.h>
using namespace Rcpp;
using namespace std;

struct Data {
  Data(DataFrame data, vector<size_t> response_indices, vector<size_t> feature_indices, vector<bool> categorical, vector<size_t> unique);

  // delete the copy constructor and the assignment operator 
  //Data(const Data&) = delete;
  //Data& operator=(const Data&) = delete;

  // extract data
  double get_x(size_t row, size_t col) const {
    return x(row, col);
  }
  double get_y(size_t row, size_t col) const {
    return y(row, col);
  }
  vector<double> get_x_row(size_t row) const {
    NumericVector res = x.row(row);
    return as<vector<double>>(res);
  }
  vector<double> get_y_row(size_t row) const {
    NumericVector res = y.row(row);
    return as<vector<double>>(res);
  }
  vector<double> get_x_col(size_t col) const {
    NumericVector res = x.column(col);
    return as<vector<double>>(res);
  }
  vector<double> get_y_col(size_t col) const {
    NumericVector res = y.column(col);
    return as<vector<double>>(res);
  }

  // for extracting possible split values
  vector<double> getValues(vector<size_t> subset_indices, size_t feature);

  // for getting misc. information
  size_t getNumberOfObs() const {
    return num_obs;
  }
  size_t getNumberOfFeatures () const {
    return num_features;
  }
  vector<bool> getCategorical () const {
    return categorical;
  }
  vector<size_t> getUniqueValues () const {
    return unique_values;
  }
  vector<size_t> getResponseIndices () const {
    return response_indices;
  }
  vector<size_t> getFeatureIndices () const {
    return feature_indices;
  }
  vector<string> getFeatureNames () const {
    return feature_names;
  }
  vector<string> getResponseNames () const {
    return response_names;
  }

private:
  // data
  NumericMatrix x;
  NumericMatrix y;

  // data attributes
  size_t num_obs;                   // number of observations
  size_t num_features;              // number of features
  vector<bool> categorical;         // for each feature, 1 if categorical, 0 otherwise
  vector<size_t> unique_values;     // number of unique values for each feature
  vector<string> feature_names;     // variable name for each feature
  vector<string> response_names;    // variable name for each response
  vector<size_t> response_indices;  // the indices of the responses
  vector<size_t> feature_indices;   // the indices of the features

  // get the ID based on a feature name
  size_t getFeatureID(const string& variable_name) const;
};

#endif // DATA_H
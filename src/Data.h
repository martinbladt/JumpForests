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
#include <utility>
using namespace Rcpp;
using namespace std;

// the following struct handles data for all types of trees and forests

struct Data {
  // constructor for creating data objects for regression, classification and survival data
  Data(DataFrame data, const vector<size_t>& response_indices, vector<size_t> feature_indices,
       const vector<bool>& categorical, const vector<size_t>& unique);
  // constructor for creating data objects for multi-states
  Data(List jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame feature_data,
       const vector<bool>& categorical, const vector<size_t>& unique);

  // delete the copy constructor and the assignment operator 
  //Data(const Data&) = delete;
  //Data& operator=(const Data&) = delete;
  
  // fetching feature values
  double get_x(size_t row, size_t col) const {
    return x[row * num_features + col];
  }
  vector<double> get_x() const {
    return x;
  }
  vector<double> get_x_row(size_t row) const {
    vector<double> res(num_features);
    for (size_t i = 0; i < num_features; ++i) {
      res[i] = x[row * num_features + i];
    }
    return res;
  }
  vector<double> get_x_col(size_t col) const {
    vector<double> res(num_obs);
    for (size_t i = 0; i < num_obs; ++i) {
      res[i] = x[i * num_features + col];
    }
    return res;
  }

  // extract y for classification and regression
  double get_y(size_t row) {
    return y[row];
  }
  // return y (regression, classification and survival)
  vector<double> get_y() const {
    return y;
  }
  // extract y for survival
  double get_y(size_t row, size_t col) {
    return y[col * num_obs + row];
  }
  // return a specific y column for survival data (0: times, 1: indicators)
  vector<double> get_y_col(size_t col) const {
    vector<double> res(num_obs);
    for (size_t i = 0; i < num_obs; ++i) {
      res[i] = y[col * num_obs + i];
    }
    return res;
  }
  // extract times and states for multi-state data
  vector<double> getTimes() const {
    return times;
  }
  vector<uint8_t> getStates() const {
    return states;
  }
  vector<double> getCensoringTimes() const {
    return censoring_times;
  }
  vector<uint8_t> getCensoringStates() const {
    return censoring_states;
  }

  // for extracting possible split values
  vector<double> getValues(const vector<size_t>& subset_indices, size_t feature);

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
  uint8_t getMaxResponseLength () const {
    return max_response_length;
  }
  uint8_t getNumberOfStates () const {
    return num_states;
  }
  // get the ID based on a feature name
  size_t getFeatureID(const string& variable_name) const;

private:
  // data
  vector<double> x;                 // the features are saved as a flattened 2D-array (counted by observation number)
  vector<double> y;                 // ditto for responses (regression, classification and survival)
  vector<double> times;             // save jump times for each trajectory as flattened 2D-array (only for multi-state data)
  vector<uint8_t> states;           // save state info for each trajectory as flattened 2D-array (only for multi-state data)
  vector<double> censoring_times;   // save the censoring times (0: no censoring, only for multi-state data)
  vector<uint8_t> censoring_states; // save the state of the censoring time (0: no censoring, only for multi-state data) 

  // data attributes
  size_t num_obs;                   // number of observations
  size_t num_features;              // number of features
  size_t num_responses;             // number of responses
  vector<bool> categorical;         // for each feature, 1 if categorical, 0 otherwise
  vector<size_t> unique_values;     // number of unique values for each feature
  vector<string> feature_names;     // variable name for each feature
  vector<string> response_names;    // variable name for each response
  vector<size_t> response_indices;  // the indices of the responses
  vector<size_t> feature_indices;   // the indices of the features
  uint8_t max_response_length;      // maximum number of jumps observed in the data (only for multi-state data)
  uint8_t num_states;               // number of states in the multi-state model (only for multi-state data)
};

// the following struct handles Multi-state models

/*
struct MMData {
  // jump_data is the response and feature_data is the DataFrame of features,
  // hence no need to have feature_indices
  MMData(List jump_data, uint8_t max_response_length, DataFrame feature_data,
       const vector<bool>& categorical, const vector<size_t>& unique);

  // extract data
  double get_x(size_t row, size_t col) const {
    return x[row * num_features + col];
  }

  vector<double> get_x_row(size_t row) const {
    vector<double> res(num_features);
    for (size_t i = 0; i < num_features; ++i) {
      res[i] = x[row * num_features + i];
    }
    return res;
  }

  vector<double> get_x_col(size_t col) const {
    vector<double> res(num_obs);
    for (size_t i = 0; i < num_obs; ++i) {
      res[i] = x[i * num_features + col];
    }
    return res;
  }

  // for extracting possible split values
  vector<double> getValues(const vector<size_t>& subset_indices, size_t feature);

  // for getting misc. information
  size_t getNumberOfObs() const {
    return num_obs;
  }
  size_t getNumberOfFeatures() const {
    return num_features;
  }
  vector<bool> getCategorical() const {
    return categorical;
  }
  vector<size_t> getUniqueValues() const {
    return unique_values;
  }
  
  vector<size_t> getResponseIndices () const {
    return response_indices;
  }
  vector<size_t> getFeatureIndices () const {
    return feature_indices;
  }
  vector<string> getResponseNames () const {
    return response_names;
  }
  
  vector<double> getTimes() const {
    return times;
  }
  vector<uint8_t> getStates() const {
    return states;
  }

  vector<string> getFeatureNames () const {
    return feature_names;
  }
  // get the ID based on a feature name
  size_t getFeatureID(const string& variable_name) const;

private:
  // data (new)
  vector<double> x;       // the features are saved as a flattened 2D-array (counted by observation number)
  vector<double> times;   // save jump times for each trajectory as flattened 2D-array
  vector<uint8_t> states;  // save state info for each trajectory as flattened 2D-array

  // data attributes
  size_t num_obs;                   // number of observations
  size_t num_features;              // number of features
  vector<bool> categorical;         // for each feature, 1 if categorical, 0 otherwise
  vector<size_t> unique_values;     // number of unique values for each feature
  vector<string> feature_names;     // variable name for each feature
};

*/


#endif // DATA_H
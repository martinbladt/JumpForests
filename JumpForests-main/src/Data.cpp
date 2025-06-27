#include "Data.h"

// it is assumed that the data has been encoded to be numeric
// response_indices are the indices for the response column(s), while categorical
// is a vector of bools indicating whether a variable (response or feature) is categorical
// unique is a vector of the number of unique values for each column
Data::Data(DataFrame data, vector<size_t> response_indices, vector<size_t> feature_indices, vector<bool> categorical,
           vector<size_t> unique) {
    this->num_obs = data.nrows();
    this->num_features = feature_indices.size();
    this->response_indices = response_indices;

    NumericMatrix x_res(num_obs, num_features);

    // fill the response matrix if response variables are supplied
    if (!response_indices.empty()) {
        NumericMatrix y_res(num_obs, response_indices.size());
        for (size_t i = 0; i < response_indices.size(); ++i) {
            NumericVector col = data[response_indices[i]];
            for (size_t j = 0; j < num_obs; ++j) {
                y_res(j, i) = col[j];
            }
        }
        vector<string> response_names(response_indices.size());
        for (size_t i = 0; i < response_indices.size(); ++i) {
            response_names[i] = as<vector<string>>(data.names())[response_indices[i]];
        }
        this->y = y_res;
        this->response_names = response_names;
    }

    vector<bool> categorical_features(num_features);
    vector<size_t> unique_values_features(num_features);
    vector<string> feature_names(num_features);

    for (size_t i = 0; i < num_features; ++i) {
        // update the feature type (categorical or continuous)
        if (categorical[feature_indices[i]] == true) {
            categorical_features[i] = true;
        } else {
            categorical_features[i] = false;
        }
        
        // update names of features and response(s)
        feature_names[i] = as<vector<string>>(data.names())[feature_indices[i]];

        // update the number of unique values
        unique_values_features[i] = unique[feature_indices[i]];
        
        // fill the feature matrix
        NumericVector col = data[feature_indices[i]];
        for (size_t j = 0; j < num_obs; ++j) {
            x_res(j, i) = col[j];
        }
    }

    this->x = x_res;
    this->categorical = categorical_features;
    this->unique_values = unique_values_features;
    this->feature_names = feature_names;
}

size_t Data::getFeatureID(const string& variable_name) const {
    for (int i = 0; i < feature_names.size(); ++i) {
        if (feature_names[i] == variable_name) {
            return i;
        }
    }
    throw runtime_error("No feature with name " + variable_name);
}

vector<double> Data::getValues(vector<size_t> subset_indices, size_t feature) {
    vector<double> unfiltered_values = get_x_col(feature);
    vector<double> result(subset_indices.size());
    for (int i = 0; i < subset_indices.size(); ++i) {
        result[i] = unfiltered_values[subset_indices[i]];
    }
    return(result);
}
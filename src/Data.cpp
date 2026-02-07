#include <set>

#include "Data.h"

// Data constructor for Regression, Classification and Survival
//--------------------------------------------------------------------------------------

/*
  it is assumed that the data has been encoded to be numeric
  response_indices are the indices for the response column(s), while categorical
  is a vector of bools indicating whether a variable (response or feature) is categorical
  unique is a vector of the number of unique values for each column
*/
Data::Data(DataFrame data, const vector<size_t>& response_indices, vector<size_t> feature_indices,
           const vector<bool>& categorical, const vector<size_t>& unique) {
    this->num_obs = data.nrows();
    this->num_features = feature_indices.size();
    this->response_indices = response_indices;
    this->num_responses = response_indices.size();
    sort(feature_indices.begin(), feature_indices.end());
    this->feature_indices = feature_indices;

    x.assign(num_features * num_obs, 0);

    // fill the response "matrix" if response variables are supplied
    if (!response_indices.empty()) {
        // classification or regression
        if (response_indices.size() == 1) {
            vector<double> y_res = as<vector<double>>(data[response_indices[0]]);
            this->y = y_res;
        }
        // survival
        else if (response_indices.size() == 2) {
            y.assign(2*num_obs, 0);
            vector<double> times = as<vector<double>>(data[response_indices[0]]);
            vector<double> ind = as<vector<double>>(data[response_indices[1]]);
            for (int i = 0; i < num_obs; ++i) {
                y[i] = times[i];
                y[num_obs + i] = ind[i];
            }
        }
        // multi-state
        else {
            // TODO
        }

        // update response names
        vector<string> response_names(response_indices.size());
        for (size_t i = 0; i < response_indices.size(); ++i) {
            response_names[i] = as<vector<string>>(data.names())[response_indices[i]];
        }
    
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
    }

    // fill the feature "matrix"
    for (size_t j = 0; j < num_features; ++j) {
        NumericVector col = data[feature_indices[j]];
        for (size_t i = 0; i < num_obs; ++i) {
            x[i * num_features + j] = col[i];
        }
    }
    /*
    for (size_t i = 0; i < num_obs; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            NumericVector col = data[feature_indices[j]];
            x[i * num_features + j] = col[i];
        }
    }
    */

    this->categorical = categorical_features;
    this->unique_values = unique_values_features;
    this->feature_names = feature_names;
}

vector<double> Data::getValues(const vector<size_t>& subset_indices, size_t feature) {
    vector<double> unfiltered_values = get_x_col(feature);
    vector<double> result(subset_indices.size());
    for (size_t i = 0; i < subset_indices.size(); ++i) {
        result[i] = unfiltered_values[subset_indices[i]];
    }
    return(result);
}

size_t Data::getFeatureID(const string& variable_name) const {
    for (size_t i = 0; i < feature_names.size(); ++i) {
        if (feature_names[i] == variable_name) {
            return i;
        }
    }
    throw runtime_error("No feature with name " + variable_name);
}

// Data constructor for Multi-state models (MM)
//--------------------------------------------------------------------------------------

/*
  jump_data contains the actual jump process data as a list of lists of times and states
  it is assumed that the data for the features has been encoded to be numeric
  categorical is a vector of bools indicating whether a variable (response or feature)
  is categorical, unique is a vector of the number of unique values for each column
*/
Data::Data(List jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame feature_data,
    vector<size_t> feature_indices, const vector<bool>& categorical, const vector<size_t>& unique) {
    this->num_obs = feature_data.nrows();
    this->num_features = feature_indices.size();
    this->max_response_length = max_response_length;
    this->num_states = num_states;
    this->feature_indices = feature_indices;
    
    // fill the response vectors if response variables are supplied
    if (jump_data.isNULL() || jump_data.size() != 0) {
        times.assign(num_obs * max_response_length, 0);
        states.assign(num_obs * max_response_length, 0);
        censoring_times.assign(num_obs, 0);
        censoring_states.assign(num_obs, 0);
        for (int i = 0; i < num_obs; ++i) {
            List obs = as<List>(jump_data[i]);
            vector<double> obs_times = as<vector<double>>(obs[0]);
            vector<uint8_t> obs_states = as<vector<uint8_t>>(obs[1]);
            size_t response_length = obs_times.size();
            for (int j = 0; j < response_length; ++j) {
                times[i * max_response_length + j] = obs_times[j];
                states[i * max_response_length + j] = obs_states[j];
            }
            // save censoring times and the corresponding state separately (eases calculations)
            if (response_length >= 2) {
                if (static_cast<uint8_t>(INTEGER(obs[1])[response_length - 1]) == static_cast<uint8_t>(INTEGER(obs[1])[response_length - 2])) {
                    censoring_times[i] = obs_times[response_length - 1];
                    censoring_states[i] = static_cast<uint8_t>(obs_states[response_length - 1]); // static_cast<uint8_t>(obs_states[response_length - 1]);
                }
            }
        }
        // we should not need response names for multi-states
    }
    // determine possible jumps
    set<pair<uint8_t, uint8_t>> possible_jumps;
    for (size_t i = 0; i < num_obs; ++i) {
        size_t index = i * max_response_length;
        for (size_t j = 0; j < max_response_length - 1; ++j) {
            uint8_t state = states[index + j];
            uint8_t next_state = states[index + j + 1];
            if (state != 0 && next_state != 0 && state != next_state) {
                possible_jumps.emplace(state, next_state);
            }
        }
    }
    valid_jumps.reserve(possible_jumps.size());
    copy(possible_jumps.begin(), possible_jumps.end(), back_inserter(valid_jumps));

    // prepare features
    x.assign(num_features * num_obs, 0);
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
        
        // update names of features
        feature_names[i] = as<vector<string>>(feature_data.names())[feature_indices[i]];

        // update the number of unique values
        unique_values_features[i] = unique[feature_indices[i]];
    }

    // fill the feature "matrix"
    for (size_t j = 0; j < num_features; ++j) {
        NumericVector col = feature_data[feature_indices[j]];
        for (size_t i = 0; i < num_obs; ++i) {
            x[i * num_features + j] = col[i];
        }
    }
    this->categorical = categorical_features;
    this->unique_values = unique_values_features;
    this->feature_names = feature_names;
}
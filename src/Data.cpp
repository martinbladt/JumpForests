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
    this->num_classes = 0;
    sort(feature_indices.begin(), feature_indices.end());
    this->feature_indices = feature_indices;
    CharacterVector column_names = data.names();

    x.assign(num_features * num_obs, 0);

    // fill the response "matrix" if response variables are supplied
    if (!response_indices.empty()) {
        // classification or regression
        if (response_indices.size() == 1) {
            vector<double> y_res = as<vector<double>>(data[response_indices[0]]);
            this->y = std::move(y_res);

            // only relevant for classification
            if (categorical[response_indices[0]]) {
                num_classes = unique[response_indices[0]];
            }
        }
        // survival
        else if (response_indices.size() == 2) {
            y.assign(2*num_obs, 0);
            vector<double> times = as<vector<double>>(data[response_indices[0]]);
            vector<double> ind = as<vector<double>>(data[response_indices[1]]);
            for (size_t i = 0; i < num_obs; ++i) {
                y[i] = times[i];
                y[num_obs + i] = ind[i];
            }
        }

        // update response names
        vector<string> response_names(response_indices.size());
        for (size_t i = 0; i < response_indices.size(); ++i) {
            response_names[i] = as<string>(column_names[response_indices[i]]);
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
        feature_names[i] = as<string>(column_names[feature_indices[i]]);

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
    this->categorical = categorical_features;
    this->unique_values = unique_values_features;
    this->feature_names = feature_names;
}

vector<double> Data::getValues(const vector<size_t>& subset_indices, size_t feature) const {
    vector<double> result(subset_indices.size());
    for (size_t i = 0; i < subset_indices.size(); ++i) {
        result[i] = get_x(subset_indices[i], feature);
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
    CharacterVector column_names = feature_data.names();
    
    // fill the response vectors if response variables are supplied
    if (!jump_data.isNULL() || jump_data.size() != 0) {
        times.assign(num_obs * max_response_length, 0);
        states.assign(num_obs * max_response_length, 0);
        last_observed_times.assign(num_obs, 0);
        censoring_states.assign(num_obs, 0);
        for (size_t i = 0; i < num_obs; ++i) {
            List obs = as<List>(jump_data[i]);
            vector<double> obs_times = as<vector<double>>(obs[0]);
            vector<uint8_t> obs_states = as<vector<uint8_t>>(obs[1]);
            size_t response_length = obs_times.size();
            for (size_t j = 0; j < response_length; ++j) {
                times[i * max_response_length + j] = obs_times[j];
                states[i * max_response_length + j] = obs_states[j];
            }

            // save last observed times and the state at the censoring time separately (eases calculations)
            if (response_length >= 2) {
                last_observed_times[i] = i * max_response_length + response_length - 1;
                if (static_cast<uint8_t>(INTEGER(obs[1])[response_length - 1]) == static_cast<uint8_t>(INTEGER(obs[1])[response_length - 2])) {
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
        feature_names[i] = as<string>(column_names[feature_indices[i]]);

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

/*

Computes a vector of bools indicating whether observation i at unique event time t is observed to be in state j
Used for computing the Brier and KL scores for multi-state trees and forests only

*/

vector<bool> Data::computeStateIndicators(const vector<size_t>& response_event_time_ids, const vector<double>& unique_event_times) {
    // initialise vectors of number of jumps and at risk
    size_t dim = num_states * num_states;
    size_t num_unique_event_times = unique_event_times.size();
    vector<bool> result(num_obs * num_unique_event_times * num_states);
    vector<size_t> jumps(num_unique_event_times * dim);
    vector<size_t> at_risk(num_unique_event_times * num_states);
    vector<size_t> jumps_acc(num_unique_event_times * dim);
    vector<size_t> censoring_contribution(num_unique_event_times * num_states);

    // all quantities are computed at the observation level
    for (size_t i = 0; i < num_obs; ++i) {
        // reassign all temporary quantities
        jumps.assign(num_unique_event_times * dim, 0);
        at_risk.assign(num_unique_event_times * num_states, 0);
        censoring_contribution.assign(num_unique_event_times * num_states, 0);
    
        // first compute starting state for current observation
        ++at_risk[states[i * max_response_length] - 1];

        // compute censoring contribution for current observation
        uint8_t censoring_state = censoring_states[i];
        if (censoring_state != 0) {
            size_t censoring_time_id = response_event_time_ids[last_observed_times[i]];
            for (size_t s = censoring_time_id; s < num_unique_event_times; ++s) {
                ++censoring_contribution[s * num_states + censoring_state - 1];
            }
        }
        size_t j = 1;
        size_t index = i * max_response_length + 1;
        // a state is 0 if and only if it is not valid e.g. a dead entry in the flattened array of observations
        while (j < max_response_length && states[index] != 0) {
            size_t id = response_event_time_ids[index];
            int current_state_index = states[index] - 1;    // states are always indexed by 1, 2, ... with 0 reserved for 'dead' entries in the flattened array
            int prev_state_index = states[index - 1] - 1;
            if (current_state_index != prev_state_index) {
                ++jumps[id * dim + prev_state_index * num_states + current_state_index];
            }
            ++j;
            ++index;
        }

        // no delay means that we compute in t and not t-
        cumulativeMatrixSumsNoDelay(jumps_acc, jumps, num_states);

        // now compute number at risk via the key decomposition
        for (size_t t = 1; t < num_unique_event_times; ++t) {   // already computed for t = 0 above
            vector<int> jump_contributions = columnSums(subtractMatrices(jumps_acc, t * dim, (t + 1) * dim - 1, transpose(jumps_acc, t * dim, (t + 1) * dim - 1)), static_cast<size_t>(num_states));
            for (size_t j = 0; j < num_states; ++j) {
                at_risk[t * num_states + j] = at_risk[j] - censoring_contribution[t * num_states + j] + jump_contributions[j];
            }
        }
        // now update the bool vector result
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            for (size_t j = 0; j < num_states; ++j) {
                size_t at_risk_index = t * num_states + j;
                result[i * num_unique_event_times * num_states + at_risk_index] = (bool) at_risk[at_risk_index];
            }
        }
    }
    return result;
}

/*

Functions for multi-state trees

*/

#include "TreeMultistate.h"


// constructor for MultistateTree
//--------------------------------------------------------------------------------------

// NB: we need to know the number of states to initialise the at risk and jumps vectors properly

// the final argument is only used for honest trees
MultistateTree::MultistateTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids, const vector<size_t>& subset_indices, const vector<size_t>& estimation_indices) : 
    unique_event_times {unique_event_times}, response_time_event_ids {response_time_event_ids} {
        this->node_obs.push_back(subset_indices);
        this->holdout_node_obs.push_back(estimation_indices);
        this->num_unique_event_times = unique_event_times->size();
        this->node_sizes.push_back(subset_indices.size());

        // initialise vector of jumps and individuals at risk
        uint8_t num_states = data->getNumberOfStates();
        this->num_jumps.resize(num_unique_event_times * num_states * num_states);
        this->num_at_risk.resize(num_unique_event_times * num_states);
}

// functions for growing multi-state trees
//--------------------------------------------------------------------------------------

void MultistateTree::computeMultistateQuantities(const vector<size_t>& indices, vector<size_t>& at_risk, vector<size_t>& jumps) {
    size_t n = indices.size();
    // fetch states and other relevant data quantities
    const vector<uint8_t>& states = data->getStates();
    const vector<double>& censoring_times = data->getCensoringTimes();
    const vector<uint8_t>& censoring_states = data->getCensoringStates();

    uint8_t max_response_length = data->getMaxResponseLength();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    
    num_jumps.assign(num_unique_event_times * dim, 0);
    num_at_risk.assign(num_unique_event_times * num_states, 0);
    vector<size_t> censoring_contribution(num_unique_event_times * num_states, 0);

    // compute number at risk at initiation I0 and the censoring contribution C
    for (size_t i : indices) {
        ++num_at_risk[states[i * max_response_length] - 1];

        // compute the censoring contribution C
        uint8_t censoring_state = censoring_states[i];
        if (censoring_state != 0) {
            double R = censoring_times[i];
            for (size_t j = 0; j < num_unique_event_times; ++j) {
                if (unique_event_times[j] > R) { 
                    // all following event times also satisfy > R
                    for (size_t k = j; k < num_unique_event_times; ++k) {
                        ++censoring_contribution[k * num_unique_event_times + censoring_state - 1];
                    }
                    break;
                }
            }
        }
    }

    // compute number of jumps
    
}

// old implementation which doesn't handle at risk calculations correctly (may also be problems with computing the jumps)
/*

void MultistateTree::computeMultistateQuantities(const vector<size_t>& indices, vector<size_t>& at_risk, vector<size_t>& jumps) {
    size_t n = indices.size();
    // fetch states and other relevant data quantities
    const vector<uint8_t>& states = data->getStates();
    uint8_t max_response_length = data->getMaxResponseLength();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    
    num_jumps.assign(num_unique_event_times * dim, 0);
    num_at_risk.assign(num_unique_event_times * num_states, 0);

    for (size_t i : indices) {
        size_t j = 0;
        size_t index = i * max_response_length;
        // a state is 0 if and only if it is not valid e.g. a dead entry in the flattened array of observations
        while (j < max_response_length && states[index] != 0) {
            size_t id = (*response_time_event_ids)[index];
            int current_state_index = states[index] - 1;     // states are always indexed by 1, 2, ... with 0 reserved for 'dead' entries in the flattened array
            int next_state_index = states[index + 1] - 1;
            // find jumps (we assume that max_response_length > 1 i.e. at least one jump occurs in the dataset)
            if (j < max_response_length - 1 && next_state_index != -1 && current_state_index != next_state_index) {
                ++num_jumps[id * dim + current_state_index * num_states + i] - next_state_index];
            }
            // find numbers at risk
            // j == 0 means that we are at the first state of an observation/path
            if (j == 0 || current_state_index != states[index - 1] - 1) {
                ++num_at_risk[id * num_states + current_state_index];
            }
            ++j;
            index = i * max_response_length + j;
        }
    }
}

*/

/*

Gemini's attempt. I do not trust this implementation. It doesn't even handle the j == 0 case

void MultistateTree::computeMultistateQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                         vector<size_t>& num_at_risk_left, vector<size_t>& num_jumps_left, size_t nsplits_final) {
    const vector<size_t>& states = data->getStates();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    uint8_t max_response_length = data->getMaxResponseLength();
    //vector<size_t> delta_num_at_risk_right(nsplits_final * num_unique_event_times * num_states);

    // initialise the number of jumps and at risk to be ones for the parent
    for (size_t s = 0; s < nsplits_final; ++s) {
        copy(num_at_risk.begin(), num_at_risk.end(), num_at_risk_left.begin() + s * num_unique_event_times * num_states);
        copy(num_jumps.begin(), num_jumps.end(), num_jumps_left.begin() + s * num_unique_event_times * dim);
    }

    // sort indices of observations by feature value
    vector<size_t> sorted_indices = node_obs[node_index];
    sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t a, size_t b) {
        return data->get_x(a, feature) < data->get_x(b, feature);
    });

    size_t current_obs_idx = 0;
    for (size_t s = 0; s < nsplits_final; ++s) {
        // carry over changes from the previous split point
        if (s > 0) {
            copy(num_at_risk_left.begin() + (s - 1) * num_unique_event_times * num_states,
                      num_at_risk_left.begin() + s * num_unique_event_times * num_states,
                      num_at_risk_left.begin() + s * num_unique_event_times * num_states);
            copy(num_jumps_left.begin() + (s - 1) * num_unique_event_times * dim,
                      num_jumps_left.begin() + s * num_unique_event_times * dim,
                      num_jumps_left.begin() + s * num_unique_event_times * dim);
        }

        // subtract observations that are to the right of split_points[s]
        while (current_obs_idx < sorted_indices.size() && data->get_x(sorted_indices[current_obs_idx], feature) > split_points[s]) {
            size_t idx = i = sorted_indices[current_obs_idx];
            ++num_obs_right[idx];

            for (size_t j = 0; j < max_response_length; ++j) {
                size_t idx = i * max_response_length + j;
                if (states[idx] == 0) break;    // end of path

                size_t id = (*response_time_event_ids)[idx];
                size_t curr_s = states[idx] - 1;

                // handle jumps
                if (j + 1 < max_response_length && states[idx + 1] != 0 && states[idx + 1] != states[idx]) {
                    size_t next_state = states[idx + 1] - 1;
                    size_t jump_idx = s * num_times * dim + id * dim + curr_s * num_states + next_state;
                    --num_jumps_left[jump_idx];
                }
                // handle at risk
                size_t risk_idx = s * num_times * num_states + id * num_states + curr_s;
                --num_at_risk_left[risk_idx];
            }
            ++current_obs_idx;
        }
    }
}

*/

// version of jan26
// for computing multi-state quantities (number at risk and number of jumps) for all splits in a node (for splits on continuous features)
void MultistateTree::computeMultistateQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                         vector<size_t>& num_at_risk_left, vector<size_t>& num_jumps_left, size_t nsplits_final) {
    const vector<uint8_t>& states = data->getStates();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    uint8_t max_response_length = data->getMaxResponseLength();
    //vector<size_t> delta_num_at_risk_right(nsplits_final * num_unique_event_times * num_states);
    
    // initialise the number of jumps and at risk to be ones for the parent
    for (size_t s = 0; s < nsplits_final; ++s) {
        copy(num_at_risk.begin(), num_at_risk.end(), num_at_risk_left.begin() + s * num_unique_event_times * num_states);
        copy(num_jumps.begin(), num_jumps.end(), num_jumps_left.begin() + s * num_unique_event_times * dim);
    }
    
    for (size_t i : node_obs[node_index]) {
        double feature_val = data->get_x(i, feature);
        for (size_t s = 0; s < nsplits_final; ++s) {
            if (feature_val > split_points[s]) {
                ++num_obs_right[s];
                // now compute the differences in numbers at risk and count number of jumps
                size_t j = 0;
                size_t index = i * max_response_length;
                while (j < max_response_length && states[index] != 0) {
                    size_t id = (*response_time_event_ids)[index];
                    int current_state_index = states[index] - 1;     // states are always indexed by 1, 2, ... with 0 reserved for 'dead' entries in the flattened array
                    int next_state_index = states[index + 1] - 1;
                    // find jumps (we assume that max_response_length > 1 i.e. at least one jump occurs in the dataset)
                    if (j == 0 && next_state_index != -1 || (j < max_response_length - 1 && next_state_index != -1 && current_state_index != next_state_index)) {
                        --num_jumps_left[s * num_unique_event_times * dim + id * dim + current_state_index * num_states + next_state_index];
                    }
                    // find numbers at risk (j == 0 means we are at the first state of an observation/path)
                    if (j == 0 || current_state_index != states[index - 1] - 1) {
                        --num_at_risk_left[s * num_unique_event_times * num_states + id * num_states + current_state_index];
                    }
                    ++j;
                    index = i * max_response_length + j;
                }
            } else {
                break;  // since the split_points are sorted
            }
        }
    }
}

void MultistateTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold) {
    const vector<size_t>& current_node_obs = node_obs[node_index];
    size_t num_states = data->getNumberOfStates();

    // samples split points
    vector<double> split_points;
    size_t nsplits_final = sampleSplitPoints(split_points, current_node_obs, feature);

    // no possible splits
    if (nsplits_final == 0) {
        return;
    }

    // initialise node info for the right daughter as flattened 2D arrays
    //vector<size_t> num_jumps_right(nsplits_final * num_unique_event_times * num_states * num_states);
    vector<size_t> num_obs_right(nsplits_final);
    //vector<size_t> num_at_risk_right(nsplits_final * num_unique_event_times * num_states);

    vector<size_t> num_jumps_left(nsplits_final * num_unique_event_times * num_states * num_states);
    vector<size_t> num_at_risk_left(nsplits_final * num_unique_event_times * num_states);

    computeMultistateQuantitiesDaughter(node_index, feature, split_points, num_obs_right, num_at_risk_left, num_jumps_left, nsplits_final);

    // now determine the best split
    for (size_t i = 0; i < nsplits_final; ++i) {
        
        // if a node is too small, skip the split
        size_t num_obs_left = current_node_obs.size() - num_obs_right[i];
        if (num_obs_left < min_node_size || num_obs_right[i] < min_node_size) {
            continue;
        }

        double split_val;
        // CHOOSE SPLITRULE
        /*
        if (splitrule == "NAME") {
            split_value = splitRuleFunction(num_jumps, num_at_risk, num_jumps_right, num_at_risk_right, i);
        }
        */

        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_feature = feature;
            // use average of split points unless it is the final split value
            if (i == nsplits_final - 1) {
                best_threshold = {split_points[i]};
            } else {
                best_threshold = {(split_points[i] + split_points[i + 1])/2.0};
            }
        }
    }
}

void MultistateTree::bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices) {
    const vector<double>& feature_values = uniqueValues(data->getValues(node_obs[node_index], feature));
    size_t num_feature_values = feature_values.size();

    unordered_set<uint64_t> partition_masks;
    // generate partitions (breaks if no possible splits)
    if (generateCategoricalPartitions(feature_values, partition_masks)) {
        return;
    }

    // consider each partition (bitmask)
    for (const auto& mask : partition_masks) {
        unordered_set<double> left_values;
        for (size_t i = 0; i < num_feature_values; ++i) {
            if ((mask >> i) & 1) {
                left_values.insert(feature_values[i]);
            }
        }

        vector<size_t> current_left_indices;
        vector<size_t> current_right_indices;
        for (size_t obs_id : node_obs[node_index]) {
            if (left_values.count(data->get_x(obs_id, feature))) {
                current_left_indices.push_back(obs_id);
            } else {
                current_right_indices.push_back(obs_id);
            }
        }

        if (current_left_indices.size() < min_node_size || current_right_indices.size() < min_node_size) {
            continue;
        }

        // here we have to compute the survival info in one of the daughters from scratch
        vector<size_t> num_jumps_left, num_at_risk_left;
        computeMultistateQuantities(current_left_indices, num_jumps_left, num_at_risk_left);
        double split_val;
        // CHOOSE SPLITRULE
        /*
        if (splitrule == "NAME") {
            split_value = splitRuleFunction(num_jumps, num_at_risk, num_jumps_right, num_at_risk_right, i);
        }
        */

        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_left_indices = move(current_left_indices);
            best_right_indices = move(current_right_indices);
            best_feature = feature;
            best_threshold.assign(left_values.begin(), left_values.end());
        }
    }
}

void MultistateTree::makeLeaf(size_t node_index) {
    // compute the matrices of Nelson--Aalen estimators
    computeNA(node_index);

    // update tree info
    feature_IDs.push_back(0);
    thresholds.push_back({});

    // since we are in a terminal node, we save the indices for the observations
    for (size_t i : node_obs[node_index]) {
        prediction_node_IDs[i] = node_index;
    }
}

// function to create a split for a multi-state tree. returns true if leaf, otherwise false
bool MultistateTree::createSplit(size_t node_index) {
    const vector<size_t>& current_node_obs = node_obs[node_index];

    // if no split is possible, make the node a leaf
    if (current_node_obs.size() < 2 * min_node_size) {
        if (!honest) {
            computeMultistateQuantities(current_node_obs, num_jumps, num_at_risk);               // for dishonest trees, use the growing indices
        } else {
            computeMultistateQuantities(holdout_node_obs[node_index], num_jumps, num_at_risk);   // for honest trees, use the holdout set for computing the CHF
        }
        makeLeaf(node_index);
        return true;
    }

    computeMultistateQuantities(current_node_obs, num_jumps, num_at_risk);   // update parent multi-state info

    double best_split_val = -1.0;
    size_t best_feature = 0;
    vector<double> best_threshold;
    vector<size_t> best_left_indices;
    vector<size_t> best_right_indices;
    
    // only used for honesty
    vector<size_t> holdout_left_indices;
    vector<size_t> holdout_right_indices;

    // sample mtry features
    size_t num_features = data->getNumberOfFeatures();
    vector<size_t> feature_indices(num_features);
    for (size_t i = 0; i < num_features; ++i) {
        feature_indices[i] = i;
    }
    vector<size_t> sampled_features = sampleIndices(feature_indices, mtry, false, random_number_generator);

    // now consider each of the sampled features
    for (size_t i : sampled_features) {
        if (data->getCategorical()[i]) {
            // finds the best split and constructs the indices of the best left and right node
            bestSplitCategorical(node_index, i, best_split_val, best_feature, best_threshold, best_left_indices, best_right_indices);
        }
        else {
            // does not return the best indices, so this has to be done later
            bestSplitContinuous(node_index, i, best_split_val, best_feature, best_threshold);
        }
    }

    // if no best split is found, make the node a leaf
    if (best_split_val < 0) {
        if (honest) {
            computeMultistateQuantities(holdout_node_obs[node_index], num_jumps, num_at_risk);   // for honest trees, use the holdout set for computing the CHF
        }
        // otherwise, use parent survival info already computed earlier
        makeLeaf(node_index);
        return true;
    }

    // for a categorical feature, the best indices are already saved, but if the feature is 
    // continuous, they should be recomputed from scratch (and only once)
    if (!(data->getCategorical()[best_feature])) {
        best_left_indices.clear();
        best_right_indices.clear();
        for (size_t i : current_node_obs) {
            if (data->get_x(i, best_feature) <= best_threshold[0]) {
                best_left_indices.push_back(i);
            } else {
                best_right_indices.push_back(i);
            }
        }
        // update the holdout index sets if the tree is honest
        if (honest) {
            const vector<size_t>& current_holdout_node_obs = holdout_node_obs[node_index];
            for (size_t i : current_holdout_node_obs) {
                if (data->get_x(i, best_feature) <= best_threshold[0]) {
                    holdout_left_indices.push_back(i);
                } else {
                    holdout_right_indices.push_back(i);
                }
            }
        }
    }

    // the best holdout index sets also need to be constructed if the split is categorical
    if (honest && data->getCategorical()[best_feature]) {
        const vector<size_t>& current_holdout_node_obs = holdout_node_obs[node_index];
        for (size_t i : current_holdout_node_obs) {
            if (find(best_threshold.begin(), best_threshold.end(), data->get_x(i, best_feature)) != best_threshold.end()) {
                holdout_left_indices.push_back(i);
            } else {
                holdout_right_indices.push_back(i);
            }
        }
    }
    
    // a best split was found, update the tree
    node_obs.push_back(best_left_indices);          // construct left daughter
    node_obs.push_back(best_right_indices);         // construct right daughter
    node_sizes.push_back(best_left_indices.size());
    node_sizes.push_back(best_right_indices.size());
    feature_IDs.push_back(best_feature);
    thresholds.push_back(best_threshold);
    na.push_back(vector<double>());

    // for honest trees, update the holdout indices
    if (honest) {
        holdout_node_obs.push_back(holdout_left_indices);
        holdout_node_obs.push_back(holdout_right_indices);
    }

    return false;
}

// computes the matrix of Nelson--Aalen estimators in node node_index
void MultistateTree::computeNA(size_t node_index) {
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    vector<double> na(num_unique_event_times * dim);
    
    // need to handle the first event time (always zero for multi-states) separately
    for (size_t j = 0; j < num_states; ++j) {
        for (size_t k = 0; k < num_states; ++k) {
            if (num_at_risk[j] != 0) {
                na[j * num_states + k] = double(num_jumps[j * num_states + k]) / double(num_at_risk[j]);
            }
        }
    }

    // for each unique event time i, loop over all entries (j, k) in the matrix
    for (size_t i = 1; i < num_unique_event_times; ++i) {
        for (size_t j = 0; j < num_states; ++j) {
            double diag = 0;
            for (size_t k = 0; k < num_states; ++k) {
                if (num_at_risk[i * num_states + j] != 0) {
                    na[i * dim + j * num_states + k] = na[(i - 1) * dim + j * num_states + k] + double(num_jumps[i * dim + j * num_states + k]) / double(num_at_risk[i * num_states + j]);
                } else {
                    na[i * dim + j * num_states + k] = na[(i - 1) * dim + j * num_states + k];
                }
                // the diagonal is minus the sum of all other row entries
                diag -= na[i * dim + j * num_states + k];
            }
            na[i * dim + j * num_states + j] = diag;
        }
    }
    this->na.push_back(move(na));
}

// splitting rules for multi-state trees
//--------------------------------------------------------------------------------------



// prediction for multi-state trees
//--------------------------------------------------------------------------------------

vector<double> MultistateTree::computePredictions(const Data& new_data) {
    uint8_t num_states = data->getNumberOfStates();
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times * num_states * num_states);
    for (size_t i = 0; i < num_obs; ++i) {
        const vector<double>& pred = get<vector<double>>(predict(new_data.get_x_row(i)));
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            for (size_t j = 0; j < num_states; ++j) {
                for (size_t k = 0; k < num_states; ++k) {
                    predictions[i * num_obs + t * num_unique_event_times + j * num_states + k] = pred[t * num_unique_event_times + j * num_states + k];
                }
            }
        }
    }
    return predictions;
}

// error estimation for multi-state trees
//--------------------------------------------------------------------------------------



// miscellaneous functions related to multi-states
//--------------------------------------------------------------------------------------

// for computing the ids in the observed times corresponding to the unique event times (including censored times)
vector<size_t> computeResponseEventTimeIDsMultistate(const vector<double>& unique_event_times, const vector<double>& times, const vector<uint8_t>& states) {
    vector<size_t> response_event_time_ids;
    size_t n = times.size();
    response_event_time_ids.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        // only difference to survival: there will be many zeroes since we flatten the times vector with the largest number of total jumps
        // the states clause means that we exclude censoring times
        if (times[i] == 0 || states[i] == states[i - 1]) {
            response_event_time_ids.push_back(0);
            continue;
        }

        // use binary search to find lower bound
        auto it = lower_bound(unique_event_times.begin(), unique_event_times.end(), times[i]);
        response_event_time_ids.push_back(static_cast<size_t>(distance(unique_event_times.begin(), it)));
    }
    return(response_event_time_ids);
}

vector<double> AalenJohansen(const vector<double>& na, uint8_t num_states) {
    vector<double> aj = vector<double>(na.size(), 0);
    size_t dim = num_states * num_states;                       // number of entries in each matrix
    size_t num_jumps = na.size() / dim;   // num_unique_event_times
    // initial value is the identity matrix
    for (size_t j = 0; j < num_states; ++j) {
        aj[j * num_states + j] = 1;
    }

    // compute the Aalen--Johansen estimator at each jump time using an optimised matrix multiplication scheme
    for (size_t i = 1; i < num_jumps; ++i) {
        for (size_t j = 0; j < num_states; ++j) {           // row of the aj matrix
            for (size_t k = 0; k < num_states; ++k) {       // column of the aj matrix
                double prev = aj[(i - 1) * dim + j * num_states + k];
                for (size_t l = 0; l < num_states; ++l) {   // column of matrix in increment
                    double contribution = na[i * dim + k * num_states + l] - na[(i - 1) * dim + k * num_states + l];
                    if (k == l) {
                        ++contribution;    // add 1 to diagonal elements
                    }
                    aj[i * dim + j * num_states + l] += prev * contribution;
                }
            }
        }
    }
    return aj;
}
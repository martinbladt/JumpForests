/*

Functions for multi-state trees

*/

#include "TreeMultistate.h"


// constructor for MultistateTree
//--------------------------------------------------------------------------------------

// the final argument is only used for honest trees
MultistateTree::MultistateTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids, const vector<size_t>& subset_indices, uint8_t num_states, const vector<size_t>& estimation_indices) : 
    unique_event_times {unique_event_times}, response_event_time_ids {response_event_time_ids} {
    this->node_obs.push_back(subset_indices);
    this->holdout_node_obs.push_back(estimation_indices);
    this->num_unique_event_times = unique_event_times->size();
    this->node_sizes.push_back(subset_indices.size());
    this->num_jumps.resize(num_unique_event_times * num_states * num_states);
    this->num_at_risk.resize(num_unique_event_times * num_states);
}

// functions for growing multi-state trees
//--------------------------------------------------------------------------------------

void MultistateTree::computeMultistateQuantities(const vector<size_t>& indices, vector<size_t>& at_risk, vector<size_t>& jumps) {
    // fetch states and other relevant data quantities
    const vector<uint8_t>& states = data->getStates();
    const vector<double>& censoring_times = data->getCensoringTimes();
    const vector<uint8_t>& censoring_states = data->getCensoringStates();

    uint8_t max_response_length = data->getMaxResponseLength();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    
    // initialise vectors of number of jumps and at risk
    num_jumps.assign(num_unique_event_times * dim, 0);
    num_at_risk.assign(num_unique_event_times * num_states, 0);
    // all temporary quantities used for the key decomposition
    vector<size_t> censoring_contribution(num_unique_event_times * num_states, 0);
    vector<size_t> num_jumps_acc(num_unique_event_times * dim, 0);

    //Rcout << "Finished initialising before computing quantities" << endl;

    // compute the initial rates, the censoring contribution C and the number of jumps across all event times and observations
    for (size_t i : indices) {
        //Rcout << "Index " << i << endl;
        // initial rates I0
        ++num_at_risk[states[i * max_response_length] - 1];
        //Rcout << "Updated I0" << endl;

        // compute the censoring contribution C
        uint8_t censoring_state = censoring_states[i];
        if (censoring_state != 0) {     // censoring actually occurs
            double R = censoring_times[i];
            for (size_t j = 0; j < num_unique_event_times; ++j) {
                if ((*unique_event_times)[j] > R) {
                    // all following event times also satisfy > R
                    for (size_t k = j; k < num_unique_event_times; ++k) {
                        ++censoring_contribution[k * num_states + censoring_state - 1];
                    }
                    break;
                }
            }
        }
        //Rcout << "Computed censoring contribution:" << endl;
        //printVector(censoring_contribution);
        // compute number of jumps (we assume that at least one event of some kind occurs so that max_response_length > 1)
        size_t j = 1;
        size_t index = i * max_response_length + 1;
        //Rcout << "j = " << j << ", index = " << index << endl;
        //Rcout << "states[index] = " << static_cast<size_t>(states[index]) << endl;
        //Rcout << "Number of elements in response_event_time_ids: " << response_event_time_ids->size() << endl;
        //Rcout << "Number of unique event times: " << unique_event_times->size() << endl;
        // a state is 0 if and only if it is not valid e.g. a dead entry in the flattened array of observations
        while (j < max_response_length && states[index] != 0) {
            size_t id = (*response_event_time_ids)[index];
            int current_state_index = states[index] - 1;     // states are always indexed by 1, 2, ... with 0 reserved for 'dead' entries in the flattened array
            int prev_state_index = states[index - 1] - 1;
            if (current_state_index != prev_state_index) {
                ++num_jumps[id * dim + prev_state_index * num_states + current_state_index];
                //num_jumps_acc[id * dim + prev_state_index * num_states + current_state_index] = num_jumps_acc[(id - 1) * dim + prev_state_index * num_states + current_state_index] + 1;
            } 
            /*
            else {
                num_jumps_acc[id * dim + prev_state_index * num_states + current_state_index] = num_jumps_acc[(id - 1) * dim + prev_state_index * num_states + current_state_index];
            }
            */
            ++j;
            ++index;
        }
        //Rcout << "Finished computing jumps for index " << i << endl;
    }
    // compute the cumulative number of jumps
    cumulativeMatrixSums(num_jumps_acc, num_jumps, num_states);

    // now compute number at risk via the key decomposition
    
    for (size_t j = 1; j < num_unique_event_times; ++j) {   // j = 1 since we already computed I0 above
        //vector<int> subtraction = subtractMatrices(num_jumps_acc, j * dim, (j + 1) * dim - 1, transpose(num_jumps_acc, j * dim, (j + 1) * dim - 1));
        vector<int> jump_contributions = columnSums(subtractMatrices(num_jumps_acc, j * dim, (j + 1) * dim - 1, transpose(num_jumps_acc, j * dim, (j + 1) * dim - 1)), static_cast<size_t>(num_states));
        for (size_t k = 0; k < num_states; ++k) {
            // key decomposition
            // for debugging thinning
            //if ((int) num_at_risk[k] - (int) censoring_contribution[j * num_states + k] + (int) jump_contributions[k] < 0) {
            //    cout << "Warning: Key decomposition negative, causing underflow in num_at_risk" << endl;
            //}
            num_at_risk[j * num_states + k] = num_at_risk[k] - censoring_contribution[j * num_states + k] + jump_contributions[k];
        }
    }
    
    // only for debugging
    /*
    Rcout << "Number of jumps: " << endl;
    printVector(num_jumps, dim);
    Rcout << "Numbers at risk: " << endl;
    printVector(num_at_risk, num_states);
    Rcout << "Computed num_jumps_acc:" << endl;
    printVector(num_jumps_acc, dim);
    Rcout << "Censoring contribution:" << endl;
    printVector(censoring_contribution, num_states);
    */
}

// version of feb10 (makes pre-sweep and checks for invalid splits before computing quantities)
// for computing multi-state quantities (number at risk and number of jumps) for all splits in a node (for splits on continuous features)
void MultistateTree::computeMultistateQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                         vector<size_t>& num_at_risk_right, vector<size_t>& num_jumps_right, size_t nsplits_final) {
    // fetch states and other relevant data quantities
    const vector<uint8_t>& states = data->getStates();
    const vector<double>& censoring_times = data->getCensoringTimes();
    const vector<uint8_t>& censoring_states = data->getCensoringStates();
    uint8_t max_response_length = data->getMaxResponseLength();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    const vector<size_t>& current_node_obs = node_obs[node_index];

    // all temporary quantities used for the key decomposition (for the right node)
    vector<size_t> censoring_contribution_right(nsplits_final * num_unique_event_times * num_states, 0);
    vector<size_t> num_jumps_acc_right(nsplits_final * num_unique_event_times * dim, 0);

    // do initial sweep to check for invalid splits
    for (size_t i : current_node_obs) {
        double feature_val = data->get_x(i, feature);
        for (size_t s = 0; s < nsplits_final; ++s) {
            if (feature_val > split_points[s]) {
                // add one to the number of observations in right node for split s
                ++num_obs_right[s];
            } else {
                break;
            }
        }
    }

    // compute initial rates, the censoring contribution C and the number of jumps across all event times and
    // observations in the right node
    for (size_t i : current_node_obs) {
        double feature_val = data->get_x(i, feature);
        for (size_t s = 0; s < nsplits_final; ++s) {
            // if one of the daughter nodes are too small, skip the computation for that split
            size_t num_obs_left = current_node_obs.size() - num_obs_right[s];
            if (num_obs_right[s] < min_node_size || num_obs_left < min_node_size) {
                continue;
            }
            if (feature_val > split_points[s]) {
                // update I0
                ++num_at_risk_right[s * num_unique_event_times * num_states + states[i * max_response_length] - 1];

                // censoring contribution
                uint8_t censoring_state = censoring_states[i];
                if (censoring_state != 0) {     // censoring actually occurs
                    double R  = censoring_times[i];
                    for (size_t j = 0; j < num_unique_event_times; ++j) {
                        if ((*unique_event_times)[j] > R) {
                            // all following event times also satisfy > R
                            for (size_t k = j; k < num_unique_event_times; ++k) {
                                ++censoring_contribution_right[s * num_unique_event_times * num_states + k * num_states + censoring_state - 1];
                            }
                            break;
                        }
                    }
                }
                // compute the number of jumps (the problem has to lie in the computation of the jumps since the accumulated jumps are too many)
                size_t j = 1;
                size_t index = i * max_response_length + 1;
                while(j < max_response_length && states[index] != 0) {
                    size_t id = (*response_event_time_ids)[index];
                    int current_state_index = states[index] - 1;
                    int prev_state_index = states[index - 1] - 1;
                    if (current_state_index != prev_state_index) {
                        ++num_jumps_right[s * num_unique_event_times * dim + id * dim + prev_state_index * num_states + current_state_index];
                        //num_jumps_acc_right[s * num_unique_event_times * dim + id * dim + prev_state_index * num_states + current_state_index] = num_jumps_acc_right[s * num_unique_event_times * dim + (id - 1) * dim + prev_state_index * num_states + current_state_index] + 1;
                    } else {
                        //num_jumps_acc_right[s * num_unique_event_times * dim + id * dim + prev_state_index * num_states + current_state_index] = num_jumps_acc_right[s * num_unique_event_times * dim + (id - 1) * dim + prev_state_index * num_states + current_state_index];
                    }
                    ++j;
                    ++index;
                }

            } else {
                break;  // since the split points are sorted
            }
        }
    }
    
    // compute the cumulative number of jumps across all splits
    cumulativeMatrixSums(num_jumps_acc_right, num_jumps_right, num_states, nsplits_final);

    // now compute number at risk for each possible split via the key decomposition
    for (size_t s = 0; s < nsplits_final; ++s) {
        size_t begin = s * num_unique_event_times * dim + dim;
        size_t end = s * num_unique_event_times * dim + 2 * dim - 1;
        for (size_t j = 1; j < num_unique_event_times; ++j) {   // j = 1 since we already computed I0 above for each split
            vector<int> jump_contributions = columnSums(subtractMatrices(num_jumps_acc_right, begin, end, transpose(num_jumps_acc_right, begin, end)), static_cast<size_t>(num_states));
            begin += dim;
            end += dim;
            for (size_t k = 0; k < num_states; ++k) {
                // key decomposition
                size_t split_stride = s * num_unique_event_times * num_states;
                // for debugging
                int addition = (int) num_at_risk_right[split_stride + k] - (int) censoring_contribution_right[split_stride + j * num_states + k] + (int) jump_contributions[k];
                if (addition < 0) {
                    cout << "Warning: Key decomposition negative, causing underflow in num_at_risk_right" << endl;  // here is the bug when thinning
                    cout << "Addition = " << addition << endl;
                    cout << "num_at_risk_0 = " << num_at_risk_right[split_stride + k] << endl;
                    cout << "Censoring contribution = " << censoring_contribution_right[split_stride + j * num_states + k] << endl;
                    cout << "Jump contribution = " << jump_contributions[k] << endl;
                }
                num_at_risk_right[split_stride + j * num_states + k] = num_at_risk_right[split_stride + k] - censoring_contribution_right[split_stride + j * num_states + k] + jump_contributions[k];
            }
        }
    }
    
    // for debugging
    /*
    Rcout << "Numbers of observations in right node " << endl;
    printVector(num_obs_right);
    Rcout << "Computed quantities for every possible split. num_at_risk_right:" << endl;
    printVector(num_at_risk_right, num_states);
    //Rcout << "num_at_risk:" << endl;
    //printVector(num_at_risk, num_states);
    //Rcout << "num_jumps_acc_right:" << endl;
    //printVector(num_jumps_acc_right, dim);
    Rcout << "num_jumps_right:" << endl;
    printVector(num_jumps_right, dim);
    */
}

// old version
// for computing multi-state quantities (number at risk and number of jumps) for all splits in a node (for splits on continuous features)
/*
void MultistateTree::computeMultistateQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                         vector<size_t>& num_at_risk_right, vector<size_t>& num_jumps_right, size_t nsplits_final) {
    // fetch states and other relevant data quantities
    const vector<uint8_t>& states = data->getStates();
    const vector<double>& censoring_times = data->getCensoringTimes();
    const vector<uint8_t>& censoring_states = data->getCensoringStates();
    uint8_t max_response_length = data->getMaxResponseLength();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;

    // all temporary quantities used for the key decomposition (for the right node)
    vector<size_t> censoring_contribution_right(nsplits_final * num_unique_event_times * num_states, 0);
    vector<size_t> num_jumps_acc_right(nsplits_final * num_unique_event_times * dim, 0);

    // compute initial rates, the censoring contribution C and the number of jumps across all event times and
    // observations in the right node
    for (size_t i : node_obs[node_index]) {
        double feature_val = data->get_x(i, feature);
        for (size_t s = 0; s < nsplits_final; ++s) {
            if (feature_val > split_points[s]) {
                // add one to the number of observations in right node for split s
                ++num_obs_right[s];
                // update I0
                ++num_at_risk_right[s * num_unique_event_times * num_states + states[i * max_response_length] - 1];

                // censoring contribution
                uint8_t censoring_state = censoring_states[i];
                if (censoring_state != 0) {     // censoring actually occurs
                    double R  = censoring_times[i];
                    for (size_t j = 0; j < num_unique_event_times; ++j) {
                        if ((*unique_event_times)[j] > R) {
                            // all following event times also satisfy > R
                            for (size_t k = j; k < num_unique_event_times; ++k) {
                                ++censoring_contribution_right[s * num_unique_event_times * num_states + k * num_states + censoring_state - 1];
                            }
                            break;
                        }
                    }
                }
                // compute the number of jumps
                size_t j = 1;
                size_t index = i * max_response_length + 1;
                while(j < max_response_length && states[index] != 0) {
                    size_t id = (*response_event_time_ids)[index];
                    int current_state_index = states[index] - 1;
                    int prev_state_index = states[index - 1] - 1;
                    if (current_state_index != prev_state_index) {
                        ++num_jumps_right[s * num_unique_event_times * dim + id * dim + prev_state_index * num_states + current_state_index];
                        //num_jumps_acc_right[s * num_unique_event_times * dim + id * dim + prev_state_index * num_states + current_state_index] = num_jumps_acc_right[s * num_unique_event_times * dim + (id - 1) * dim + prev_state_index * num_states + current_state_index] + 1;
                    } else {
                        //num_jumps_acc_right[s * num_unique_event_times * dim + id * dim + prev_state_index * num_states + current_state_index] = num_jumps_acc_right[s * num_unique_event_times * dim + (id - 1) * dim + prev_state_index * num_states + current_state_index];
                    }
                    ++j;
                    ++index;
                }

            } else {
                break;  // since the split points are sorted
            }
        }
    }
    
    // compute the cumulative number of jumps across all splits
    cumulativeMatrixSums(num_jumps_acc_right, num_jumps_right, num_states, nsplits_final);

    // now compute number at risk for each possible split via the key decomposition
    for (size_t s = 0; s < nsplits_final; ++s) {
        size_t begin = s * num_unique_event_times * dim + dim;
        size_t end = s * num_unique_event_times * dim + 2 * dim - 1;
        for (size_t j = 1; j < num_unique_event_times; ++j) {   // j = 1 since we already computed I0 above for each split
            vector<int> jump_contributions = columnSums(subtractMatrices(num_jumps_acc_right, begin, end, transpose(num_jumps_acc_right, begin, end)), static_cast<size_t>(num_states));
            begin += dim;
            end += dim;
            for (size_t k = 0; k < num_states; ++k) {
                // key decomposition
                size_t split_stride = s * num_unique_event_times * num_states;
                num_at_risk_right[split_stride + j * num_states + k] = num_at_risk_right[split_stride + k] - censoring_contribution_right[split_stride + j * num_states + k] + jump_contributions[k];
            }
        }
    }
    
    // for debugging
    
    Rcout << "Numbers of observations in right node " << endl;
    printVector(num_obs_right);
    Rcout << "Computed quantities for every possible split. num_at_risk_right:" << endl;
    printVector(num_at_risk_right, num_states);
    //Rcout << "num_at_risk:" << endl;
    //printVector(num_at_risk, num_states);
    //Rcout << "num_jumps_acc_right:" << endl;
    //printVector(num_jumps_acc_right, dim);
    Rcout << "num_jumps_right:" << endl;
    printVector(num_jumps_right, dim);
    
}
*/

/*

void MultistateTree::computeMultistateQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                         vector<size_t>& num_at_risk_left, vector<size_t>& num_jumps_left, size_t nsplits_final) {
    const vector<uint8_t>& states = data->getStates();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    uint8_t max_response_length = data->getMaxResponseLength();
    //vector<size_t> delta_num_at_risk_right(nsplits_final * num_unique_event_times * num_states);
    
    // initialise the number of jumps and at risk to be the ones for the parent
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

*/

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

    vector<size_t> num_jumps_right(nsplits_final * num_unique_event_times * num_states * num_states);
    vector<size_t> num_at_risk_right(nsplits_final * num_unique_event_times * num_states);

    computeMultistateQuantitiesDaughter(node_index, feature, split_points, num_obs_right, num_at_risk_right, num_jumps_right, nsplits_final);

    // now determine the best split
    for (size_t i = 0; i < nsplits_final; ++i) {
        // if a node is too small, skip the split
        size_t num_obs_left = current_node_obs.size() - num_obs_right[i];
        if (num_obs_left < min_node_size || num_obs_right[i] < min_node_size) {
            continue;
        }

        double split_val;

        // choose splitrule
        if (splitrule == "logrank") {
            split_val = logRank(num_jumps, num_at_risk, num_jumps_right, num_at_risk_right, i);
        }
        if (splitrule == "gehan") {
            split_val = Gehan(num_jumps, num_at_risk, num_jumps_right, num_at_risk_right, i);
        }
        if (splitrule == "taroneware") {
            split_val = TaroneWare(num_jumps, num_at_risk, num_jumps_right, num_at_risk_right, i);
        }
        if (splitrule == "conserve") {
            split_val = conserve(num_jumps, num_at_risk, num_jumps_right, num_at_risk_right, i);
        }
        if (splitrule == "approxlogrank") {
            split_val = approxLogRank(num_jumps, num_at_risk, num_jumps_right, num_at_risk_right, i);
        }
        //Rcout << "Split value for split " << i << ":" << split_val << endl;

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
        // choose splitrule
        if (splitrule == "logrank") {
            split_val = logRank(num_jumps, num_at_risk, num_jumps_left, num_at_risk_left);
        }
        if (splitrule == "gehan") {
            split_val = Gehan(num_jumps, num_at_risk, num_jumps_left, num_at_risk_left);
        }
        if (splitrule == "taroneware") {
            split_val = TaroneWare(num_jumps, num_at_risk, num_jumps_left, num_at_risk_left);
        }
        if (splitrule == "conserve") {
            split_val = conserve(num_jumps, num_at_risk, num_jumps_left, num_at_risk_left);
        }
        if (splitrule == "approxlogrank") {
            split_val = approxLogRank(num_jumps, num_at_risk, num_jumps_left, num_at_risk_left);
        }

        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_left_indices = std::move(current_left_indices);
            best_right_indices = std::move(current_right_indices);
            best_feature = feature;
            best_threshold.assign(left_values.begin(), left_values.end());
        }
    }
}

void MultistateTree::makeLeaf(size_t node_index) {
    // compute the matrices of Nelson--Aalen estimators
    computeNA(node_index);
    computeInitialDist(node_index);

    // update tree info
    feature_IDs.push_back(0);
    thresholds.push_back({});

    // since we are in a terminal node, we save the indices for the observations
    for (size_t i : node_obs[node_index]) {
        prediction_node_IDs[i] = node_index;
    }
    if (honest) {
        for (size_t i : holdout_node_obs[node_index]) {
            prediction_node_IDs[i] = node_index;
        }
    }
}

// function to create a split for a multi-state tree. returns true if leaf, otherwise false
bool MultistateTree::createSplit(size_t node_index) {
    //Rcout << "Creating split on node " << node_index << endl;
    const vector<size_t>& current_node_obs = node_obs[node_index];

    // if no split is possible, make the node a leaf
    if (current_node_obs.size() < 2 * min_node_size) {
        if (!honest) {
            //Rcout << "Computing multi-state quantities in node (leaf)" << endl;
            computeMultistateQuantities(current_node_obs, num_jumps, num_at_risk);               // for dishonest trees, use the growing indices
        } else {
            computeMultistateQuantities(holdout_node_obs[node_index], num_jumps, num_at_risk);   // for honest trees, use the holdout set for computing the CHF
        }
        makeLeaf(node_index);
        return true;
    }

    //Rcout << "Computing multi-state quantities in node (not leaf)" << endl;
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
    //Rcout << "Sampled features: ";
    //printVector(sampled_features);

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

    //Rcout << "best_split_val = " << best_split_val << endl;

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
    init_dist.push_back(vector<double>());

    // for honest trees, update the holdout indices
    if (honest) {
        holdout_node_obs.push_back(holdout_left_indices);
        holdout_node_obs.push_back(holdout_right_indices);
    }

    return false;
}

// computes the initial distribution in a node
void MultistateTree::computeInitialDist(size_t node_index) {
    uint8_t num_states = data->getNumberOfStates();
    vector<double> init_dist(num_states, 0);
    double num_obs = node_obs[node_index].size();
    for (size_t j = 0; j < num_states; ++j) {
        init_dist[j] = (double) num_at_risk[j] / num_obs;
    }
    this->init_dist.push_back(std::move(init_dist));
}

// computes the matrix of Nelson--Aalen estimators in node node_index
void MultistateTree::computeNA(size_t node_index) {
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    vector<double> na(num_unique_event_times * dim, 0);
    
    // need to handle the first event time (always zero for multi-states) separately
    /*
    // no need for the zero case since a jump never takes place at time zero
    for (size_t j = 0; j < num_states; ++j) {
        for (size_t k = 0; k < num_states; ++k) {
            if (num_at_risk[j] != 0) {
                na[j * num_states + k] = double(num_jumps[j * num_states + k]) / double(num_at_risk[j]);
            }
        }
    }
    */

    // for each unique event time i, loop over all entries (j, k) in the matrix
    // we start at i = 1 since 0 is always the first unique event time and no jump takes place at time zero
    for (size_t i = 1; i < num_unique_event_times; ++i) {
        for (size_t j = 0; j < num_states; ++j) {
            size_t index = i * dim + j * num_states;
            double diag = 0;
            for (size_t k = 0; k < num_states; ++k) {
                if (j != k) {
                    na[index + k] = na[index - dim + k];
                    // for debugging only
                    if (num_at_risk[i * num_states + j] == 0 && num_jumps[index + k] > 0) {
                        Rcout << "Warning in computeNA: num_at_risk[i * num_states + j] = 0, but num_jumps[index + k] = " << num_jumps[index + k] << endl; 
                    }

                    if (num_at_risk[i * num_states + j] != 0) {
                        na[index + k] += double(num_jumps[index + k]) / double(num_at_risk[i * num_states + j]);
                    }
                    diag -= na[index + k];
                }
                /*
                else {
                    na[i * dim + j * num_states + k] = na[(i - 1) * dim + j * num_states + k];
                }
                */
                // the diagonal is minus the sum of all other row entries
                
            }
            na[index + j] = diag;
        }
    }
    //Rcout << "Nelson-Aalen estimator in terminal node:" << endl;
    //printVector(na);
    this->na.push_back(std::move(na));
}

// splitting rules for multi-state trees
//--------------------------------------------------------------------------------------

double MultistateTree::logRank(const vector<size_t>& num_jumps, const vector<size_t>& num_at_risk, const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id) {
    // fetch relevant data quantities
    const vector<pair<uint8_t, uint8_t>>& valid_jumps = data->getValidJumps();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;

    double LR = 0;
    for (auto jump : valid_jumps) {
        uint8_t j = jump.first - 1;
        uint8_t k = jump.second - 1;
        double sum_num = 0;
        double sum_den = 0;
        size_t jump_index = split_id * num_unique_event_times * dim;
        size_t at_risk_index = split_id * num_unique_event_times * num_states;
        for (size_t i = 0; i < num_unique_event_times; ++i) {
            const double d = (double) num_jumps[i * dim + j * num_states + k];
            const double d1 = (double) num_jumps_daughter[jump_index + i * dim + j * num_states + k];
            const double Y = (double) num_at_risk[i * num_states + j];
            const double Y1 = (double) num_at_risk_daughter[at_risk_index + i * num_states + j];
            
            // temporary for debugging
            if (Y < Y1) {
                Rcout << "Warning: Y = " << Y << " < Y1 = " << Y1 << endl; 
            }
            if (d > Y) {
                Rcout << "Warning: Number of jumps d = " << d << ", but Y = " << Y << " at event time i = " << i << " and jump (" << static_cast<size_t>(j + 1) << ", " << static_cast<size_t>(k + 1) << ")" << endl;
            }
            if (d1 > Y1) {
                Rcout << "Warning: Number of jumps d1 = " << d1 << ", but Y1 = " << Y1 << " at event time i = " << i << " and jump (" << static_cast<size_t>(j + 1) << ", " << static_cast<size_t>(k + 1) << ")" << endl;
            }

            // prevent division by zero in the log-rank test
            if (Y < 2 || Y1 < 1) {
                break;  // since the event times are ordered, all subsequent numbers at risk will also be too small 
            }
            if (d > 0) {
                double at_risk_frac = Y1 / Y;
                sum_num += d1 - d * at_risk_frac;
                sum_den += d * at_risk_frac * (1.0 - at_risk_frac) * (Y - d) / (Y - 1.0);
            }
        }

        // update the final log-rank statistic
        if (sum_den != 0) {
            LR += sum_num * sum_num / sum_den;
        }
    }
    if (LR > 0) {
        return LR;
    } else {
        return -1;  // if a non-sensical value has been computed, treat as unvalid split
    }
}

double MultistateTree::Gehan(const vector<size_t>& num_jumps, const vector<size_t>& num_at_risk, const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id) {
    // fetch relevant data quantities
    const vector<pair<uint8_t, uint8_t>>& valid_jumps = data->getValidJumps();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;

    double G = 0;
    for (auto jump : valid_jumps) {
        uint8_t j = jump.first - 1;
        uint8_t k = jump.second - 1;
        double sum_num = 0;
        double sum_den = 0;
        size_t jump_index = split_id * num_unique_event_times * dim;
        size_t at_risk_index = split_id * num_unique_event_times * num_states;
        for (size_t i = 0; i < num_unique_event_times; ++i) {
            const double d = (double) num_jumps[i * dim + j * num_states + k];
            const double d1 = (double) num_jumps_daughter[jump_index + i * dim + j * num_states + k];
            const double Y = (double) num_at_risk[i * num_states + j];
            const double Y1 = (double) num_at_risk_daughter[at_risk_index + i * num_states + j];

            // prevent division by zero in the log-rank test
            if (Y < 2 || Y1 < 1) {
                break;  // since the event times are ordered, all subsequent numbers at risk will also be too small 
            }
            if (d > 0) {
                sum_num += Y * d1 - d * Y1;
                sum_den += d * Y1 * (Y - Y1) * (Y - d) / (Y - 1.0);
            }
        }

        // update the final log-rank statistic
        if (sum_den != 0) {
            G += sum_num * sum_num / sum_den;
        }
    }
    if (G > 0) {
        return G;
    } else {
        return -1;  // if a non-sensical value has been computed, treat as unvalid split
    }
}

double MultistateTree::TaroneWare(const vector<size_t>& num_jumps, const vector<size_t>& num_at_risk, const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id) {
    // fetch relevant data quantities
    const vector<pair<uint8_t, uint8_t>>& valid_jumps = data->getValidJumps();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;

    double TW = 0;
    for (auto jump : valid_jumps) {
        uint8_t j = jump.first - 1;
        uint8_t k = jump.second - 1;
        double sum_num = 0;
        double sum_den = 0;
        size_t jump_index = split_id * num_unique_event_times * dim;
        size_t at_risk_index = split_id * num_unique_event_times * num_states;
        for (size_t i = 0; i < num_unique_event_times; ++i) {
            const double d = (double) num_jumps[i * dim + j * num_states + k];
            const double d1 = (double) num_jumps_daughter[jump_index + i * dim + j * num_states + k];
            const double Y = (double) num_at_risk[i * num_states + j];
            const double Y1 = (double) num_at_risk_daughter[at_risk_index + i * num_states + j];

            // prevent division by zero in the log-rank test
            if (Y < 2 || Y1 < 1) {
                break;  // since the event times are ordered, all subsequent numbers at risk will also be too small 
            }
            if (d > 0) {
                double at_risk_frac = Y1 / Y;
                sum_num += sqrt(Y) * (d1 - d * at_risk_frac);
                sum_den += d * Y1 * (1.0 - at_risk_frac) * (Y - d) / (Y - 1.0);
            }
        }

        // update the final log-rank statistic
        if (sum_den != 0) {
            TW += sum_num * sum_num / sum_den;
        }
    }
    if (TW > 0) {
        return TW;
    } else {
        return -1;  // if a non-sensical value has been computed, treat as unvalid split
    }
}

// WARNING: This splitting rule may be completely nonsensical for multi-states (but it seems to work with the absolute value fix)
double MultistateTree::conserve(const vector<size_t>& num_jumps, const vector<size_t>& num_at_risk, const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id) {
    // fetch relevant data quantities
    const vector<pair<uint8_t, uint8_t>>& valid_jumps = data->getValidJumps();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;

    //initialise vectors to hold the sums and the numbers at risk in the other daughter
    vector<double> NAsum1(num_unique_event_times, 0);
    vector<double> NAsum2(num_unique_event_times, 0);
    vector<size_t> num_at_risk_daughter_2(num_unique_event_times, 0);

    double cons = 0;
    for (auto jump : valid_jumps) {
        // save indices
        uint8_t j = jump.first - 1;
        uint8_t k = jump.second - 1;
        size_t jump_index = split_id * num_unique_event_times * dim;
        size_t at_risk_index = split_id * num_unique_event_times * num_states;

        // temporary quantities, including info on the other daughter computed residually
        double sum1_jump = 0;
        double sum2_jump = 0;
        size_t num_jumps_daughter_2 = num_jumps[j * num_states + k] - num_jumps_daughter[jump_index + j * num_states + k];
        num_at_risk_daughter_2[0] = num_at_risk[j] - num_at_risk_daughter[at_risk_index + j];

        // compute vectors containing the innermost sum in the approximation used by Ishwaran and Kogalur
        if (num_jumps_daughter[jump_index + j * num_states + k] > 0) {
            NAsum1[0] = (double) num_jumps_daughter[jump_index + j * num_states + k] / num_at_risk_daughter[at_risk_index + j];
        } else {
            NAsum1[0] = 0;  // ensures that NAsum1 gets reset for every jump
        }
        if (num_jumps_daughter_2 > 0) {
            NAsum2[0] = (double) num_jumps_daughter_2 / num_at_risk_daughter_2[0];
        } else {
            NAsum2[0] = 0;  // ensures that NAsum2 gets reset for every jump
        }

        for (size_t i = 1; i < num_unique_event_times; ++i) {
            NAsum1[i] = NAsum1[i - 1];
            NAsum2[i] = NAsum2[i - 1];

            // update quantities for the other daughter
            size_t d1 = num_jumps_daughter[jump_index + i * dim + j * num_states + k];
            num_jumps_daughter_2 = num_jumps[i * dim + j * num_states + k] - d1;
            num_at_risk_daughter_2[i] = num_at_risk[i * num_states + j] - num_at_risk_daughter[at_risk_index + i * num_states + j];
            Rcout << "num_at_risk_daughter_2[i] = " << num_at_risk_daughter_2[i] << endl;

            if (d1 > 0) {
                NAsum1[i] += (double) d1 / num_at_risk_daughter[at_risk_index + i * num_states + j];
            }
            if (num_jumps_daughter_2 > 0) {
                NAsum2[i] += (double) num_jumps_daughter_2 / num_at_risk_daughter_2[i];     // something is weird here...
            }
        }

        // now compute the outer sums
        for (size_t i = 0; i < num_unique_event_times - 1; ++i) {
            // in survival, you would not need the abs (maybe this adaptation to multi-states does not even make sense)
            Rcout << "num_at_risk_daughter_2[i + 1] = " << num_at_risk_daughter_2[i + 1] << ", NAsum2[i] = " << NAsum1[i] << endl;
            sum1_jump += abs(static_cast<int>(num_at_risk_daughter[at_risk_index + i * num_states + j]) - static_cast<int>(num_at_risk_daughter[at_risk_index + (i + 1) * num_states + j])) * num_at_risk_daughter[at_risk_index + (i + 1) * num_states + j] * NAsum1[i];
            sum2_jump += abs(static_cast<int>(num_at_risk_daughter_2[i]) - static_cast<int>(num_at_risk_daughter_2[i + 1])) * num_at_risk_daughter_2[i + 1] * NAsum2[i];
        }
        Rcout << "Added to cons: " << (num_at_risk_daughter[at_risk_index + j] * sum1_jump + num_at_risk_daughter_2[0] * sum2_jump) / num_at_risk[j] << endl;
        cons += (num_at_risk_daughter[at_risk_index + j] * sum1_jump + num_at_risk_daughter_2[0] * sum2_jump) / num_at_risk[j];
        Rcout << "cons = " << cons << endl;

        // reset NAsum vectors
        //fill(NAsum1.begin(), NAsum1.end(), 0);
        //fill(NAsum2.begin(), NAsum2.end(), 0);
    }
    return 1/(1 + cons);
}

double MultistateTree::approxLogRank(const vector<size_t>& num_jumps, const vector<size_t>& num_at_risk, const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id) {
    // fetch relevant data quantities
    const vector<pair<uint8_t, uint8_t>>& valid_jumps = data->getValidJumps();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;

    double aLR = 0;
    for (auto jump : valid_jumps) {
        uint8_t j = jump.first - 1;
        uint8_t k = jump.second - 1;
        double D1 = 0;
        double D = 0;
        double sum_num = 0;
        size_t jump_index = split_id * num_unique_event_times * dim;
        size_t at_risk_index = split_id * num_unique_event_times * num_states;
        for (size_t i = 0; i < num_unique_event_times; ++i) {
            const double d = (double) num_jumps[i * dim + j * num_states + k];
            const double d1 = (double) num_jumps_daughter[jump_index + i * dim + j * num_states + k];
            const double Y = (double) num_at_risk[i * num_states + j];
            const double Y1 = (double) num_at_risk_daughter[at_risk_index + i * num_states + j];
            sum_num += d1 - Y1 * d / Y;
            D1 += d1;
            D += d;     // I think this is ok, check that it makes sense
        }

        double den = sqrt((D1 - sum_num) * (D - D1 + sum_num));
        if (den != 0) {
            aLR += abs((sqrt(D) * sum_num ) / den); // alternative: take absolute values in the end, but this is more in line with the log-rank test
        }
    }
    if (aLR > 0) {
        return aLR;
    } else {
        return -1;  // if a non-sensical value has been computed, treat as unvalid split
    }
}

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

vector<double> MultistateTree::computePredictedInitialDistributions(const Data& new_data) {
    uint8_t num_states = data->getNumberOfStates();
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_states);
    for (size_t i = 0; i < num_obs; ++i) {
        const vector<double>& pred = predictInitDist(new_data.get_x_row(i));
        for (size_t j = 0; j < num_states; ++j) {
            predictions[i * num_states + j] = pred[j];
        }
    }
    return predictions;
}

// error estimation for multi-state trees
//--------------------------------------------------------------------------------------



// miscellaneous functions related to multi-states
//--------------------------------------------------------------------------------------

// computes the vector of unique event times for multi-state data, excluding censoring times
vector<double> uniqueEventTimesMultistate(const vector<double>& times, const vector<uint8_t>& states) {
    vector<double> result;
    result.push_back(0);    // 0 is always included

    // start by picking out the times that are not only censoring times
    // since 0 is the first time of every observation, we can also skip 0
    for (size_t i = 1; i < times.size(); ++i) {
        if (times[i] != 0 && states[i] != states[i - 1]) {
            result.push_back(times[i]);
        }
    }

    // now sort the values and remove duplicates
    sort(result.begin(), result.end());
    auto last = unique(result.begin(), result.end());
    result.erase(last, result.end());   // save memory
    return result;
}

// for computing the ids in the observed times corresponding to the unique event times (not counting censored times)
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
        size_t idx = static_cast<size_t>(distance(unique_event_times.begin(), it));
        if (idx >= unique_event_times.size()) {
            idx = unique_event_times.size() - 1;
        }
        response_event_time_ids.push_back(idx);
    }
    return response_event_time_ids;
}

vector<double> AalenJohansen(const vector<double>& na, uint8_t num_states) {
    vector<double> aj = vector<double>(na.size(), 0);
    size_t dim = num_states * num_states;   // number of entries in each matrix
    size_t num_jumps = na.size() / dim;     // num_unique_event_times
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
/*

Functions for multi-state trees

*/

#include "TreeMultistate.h"


// constructor for MultistateTree
//--------------------------------------------------------------------------------------

// the final argument is only used for honest trees
MultistateTree::MultistateTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids, 
                               const vector<size_t>& subset_indices, uint8_t num_states, bool save_predictions, const vector<size_t>& estimation_indices) : 
    unique_event_times {unique_event_times}, response_event_time_ids {response_event_time_ids}, save_predictions {save_predictions} {
    this->node_obs.push_back(subset_indices);
    this->holdout_node_obs.push_back(estimation_indices);
    this->num_unique_event_times = unique_event_times->size();
    this->node_sizes.push_back(subset_indices.size());
    this->num_jumps.resize(num_unique_event_times * num_states * num_states);
    this->num_at_risk.resize(num_unique_event_times * num_states);
}

// functions for growing multi-state trees
//--------------------------------------------------------------------------------------

void MultistateTree::computeMultistateQuantities(const vector<size_t>& indices, vector<size_t>& jumps, vector<size_t>& at_risk) {
    // fetch states and other relevant data quantities
    const vector<double>& times = data->getTimes();
    const vector<uint8_t>& states = data->getStates();
    const vector<size_t>& last_observed_times = data->getLastObservedTimes();
    //const vector<double>& last_observed_times = data->getLastObservedTimes();
    //const vector<double>& censoring_times = data->getCensoringTimes();
    const vector<uint8_t>& censoring_states = data->getCensoringStates();

    uint8_t max_response_length = data->getMaxResponseLength();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    
    // initialise vectors of number of jumps and at risk
    jumps.assign(num_unique_event_times * dim, 0);
    at_risk.assign(num_unique_event_times * num_states, 0);
    censoring_contribution.assign(num_unique_event_times * num_states, 0);
    // all temporary quantities used for the key decomposition
    //vector<size_t> censoring_contribution(num_unique_event_times * num_states, 0);
    vector<size_t> num_jumps_acc(num_unique_event_times * dim, 0);

    //Rcout << "Finished initialising before computing quantities" << endl;

    // compute the initial rates, the censoring contribution C and the number of jumps across all event times and observations
    for (size_t i : indices) {
        //Rcout << "Index " << i << endl;
        // initial rates I0
        ++at_risk[states[i * max_response_length] - 1];
        //Rcout << "Updated I0" << endl;

        // compute the censoring contribution C
        uint8_t censoring_state = censoring_states[i];
        if (censoring_state != 0) {     // censoring actually occurs
            double R = times[last_observed_times[i]];
            //double R = last_observed_times[i];    // old: when we saved the last observed time itself
            //double R = censoring_times[i];        // old: when we saved the censoring times only
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
                ++jumps[id * dim + prev_state_index * num_states + current_state_index];
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
    cumulativeMatrixSums(num_jumps_acc, jumps, num_states);

    // now compute number at risk via the key decomposition
    for (size_t j = 1; j < num_unique_event_times; ++j) {   // j = 1 since we already computed I0 above
        //vector<int> subtraction = subtractMatrices(num_jumps_acc, j * dim, (j + 1) * dim - 1, transpose(num_jumps_acc, j * dim, (j + 1) * dim - 1));
        vector<int> jump_contributions = columnSums(subtractMatrices(num_jumps_acc, j * dim, (j + 1) * dim - 1, transpose(num_jumps_acc, j * dim, (j + 1) * dim - 1)), static_cast<size_t>(num_states));
        for (size_t k = 0; k < num_states; ++k) {
            // key decomposition
            // for debugging thinning
            //if ((int) at_risk[k] - (int) censoring_contribution[j * num_states + k] + (int) jump_contributions[k] < 0) {
            //    cout << "Warning: Key decomposition negative, causing underflow in at_risk" << endl;
            //}
            at_risk[j * num_states + k] = at_risk[k] - censoring_contribution[j * num_states + k] + jump_contributions[k];
        }
    }
    
    // only for debugging
    /*
    Rcout << "Number of jumps: " << endl;
    printVector(num_jumps, dim);
    Rcout << "Numbers at risk: " << endl;
    printVector(at_risk, num_states);
    Rcout << "Computed num_jumps_acc:" << endl;
    printVector(num_jumps_acc, dim);
    Rcout << "Censoring contribution:" << endl;
    printVector(censoring_contribution, num_states);
    */

    // for debugging purposes
    /*
    size_t total_number_at_risk = 0;
    for (size_t j = 0; j < data->getNumberOfStates(); ++j) {
        total_number_at_risk += at_risk[j];
    }
    cout << "Total initial number at risk in computeMultistateQuantities: " << total_number_at_risk << endl;
    cout << "Total initial number of observations in computeMultistateQuantities: " << indices.size() << endl;
    */
    // these two always concur here as they should
}

// version of feb10 (makes pre-sweep and checks for invalid splits before computing quantities)
// for computing multi-state quantities (number at risk and number of jumps) for all splits in a node (for splits on continuous features)
void MultistateTree::computeMultistateQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                         vector<size_t>& num_at_risk_right, vector<size_t>& num_jumps_right, size_t nsplits_final) {
    // fetch states and other relevant data quantities
    const vector<double>& times = data->getTimes();
    const vector<uint8_t>& states = data->getStates();
    const vector<size_t>& last_observed_times = data->getLastObservedTimes();
    //const vector<double>& last_observed_times = data->getLastObservedTimes();
    //const vector<double>& censoring_times = data->getCensoringTimes();
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
                    double R = times[last_observed_times[i]];
                    //double R = last_observed_times[i];
                    //double R  = censoring_times[i];
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
            //cout << "Computing log-rank splitting value in bestSplitContinuous" << endl;
            split_val = logRank(num_jumps, num_at_risk, num_jumps_right, num_at_risk_right, i);
            //cout << "Done computing log-rank splitting value in bestSplitContinuous, split_val = " << split_val << endl;
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
    size_t num_states = data->getNumberOfStates();

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

        // here we have to compute the multi-state info in one of the daughters from scratch
        vector<size_t> num_jumps_left(num_unique_event_times * num_states * num_states);
        vector<size_t> num_at_risk_left(num_unique_event_times * num_states);
        computeMultistateQuantities(current_left_indices, num_jumps_left, num_at_risk_left);
        double split_val;
        // choose splitrule
        if (splitrule == "logrank") {
            //cout << "Computing log-rank splitting value in bestSplitCategorical" << endl;
            split_val = logRank(num_jumps, num_at_risk, num_jumps_left, num_at_risk_left);
            //cout << "Done computing log-rank splitting value in bestSplitCategorical, split_val = " << split_val << endl;
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
    // compute the matrices of Nelson--Aalen estimators and the vector of predicted initial distributions
    computeNA(node_index);
    computeInitialDist(node_index);

    if (save_predictions) {
        computeCensoringKM();
    }

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

        // for debugging purposes
        
        size_t total_number_at_risk = 0;
        for (size_t j = 0; j < data->getNumberOfStates(); ++j) {
            total_number_at_risk += num_at_risk[j];
        }
        //cout << "Total initial number at risk in createSplit (making leaf): " << total_number_at_risk << endl;
        //cout << "Total number of observations in createSplit (making leaf): " << current_node_obs.size() << endl;
        

        makeLeaf(node_index);
        return true;
    }

    //Rcout << "Computing multi-state quantities in node (not leaf)" << endl;
    computeMultistateQuantities(current_node_obs, num_jumps, num_at_risk);   // update parent multi-state info
    //cout << "num_at_risk in parent node:" << endl;
    //printVector(num_at_risk);
    //cout << "num_jumps in parent node:" << endl;
    //printVector(num_jumps);
    // for debugging purposes
    /*
    size_t total_number_at_risk = 0;
    for (size_t j = 0; j < data->getNumberOfStates(); ++j) {
        total_number_at_risk += num_at_risk[j];
    }
    cout << "Total initial number at risk in createSplit: " << total_number_at_risk << endl;
    cout << "Total initial number of observations in createSplit: " << current_node_obs.size() << endl;
    */

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
            //cout << "About to call bestSplitCategorical" << endl;
            bestSplitCategorical(node_index, i, best_split_val, best_feature, best_threshold, best_left_indices, best_right_indices);
        }
        else {
            // does not return the best indices, so this has to be done later
            //cout << "About to call bestSplitContinuous" << endl;
            bestSplitContinuous(node_index, i, best_split_val, best_feature, best_threshold);
        }
    }

    // cout << "best_split_val = " << best_split_val << endl;

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
    if (save_predictions) {
        KM_censoring.push_back(vector<double>());
    }

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
    double num_obs;
    if (honest) {
        num_obs = holdout_node_obs[node_index].size();
    } else {
        num_obs = node_obs[node_index].size();
    }
    //size_t total_num_at_risk = 0;
    for (size_t j = 0; j < num_states; ++j) {
        //cout << "num_at_risk initially in state " << j << ": " << num_at_risk[j] << endl;
        init_dist[j] = (double) num_at_risk[j] / num_obs;
        //total_num_at_risk += num_at_risk[j];
    }
    //cout << "Total number at risk: " << total_num_at_risk << ", total obs in node: " << num_obs << endl;
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
    // only for debugging
    //Rcout << "Nelson-Aalen estimator in terminal node " << node_index << endl;
    //printVector(na);
    this->na.push_back(std::move(na));
}

void MultistateTree::computeCensoringKM() {
    // initalise and fetch data
    vector<double>KM_censoring(num_unique_event_times, 1);
    size_t num_states = data->getNumberOfStates();
    //const vector<size_t>& last_observed_times = data->getLastObservedTimes();
    //const vector<uint8_t>& censoring_states = data->getCensoringStates();
    //const vector<double>& unique_event_times = this->unique_event_times.get();
    //const vector<size_t>& response_event_time_ids = this->response_event_time_ids.get();
    
    // compute total censoring and at risk contributions across states (integrator and 1/integrand of the NA estimator for censoring, respectively)
    vector<size_t> censoring_contribution_total = sum_vectors(censoring_contribution, num_unique_event_times);
    vector<size_t> num_at_risk_total = sum_vectors(num_at_risk, num_unique_event_times);
    vector<size_t> num_jumps_total = sum_vectors(num_jumps, num_unique_event_times);

    // possibly relevant for future debugging
    /*
    cout << "num_at_risk_total has length " << num_at_risk_total.size() << endl;
    printVector(num_at_risk_total);
    cout << "censoring_contribution_total has length " << censoring_contribution_total.size() << endl;
    printVector(censoring_contribution_total);
    */

    // now compute the Kaplan-Meier estimator on the observed last event times
    double jump_size;
    for (size_t i = 1; i < num_unique_event_times; ++i) {
        if (num_at_risk_total[i - 1] > 0) {
            jump_size = censoring_contribution_total[i] - censoring_contribution_total[i - 1];
            KM_censoring[i] = KM_censoring[i - 1] * (1 - jump_size / double(num_at_risk_total[i - 1] - num_jumps_total[i - 1]));
        } else {
            KM_censoring[i] = KM_censoring[i - 1];
        }
    }
    //printVector(KM_censoring);
    this->KM_censoring.push_back(std::move(KM_censoring));
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
        //cout << "Considering jump (" << static_cast<size_t>(j) << "," << static_cast<size_t>(k) << ")" << endl;
        double sum_num = 0;
        double sum_den = 0;
        size_t jump_index = split_id * num_unique_event_times * dim;
        size_t at_risk_index = split_id * num_unique_event_times * num_states;
        //cout << "num_unique_event_times = " << num_unique_event_times << endl;
        //cout << "num_jumps.size() = " << num_jumps.size() << ", num_jumps_daughter.size() = " << num_jumps_daughter.size() << ", num_at_risk.size() = " << num_at_risk.size() << ", num_at_risk_daughter.size() = " << num_at_risk_daughter.size() << endl;
        for (size_t i = 0; i < num_unique_event_times; ++i) {
            //cout << "Corresponding indices: " << i * dim + j * num_states + k << ", " << jump_index + i * dim + j * num_states + k << ", " << i * num_states + j << ", " << at_risk_index + i * num_states + j << endl;
            const double d = (double) num_jumps[i * dim + j * num_states + k];
            const double d1 = (double) num_jumps_daughter[jump_index + i * dim + j * num_states + k];
            const double Y = (double) num_at_risk[i * num_states + j];
            const double Y1 = (double) num_at_risk_daughter[at_risk_index + i * num_states + j];
            //cout << "Fetched quantities for event time " << i << endl;
            
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
            //cout << "Done updating sum_num and sum_den for event time " << i << endl;
        }

        // update the final log-rank statistic
        if (sum_den != 0) {
            LR += sum_num * sum_num / sum_den;
        }
    }
    if (LR > 0) {
        return LR;
    } else {
        return -1;  // if a non-sensical value has been computed, treat as invalid split
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

// compute a flattened vector of predictions (the Nelson-Aalen estimators at each event time)
vector<double> MultistateTree::computePredictions(const Data& new_data) {
    uint8_t num_states = data->getNumberOfStates();
    size_t dim = num_states * num_states;
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times * dim);
    for (size_t i = 0; i < num_obs; ++i) {
        const vector<double>& pred = get<vector<double>>(predict(new_data.get_x_row(i)));   // the na vector has for unknown reasons been destroyed...
        //cout << "Prediction for observation " << i << ":" << endl;
        //printVector(pred);
        //cout << endl;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            for (size_t j = 0; j < num_states; ++j) {
                for (size_t k = 0; k < num_states; ++k) {
                    predictions[i * num_unique_event_times * dim + t * dim + j * num_states + k] = pred[t * dim + j * num_states + k];
                }
            }
        }
    }
    return predictions;
}

// compute a flattened vector of predictions and of the predicted initial distributions
pair<vector<double>, vector<double>> MultistateTree::computePredictedInitialDistributions(const Data& new_data) {
    uint8_t num_states = data->getNumberOfStates();
    size_t dim = num_states * num_states;
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times * num_states * num_states);
    vector<double> predictions_init(num_obs * num_states);
    for (size_t i = 0; i < num_obs; ++i) {
        size_t leaf_id = predictionLeafID(new_data.get_x_row(i));
        const vector<double>& pred = na[leaf_id];
        const vector<double>& init = init_dist[leaf_id];
        // first save initial distributions
        for (size_t j = 0; j < num_states; ++j) {
            predictions_init[i * num_states + j] = init[j];
        }

        // now save the predicted NA-estimators
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            for (size_t j = 0; j < num_states; ++j) {
                for (size_t k = 0; k < num_states; ++k) {
                    predictions[i * num_unique_event_times * dim + t * dim + j * num_states + k] = pred[t * dim + j * num_states + k];
                }
            }
        }
    }
    return {predictions, predictions_init};
}

// computes a pair of flattened vectors, the first predictions and the second the censoring KM estimators
pair<vector<double>, vector<double>> MultistateTree::computePredictionsCensoring(const Data& new_data) {
    uint8_t num_states = data->getNumberOfStates();
    size_t dim = num_states * num_states;
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times * num_states * num_states);
    vector<double> censoring(num_obs * num_unique_event_times);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t leaf_id = predictionLeafID(new_data.get_x_row(i));
        const vector<double>& pred = na[leaf_id];
        const vector<double>& cens = KM_censoring[leaf_id];
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            // first save the predicted Nelson-Aalen estimator
            for (size_t j = 0; j < num_states; ++j) {
                for (size_t k = 0; k < num_states; ++k) {
                    predictions[i * num_unique_event_times * dim + t * dim + j * num_states + k] = pred[t * dim + j * num_states + k];
                    //predictions[i * num_obs + t * num_unique_event_times + j * num_states + k] = pred[t * num_unique_event_times + j * num_states + k];
                }
            }
            // now save the censoring Kaplan-Meier estimator
            censoring[i * num_unique_event_times + t] = cens[t];
        }
    }
    return {predictions, censoring};
}

// computes three flattened vectors of all predictions (NA estimators, initial distributions and censoring KM estimators)
vector<vector<double>> MultistateTree::computeAllPredictions(const Data& new_data) {
    uint8_t num_states = data->getNumberOfStates();
    size_t dim = num_states * num_states;
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times * num_states * num_states);
    vector<double> predictions_init(num_obs * num_unique_event_times * num_states);
    vector<double> censoring(num_obs * num_unique_event_times);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t leaf_id = predictionLeafID(new_data.get_x_row(i));
        const vector<double>& pred = na[leaf_id];
        const vector<double>& cens = KM_censoring[leaf_id];
        const vector<double>& init = init_dist[leaf_id];
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            // first save predicted NA estimators
            for (size_t j = 0; j < num_states; ++j) {
                for (size_t k = 0; k < num_states; ++k) {
                    predictions[i * num_unique_event_times * dim + t * dim + j * num_states + k] = pred[t * dim + j * num_states + k];
                }
            }
            // save censoring distribution
            censoring[i * num_unique_event_times + t] = cens[t];
        }
        // save initial distribution
        for (size_t j = 0; j < num_states; ++j) {
            predictions_init[i * num_states + j] = init[j];
        }
    }
    return {predictions, predictions_init, censoring};
}

// error estimation for multi-state trees
//--------------------------------------------------------------------------------------

// computes a vector of the Brier score using given IPCW weights for multi-state predictions 
// (states_ind is a flattened vector of boolean indicators of whether observation i at event time t is in state j)
vector<double> computeBrierScoreMM(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times,
                                   const List& occupation_probs, const vector<double>& state_weights) {
    size_t num_unique_event_times = unique_event_times.size();
    uint8_t num_states = state_weights.size();
    size_t num_obs = weights.size() / num_unique_event_times;
    vector<double> brier(num_unique_event_times * num_states, 0);

    for (size_t i = 0; i < num_obs; ++i) {
        const List& occ_probs_obs = as<List>(occupation_probs[i]);
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            const vector<double> occ_probs_obs_t = as<vector<double>>(occ_probs_obs[t]);
            double ipcw = weights[i * num_unique_event_times + t];
            // difference to survival: need to compute a contribution to the score across all states
            for (size_t j = 0; j < num_states; ++j) {
                if (states_ind[i * num_unique_event_times * num_states + j]) {
                    brier[t * num_states + j] += ipcw * (1 - occ_probs_obs_t[j]) * (1 - occ_probs_obs_t[j]) * state_weights[j];
                } else {
                    brier[t * num_states + j] += ipcw * occ_probs_obs_t[j] * occ_probs_obs_t[j] * state_weights[j];
                }
            }
        }
    }
    return brier;
}

// same function as above but where the occupation probabilities are instead given by a flattened vector
vector<double> computeBrierScoreCppMM(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times, 
                                      const vector<double>& occupation_probs, const vector<double>& state_weights) {
    size_t num_unique_event_times = unique_event_times.size();
    uint8_t num_states = state_weights.size();
    size_t num_obs = weights.size() / num_unique_event_times;
    vector<double> brier(num_unique_event_times * num_states, 0);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t obs_index = i * num_unique_event_times * num_states;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            size_t time_index = obs_index + t * num_states;
            double ipcw = weights[i * num_unique_event_times + t];
            // difference to survival: need to compute a contribution to the score across all states
            for (size_t j = 0; j < num_states; ++j) {
                if (states_ind[i * num_unique_event_times * num_states + j]) {
                    brier[t * num_states + j] += ipcw * (1 - occupation_probs[time_index + j]) * (1 - occupation_probs[time_index + j]) * state_weights[j];
                } else {
                    brier[t * num_states + j] += ipcw * occupation_probs[time_index + j] * occupation_probs[time_index + j] * state_weights[j];
                }
            }
        }
    }
    return brier;
}

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
    size_t num_unique_event_times = na.size() / dim;
    // initial value is the identity matrix
    for (size_t j = 0; j < num_states; ++j) {
        aj[j * num_states + j] = 1;
    }

    // compute the Aalen--Johansen estimator at each jump time using an optimised matrix multiplication scheme
    for (size_t i = 1; i < num_unique_event_times; ++i) {
        for (size_t j = 0; j < num_states; ++j) {           // row of the aj matrix
            for (size_t k = 0; k < num_states; ++k) {       // column of the aj matrix
                double prev = aj[(i - 1) * dim + j * num_states + k];
                for (size_t l = 0; l < num_states; ++l) {   // column of matrix in increment
                    double contribution = na[i * dim + k * num_states + l] - na[(i - 1) * dim + k * num_states + l];
                    if (k == l) {
                        ++contribution;
                    }
                    aj[i * dim + j * num_states + l] += prev * contribution;
                }
            }
        }
    }
    return aj;
}

// computes the occupation probabilities given a Nelson-Aalen estimator and an initial distribution
// if na and init are flattened vectors, provide the number of estimators (na and init need to have the same 'dimensions')
vector<double> occupationProbabilitiesCpp(const vector<double>& na, const vector<double>& init, size_t num_states, size_t num_estimators) {
    size_t stride_length = na.size() / num_estimators;
    size_t dim = num_states * num_states;
    size_t num_unique_event_times = stride_length / dim;
    vector<double> result(num_estimators * num_unique_event_times * num_states, 0);     // num_estimators corresponds to num_obs
    vector<double> aj(num_estimators * num_unique_event_times * dim, 0);

    // across all observations, the initial value of the product integral (aj) is the identity matrix
    for (size_t i = 0; i < num_estimators; ++i) {
        for (size_t j = 0; j < num_states; ++j) {
            aj[i * num_unique_event_times * dim + j * num_states + j] = 1;
        }
    }

    // now compute the product integral across all event times for every observation
    for (size_t i = 0; i < num_estimators; ++i) {
        size_t obs_index = i * num_unique_event_times * dim;
        for (size_t t = 1; t < num_unique_event_times; ++t) {
            for (size_t j = 0; j < num_states; ++j) {           // row of the aj matrix
                for (size_t k = 0; k < num_states; ++k) {       // col of the aj matrix
                    double prev = aj[obs_index + (t - 1) * dim + j * num_states + k];
                    for (size_t l = 0; l < num_states; ++l) {   // column of matrix in increment
                        double contribution = na[obs_index + t * dim + k * num_states + l] - na[obs_index + (t - 1) * dim + k * num_states + l];
                        if (k == l) {
                            ++contribution;
                        }
                        aj[obs_index + t * dim + j * num_states + l] += prev * contribution;
                    }
                }
            }
        }
    }

    // now multiply with the initial distribution
    for (size_t i = 0; i < num_estimators; ++i) {
        size_t obs_index = i * num_unique_event_times * num_states;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            for (size_t j = 0; j < num_states; ++j) {
                double init_val = init[obs_index + j];
                for (size_t k = 0; k < num_states; ++k) {
                    result[obs_index + t * num_states + k] += init_val * aj[obs_index * num_states + t * dim + j * num_states + k];
                }
            }
        }
    }
    return result;
}

// same function as before, but now accepts List inputs instead for na and init (each is a List of Lists)
vector<double> occupationProbabilities(const List& na, const List& init, size_t num_states) {
    size_t num_obs = na.size();
    size_t dim = num_states * num_states;
    size_t num_unique_event_times = as<List>(na[0]).size();
    vector<double> result(num_obs * num_unique_event_times * num_states, 0);
    vector<double> aj(num_obs * num_unique_event_times * dim, 0);

    // across all observations, the initial value of the product integral (aj) is the identity matrix
    for (size_t i = 0; i < num_obs; ++i) {
        for (size_t j = 0; j < num_states; ++j) {
            aj[i * num_unique_event_times * dim + j * num_states + j] = 1;
        }
    }

    // now compute the product integral across all event times for every observation
    for (size_t i = 0; i < num_obs; ++i) {
        const List& na_obs = as<List>(na[i]);
        size_t obs_index = i * num_unique_event_times * dim;
        for (size_t t = 1; t < num_unique_event_times; ++t) {
            const NumericMatrix& na_obs_cur = na_obs[t];
            const NumericMatrix& na_obs_prev = na_obs[t - 1];
            for (size_t j = 0; j < num_states; ++j) {           // row of the aj matrix
                for (size_t k = 0; k < num_states; ++k) {       // col of the aj matrix
                    double prev = aj[obs_index + (t - 1) * dim + j * num_states + k];
                    for (size_t l = 0; l < num_states; ++l) {   // column of matrix in increment
                        //double contribution = na[obs_index + t * dim + k * num_states + l] - na[obs_index + (t - 1) * dim + k * num_states + l];
                        double contribution = na_obs_cur(k, l) - na_obs_prev(k, l);
                        if (k == l) {
                            ++contribution;
                        }
                        aj[obs_index + t * dim + j * num_states + l] += prev * contribution;
                    }
                }
            }
        }
    }

    // now multiply with the initial distribution
    for (size_t i = 0; i < num_obs; ++i) {
        const vector<double>& init_obs = as<vector<double>>(init[i]);
        size_t obs_index = i * num_unique_event_times * num_states;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            for (size_t j = 0; j < num_states; ++j) {
                double init_val = init_obs[j];
                for (size_t k = 0; k < num_states; ++k) {
                    result[obs_index + t * num_states + k] += init_val * aj[obs_index * num_states + t * dim + j * num_states + k];
                }
            }
        }
    }
    return result;
}
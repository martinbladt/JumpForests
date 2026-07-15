/*

Functions for multi-state trees

*/

#include "TreeMultistate.h"
#include "TreeSurvival.h"


// constructor for MultistateTree
//--------------------------------------------------------------------------------------

// the second to final argument is only used for honest trees
MultistateTree::MultistateTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids, 
                               const vector<size_t>& subset_indices, uint8_t num_states, bool save_predictions, const vector<size_t>& estimation_indices, shared_ptr<vector<double>> censoring_times) : 
    unique_event_times {unique_event_times}, response_event_time_ids {response_event_time_ids}, save_predictions {save_predictions} {
    this->node_obs.push_back(subset_indices);
    this->holdout_node_obs.push_back(estimation_indices);
    this->num_unique_event_times = unique_event_times->size();
    this->censoring_times = censoring_times == nullptr ? unique_event_times : censoring_times;
    this->num_censoring_times = this->censoring_times->size();
    this->node_sizes.push_back(subset_indices.size());
    this->num_jumps.resize(num_unique_event_times * num_states * num_states);
    this->num_at_risk.resize(num_unique_event_times * num_states);
}

// functions for growing multi-state trees
//--------------------------------------------------------------------------------------

void MultistateTree::computeMultistateQuantities(const vector<size_t>& indices, vector<size_t>& jumps, vector<size_t>& at_risk) {
    // fetch states and other relevant data quantities
    const vector<uint8_t>& states = data->getStates();
    const vector<size_t>& last_observed_times = data->getLastObservedTimes();
    const vector<uint8_t>& censoring_states = data->getCensoringStates();

    uint8_t max_response_length = data->getMaxResponseLength();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    
    // initialise vectors of number of jumps and at risk
    jumps.assign(num_unique_event_times * dim, 0);
    at_risk.assign(num_unique_event_times * num_states, 0);
    censoring_contribution.assign(num_unique_event_times * num_states, 0);
    vector<size_t> num_jumps_acc(num_unique_event_times * dim, 0);

    // compute the initial rates, the censoring contribution C and the number of jumps across all event times and observations
    for (size_t i : indices) {
        // initial rates I0
        ++at_risk[states[i * max_response_length] - 1];

        // compute the censoring contribution C
        uint8_t censoring_state = censoring_states[i];
        if (censoring_state != 0) {     // censoring actually occurs
            size_t censoring_time_id = (*response_event_time_ids)[last_observed_times[i]];
            for (size_t k = censoring_time_id; k < num_unique_event_times; ++k) {
                ++censoring_contribution[k * num_states + censoring_state - 1];
            }
        }
        // compute number of jumps (we assume that at least one event of some kind occurs so that max_response_length > 1)
        size_t j = 1;
        size_t index = i * max_response_length + 1;
        // a state is 0 if and only if it is not valid e.g. a dead entry in the flattened array of observations
        while (j < max_response_length && states[index] != 0) {
            size_t id = (*response_event_time_ids)[index];
            int current_state_index = states[index] - 1;     // states are always indexed by 1, 2, ... with 0 reserved for 'dead' entries in the flattened array
            int prev_state_index = states[index - 1] - 1;
            if (current_state_index != prev_state_index) {
                ++jumps[id * dim + prev_state_index * num_states + current_state_index];
            } 
            ++j;
            ++index;
        }
    }
    // compute the cumulative number of jumps
    cumulativeMatrixSums(num_jumps_acc, jumps, num_states);

    // now compute number at risk via the key decomposition
    for (size_t j = 1; j < num_unique_event_times; ++j) {   // j = 1 since we already computed I0 above
        vector<int> jump_contributions = columnSums(subtractMatrices(num_jumps_acc, j * dim, (j + 1) * dim - 1, transpose(num_jumps_acc, j * dim, (j + 1) * dim - 1)), static_cast<size_t>(num_states));
        for (size_t k = 0; k < num_states; ++k) {
            // key decomposition
            at_risk[j * num_states + k] = at_risk[k] - censoring_contribution[j * num_states + k] + jump_contributions[k];
        }
    }
}

// version of feb10 (makes pre-sweep and checks for invalid splits before computing quantities)
// for computing multi-state quantities (number at risk and number of jumps) for all splits in a node (for splits on continuous features)
void MultistateTree::computeMultistateQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                         vector<size_t>& num_at_risk_right, vector<size_t>& num_jumps_right, size_t nsplits_final) {
    // fetch states and other relevant data quantities
    const vector<uint8_t>& states = data->getStates();
    const vector<size_t>& last_observed_times = data->getLastObservedTimes();
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
                    size_t censoring_time_id = (*response_event_time_ids)[last_observed_times[i]];
                    for (size_t k = censoring_time_id; k < num_unique_event_times; ++k) {
                        ++censoring_contribution_right[s * num_unique_event_times * num_states + k * num_states + censoring_state - 1];
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
                    cout << "Warning: Key decomposition negative, causing underflow in num_at_risk_right" << endl;
                    cout << "Addition = " << addition << endl;
                    cout << "num_at_risk_0 = " << num_at_risk_right[split_stride + k] << endl;
                    cout << "Censoring contribution = " << censoring_contribution_right[split_stride + j * num_states + k] << endl;
                    cout << "Jump contribution = " << jump_contributions[k] << endl;
                }
                num_at_risk_right[split_stride + j * num_states + k] = num_at_risk_right[split_stride + k] - censoring_contribution_right[split_stride + j * num_states + k] + jump_contributions[k];
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
    vector<size_t> num_obs_right(nsplits_final);
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
    // compute the matrices of Nelson--Aalen estimators and the vector of predicted initial distributions
    computeNA(node_index);
    computeInitialDist(node_index);

    if (save_predictions) {
        computeCensoringKM(node_index);
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
    const vector<size_t>& current_node_obs = node_obs[node_index];

    // if no split is possible, make the node a leaf
    if (current_node_obs.size() < 2 * min_node_size) {
        if (!honest) {
            computeMultistateQuantities(current_node_obs, num_jumps, num_at_risk);               // for dishonest trees, use the growing indices
        } else {
            computeMultistateQuantities(holdout_node_obs[node_index], num_jumps, num_at_risk);   // for honest trees, use the holdout set for computing the CHF
        }

        // for debugging purposes
        
        size_t total_number_at_risk = 0;
        for (size_t j = 0; j < data->getNumberOfStates(); ++j) {
            total_number_at_risk += num_at_risk[j];
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
    init_dist.push_back(vector<double>());
    if (save_predictions) {
        KM_censoring.push_back(vector<double>());
        KM_censoring_full.push_back(vector<double>());
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
            }
            // the diagonal is minus the sum of all other row entries
            na[index + j] = diag;
        }
    }
    this->na.push_back(std::move(na));
}

void MultistateTree::computeCensoringKM(size_t node_index) {
    const vector<size_t>& indices = honest ? holdout_node_obs[node_index] : node_obs[node_index];
    computeCensoringKMExternal(indices, node_index);
}

void MultistateTree::computeCensoringKMExternal(const vector<size_t>& indices, size_t node_index) {
    // initialise and fetch data
    size_t num_obs = data->getNumberOfObs();
    vector<double> endpoint_times(num_obs, 0);
    vector<double> uncensored_indicators(num_obs, 0);
    const vector<double>& times = data->getTimes();
    const vector<size_t>& last_observed_times = data->getLastObservedTimes();
    const vector<uint8_t>& censoring_states = data->getCensoringStates();

    for (size_t i = 0; i < num_obs; ++i) {
        size_t last_observed_time_id = last_observed_times[i];
        endpoint_times[i] = times[last_observed_time_id];
        uncensored_indicators[i] = censoring_states[i] == 0 ? 1 : 0;
    }

    vector<double> KM_full = computeCensoringKMFromEndpoints(endpoint_times, uncensored_indicators, *censoring_times, indices);
    vector<double> KM_event = selectCensoringAtTimes(KM_full, *censoring_times, *unique_event_times, 1);
    if (node_index < KM_censoring.size()) {
        KM_censoring_full[node_index] = std::move(KM_full);
        KM_censoring[node_index] = std::move(KM_event);
    } else {
        KM_censoring_full.push_back(std::move(KM_full));
        KM_censoring.push_back(std::move(KM_event));
    }
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

/*

Computes all in-bag predictions, result is a vector with one, two or three flattened vectors
depending on the parameters. First two vectors are always in-bag Nelson-Aalen estimators, and the
same structure applies to the remaining output vectors. If compute_initial = true, the next vector is
in-bag predicted initial distributions, and if compute_censoring = true, the censoring predictions
are added to the result vector.

*/

vector<vector<double>> MultistateTree::computePredictions(bool compute_initial, bool compute_censoring, const Data& new_data) {
    size_t num_obs;
    bool new_data_provided;
    if (new_data.getNumberOfObs() == 0) {
        new_data_provided = false;
    } else {
        new_data_provided = true;
    }

    // if new data has not been provided, compute in-bag predictions
    if (new_data_provided) {
        num_obs = new_data.getNumberOfObs();
    } else {
        num_obs = data->getNumberOfObs();
    }
    uint8_t num_states = data->getNumberOfStates();
    size_t dim = num_states * num_states;

    // initialise vectors of predictions
    vector<double> predictions(num_obs * num_unique_event_times * num_states * num_states);
    vector<double> predictions_init(num_obs * num_states);
    vector<double> censoring(num_obs * num_censoring_times);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t leaf_id;
        if (new_data_provided) {
            leaf_id = predictionLeafID(new_data.get_x_row(i));
        } else {
            leaf_id = predictionLeafID(data->get_x_row(i));
        }
        const vector<double>& pred = na[leaf_id];
        vector<double> init;
        if (compute_initial) {
            init = init_dist[leaf_id];
        }
        vector<double> cens;
        if (compute_censoring) {
            cens = KM_censoring_full[leaf_id];
        }
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            // first save predicted NA estimators
            for (size_t j = 0; j < num_states; ++j) {
                for (size_t k = 0; k < num_states; ++k) {
                    predictions[i * num_unique_event_times * dim + t * dim + j * num_states + k] = pred[t * dim + j * num_states + k];
                }
            }
        }
        // save censoring distribution
        if (compute_censoring) {
            for (size_t t = 0; t < num_censoring_times; ++t) {
                censoring[i * num_censoring_times + t] = cens[t];
            }
        }
        // save initial distribution
        if (compute_initial) {
            for (size_t j = 0; j < num_states; ++j) {
                predictions_init[i * num_states + j] = init[j];
            }
        }
    }

    if (compute_initial && compute_censoring) {
        return {predictions, predictions_init, censoring};
    }
    else if (compute_initial) {
        return {predictions, predictions_init};
    }
    else if (compute_censoring) {
        return {predictions, censoring};
    } else {
        return {predictions};
    }
}

// error estimation for multi-state trees
//--------------------------------------------------------------------------------------

// computes a vector of the Brier score using given IPCW weights for multi-state predictions 
// (states_ind is a flattened vector of boolean indicators of whether observation i at event time t is in state j)
vector<double> computeBrierScoreMultistate(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times,
                                   const List& occupation_probs, const vector<double>& state_weights) {
    size_t num_unique_event_times = unique_event_times.size();
    uint8_t num_states = state_weights.size();
    size_t num_obs = weights.size() / num_unique_event_times;
    vector<double> brier(num_unique_event_times * num_states, 0);

    for (size_t i = 0; i < num_obs; ++i) {
        const List& occ_probs_obs = as<List>(occupation_probs[i]);
        size_t obs_index = i * num_unique_event_times * num_states;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            const vector<double> occ_probs_obs_t = as<vector<double>>(occ_probs_obs[t]);
            size_t time_index = obs_index + t * num_states;
            double ipcw = weights[i * num_unique_event_times + t];
            // difference to survival: need to compute a contribution to the score across all states
            for (size_t j = 0; j < num_states; ++j) {
                if (states_ind[time_index + j]) {
                    brier[t * num_states + j] += ipcw * (1 - occ_probs_obs_t[j]) * (1 - occ_probs_obs_t[j]) * state_weights[j];
                } else {
                    brier[t * num_states + j] += ipcw * occ_probs_obs_t[j] * occ_probs_obs_t[j] * state_weights[j];
                }
            }
        }
    }
    return brier;
}

// computes a vector of the Kullback-Leibler score using given IPCW weights for multi-state predictions
// (states_ind is a flattened vector of boolean indicators of whether observation i at event time t is in state j)
vector<double> computeKLScoreMultistate(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times,
                            const List& occupation_probs, const vector<double>& state_weights) {
    size_t num_unique_event_times = unique_event_times.size();
    uint8_t num_states = state_weights.size();
    size_t num_obs = weights.size() / num_unique_event_times;
    vector<double> kl(num_unique_event_times * num_states, 0);

    for (size_t i = 0; i < num_obs; ++i) {
        const List& occ_probs_obs = as<List>(occupation_probs[i]);
        size_t obs_index = i * num_unique_event_times * num_states;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            const vector<double> occ_probs_obs_t = as<vector<double>>(occ_probs_obs[t]);
            size_t time_index = obs_index + t * num_states;
            double ipcw = weights[i * num_unique_event_times + t];
            if (ipcw == 0) {
                continue;
            }
            for (size_t j = 0; j < num_states; ++j) {
                if (states_ind[time_index + j] && state_weights[j] != 0) {
                    double occupation_probability = probabilityForLogScore(occ_probs_obs_t[j]);
                    kl[t * num_states + j] -= ipcw * log(occupation_probability) * state_weights[j];
                }
            }
        }
    }
    return kl;
}

// same function as above but where the occupation probabilities are instead given by a flattened vector
vector<double> computeBrierScoreCppMultistate(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times,
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
                if (states_ind[time_index + j]) {
                    brier[t * num_states + j] += ipcw * (1 - occupation_probs[time_index + j]) * (1 - occupation_probs[time_index + j]) * state_weights[j];
                } else {
                    brier[t * num_states + j] += ipcw * occupation_probs[time_index + j] * occupation_probs[time_index + j] * state_weights[j];
                }
            }
        }
    }
    return brier;
}

// same function as above but where the occupation probabilities are instead given by a flattened vector
vector<double> computeKLScoreCppMultistate(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times,
                                   const vector<double>& occupation_probs, const vector<double>& state_weights) {
    size_t num_unique_event_times = unique_event_times.size();
    uint8_t num_states = state_weights.size();
    size_t num_obs = weights.size() / num_unique_event_times;
    vector<double> kl(num_unique_event_times * num_states, 0);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t obs_index = i * num_unique_event_times * num_states;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            size_t time_index = obs_index + t * num_states;
            double ipcw = weights[i * num_unique_event_times + t];
            if (ipcw == 0) {
                continue;
            }
            for (size_t j = 0; j < num_states; ++j) {
                if (states_ind[time_index + j] && state_weights[j] != 0) {
                    double occupation_probability = probabilityForLogScore(occupation_probs[time_index + j]);
                    kl[t * num_states + j] -= ipcw * log(occupation_probability) * state_weights[j];
                }
            }
        }
    }
    return kl;
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

vector<double> uniqueCensoringTimesMultistate(const vector<double>& unique_event_times, const vector<double>& times, const vector<size_t>& last_observed_times) {
    vector<double> result = unique_event_times;
    result.reserve(unique_event_times.size() + last_observed_times.size());
    for (size_t id : last_observed_times) {
        result.push_back(times[id]);
    }

    sort(result.begin(), result.end());
    auto last = unique(result.begin(), result.end());
    result.erase(last, result.end());
    return result;
}

/*
  Maps each response time to the fitted event-time grid. When the grid is thinned, several successive transitions can
  otherwise map to the same time. Keep the mapped transitions strictly ordered within each observation, and delay censoring
  until after the final mapped transition, so the binned paths remain valid paths.
*/
vector<size_t> computeResponseEventTimeIDsMultistate(const vector<double>& unique_event_times, const vector<double>& times,
                                                     const vector<uint8_t>& states, uint8_t max_response_length) {
    if (times.size() != states.size() || max_response_length == 0 || times.size() % max_response_length != 0) {
        throw invalid_argument("Invalid flattened multi-state response data");
    }
    if (unique_event_times.empty()) {
        throw invalid_argument("The multi-state event-time grid cannot be empty");
    }

    const size_t num_event_times = unique_event_times.size();
    const size_t num_obs = times.size() / max_response_length;
    vector<size_t> response_event_time_ids(times.size(), 0);

    for (size_t obs = 0; obs < num_obs; ++obs) {
        const size_t offset = obs * max_response_length;
        size_t response_length = 1;
        while (response_length < max_response_length && states[offset + response_length] != 0) {
            ++response_length;
        }

        vector<size_t> transition_positions;
        vector<size_t> mapped_transition_ids;
        transition_positions.reserve(response_length - 1);
        mapped_transition_ids.reserve(response_length - 1);

        for (size_t j = 1; j < response_length; ++j) {
            const size_t index = offset + j;
            if (states[index] == states[index - 1]) {
                continue;
            }

            auto it = lower_bound(unique_event_times.begin(), unique_event_times.end(), times[index]);
            size_t event_time_id = static_cast<size_t>(distance(unique_event_times.begin(), it));
            if (event_time_id >= num_event_times) {
                event_time_id = num_event_times - 1;
            }
            transition_positions.push_back(index);
            mapped_transition_ids.push_back(event_time_id);
        }

        const size_t num_transitions = mapped_transition_ids.size();
        if (num_transitions >= num_event_times) {
            throw invalid_argument("num_event_times is too small to preserve the order of a multi-state response path");
        }

        // Prefer the existing lower-bound mapping and only move a transition when
        // thinning would collapse it onto an earlier transition from the same path.
        for (size_t j = 0; j < num_transitions; ++j) {
            const size_t earliest_id = j == 0 ? 1 : mapped_transition_ids[j - 1] + 1;
            mapped_transition_ids[j] = max(mapped_transition_ids[j], earliest_id);
        }

        // Near the right edge there may not be room to move transitions forward.
        // Move the affected tail backwards while retaining strict ordering.
        if (num_transitions > 0 && mapped_transition_ids.back() >= num_event_times) {
            mapped_transition_ids.back() = num_event_times - 1;
            for (size_t j = num_transitions - 1; j > 0; --j) {
                mapped_transition_ids[j - 1] = min(mapped_transition_ids[j - 1], mapped_transition_ids[j] - 1);
            }
        }

        for (size_t j = 0; j < num_transitions; ++j) {
            response_event_time_ids[transition_positions[j]] = mapped_transition_ids[j];
        }

        // A repeated final state denotes censoring. Store the first grid index at
        // which censoring should remove the observation. num_event_times is a valid
        // sentinel meaning that censoring occurs beyond the fitted grid.
        if (response_length >= 2) {
            const size_t last_index = offset + response_length - 1;
            if (states[last_index] == states[last_index - 1]) {
                auto it = upper_bound(unique_event_times.begin(), unique_event_times.end(), times[last_index]);
                size_t censoring_time_id = static_cast<size_t>(distance(unique_event_times.begin(), it));
                if (num_transitions > 0) {
                    censoring_time_id = max(censoring_time_id, mapped_transition_ids.back() + 1);
                }
                response_event_time_ids[last_index] = censoring_time_id;
            }
        }
    }
    return response_event_time_ids;
}

// computes occupation probabilities from flattened Nelson-Aalen estimators and one initial distribution per estimator
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
        size_t init_index = i * num_states;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            for (size_t j = 0; j < num_states; ++j) {
                double init_val = init[init_index + j];
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

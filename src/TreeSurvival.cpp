/*

Functions for survival trees

*/

#include "TreeSurvival.h"

// constructor for SurvivalTree
//--------------------------------------------------------------------------------------

// the final argument is only used for honest trees
SurvivalTree::SurvivalTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids, shared_ptr<vector<size_t>> true_event_time_ids, const vector<size_t>& subset_indices, bool save_predictions, const vector<size_t>& estimation_indices) :
    unique_event_times {unique_event_times}, true_event_time_ids {true_event_time_ids}, response_event_time_ids {response_event_time_ids}, save_predictions {save_predictions} {
        this->node_obs.push_back(subset_indices);
        this->holdout_node_obs.push_back(estimation_indices);
        this->num_unique_event_times = unique_event_times->size();
        this->node_sizes.push_back(subset_indices.size());

        // initialise the vector of deaths and individuals at risk
        this->num_deaths.resize(num_unique_event_times);
        this->num_at_risk.resize(num_unique_event_times);
}

/*
SurvivalTree::SurvivalTree(const vector<double> unique_event_times, const vector<size_t>& response_event_time_ids, const vector<size_t>& true_event_time_ids, const vector<size_t> subset_indices) :
    unique_event_times {unique_event_times}, subset_indices {subset_indices}, true_event_time_ids {true_event_time_ids}, response_event_time_ids {response_event_time_ids} {
        this->node_obs.push_back(subset_indices);
        this->num_unique_event_times = unique_event_times.size();

        // initialise the vector of deaths and individuals at risk
        this->num_deaths.resize(num_unique_event_times);
        this->num_at_risk.resize(num_unique_event_times);
}
*/

// functions for growing survival trees
//--------------------------------------------------------------------------------------

// computes the number of deaths and the number at risk in each event time
// for the observations given by indices
void SurvivalTree::computeSurvivalQuantities(const vector<size_t>& indices, vector<size_t>& deaths, vector<size_t>& at_risk) {
    size_t n = indices.size();
    vector<double> times(n);
    vector<double> indicators(n);

    deaths.assign(num_unique_event_times, 0);
    at_risk.assign(num_unique_event_times, 0);

    // save times and indicators instead of fetching each time
    for (size_t i = 0; i < n; ++i) {
        times[i] = data->get_y(indices[i], 0);
        indicators[i] = data->get_y(indices[i], 1);
    }
    
    /*
    for (size_t i : indices) {
        double time = data->get_y(i, 0);
        
        size_t t = 0;
        while (t < num_unique_event_times && unique_event_times[t] < time) {
            ++at_risk[t];
            ++t;
        }

        if (t < num_unique_event_times) {
            ++at_risk[t];
            if (data->get_y(i, 1) == 1) {
                ++deaths[t];
            }
        }
    }
    */
    
    // sort the indices by their times
    vector<size_t> sorted_indices(n);
    iota(sorted_indices.begin(), sorted_indices.end(), 0);
    sort(sorted_indices.begin(), sorted_indices.end(),
        [&](size_t a, size_t b) {return times[a] < times[b]; });

    size_t obs_idx = 0;
    for (size_t t_idx = 0; t_idx < num_unique_event_times; ++t_idx) {
        double current_event_time = (*unique_event_times)[t_idx];

        // move pointer obs_idx to the first time larger than or equal to the current event time
        while (obs_idx < n && times[sorted_indices[obs_idx]] < current_event_time) {
            obs_idx++;
        }
        at_risk[t_idx] = n - obs_idx;

        // count the deaths
        size_t j = obs_idx;
        while (j < n && times[sorted_indices[j]] == current_event_time) {
            if (indicators[sorted_indices[j]] == 1) {
                deaths[t_idx]++;
            }
            j++;
        }
    }

    // sort indices based on the times
    /*
    vector<size_t> sorted_indices(n);
    iota(sorted_indices.begin(), sorted_indices.end(), 0);
    sort(sorted_indices.begin(), sorted_indices.end(),
        [&](size_t a, size_t b) {return times[a] < times[b]; });

    size_t obs_idx = 0;
    for (size_t t_idx = 0; t_idx < num_unique_event_times; ++t_idx) {
        double current_event_time = unique_event_times[t_idx];

        // move pointer obs_idx to the first time larger than or equal to the current event time
        while (obs_idx < n && times[sorted_indices[obs_idx]] < current_event_time) {
            obs_idx++;
        }
        at_risk[t_idx] = n - obs_idx;

        // count the deaths
        size_t j = obs_idx;
        while (j < n && times[sorted_indices[j]] == current_event_time) {
            if (indicators[sorted_indices[j]] == 1) {
                deaths[t_idx]++;
            }
            j++;
        }
    }
    */

    /*
    // compute number of deaths
    for (size_t i = 0; i < n; ++i) {
        if (indicators[i] == 1) {
            size_t pos = lower_bound(unique_event_times.begin(),
                                          unique_event_times.end(),
                                          times[i]) - unique_event_times.begin();
            if (pos < num_unique_event_times && unique_event_times[pos] == times[i]) {
                num_deaths[pos] += 1;
            }
        }
    }
    // compute number at risk
    size_t j = 0;  // index for the first observation larger than or equal to the event time
    for (size_t i = 0; i < num_unique_event_times; ++i) {
        double t = unique_event_times[i];
        while (j < n && times[sorted_indices[j]] < t) {
            ++j;
        }
        num_at_risk[i] = n - j;
    }
    */
}

/*
void SurvivalTree::computeSurvivalQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                                     vector<size_t>& num_at_risk_right, vector<size_t>& num_deaths_right, size_t nsplits_final) {
    const vector<size_t>& current_node_obs = node_obs[node_index];
    for (size_t i : current_node_obs) {
        double feature_val = data->get_x(i, feature);
        size_t time_id = (*response_event_time_ids)[i];

        for (size_t j = 0; j < nsplits_final; ++j) {
            if (feature_val > split_points[j]) {
                ++num_obs_right[j];
                ++num_at_risk_right[j * num_unique_event_times + time_id];

                if (data->get_y(i, 1) == 1) {
                    ++num_deaths_right[j * num_unique_event_times + time_id];
                }
            } else {
                break;
            }
        }
    }
    // compute number at risk in the right node (Gemini's approach, doesn't work)
    for (size_t i = 0; i < nsplits_final; ++i) {
        size_t index = i * num_unique_event_times;
        for (size_t j = num_unique_event_times - 2; j >= 0; --j) {
            num_at_risk_right[index + j] += num_at_risk_right[index + j + 1];
        }
    }
}
*/

// for computing survival quantities (number at risk and number of deaths) for all splits in a node (for splits on continuous features)
void SurvivalTree::computeSurvivalQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                                     vector<size_t>& num_at_risk_right, vector<size_t>& num_deaths_right, size_t nsplits_final) {
    
    vector<size_t> delta_num_at_risk_right(nsplits_final * num_unique_event_times);
    // counts number of deaths in right daughter at every event time for every possible split
    const vector<size_t>& current_node_obs = node_obs[node_index];
    for (size_t i : current_node_obs) {
        double feature_val = data->get_x(i, feature);
        size_t time_id = (*response_event_time_ids)[i];

        for (size_t j = 0; j < nsplits_final; ++j) {
            if (feature_val > split_points[j]) {
                ++num_obs_right[j];
                ++delta_num_at_risk_right[j * num_unique_event_times + time_id];
                //Rcout << "Line 152: death? " << data->get_y(i, 1) << endl;
                if (data->get_y(i, 1) == 1) {
                    ++num_deaths_right[j * num_unique_event_times + time_id];
                }
            } else {
                break;  // since the split_points are sorted
            }
        }
    }
    // compute number at risk in the right node
    for (size_t i = 0; i < nsplits_final; ++i) {
        size_t total_events = 0;
        for (size_t j = 0; j < num_unique_event_times; ++j) {
            num_at_risk_right[i * num_unique_event_times + j] = num_obs_right[i] - total_events;
            total_events += delta_num_at_risk_right[i * num_unique_event_times + j];
        }
    }
}

void SurvivalTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold) {
    const vector<size_t>& current_node_obs = node_obs[node_index];

    // samples and sorts split points
    vector<double> split_points;
    size_t nsplits_final = sampleSplitPoints(split_points, current_node_obs, feature);

    // no possible splits
    if (nsplits_final == 0) {
        return;
    }

    // initialise node info for the right daughter as flattened 2D arrays
    vector<size_t> num_deaths_right(nsplits_final * num_unique_event_times);
    vector<size_t> num_obs_right(nsplits_final);
    vector<size_t> num_at_risk_right(nsplits_final * num_unique_event_times);

    computeSurvivalQuantitiesDaughter(node_index, feature, split_points, num_obs_right, num_at_risk_right, num_deaths_right, nsplits_final);

    // now determine the best split
    for (size_t i = 0; i < nsplits_final; ++i) {

        // if a node is too small, skip the split
        size_t num_obs_left = current_node_obs.size() - num_obs_right[i];
        if (num_obs_left < min_node_size || num_obs_right[i] < min_node_size) {
            continue;
        }

        double split_val;
        if (splitrule == "logrank") {
            split_val = logRank(num_deaths, num_at_risk, num_deaths_right, num_at_risk_right, i);
        }
        if (splitrule == "conserve") {
            split_val = conserve(num_deaths, num_at_risk, num_deaths_right, num_at_risk_right, i);
        }
        if (splitrule == "approxlogrank") {
            split_val = approxLogRank(num_deaths, num_at_risk, num_deaths_right, num_at_risk_right, i);
        }
        //Rcout << "Line 193: log-rank value: " << split_val << endl;
        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_feature = feature;
            // use average of split points unless it is the final split_value
            if (i == nsplits_final - 1) {
                best_threshold = {split_points[i]};
            } else {
                best_threshold = {(split_points[i] + split_points[i + 1]) / 2.0};
            }
        }
    }
}

void SurvivalTree::bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature,
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
        vector<size_t> num_deaths_left, num_at_risk_left;
        computeSurvivalQuantities(current_left_indices, num_deaths_left, num_at_risk_left);
        double split_val;
        if (splitrule == "logrank") {
            split_val = logRank(num_deaths, num_at_risk, num_deaths_left, num_at_risk_left);
        }
        if (splitrule == "conserve") {
            split_val = conserve(num_deaths, num_at_risk, num_deaths_left, num_at_risk_left);
        }
        if (splitrule == "approxlogrank") {
            split_val = approxLogRank(num_deaths, num_at_risk, num_deaths_left, num_at_risk_left);
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

void SurvivalTree::makeLeaf(size_t node_index) {
    // compute the chf
    computeChf(node_index);

    // if we save predictions, we also compute the KM estimator of the censoring distribution
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
}

// function to create a split for a survival tree. returns true if leaf, otherwise false
bool SurvivalTree::createSplit(size_t node_index) {
    const vector<size_t>& current_node_obs = node_obs[node_index];

    // if no split is possible, make the node a leaf
    if (current_node_obs.size() < 2 * min_node_size) {
        if (!honest) {
            computeSurvivalQuantities(current_node_obs, num_deaths, num_at_risk);               // for dishonest trees, use the growing indices
        } else {
            computeSurvivalQuantities(holdout_node_obs[node_index], num_deaths, num_at_risk);   // for honest trees, use the holdout set for computing the CHF
        }
        makeLeaf(node_index);
        return true;
    }

    computeSurvivalQuantities(current_node_obs, num_deaths, num_at_risk);   // update parent survival info

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
            computeSurvivalQuantities(holdout_node_obs[node_index], num_deaths, num_at_risk);   // for honest trees, use the holdout set for computing the CHF
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
    chf.push_back(vector<double>());
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

// computes the cumulative hazard estimate in node node_index
void SurvivalTree::computeChf(size_t node_index) {
    vector<double>chf = vector<double>(num_unique_event_times, 0);
    if (num_at_risk[0] != 0) {
        chf[0] = double(num_deaths[0])/double(num_at_risk[0]);
    }
    for (size_t i = 1; i < num_unique_event_times; ++i) {
        if (num_at_risk[i] != 0) {
            chf[i] = chf[i - 1] + double(num_deaths[i]) / double(num_at_risk[i]);
        }
        else {
            chf[i] = chf[i - 1];
        }
    }
    this->chf.push_back(std::move(chf));
}

// computes the Kaplan-Meier estimator for the censoring distribution in node node_index
void SurvivalTree::computeCensoringKM(size_t node_index) {
    vector<double>KM_censoring = vector<double>(num_unique_event_times, 1);
    
    // now compute the Kaplan-Meier estimator
    double num_censored;
    for (size_t i = 1; i < num_unique_event_times; ++i) {
        if (num_at_risk[i] - num_deaths[i] > 0) {
            num_censored = num_at_risk[i] - num_at_risk[i + 1] - num_deaths[i];
            KM_censoring[i] = KM_censoring[i - 1] * (1 - double(num_censored) / double(num_at_risk[i] - num_deaths[i]));    // may be other ways of handling ties
        } else {
            KM_censoring[i] = KM_censoring[i - 1];
        }
    }
    // for debugging
    cout << "KM: ";
    printVector(KM_censoring);
    this->KM_censoring.push_back(std::move(KM_censoring));
}

/*
// function to grow a survival tree
void SurvivalTree::grow() {
  // maybe bootstrap weights should be here if we choose to implement general bootstrap schemes

  size_t num_queue = 1;
  size_t depth = 0;
  size_t left_most_node = 0;

  // while not all nodes terminal, continue growing the tree
  size_t i = 0;
  depths.push_back(depth);
  while (num_queue > 0) {
    //Rcout << "Line 406: Current node ID:" << i << endl;
    bool is_leaf = createSplit(i);
    if (is_leaf) {
        num_queue--;
        left_daughters.push_back(0);  // 0 indicates no daughters
        num_terminal_nodes++;
    }
    else {
        num_queue++;
        left_daughters.push_back(num_nodes);
        num_nodes += 2;
        if (i >= left_most_node) {
            depth++;
            left_most_node = num_nodes - 2;
        }
        // only for info, technically redundant
        depths.push_back(depth);
        depths.push_back(depth);
    }
    i++;
  }

  tree_depth = depth;
  cleanUpTree();
}
*/

// splitting rules for survival trees
//--------------------------------------------------------------------------------------

// split_id = 0 for categorical splits
double SurvivalTree::logRank(const vector<size_t>& num_deaths, const vector<size_t>& num_at_risk, 
                             const vector<size_t>& num_deaths_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id) {
    double sum_num = 0;
    double sum_den = 0;
    size_t array_index = split_id * num_unique_event_times;
    for (size_t i = 0; i < num_unique_event_times; ++i) {
        const double d = (double) num_deaths[i];
        const double d1 = (double) num_deaths_daughter[array_index + i];  // we use a single sweep which considers every split at once, hence the flattened array
        const double Y = (double) num_at_risk[i];
        const double Y1 = (double) num_at_risk_daughter[array_index + i];

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

    if (sum_den != 0) {
        // return the squared log-rank test
        return(sum_num * sum_num / sum_den);
    } else {
        return -1;
    }
}

double SurvivalTree::conserve(const vector<size_t>& num_deaths, const vector<size_t>& num_at_risk, 
                              const vector<size_t>& num_deaths_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id) {
    vector<double> NAsum1(num_unique_event_times, 0);
    vector<double> NAsum2(num_unique_event_times, 0);
    vector<size_t> num_deaths_daughter_2(num_unique_event_times);
    vector<size_t> num_at_risk_daughter_2(num_unique_event_times);

    // compute number at risk and number of deaths in the second daughter
    size_t array_index = split_id * num_unique_event_times;
    num_deaths_daughter_2[0] = num_deaths[0] - num_deaths_daughter[array_index];
    num_at_risk_daughter_2[0] = num_at_risk[0] - num_at_risk_daughter[array_index];
    for (size_t i = 1; i < num_unique_event_times; ++i) {
        num_deaths_daughter_2[i] = num_deaths[i] - num_deaths_daughter[array_index + i];
        num_at_risk_daughter_2[i] = num_at_risk[i] - num_at_risk_daughter[array_index + i];
    }

    // compute vectors containing the innermost sum in the approximation used by Ishwaran and Kogalur
    if (num_deaths_daughter[array_index] > 0) {
        NAsum1[0] = (double) num_deaths_daughter[array_index]/num_at_risk_daughter[array_index];
    }
    if (num_deaths_daughter_2[0] > 0) {
        NAsum2[0] = (double) num_deaths_daughter_2[0]/num_at_risk_daughter_2[0];
    }

    for (size_t i = 1; i < num_unique_event_times - 1; ++i) {
        if (num_deaths_daughter[array_index + i] > 0) {
            NAsum1[i] = NAsum1[i - 1] + (double) num_deaths_daughter[array_index + i]/num_at_risk_daughter[array_index + i];
        } else {
            NAsum1[i] = NAsum1[i - 1];
        }
        if (num_deaths_daughter_2[i] > 0) {
            NAsum2[i] = NAsum2[i - 1] + (double) num_deaths_daughter_2[i]/num_at_risk_daughter_2[i];
        } else {
            NAsum2[i] = NAsum2[i - 1];
        }
    }

    // now compute the sums running from 1 to N - 1 for each daughter
    double sum1 = 0;
    double sum2 = 0;
    for (size_t i = 0; i < num_unique_event_times - 1; ++i) {
        sum1 += (num_at_risk_daughter[array_index + i] - num_at_risk_daughter[array_index + i + 1]) * num_at_risk_daughter[array_index + i + 1] * NAsum1[i];
        sum2 += (num_at_risk_daughter_2[i] - num_at_risk_daughter_2[i + 1]) * num_at_risk_daughter_2[i + 1] * NAsum2[i];
    }

    double cons = (num_at_risk_daughter[0] * sum1 + num_at_risk_daughter_2[0] * sum2) / num_at_risk[0];
    return 1/(1 + cons);
}

double SurvivalTree::approxLogRank(const vector<size_t>& num_deaths, const vector<size_t>& num_at_risk, 
                                   const vector<size_t>& num_deaths_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id) {
    double D = (double) vector_sum(num_deaths);
    double D1 = (double) vector_sum(num_deaths_daughter);
    
    // compute the sum in the numerator
    double sum_num = 0;
    size_t array_index = split_id * num_unique_event_times;
    for (size_t i = 0; i < num_unique_event_times; ++i) {
        const double d = (double) num_deaths[i];
        const double d1 = (double) num_deaths_daughter[array_index + i];  // we use a single sweep which considers every split at once, hence the flattened array
        const double Y = (double) num_at_risk[i];
        const double Y1 = (double) num_at_risk_daughter[array_index + i];
        sum_num += d1 - Y1 * d / Y;
    }
    double den = sqrt((D1 - sum_num) * (D - D1 + sum_num));
    if (den != 0) {
        return abs((sqrt(D) * sum_num) / den);
    } else {
        return -1;
    }
}

// prediction for survival trees
//--------------------------------------------------------------------------------------

// compute a flattened vector of predictions
vector<double> SurvivalTree::computePredictions(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);

    for (size_t i = 0; i < num_obs; ++i) {
        const vector<double>& pred = get<vector<double>>(predict(new_data.get_x_row(i)));
        for (size_t j = 0; j < num_unique_event_times; ++j) {
            predictions[i * num_unique_event_times + j] = pred[j];
        }
    }
    return predictions;
}

// computes a pair of flattened vectors, the first predictions and the second the censoring KM estimators
pair<vector<double>, vector<double>> SurvivalTree::computePredictionsCensoring(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);
    vector<double> censoring(num_obs * num_unique_event_times);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t leaf_id = predictionLeafID(new_data.get_x_row(i));
        const vector<double>& pred = chf[leaf_id];
        const vector<double>& cens = KM_censoring[leaf_id];
        for (size_t j = 0; j < num_unique_event_times; ++j) {
            predictions[i * num_unique_event_times + j] = pred[j];
            censoring[i * num_unique_event_times + j] = cens[j];
        }
    }
    return {predictions, censoring};
}

// error estimation for survival trees
//--------------------------------------------------------------------------------------

// computes the outcomes used in the concordance index calculations using a NumericMatrix as input
vector<double> computeOutcomes(const NumericMatrix& predictions) {
    size_t n = predictions.nrow();
    size_t N = predictions.ncol();
    vector<double> outcomes(n);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0;
        for (size_t j = 0; j < N; ++j) {
            sum += predictions(i, j);
        }
        outcomes[i] = sum;
    }
    return outcomes;
}

// computes the outcomes used in the concordance index calculations using a (flattened) vector as input
vector<double> computeOutcomes(const vector<double>& predictions, size_t num_unique_event_times) {
    size_t n = predictions.size() / num_unique_event_times;
    vector<double> outcomes(n);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0;
        for (size_t j = 0; j < num_unique_event_times; ++j) {
            sum += predictions[i * num_unique_event_times + j];
        }
        outcomes[i] = sum;
    }
    return outcomes;
}

// miscellaneous functions related to survival
//--------------------------------------------------------------------------------------

// for computing the ids in the observed times corresponding to the unique event times (including censored times)
vector<size_t> computeResponseEventTimeIDs(const vector<double>& unique_event_times, const vector<double>& times) {
    vector<size_t> response_event_time_ids;
    response_event_time_ids.reserve(times.size());
    for (const double& time : times) {
        // use binary search to find lower bound
        auto it = lower_bound(unique_event_times.begin(), unique_event_times.end(), time);
        response_event_time_ids.push_back(static_cast<size_t>(distance(unique_event_times.begin(), it)));
    }
    return(response_event_time_ids);
}

// for computing the indices of the unique event times where times with only censored observations are removed
vector<size_t> computeTrueEventTimeIDs(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, const vector<double>& ind) {
    // start by summing over the indices belonging to the same event times
    vector<double> sum_ind;
    size_t num_unique_event_times = unique_event_times.size();
    sum_ind.assign(num_unique_event_times, 0);
    for (size_t i = 0; i < ind.size(); ++i) {
        sum_ind[response_event_time_ids[i]] += ind[i];
    }
    // now select the indices for the times to keep
    vector<size_t> final_time_indices;
    for (size_t i = 0; i < num_unique_event_times; ++i) {
        if (sum_ind[i] > 0) {
            final_time_indices.push_back(i);
        }
    }
    return final_time_indices;
}

// NB: this function is no longer used since we don't remove times without an observed event
vector<double> computeUniqueEventTimes(const vector<double>& times, const vector<size_t>& ind) {
    // remove censored times 
    vector<double> observed_times;
    for (size_t i = 0; i < times.size(); ++i) {
      if (ind[i] == 1) {
        observed_times.push_back(times[i]);
      }
    }
    
    // sort and remove duplicated observed times
    sort(observed_times.begin(), observed_times.end());
    /*
    observed_times.push_back(-1); // to ensure the last observed time is included
    for (int i = 0; i < observed_times.size() - 1; ++i) {
      if (observed_times[i] != observed_times[i + 1]) {
        unique_event_times.push_back(observed_times[i]);
      }
    }
    return(unique_event_times);
    */
    observed_times.erase(unique(observed_times.begin(), observed_times.end()), observed_times.end());
    return observed_times; 
}

// computes the Kaplan-Meier estimator given a Nelson-Aalen estimator
vector<double> KaplanMeier(const vector<double>& na) {
    vector<double> KM = vector<double>(na.size() + 1, 1);
    KM[0] = 1;

    // compute using the recursion given by the product integral
    for (size_t i = 1; i < na.size() + 1; ++i) {
        KM[i] = KM[i - 1] * (1 - (na[i] - na[i - 1]));
    }
    return KM;
}

// computes Harrell's C-index for a list of outcomes and survival data
double computeConcordanceIndex(const vector<double>& outcomes, const vector<double>& times, const vector<double>& ind) {
    double concordance = 0;
    double permissible = 0;
    size_t n = times.size();

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            // if T_i < T_j and delta_i = 0, the pair is not comparable
            if (times[i] < times[j] && ind[i] == 0) {
                continue;
            }
            // similarly with i and j reversed
            if (times[j] < times[i] && ind[j] == 0) {
                continue;
            }
            // if T_i = T_j and delta_i = delta_j, the pair is also considered incomparable
            if (ind[i] == ind[j] && times[i] == times[j]) {
                continue;
            }

            // if T_i < T_j and outcomes[i] > outcomes[j], the model predicts correctly
            // (similarly with i and j reversed), so add 1 to concordance
            // if the outcomes are identical, the model is indecisive so add 0.5 to concordance
            if (times[i] < times[j] && outcomes[i] > outcomes[j]) {
                concordance += 1;
            }
            else if (times[j] < times[i] && outcomes[j] > outcomes[i]) {
                concordance += 1;
            }
            else if (outcomes[i] == outcomes[j]) {
                concordance += 0.5;
            }
            permissible += 1;
        }
    }

    return(concordance/permissible);
}
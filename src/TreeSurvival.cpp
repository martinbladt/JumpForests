/*

Functions for survival trees

*/

#include "TreeSurvival.h"

#include <unordered_map>

// constructor for SurvivalTree
//--------------------------------------------------------------------------------------

// estimation_indices are only used for honest trees
SurvivalTree::SurvivalTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids, shared_ptr<vector<size_t>> true_event_time_ids,
                           vector<size_t> subset_indices, bool save_predictions, vector<size_t> estimation_indices,
                           shared_ptr<vector<double>> censoring_times, shared_ptr<vector<size_t>> removal_time_ids) :
    unique_event_times {unique_event_times}, true_event_time_ids {true_event_time_ids}, response_event_time_ids {response_event_time_ids},
    removal_time_ids {removal_time_ids}, save_predictions {save_predictions} {
        this->node_sizes.push_back(subset_indices.size());
        this->node_obs.push_back(std::move(subset_indices));
        this->holdout_node_obs.push_back(std::move(estimation_indices));
        this->num_unique_event_times = unique_event_times->size();
        this->censoring_times = censoring_times == nullptr ? unique_event_times : censoring_times;
        this->num_censoring_times = this->censoring_times->size();

        // initialise the vector of deaths and individuals at risk
        this->num_deaths.resize(num_unique_event_times);
        this->num_at_risk.resize(num_unique_event_times);
}

// functions for growing survival trees
//--------------------------------------------------------------------------------------

// computes the vector of IDs for the event time where an observation is no longer at risk
vector<size_t> computeRemovalTimeIDs(const Data& data, const vector<double>& unique_event_times,
                                     const vector<size_t>& response_event_time_ids) {
    vector<size_t> removal_time_ids(data.getNumberOfObs());
    for (size_t i = 0; i < data.getNumberOfObs(); ++i) {
        if (data.get_y(i, 1) == 1) {
            // an observation experiencing the event is still at risk at the event time
            removal_time_ids[i] = response_event_time_ids[i] + 1;
        } else {
            // censoring occurs after the last retained time which is smaller than or equal to the censoring time
            auto it = upper_bound(unique_event_times.begin(), unique_event_times.end(), data.get_y(i, 0));
            removal_time_ids[i] = static_cast<size_t>(distance(unique_event_times.begin(), it));
        }
    }
    return removal_time_ids;
}

void SurvivalTree::reserveTreeMemory(size_t num_obs) {
    // use the minimal node size to estimate the number of nodes in the tree
    size_t max_terminal_nodes = min_node_size == 0 ? max(static_cast<size_t>(1), num_obs) :
      max(static_cast<size_t>(1), num_obs / min_node_size);
    size_t max_num_nodes = 2 * max_terminal_nodes - 1;
    // avoid excessive reservation for unusually large shallow trees
    max_num_nodes = min(max_num_nodes, static_cast<size_t>(1024));

    node_obs.reserve(max_num_nodes);
    if (honest) {
        holdout_node_obs.reserve(max_num_nodes);
    }
    node_sizes.reserve(max_num_nodes);
    left_daughters.reserve(max_num_nodes);
    feature_IDs.reserve(max_num_nodes);
    thresholds.reserve(max_num_nodes);
    depths.reserve(max_num_nodes);
    chf.reserve(max_num_nodes);
    chf_outcomes.reserve(max_num_nodes);
    if (save_predictions) {
        KM_censoring.reserve(max_num_nodes);
        KM_censoring_full.reserve(max_num_nodes);
    }
}

// computes the vector of the unique event time ID where an observation is no longer at risk if this is not already done
void SurvivalTree::prepareSurvivalData() {
    if (removal_time_ids == nullptr) {
        removal_time_ids = make_shared<vector<size_t>>(
          computeRemovalTimeIDs(*data, *unique_event_times, *response_event_time_ids));
    }
}

// computes the number of deaths and the number at risk in each event time
// for the observations given by indices
void SurvivalTree::computeSurvivalQuantities(const vector<size_t>& indices, vector<size_t>& deaths, vector<size_t>& at_risk) {
    deaths.assign(num_unique_event_times, 0);
    at_risk.assign(num_unique_event_times, 0);

    // keeps track of how many individuals are removed from the at risk set at each unique event time
    // the last element is for observations which remain at risk throughout the event-time grid
    num_removed.assign(num_unique_event_times + 1, 0);

    for (size_t i : indices) {
        if (data->get_y(i, 1) == 1) {
            size_t event_time_id = (*response_event_time_ids)[i];
            ++deaths[event_time_id];
        }
        ++num_removed[(*removal_time_ids)[i]];
    }

    // can now compute the at risk vector residually using the number of removed individuals
    size_t current_num_at_risk = indices.size();
    for (size_t t = 0; t < num_unique_event_times; ++t) {
        current_num_at_risk -= num_removed[t];
        at_risk[t] = current_num_at_risk;
    }
}

// for computing survival quantities (number at risk and number of deaths) for all splits in a node (for splits on continuous features)
void SurvivalTree::computeSurvivalQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, size_t nsplits_final) {
    size_t removal_stride = num_unique_event_times + 1;
    num_obs_right.assign(nsplits_final, 0);
    num_removed_right.assign(nsplits_final * removal_stride, 0);
    num_deaths_right.assign(nsplits_final * num_unique_event_times, 0);
    num_at_risk_right.assign(nsplits_final * num_unique_event_times, 0);

    // first place each observation at the last split point where it belongs to the right node
    const vector<size_t>& current_node_obs = node_obs[node_index];
    for (size_t i : current_node_obs) {
        double feature_val = data->get_x(i, feature);
        // missing feature values are sent to the right daughter during prediction
        size_t num_splits_right = std::isnan(feature_val) ? nsplits_final : lower_bound(split_points.begin(), split_points.end(), feature_val) - split_points.begin();
        if (num_splits_right > 0) {
            size_t split_id = num_splits_right - 1;
            ++num_obs_right[split_id];
            ++num_removed_right[split_id * removal_stride + (*removal_time_ids)[i]];
            if (data->get_y(i, 1) == 1) {
                ++num_deaths_right[split_id * num_unique_event_times + (*response_event_time_ids)[i]];
            }
        }
    }

    // accumulate backwards since an observation to the right of one split point
    // also belongs to the right of every smaller split point
    for (size_t s = nsplits_final - 1; s > 0; --s) {
        num_obs_right[s - 1] += num_obs_right[s];
        size_t index = s * num_unique_event_times;
        size_t previous_index = index - num_unique_event_times;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            num_deaths_right[previous_index + t] += num_deaths_right[index + t];
        }
        index = s * removal_stride;
        previous_index = index - removal_stride;
        for (size_t t = 0; t < removal_stride; ++t) {
            num_removed_right[previous_index + t] += num_removed_right[index + t];
        }
    }

    // compute number at risk in the right node
    for (size_t s = 0; s < nsplits_final; ++s) {
        size_t current_num_at_risk = num_obs_right[s];
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            current_num_at_risk -= num_removed_right[s * removal_stride + t];
            num_at_risk_right[s * num_unique_event_times + t] = current_num_at_risk;
        }
    }
}

double SurvivalTree::computeSplitValue(const vector<size_t>& deaths_daughter, const vector<size_t>& at_risk_daughter, size_t split_id) {
    switch (splitrule_id) {
        case SurvivalSplitRule::LogRank:
            return logRank(num_deaths, num_at_risk, deaths_daughter, at_risk_daughter, split_id);
        case SurvivalSplitRule::Conserve:
            return conserve(num_deaths, num_at_risk, deaths_daughter, at_risk_daughter, split_id);
        case SurvivalSplitRule::ApproxLogRank:
            return approxLogRank(num_deaths, num_at_risk, deaths_daughter, at_risk_daughter, split_id);
    }
    return -1;
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

    computeSurvivalQuantitiesDaughter(node_index, feature, split_points, nsplits_final);

    // now determine the best split
    for (size_t i = 0; i < nsplits_final; ++i) {

        // if a node is too small, skip the split
        size_t num_obs_left = current_node_obs.size() - num_obs_right[i];
        if (num_obs_left < min_node_size || num_obs_right[i] < min_node_size) {
            continue;
        }

        double split_val = computeSplitValue(num_deaths_right, num_at_risk_right, i);
        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_feature = feature;
            best_threshold = {split_points[i]};
        }
    }
}

void SurvivalTree::bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature,
                                        vector<double>& best_threshold) {
    const vector<size_t>& current_node_obs = node_obs[node_index];
    vector<double> feature_values = data->getValues(current_node_obs, feature);
    feature_values.erase(remove_if(feature_values.begin(), feature_values.end(),
      [](double value) { return std::isnan(value); }), feature_values.end());
    feature_values = uniqueValues(std::move(feature_values));
    size_t num_feature_values = feature_values.size();

    unordered_set<uint64_t> partition_masks;
    // generate partitions (breaks if no possible splits)
    if (generateCategoricalPartitions(feature_values, partition_masks)) {
        return;
    }

    // compute survival quantities for each category once
    size_t removal_stride = num_unique_event_times + 1;
    vector<size_t> category_counts(num_feature_values, 0);
    vector<size_t> category_removed(num_feature_values * removal_stride, 0);
    vector<size_t> category_deaths(num_feature_values * num_unique_event_times, 0);
    vector<size_t> category_at_risk(num_feature_values * num_unique_event_times, 0);
    for (size_t obs_id : current_node_obs) {
        double feature_value = data->get_x(obs_id, feature);
        // missing feature values are sent to the right daughter during prediction
        size_t category = std::isnan(feature_value) ? num_feature_values : lower_bound(feature_values.begin(), feature_values.end(), feature_value) - feature_values.begin();
        if (category < num_feature_values) {
            ++category_counts[category];
            ++category_removed[category * removal_stride + (*removal_time_ids)[obs_id]];
            if (data->get_y(obs_id, 1) == 1) {
                ++category_deaths[category * num_unique_event_times + (*response_event_time_ids)[obs_id]];
            }
        }
    }

    for (size_t category = 0; category < num_feature_values; ++category) {
        size_t current_num_at_risk = category_counts[category];
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            current_num_at_risk -= category_removed[category * removal_stride + t];
            category_at_risk[category * num_unique_event_times + t] = current_num_at_risk;
        }
    }

    vector<size_t> num_deaths_left(num_unique_event_times, 0);
    vector<size_t> num_at_risk_left(num_unique_event_times, 0);

    // consider each partition (bitmask)
    for (const auto& mask : partition_masks) {
        size_t num_obs_left = 0;
        fill(num_deaths_left.begin(), num_deaths_left.end(), 0);
        fill(num_at_risk_left.begin(), num_at_risk_left.end(), 0);
        for (size_t category = 0; category < num_feature_values; ++category) {
            if ((mask >> category) & 1) {
                num_obs_left += category_counts[category];
                size_t index = category * num_unique_event_times;
                for (size_t t = 0; t < num_unique_event_times; ++t) {
                    num_deaths_left[t] += category_deaths[index + t];
                    num_at_risk_left[t] += category_at_risk[index + t];
                }
            }
        }

        size_t num_obs_right = current_node_obs.size() - num_obs_left;
        if (num_obs_left < min_node_size || num_obs_right < min_node_size) {
            continue;
        }

        double split_val = computeSplitValue(num_deaths_left, num_at_risk_left);

        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_feature = feature;
            unordered_set<double> left_values;
            for (size_t i = 0; i < num_feature_values; ++i) {
                if ((mask >> i) & 1) {
                    left_values.insert(feature_values[i]);
                }
            }
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
    if (honest) {
        for (size_t i : holdout_node_obs[node_index]) {
            prediction_node_IDs[i] = node_index;
        }
    }

    // the observation indices are no longer needed after the node is made terminal
    vector<size_t>().swap(node_obs[node_index]);
    if (honest) {
        vector<size_t>().swap(holdout_node_obs[node_index]);
    }
}

// function to create a split for a survival tree, returns true if leaf, otherwise false
bool SurvivalTree::createSplit(size_t node_index) {
    // initialise quantities which are reused throughout the tree
    if (feature_indices.empty()) {
        reserveTreeMemory(node_obs[node_index].size());
        prepareSurvivalData();

        // retain the correct estimation sample if censoring estimators are requested later
        if (!save_predictions) {
            censoring_indices = honest ? holdout_node_obs[node_index] : node_obs[node_index];
        }

        size_t num_features = data->getNumberOfFeatures();
        feature_indices.resize(num_features);
        for (size_t i = 0; i < num_features; ++i) {
            feature_indices[i] = i;
        }

        if (splitrule == "conserve") {
            splitrule_id = SurvivalSplitRule::Conserve;
        } else if (splitrule == "approxlogrank") {
            splitrule_id = SurvivalSplitRule::ApproxLogRank;
        } else {
            splitrule_id = SurvivalSplitRule::LogRank;
        }
    }

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
    vector<size_t> sampled_features = sampleIndices(feature_indices, mtry, false, random_number_generator);
    const vector<bool>& categorical = data->getCategorical();

    // now consider each of the sampled features
    for (size_t i : sampled_features) {
        if (categorical[i]) {
            bestSplitCategorical(node_index, i, best_split_val, best_feature, best_threshold);
        }
        else {
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

    // construct the indices of the two daughters once for the best split
    bool categorical_split = categorical[best_feature];
    best_left_indices.reserve(current_node_obs.size());
    best_right_indices.reserve(current_node_obs.size());
    for (size_t i : current_node_obs) {
        double feature_value = data->get_x(i, best_feature);
        // if the split is categorical, the observation goes to the left node if it is in the threshold set
        bool goes_left = categorical_split ? find(best_threshold.begin(), best_threshold.end(), feature_value) != best_threshold.end() : feature_value <= best_threshold[0];
        if (goes_left) {
            best_left_indices.push_back(i);
        } else {
            best_right_indices.push_back(i);
        }
    }

    // update the holdout index sets if the tree is honest
    if (honest) {
        const vector<size_t>& current_holdout_node_obs = holdout_node_obs[node_index];
        holdout_left_indices.reserve(current_holdout_node_obs.size());
        holdout_right_indices.reserve(current_holdout_node_obs.size());
        for (size_t i : current_holdout_node_obs) {
            double feature_value = data->get_x(i, best_feature);
            bool goes_left = categorical_split ?
              find(best_threshold.begin(), best_threshold.end(), feature_value) != best_threshold.end() :
              feature_value <= best_threshold[0];
            if (goes_left) {
                holdout_left_indices.push_back(i);
            } else {
                holdout_right_indices.push_back(i);
            }
        }
    }
    
    // a best split was found, update the tree
    node_sizes.push_back(best_left_indices.size());
    node_sizes.push_back(best_right_indices.size());
    node_obs.push_back(std::move(best_left_indices));          // construct left daughter
    node_obs.push_back(std::move(best_right_indices));         // construct right daughter
    feature_IDs.push_back(best_feature);
    thresholds.push_back(std::move(best_threshold));
    chf.push_back(vector<double>());
    chf_outcomes.push_back(0);
    if (save_predictions) {
        KM_censoring.push_back(vector<double>());
        KM_censoring_full.push_back(vector<double>());
    }

    // for honest trees, update the holdout indices
    if (honest) {
        holdout_node_obs.push_back(std::move(holdout_left_indices));
        holdout_node_obs.push_back(std::move(holdout_right_indices));
    }

    // the observation indices in the parent node are no longer needed
    vector<size_t>().swap(node_obs[node_index]);
    if (honest) {
        vector<size_t>().swap(holdout_node_obs[node_index]);
    }
    return false;
}

// computes the cumulative hazard estimate in node node_index
void SurvivalTree::computeChf(size_t node_index) {
    vector<double>chf = vector<double>(num_unique_event_times, 0);
    double chf_outcome = 0;
    if (num_at_risk[0] != 0) {
        chf[0] = double(num_deaths[0])/double(num_at_risk[0]);
    }
    chf_outcome += chf[0];
    for (size_t i = 1; i < num_unique_event_times; ++i) {
        if (num_at_risk[i] != 0) {
            chf[i] = chf[i - 1] + double(num_deaths[i]) / double(num_at_risk[i]);
        }
        else {
            chf[i] = chf[i - 1];
        }
        chf_outcome += chf[i];
    }
    chf_outcomes.push_back(chf_outcome);
    this->chf.push_back(std::move(chf));
}

void SurvivalTree::prepareSurvivalProbabilities() {
    if (!survival_probabilities.empty()) {
        return;
    }

    survival_probabilities.resize(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        if (!chf[i].empty()) {
            survival_probabilities[i] = KaplanMeier(chf[i]);
        }
    }
}

// computes the Kaplan-Meier estimator for the censoring distribution in node node_index
void SurvivalTree::computeCensoringKM(size_t node_index) {
    const vector<size_t>& indices = honest ? holdout_node_obs[node_index] : node_obs[node_index];

    vector<double> KM_full = computeCensoringKMFromEndpoints(data->get_y_col_ptr(0), data->get_y_col_ptr(1), *censoring_times, indices);
    vector<double> KM_event = selectCensoringAtTimes(KM_full, *censoring_times, *unique_event_times, 1);
    this->KM_censoring_full.push_back(std::move(KM_full));
    this->KM_censoring.push_back(std::move(KM_event));
}

// computes the Kaplan-Meier estimator for the censoring distribution in the node given by node_index based on the training data given by indices
void SurvivalTree::computeCensoringKMExternal(const vector<size_t>& indices, size_t node_index) {
    vector<double> KM_full = computeCensoringKMFromEndpoints(data->get_y_col_ptr(0), data->get_y_col_ptr(1), *censoring_times, indices);
    this->KM_censoring[node_index] = selectCensoringAtTimes(KM_full, *censoring_times, *unique_event_times, 1);
    this->KM_censoring_full[node_index] = std::move(KM_full);
}

// computes censoring estimators after fitting from the same estimation sample used by eager fitting
void SurvivalTree::computeCensoringKMLazy() {
    KM_censoring.assign(num_nodes, vector<double>());
    KM_censoring_full.assign(num_nodes, vector<double>());

    vector<vector<size_t>> leaf_groups(num_nodes);
    for (size_t obs_id : censoring_indices) {
        leaf_groups[prediction_node_IDs[obs_id]].push_back(obs_id);
    }

    for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
        if (left_daughters[node_id] == 0) {
            computeCensoringKMExternal(leaf_groups[node_id], node_id);
        }
    }
    vector<size_t>().swap(censoring_indices);
}

// splitting rules for survival trees
//--------------------------------------------------------------------------------------

// split_id = 0 for categorical splits

double SurvivalTree::logRank(const vector<size_t>& num_deaths, const vector<size_t>& num_at_risk, 
                             const vector<size_t>& num_deaths_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id) {
    double sum_num = 0;
    double sum_den = 0;
    size_t array_index = split_id * num_unique_event_times;
    // only event times can contribute to the log-rank statistic
    for (size_t i : *true_event_time_ids) {
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
    size_t array_index = split_id * num_unique_event_times;
    double NAsum1 = 0;
    double NAsum2 = 0;
    double sum1 = 0;
    double sum2 = 0;
    for (size_t i = 0; i < num_unique_event_times - 1; ++i) {
        size_t deaths1 = num_deaths_daughter[array_index + i];
        size_t at_risk1 = num_at_risk_daughter[array_index + i];
        size_t deaths2 = num_deaths[i] - deaths1;
        size_t at_risk2 = num_at_risk[i] - at_risk1;
        if (deaths1 > 0) {
            NAsum1 += (double) deaths1 / at_risk1;
        }
        if (deaths2 > 0) {
            NAsum2 += (double) deaths2 / at_risk2;
        }

        size_t at_risk1_next = num_at_risk_daughter[array_index + i + 1];
        size_t at_risk2_next = num_at_risk[i + 1] - at_risk1_next;
        sum1 += (at_risk1 - at_risk1_next) * at_risk1_next * NAsum1;
        sum2 += (at_risk2 - at_risk2_next) * at_risk2_next * NAsum2;
    }

    size_t at_risk1 = num_at_risk_daughter[array_index];
    size_t at_risk2 = num_at_risk[0] - at_risk1;
    double cons = (at_risk1 * sum1 + at_risk2 * sum2) / num_at_risk[0];
    return 1/(1 + cons);
}

double SurvivalTree::approxLogRank(const vector<size_t>& num_deaths, const vector<size_t>& num_at_risk, 
                                   const vector<size_t>& num_deaths_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id) {
    double D = 0;
    double D1 = 0;
    double sum_num = 0;
    size_t array_index = split_id * num_unique_event_times;
    // only event times can contribute to the approximate log-rank statistic
    for (size_t i : *true_event_time_ids) {
        const double d = (double) num_deaths[i];
        const double d1 = (double) num_deaths_daughter[array_index + i];  // we use a single sweep which considers every split at once, hence the flattened array
        const double Y = (double) num_at_risk[i];
        const double Y1 = (double) num_at_risk_daughter[array_index + i];
        if (d > 0) {
            sum_num += d1 - Y1 * d / Y;
            D += d;
            D1 += d1;
        }
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
        size_t leaf_id = predictionLeafID(new_data, i);
        const vector<double>& pred = chf[leaf_id];
        copy(pred.begin(), pred.end(), predictions.begin() + i * num_unique_event_times);
    }
    return predictions;
}

// computes a pair of flattened vectors, the first predictions and the second the censoring KM estimators
pair<vector<double>, vector<double>> SurvivalTree::computePredictionsCensoring(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);
    vector<double> censoring(num_obs * num_censoring_times);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t leaf_id = predictionLeafID(new_data, i);
        const vector<double>& pred = chf[leaf_id];
        const vector<double>& cens = KM_censoring_full[leaf_id];
        copy(pred.begin(), pred.end(), predictions.begin() + i * num_unique_event_times);
        copy(cens.begin(), cens.end(), censoring.begin() + i * num_censoring_times);
    }
    return {predictions, censoring};
}

// computes the Kaplan-Meier estimator given a NumericMatrix of Nelson-Aalen estimators
NumericMatrix KaplanMeier(const NumericMatrix& na) {
  size_t num_obs = na.nrow();
  size_t num_unique_event_times = na.ncol();
  NumericMatrix km(num_obs, num_unique_event_times);
  km.fill(1.0);

  if (num_unique_event_times == 0) {
    return km;
  }

  /*
    na contains cumulative hazards at the retained event times, and the first retained time does not generally equal zero.
    Consequently, the first increment is na(i, 0) - 0 and must be included before continuing with the usual recursion.
  */
  for (size_t i = 0; i < num_obs; ++i) {
    km(i, 0) = 1.0 - na(i, 0);
  }

  for (size_t j = 1; j < num_unique_event_times; ++j) {
    for (size_t i = 0; i < num_obs; ++i) {
      km(i, j) = km(i, j - 1) * (1 - (na(i, j) - na(i, j - 1)));
    }
  }
  return km;
}

// error estimation for survival trees
//--------------------------------------------------------------------------------------

// computes the outcomes used in the concordance index calculations using a NumericMatrix as input
vector<double> computeOutcomes(const NumericMatrix& predictions) {
    size_t n = predictions.nrow();
    size_t N = predictions.ncol();
    vector<double> outcomes(n);
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < n; ++i) {
            outcomes[i] += predictions(i, j);
        }
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

vector<double> computeCensoringKMFromEndpoints(const vector<double>& times, const vector<double>& ind, const vector<double>& censoring_times, const vector<size_t>& indices) {
    return computeCensoringKMFromEndpoints(times.data(), ind.data(), censoring_times, indices);
}

vector<double> computeCensoringKMFromEndpoints(const double* times, const double* ind, const vector<double>& censoring_times, const vector<size_t>& indices) {
    vector<double> KM(censoring_times.size(), 1);
    if (censoring_times.empty()) {
        return KM;
    }

    // bucket the time where observations leave the risk set and exact censoring-grid endpoints
    vector<size_t> num_removed(censoring_times.size() + 1, 0);
    vector<size_t> num_events(censoring_times.size(), 0);
    vector<size_t> num_censored(censoring_times.size(), 0);
    for (size_t i : indices) {
        size_t removal_id = upper_bound(censoring_times.begin(), censoring_times.end(), times[i]) - censoring_times.begin();
        ++num_removed[removal_id];

        auto it = lower_bound(censoring_times.begin(), censoring_times.end(), times[i]);
        if (it != censoring_times.end() && *it == times[i]) {
            size_t time_id = static_cast<size_t>(distance(censoring_times.begin(), it));
            if (ind[i] == 1) {
                ++num_events[time_id];
            } else {
                ++num_censored[time_id];
            }
        }
    }

    size_t num_at_risk = indices.size();
    for (size_t t = 0; t < censoring_times.size(); ++t) {
        double previous_KM = t == 0 ? 1.0 : KM[t - 1];
        num_at_risk -= num_removed[t];
        double denominator = (double) num_at_risk - num_events[t];
        if (denominator > 0) {
            KM[t] = previous_KM * (1 - num_censored[t] / denominator);
        } else {
            KM[t] = previous_KM;
        }
    }
    return KM;
}

double censoringValueAtTime(const vector<double>& KM_cens, const vector<double>& censoring_times, size_t row, size_t row_length, double time, bool left_limit) {
    if (censoring_times.empty()) {
        return 1;
    }

    auto it = left_limit ? lower_bound(censoring_times.begin(), censoring_times.end(), time) :
                           upper_bound(censoring_times.begin(), censoring_times.end(), time);
    if (it == censoring_times.begin()) {
        return 1;
    }

    size_t time_index = static_cast<size_t>(distance(censoring_times.begin(), it) - 1);
    return KM_cens[row * row_length + time_index];
}

vector<double> selectCensoringAtTimes(const vector<double>& KM_cens, const vector<double>& censoring_times, const vector<double>& output_times, size_t num_obs) {
    vector<double> result(num_obs * output_times.size());
    size_t row_length = censoring_times.size();

    // the source position for a given output time is the same in every row
    vector<size_t> source_ids(output_times.size(), row_length);
    for (size_t t = 0; t < output_times.size(); ++t) {
        auto it = upper_bound(censoring_times.begin(), censoring_times.end(), output_times[t]);
        if (it != censoring_times.begin()) {
            source_ids[t] = static_cast<size_t>(distance(censoring_times.begin(), it) - 1);
        }
    }

    for (size_t i = 0; i < num_obs; ++i) {
        for (size_t t = 0; t < output_times.size(); ++t) {
            result[i * output_times.size() + t] = source_ids[t] == row_length ? 1 : KM_cens[i * row_length + source_ids[t]];
        }
    }
    return result;
}

// computes the IPCW weights for each observation and event time (useful if one wants to extend to other error metrics)
vector<double> computeIPCW(const vector<double>& ind, const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids,
                           const NumericMatrix& KM_cens, const vector<double>& times, const vector<size_t>& last_observed_time_ids,
                           const vector<double>& censoring_times, const vector<double>& censoring_before_event) {
    // censoring_before_event supplies exact left limits when the event-time grid has been thinned
    size_t num_obs;
    bool multi_state;
    if (last_observed_time_ids.empty()) {
        multi_state = false;                        // survival model (times are supplied)
        num_obs = times.size();

    } else {
        multi_state = true;                         // multi-state model (last_observed_time_ids are supplied)
        num_obs = last_observed_time_ids.size();
    }
    size_t num_unique_event_times = unique_event_times.size();
    vector<double> weights(num_obs * num_unique_event_times, 0);    // result is a flattened vector

    // when a separate censoring grid is used, its source positions are shared by all rows
    vector<size_t> censoring_time_ids;
    if (!censoring_times.empty()) {
        censoring_time_ids.assign(num_unique_event_times, censoring_times.size());
        for (size_t j = 0; j < num_unique_event_times; ++j) {
            auto it = upper_bound(censoring_times.begin(), censoring_times.end(), unique_event_times[j]);
            if (it != censoring_times.begin()) {
                censoring_time_ids[j] = static_cast<size_t>(distance(censoring_times.begin(), it) - 1);
            }
        }
    }

    for (size_t i = 0; i < num_obs; ++i) {
        // when determining the index for the corresponding unique event time for the given observation,
        // we need to take the id for the last observed time for a multi-state tree/forest
        size_t event_time_index;
        size_t obs_index;
        if (multi_state) {
            obs_index = last_observed_time_ids[i];
            event_time_index = response_event_time_ids[obs_index];
        } else {
            obs_index = i;
            event_time_index = response_event_time_ids[i];
        }
        size_t event_censoring_index = event_time_index == 0 ? 0 : event_time_index - 1;
        double censoring_before = 1;
        if (!censoring_before_event.empty()) {
            censoring_before = censoring_before_event[i];
        } else if (!censoring_times.empty()) {
            auto it = lower_bound(censoring_times.begin(), censoring_times.end(), times[obs_index]);
            if (it != censoring_times.begin()) {
                censoring_before = KM_cens(i, distance(censoring_times.begin(), it) - 1);
            }
        } else {
            censoring_before = KM_cens(i, event_censoring_index);
        }

        for (size_t j = 0; j < num_unique_event_times; ++j) {
            if (times[obs_index] <= unique_event_times[j] && ind[i] == 1) {
                if (censoring_before > 0) {
                    weights[i * num_unique_event_times + j] = 1 / (num_obs * censoring_before);
                }
            }
            else if (times[obs_index] > unique_event_times[j]) {
                double cens = KM_cens(i, j);
                if (!censoring_times.empty()) {
                    size_t censoring_time_id = censoring_time_ids[j];
                    cens = censoring_time_id == censoring_times.size() ? 1 : KM_cens(i, censoring_time_id);
                }
                if (cens > 0) {
                    weights[i * num_unique_event_times + j] = 1 / (num_obs * cens);
                }
            }
            // the third case (censored before the event time) has weight zero
        }
    }
    return weights;
}

// computes a vector of the Brier score using given IPCW weights
vector<double> computeBrierScore(const vector<double>& times, const vector<double>& weights, const vector<double>& unique_event_times, const NumericMatrix& KM_pred) {
    size_t num_obs = times.size();
    size_t num_unique_event_times = unique_event_times.size();
    vector<double> brier(num_unique_event_times, 0);

    for (size_t j = 0; j < num_unique_event_times; ++j) {
        for (size_t i = 0; i < num_obs; ++i) {
            if (times[i] <= unique_event_times[j]) {
                brier[j] += KM_pred(i, j) * KM_pred(i, j) * weights[i * num_unique_event_times + j];
            } else if (times[i] > unique_event_times[j]) {
                brier[j] += (1 - KM_pred(i, j)) * (1 - KM_pred(i, j)) * weights[i * num_unique_event_times + j];
            }
        }
    }
    // for debugging
    return brier;
}

// computes a vector of the Kullback-Leibler score using given IPCW weights
double probabilityForLogScore(double probability) {
    constexpr double probability_tolerance = 1e-12;
    constexpr double probability_epsilon = 1e-15;

    if (!isfinite(probability)) {
        throw runtime_error("Non-finite predicted probability in Kullback-Leibler score");
    }
    if (probability < -probability_tolerance || probability > 1 + probability_tolerance) {
        throw runtime_error("Predicted probability outside [0, 1] in Kullback-Leibler score");
    }
    return min(max(probability, probability_epsilon), 1 - probability_epsilon);
}

vector<double> computeKLScore(const vector<double>& times, const vector<double>& weights, const vector<double>& unique_event_times, const NumericMatrix& KM_pred) {
    size_t num_obs = times.size();
    size_t num_unique_event_times = unique_event_times.size();
    vector<double> kl_score(num_unique_event_times, 0);

    for (size_t j = 0; j < num_unique_event_times; ++j) {
        for (size_t i = 0; i < num_obs; ++i) {
            double weight = weights[i * num_unique_event_times + j];
            if (weight == 0) {
                continue;
            }
            double survival_probability = probabilityForLogScore(KM_pred(i, j));
            if (times[i] <= unique_event_times[j]) {
                kl_score[j] -= log1p(-survival_probability) * weight;
            } else if (times[i] > unique_event_times[j]) {
                kl_score[j] -= log(survival_probability) * weight;
            }
        }
    }
    return kl_score;
}

// the following functions are identical except that they accept KM_cens and KM_pred as flattened arrays instead of NumericMatrices

// computes the IPCW weights for each observation and event time (useful if one wants to extend to other error metrics)
vector<double> computeIPCWCpp(const vector<double>& times, const vector<double>& ind, const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, 
                              const vector<double>& KM_cens, vector<size_t> obs_indices, const vector<size_t>& last_observed_time_ids,
                              const vector<double>& censoring_times, const vector<double>& censoring_before_events) {
    // if obs_indices is not provided, it means that all observations should be used
    bool multi_state = !last_observed_time_ids.empty();
    size_t num_obs;
    if (obs_indices.empty()) {
        num_obs = multi_state ? last_observed_time_ids.size() : times.size();
        obs_indices.resize(num_obs);
        for (size_t i = 0; i < num_obs; ++i) {
            obs_indices[i] = i;
        }
    } else {
        num_obs = obs_indices.size();
    }

    size_t num_unique_event_times = unique_event_times.size();
    vector<double> weights(num_obs * num_unique_event_times, 0);    // result is a flattened vector

    // when a separate censoring grid is used, its source positions are shared by all rows
    vector<size_t> censoring_time_ids;
    if (!censoring_times.empty()) {
        censoring_time_ids.assign(num_unique_event_times, censoring_times.size());
        for (size_t j = 0; j < num_unique_event_times; ++j) {
            auto it = upper_bound(censoring_times.begin(), censoring_times.end(), unique_event_times[j]);
            if (it != censoring_times.begin()) {
                censoring_time_ids[j] = static_cast<size_t>(distance(censoring_times.begin(), it) - 1);
            }
        }
    }

    for (size_t row = 0; row < num_obs; ++row) {
        size_t obs_id = obs_indices[row];
        // when determining the index for the corresponding unique event time for the given observation,
        // we need to take the id for the last observed time for a multi-state tree/forest
        size_t event_time_index;
        double time;
        double indicator;
        if (multi_state) {
            size_t last_observed_time_id = last_observed_time_ids[obs_id];
            event_time_index = response_event_time_ids[last_observed_time_id];
            time = times[last_observed_time_id];
            indicator = ind[obs_id];
        } else {
            event_time_index = response_event_time_ids[obs_id];
            time = times.size() == num_obs ? times[row] : times[obs_id];
            indicator = ind.size() == num_obs ? ind[row] : ind[obs_id];
        }

        size_t event_censoring_index = event_time_index == 0 ? 0 : event_time_index - 1;
        size_t censoring_row_length = censoring_times.empty() ? num_unique_event_times : censoring_times.size();
        double censoring_before_event;
        if (!censoring_before_events.empty()) {
            censoring_before_event = censoring_before_events[row];
        } else if (!censoring_times.empty()) {
            censoring_before_event = 1;
            auto it = lower_bound(censoring_times.begin(), censoring_times.end(), time);
            if (it != censoring_times.begin()) {
                size_t source_id = static_cast<size_t>(distance(censoring_times.begin(), it) - 1);
                censoring_before_event = KM_cens[row * censoring_row_length + source_id];
            }
        } else {
            censoring_before_event = KM_cens[row * num_unique_event_times + event_censoring_index];
        }

        for (size_t j = 0; j < num_unique_event_times; ++j) {
            size_t index = row * num_unique_event_times + j;
            if (time <= unique_event_times[j] && indicator == 1) {
                if (censoring_before_event > 0) {
                    weights[index] = 1 / (num_obs * censoring_before_event);
                }
            }
            else if (time > unique_event_times[j]) {
                double cens = censoring_times.empty() ? KM_cens[index] :
                  (censoring_time_ids[j] == censoring_row_length ? 1 : KM_cens[row * censoring_row_length + censoring_time_ids[j]]);
                if (cens > 0) {
                    weights[index] = 1 / (num_obs * cens);
                }
            }
            // the third case (censored before the event time) has weight zero
        }
    }
    return weights;
}

// computes a vector of the Brier score using given IPCW weights
vector<double> computeBrierScoreCpp(const vector<double>& times, const vector<double>& weights, const vector<double>& unique_event_times, const vector<double>& KM_pred) {
    size_t num_obs = times.size();
    size_t num_unique_event_times = unique_event_times.size();
    vector<double> brier(num_unique_event_times, 0);

    for (size_t i = 0; i < num_obs; ++i) {
        for (size_t j = 0; j < num_unique_event_times; ++j) {
            size_t index = i * num_unique_event_times + j;
            if (times[i] <= unique_event_times[j]) {
                brier[j] += KM_pred[index] * KM_pred[index] * weights[index];
            } else if (times[i] > unique_event_times[j]) {
                brier[j] += (1 - KM_pred[index]) * (1 - KM_pred[index]) * weights[index];
            }
        }
    }
    // for debugging
    return brier;
}

// computes a vector of the Kullback-Leibler score using given IPCW weights
vector<double> computeKLScoreCpp(const vector<double>& times, const vector<double>& weights, const vector<double>& unique_event_times, const vector<double>& KM_pred) {
    size_t num_obs = times.size();
    size_t num_unique_event_times = unique_event_times.size();
    vector<double> kl_score(num_unique_event_times, 0);

    for (size_t i = 0; i < num_obs; ++i) {
        for (size_t j = 0; j < num_unique_event_times; ++j) {
            size_t index = i * num_unique_event_times + j;
            double weight = weights[index];
            if (weight == 0) {
                continue;
            }
            double survival_probability = probabilityForLogScore(KM_pred[index]);
            if (times[i] <= unique_event_times[j]) {
                kl_score[j] -= log1p(-survival_probability) * weight;
            } else if (times[i] > unique_event_times[j]) {
                kl_score[j] -= log(survival_probability) * weight;
            }
        }
    }
    return kl_score;
}

// computes the integrated Brier Score (IBS) and the normalised IBS (using the trapezoidal rule)
// for a multi-state model (multi_state == true), score is a flattened vector of length num_unique_event_times * num_states
pair<double, double> computeIntegratedScore(const vector<double>& score, const vector<double>& unique_event_times, bool multi_state) {
    size_t num_unique_event_times = unique_event_times.size();
    double iscore = 0;

    if (num_unique_event_times == 0) {
        return {0, 0};
    }

    if (!multi_state) { // survival
        /*
          survival grids contain true event times and therefore usually start after zero. The loss at time zero is zero
          since everybody is alive with predicted survival one, so include the missing first trapezoid explicitly.
          Multi-state grids already contain zero and consequently need no corresponding correction below.
        */
        iscore += score[0] * unique_event_times[0] / 2;

        // apply trapezoidal rule
        for (size_t j = 1; j < num_unique_event_times; ++j) {
            iscore += (score[j] + score[j - 1]) * (unique_event_times[j] - unique_event_times[j - 1]) / 2;
        }
    } else {            // for multi-state models, aggregate over all states
        size_t num_states = score.size() / num_unique_event_times;
        // first compute the score across all states
        vector<double> state_contributions(num_unique_event_times, 0);
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            for (size_t j = 0; j < num_states; ++j) {
                state_contributions[t] += score[t * num_states + j];
            }
        }
        // now compute the integrated score via the trapezoidal rule
        for (size_t t = 1; t < num_unique_event_times; ++t) {
            iscore += (state_contributions[t] + state_contributions[t - 1]) * (unique_event_times[t] - unique_event_times[t - 1]) / 2;
        }
    }
    return {iscore, iscore / unique_event_times.back()};
}



// miscellaneous functions related to survival
//--------------------------------------------------------------------------------------

// for computing the ids in the observed times corresponding to the unique event times (including censored times)
vector<size_t> computeResponseEventTimeIDs(const vector<double>& unique_event_times, const vector<double>& times) {
    if (unique_event_times.empty()) {
        throw invalid_argument("The survival event-time grid cannot be empty");
    }

    vector<size_t> response_event_time_ids;
    response_event_time_ids.reserve(times.size());
    for (const double& time : times) {
        // use binary search to find lower bound
        auto it = lower_bound(unique_event_times.begin(), unique_event_times.end(), time);
        size_t event_time_id = static_cast<size_t>(distance(unique_event_times.begin(), it));
        // ensures that we don't choose an event time which is out of bounds
        if (event_time_id >= unique_event_times.size()) {
            event_time_id = unique_event_times.size() - 1;
        }
        response_event_time_ids.push_back(event_time_id);
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

// computes the Kaplan-Meier estimator given a Nelson-Aalen estimator
// if the Nelson-Aalen estimator is a flattened vector, provide the number of estimators in num_estimators
vector<double> KaplanMeier(const vector<double>& na, size_t num_estimators) {
    if (num_estimators == 0) {
        throw invalid_argument("The number of Kaplan-Meier estimators must be positive");
    }
    size_t stride_length = na.size() / num_estimators;
    vector<double> KM = vector<double>(na.size(), 1);

    if (stride_length == 0) {
        return KM;
    }

    for (size_t i = 0; i < num_estimators; ++i) {
        size_t first_index = i * stride_length;
        KM[first_index] = 1.0 - na[first_index];

        // compute using the recursion given by the product integral
        for (size_t j = 1; j < stride_length; ++j) {
            size_t index = i * stride_length + j;
            KM[index] = KM[index - 1] * (1 - (na[index] - na[index - 1]));
        }
    }
    return KM;
}

namespace {
double computeConcordanceIndexQuadratic(const vector<double>& outcomes, const vector<double>& times, const vector<double>& ind) {
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

class FenwickCounts {
public:
    explicit FenwickCounts(size_t size) : counts(size + 1, 0) {
    }

    void add(size_t index) {
        for (++index; index < counts.size(); index += index & -index) {
            ++counts[index];
        }
    }

    size_t sum(size_t end) const {
        size_t result = 0;
        for (; end > 0; end -= end & -end) {
            result += counts[end];
        }
        return result;
    }

private:
    vector<size_t> counts;
};
}

// computes Harrell's C-index for a list of outcomes and survival data
double computeConcordanceIndex(const vector<double>& outcomes, const vector<double>& times, const vector<double>& ind) {
    size_t n = times.size();
    for (size_t i = 0; i < n; ++i) {
        // the faster counting method assumes ordinary finite survival data with a binary indicator
        if (!std::isfinite(outcomes[i]) || !std::isfinite(times[i]) || (ind[i] != 0 && ind[i] != 1)) {
            return computeConcordanceIndexQuadratic(outcomes, times, ind);
        }
    }

    vector<double> outcome_values = outcomes;
    sort(outcome_values.begin(), outcome_values.end());
    outcome_values.erase(unique(outcome_values.begin(), outcome_values.end()), outcome_values.end());

    vector<size_t> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](size_t first, size_t second) {
        if (times[first] == times[second]) {
            return first < second;
        }
        return times[first] > times[second];
    });

    FenwickCounts later_outcomes(outcome_values.size());
    double concordance = 0;
    double permissible = 0;
    size_t num_later = 0;
    vector<size_t> group_ranks;

    // observations at later times are added first, so rank counts replace pairwise comparisons
    for (size_t begin = 0; begin < n;) {
        size_t end = begin + 1;
        while (end < n && times[order[end]] == times[order[begin]]) {
            ++end;
        }

        size_t num_events = 0;
        size_t num_censored = 0;
        group_ranks.clear();
        group_ranks.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            size_t obs_id = order[i];
            size_t rank = lower_bound(outcome_values.begin(), outcome_values.end(), outcomes[obs_id]) - outcome_values.begin();
            group_ranks.push_back(rank);
            if (ind[obs_id] == 1) {
                size_t num_smaller = later_outcomes.sum(rank);
                size_t num_equal = later_outcomes.sum(rank + 1) - num_smaller;
                concordance += num_smaller + 0.5 * num_equal;
                permissible += num_later;
                ++num_events;
            } else {
                ++num_censored;
            }
        }

        // at tied times, an event-censoring pair is comparable and receives credit only for tied outcomes
        permissible += (double) num_events * num_censored;
        if (num_events > 0 && num_censored > 0) {
            unordered_map<size_t, pair<size_t, size_t>> tied_outcomes;
            for (size_t i = begin; i < end; ++i) {
                pair<size_t, size_t>& counts = tied_outcomes[group_ranks[i - begin]];
                if (ind[order[i]] == 1) {
                    ++counts.first;
                } else {
                    ++counts.second;
                }
            }
            for (const auto& entry : tied_outcomes) {
                concordance += 0.5 * (double) entry.second.first * entry.second.second;
            }
        }

        for (size_t rank : group_ranks) {
            later_outcomes.add(rank);
        }
        num_later += end - begin;
        begin = end;
    }

    return concordance / permissible;
}

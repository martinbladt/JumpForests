/*

Functions for multi-state trees

*/

#include "TreeMultistate.h"
#include "TreeSurvival.h"
#include <cmath>

namespace {

/*
  Fleming--Harrington statistics are unchanged when all weights for one transition are multiplied by a common
  positive constant. Work on the log scale and normalise the largest usable weight to one. This avoids losing an
  otherwise valid statistic when large exponents make every raw pow() result underflow to zero.
*/
long double flemingHarringtonLogWeight(double occupation_probability, double a, double b) {
    long double probability = static_cast<long double>(occupation_probability);
    long double log_weight = 0;
    if (a > 0) {
        if (probability <= 0) {
            return -numeric_limits<long double>::infinity();
        }
        log_weight += static_cast<long double>(a) * log(probability);
    }
    if (b > 0) {
        if (probability >= 1) {
            return -numeric_limits<long double>::infinity();
        }
        log_weight += static_cast<long double>(b) * log1p(-probability);
    }
    return log_weight;
}

}


// constructor for MultistateTree
//--------------------------------------------------------------------------------------

// estimation_indices is only used for honest trees; the final two shared lookups are supplied by forests
MultistateTree::MultistateTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids, 
                               vector<size_t> subset_indices, uint8_t num_states, bool save_predictions,
                               vector<size_t> estimation_indices, shared_ptr<vector<double>> censoring_times,
                               shared_ptr<const MultistateResponseData> response_data,
                               shared_ptr<const vector<size_t>> censoring_endpoint_ids,
                               shared_ptr<const vector<double>> fh_weights_a,
                               shared_ptr<const vector<double>> fh_weights_b) :
    unique_event_times {unique_event_times}, response_event_time_ids {response_event_time_ids}, num_states {num_states},
    dim {static_cast<size_t>(num_states) * num_states}, save_predictions {save_predictions},
    forest_tree {response_data != nullptr},
    fh_weights_a {std::move(fh_weights_a)}, fh_weights_b {std::move(fh_weights_b)},
    response_data {std::move(response_data)},
    censoring_endpoint_ids {std::move(censoring_endpoint_ids)} {
    this->node_obs.push_back(std::move(subset_indices));
    this->holdout_node_obs.push_back(std::move(estimation_indices));
    this->num_unique_event_times = unique_event_times->size();
    this->censoring_times = censoring_times == nullptr ? unique_event_times : censoring_times;
    this->num_censoring_times = this->censoring_times->size();
    this->node_sizes.push_back(this->node_obs[0].size());
    this->num_jumps.resize(num_unique_event_times * dim);
    this->num_at_risk.resize(num_unique_event_times * num_states);
    this->num_censored.resize(num_unique_event_times * num_states);

    /*
      A forest stores full censoring curves and performs this common lookup once at forest level. Standalone trees retain
      their own map because their public prediction output also includes censoring on the event-time grid.
    */
    if (!forest_tree) {
        this->event_censoring_time_ids.resize(num_unique_event_times, num_censoring_times);
        size_t censoring_time_id = 0;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            // both grids are sorted, so one forward pass replaces a separate binary search for every event time
            while (censoring_time_id + 1 < num_censoring_times &&
                   (*this->censoring_times)[censoring_time_id + 1] <= (*unique_event_times)[t]) {
                ++censoring_time_id;
            }
            if (censoring_time_id < num_censoring_times &&
                (*this->censoring_times)[censoring_time_id] <= (*unique_event_times)[t]) {
                this->event_censoring_time_ids[t] = censoring_time_id;
            }
        }
    }
}

shared_ptr<const MultistateResponseData> prepareMultistateResponseData(const Data& data, const vector<size_t>& response_event_time_ids,
                                                                       size_t num_unique_event_times, size_t num_states) {
    auto result = make_shared<MultistateResponseData>();
    size_t dim = num_states * num_states;

    // save both the state indices and the flattened matrix index for each transition which can occur
    const vector<pair<uint8_t, uint8_t>> data_valid_jumps = data.getValidJumps();
    result->valid_jumps.reserve(data_valid_jumps.size());
    for (const auto& jump : data_valid_jumps) {
        size_t from_state = static_cast<size_t>(jump.first - 1);
        size_t to_state = static_cast<size_t>(jump.second - 1);
        result->valid_jumps.push_back({from_state, to_state, from_state * num_states + to_state});
    }

    result->feature_indices.resize(data.getNumberOfFeatures());
    for (size_t feature = 0; feature < result->feature_indices.size(); ++feature) {
        result->feature_indices[feature] = feature;
    }

    /*
      convert every padded response path to a compact range of flattened transition indices. A forest consequently pays
      this cost once, rather than once for every tree, and parallel trees only retain a small shared pointer to the result.
    */
    const vector<uint8_t>& states = data.getStates();
    const vector<size_t>& last_observed_times = data.getLastObservedTimes();
    const vector<uint8_t>& censoring_states = data.getCensoringStates();
    size_t max_response_length = data.getMaxResponseLength();
    size_t num_obs = data.getNumberOfObs();
    size_t risk_array_size = num_unique_event_times * num_states;

    result->initial_state_ids.resize(num_obs);
    result->transition_offsets.resize(num_obs + 1);
    result->transition_ids.reserve(num_obs);   // roughly one transition per observation is a useful initial estimate
    result->censoring_risk_ids.assign(num_obs, risk_array_size);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t response_index = i * max_response_length;
        result->initial_state_ids[i] = states[response_index] - 1;
        result->transition_offsets[i] = result->transition_ids.size();

        for (size_t j = 1; j < max_response_length && states[response_index + j] != 0; ++j) {
            size_t current_state = static_cast<size_t>(states[response_index + j] - 1);
            size_t previous_state = static_cast<size_t>(states[response_index + j - 1] - 1);
            if (current_state != previous_state) {
                size_t event_time_id = response_event_time_ids[response_index + j];
                result->transition_ids.push_back(
                  event_time_id * dim + previous_state * num_states + current_state);
            }
        }

        uint8_t censoring_state = censoring_states[i];
        // censoring occurs
        if (censoring_state != 0) {
            size_t censoring_time_id = response_event_time_ids[last_observed_times[i]];
            if (censoring_time_id < num_unique_event_times) {
                result->censoring_risk_ids[i] = censoring_time_id * num_states + censoring_state - 1;
            }
        }
    }
    result->transition_offsets[num_obs] = result->transition_ids.size();
    result->transition_ids.shrink_to_fit();
    return result;
}

// functions for growing multi-state trees
//--------------------------------------------------------------------------------------

void MultistateTree::reserveTreeMemory(size_t num_obs) {
    // use the minimal node size to estimate the largest plausible tree and cap unusually large reservations
    size_t max_terminal_nodes = min_node_size == 0 ? max(static_cast<size_t>(1), num_obs) :
      max(static_cast<size_t>(1), num_obs / min_node_size);
    size_t max_num_nodes = min(2 * max_terminal_nodes - 1, static_cast<size_t>(1024));

    node_obs.reserve(max_num_nodes);
    if (honest) {
        holdout_node_obs.reserve(max_num_nodes);
    }
    node_sizes.reserve(max_num_nodes);
    left_daughters.reserve(max_num_nodes);
    feature_IDs.reserve(max_num_nodes);
    thresholds.reserve(max_num_nodes);
    depths.reserve(max_num_nodes);
    na.reserve(max_num_nodes);
    init_dist.reserve(max_num_nodes);
    if (save_predictions) {
        KM_censoring_full.reserve(max_num_nodes);
        if (!forest_tree) {
            KM_censoring.reserve(max_num_nodes);
        }
    }
}

void MultistateTree::prepareMultistateData() {
    if (multistate_data_prepared) {
        return;
    }

    reserveTreeMemory(node_obs[0].size());

    // standalone trees construct this once here; all trees in a forest receive the same immutable representation
    if (response_data == nullptr) {
        if (response_event_time_ids == nullptr) {
            throw runtime_error("Multi-state response-time IDs are missing");
        }
        response_data = prepareMultistateResponseData(*data, *response_event_time_ids, num_unique_event_times, num_states);
    }

    // retain the exact estimation sample, including bootstrap multiplicities, if censoring is computed lazily
    if (!save_predictions) {
        const vector<size_t>& estimation_sample = honest ? holdout_node_obs[0] : node_obs[0];
        if (data->getNumberOfObs() > numeric_limits<uint32_t>::max()) {
            throw runtime_error("Lazy multi-state censoring supports at most 2^32 - 1 observations");
        }
        censoring_indices.reserve(estimation_sample.size());
        for (size_t observation : estimation_sample) {
            censoring_indices.push_back(static_cast<uint32_t>(observation));
        }
    }

    // turn the splitting-rule string into a small integer once instead of comparing strings for every split
    if (splitrule == "gehan") {
        splitrule_id = MultistateSplitRule::Gehan;
    } else if (splitrule == "taroneware") {
        splitrule_id = MultistateSplitRule::TaroneWare;
    } else if (splitrule == "conserve") {
        splitrule_id = MultistateSplitRule::Conserve;
    } else if (splitrule == "approxlogrank") {
        splitrule_id = MultistateSplitRule::ApproxLogRank;
    } else if (splitrule == "petoprentice") {
        splitrule_id = MultistateSplitRule::PetoPrentice;
    } else if (splitrule == "flemingharrington") {
        splitrule_id = MultistateSplitRule::FlemingHarrington;
    } else {
        splitrule_id = MultistateSplitRule::LogRank;
    }

    if (splitrule_id == MultistateSplitRule::FlemingHarrington) {
        if (fh_weights_a == nullptr || fh_weights_b == nullptr ||
            fh_weights_a->size() != num_states || fh_weights_b->size() != num_states) {
            throw runtime_error(
              "Fleming--Harrington exponents must contain one value per state");
        }
        for (size_t j = 0; j < num_states; ++j) {
            if (!isfinite((*fh_weights_a)[j]) || (*fh_weights_a)[j] < 0 ||
                !isfinite((*fh_weights_b)[j]) || (*fh_weights_b)[j] < 0) {
                throw runtime_error(
                  "Fleming--Harrington exponents must contain finite non-negative values");
            }
        }
    }
    multistate_data_prepared = true;
}

void MultistateTree::addObservationToQuantities(size_t observation, vector<size_t>& jumps, vector<size_t>& at_risk, vector<size_t>& censored) {
    // the first row of at_risk temporarily holds the initial state counts and remains the time-zero row
    ++at_risk[response_data->initial_state_ids[observation]];

    // censoring actually occurs
    size_t censoring_id = response_data->censoring_risk_ids[observation];
    if (censoring_id < censored.size()) {
        ++censored[censoring_id];
    }

    // use the compact representation with offsets and IDs to update number of jumps
    for (size_t j = response_data->transition_offsets[observation]; j < response_data->transition_offsets[observation + 1]; ++j) {
        ++jumps[response_data->transition_ids[j]];
    }
}

void MultistateTree::materialiseAtRisk(const vector<size_t>& jumps, const vector<size_t>& censored, vector<size_t>& at_risk) {
    // start with the state counts at time zero and update these counts as time advances
    current_num_at_risk.assign(at_risk.begin(), at_risk.begin() + num_states);
    for (size_t t = 1; t < num_unique_event_times; ++t) {
        size_t previous_jump_index = (t - 1) * dim;

        // a jump at the preceding time removes one individual from its old state and adds it to its new state
        for (const MultistateValidJumpInfo& jump : response_data->valid_jumps) {
            size_t count = jumps[previous_jump_index + jump.matrix_index];
            if (count != 0) {
                current_num_at_risk[jump.from_state] -= count;
                current_num_at_risk[jump.to_state] += count;
            }
        }

        // censoring at t removes an individual before the at-risk counts at t are stored
        size_t risk_index = t * num_states;
        for (size_t state = 0; state < num_states; ++state) {
            current_num_at_risk[state] -= censored[risk_index + state];
            at_risk[risk_index + state] = current_num_at_risk[state];
        }
    }
}

void MultistateTree::computeMultistateQuantities(const vector<size_t>& indices, vector<size_t>& jumps, vector<size_t>& at_risk) {
    jumps.assign(num_unique_event_times * dim, 0);
    at_risk.assign(num_unique_event_times * num_states, 0);
    num_censored.assign(num_unique_event_times * num_states, 0);

    for (size_t i : indices) {
        addObservationToQuantities(i, jumps, at_risk, num_censored);
    }
    materialiseAtRisk(jumps, num_censored, at_risk);
}

void MultistateTree::prepareJumpEventTimeIDs() {
    if (jump_event_time_ids.size() != response_data->valid_jumps.size()) {
        jump_event_time_ids.resize(response_data->valid_jumps.size());
    }
    for (size_t jump_id = 0; jump_id < response_data->valid_jumps.size(); ++jump_id) {
        const MultistateValidJumpInfo& jump = response_data->valid_jumps[jump_id];
        vector<size_t>& event_ids = jump_event_time_ids[jump_id];
        event_ids.clear();   // retain capacity from the preceding node instead of reallocating the same sparse list
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            if (num_jumps[t * dim + jump.matrix_index] != 0) {
                event_ids.push_back(t);
            }
        }
    }
}

void MultistateTree::prepareSplitOccupationProbabilities() {
    parent_occupation_probs.assign(num_unique_event_times * num_states, 0);
    if (num_unique_event_times == 0) {
        return;
    }

    double num_obs = 0;
    for (size_t j = 0; j < num_states; ++j) {
        num_obs += static_cast<double>(num_at_risk[j]);
    }
    if (num_obs == 0) {
        return;
    }
    // compute parent-node initial probabilities
    for (size_t j = 0; j < num_states; ++j) {
        parent_occupation_probs[j] = static_cast<double>(num_at_risk[j]) / num_obs;
    }

    /*
      Update the probability row directly with the pooled Nelson--Aalen increments.
      This is the same Aalen--Johansen recursion used for predictions, but avoids
      constructing a complete cumulative transition-hazard array for every node.
    */
    for (size_t t = 1; t < num_unique_event_times; ++t) {
        size_t probability_index = t * num_states;
        size_t previous_probability_index = probability_index - num_states;
        copy(parent_occupation_probs.begin() + previous_probability_index,
             parent_occupation_probs.begin() + probability_index,
             parent_occupation_probs.begin() + probability_index);

        for (const MultistateValidJumpInfo& jump : response_data->valid_jumps) {
            size_t count = num_jumps[t * dim + jump.matrix_index];
            size_t at_risk = num_at_risk[t * num_states + jump.from_state];
            if (count == 0 || at_risk == 0) {
                continue;
            }
            double probability_change = parent_occupation_probs[previous_probability_index + jump.from_state] * static_cast<double>(count) / static_cast<double>(at_risk);
            parent_occupation_probs[probability_index + jump.from_state] -= probability_change;
            parent_occupation_probs[probability_index + jump.to_state] += probability_change;
        }
    }
}

void MultistateTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold) {
    const vector<size_t>& current_node_obs = node_obs[node_index];

    // samples split points
    vector<double> split_points;
    size_t nsplits_final = sampleSplitPoints(split_points, current_node_obs, feature);

    // no possible splits
    if (nsplits_final == 0) {
        return;
    }

    /*
      place each observation in the largest split where it belongs to the right daughter. Sweeping the buckets backwards
      then adds every response path exactly once, while the old implementation added it separately to every smaller split
    */
    vector<vector<size_t>> split_buckets(nsplits_final);
    for (size_t i : current_node_obs) {
        double feature_value = data->get_x(i, feature);
        /*
          missing feature values are sent right, which agrees with prediction where NaN <= threshold is false.
          Otherwise, determine the index of the first split point larger than or equal to the feature value.
          This index is also the number right nodes, the observation belongs to and this number minus 1 is
          the index of the largest threshold where the observation still goes to the right
        */
        size_t num_splits_right = std::isnan(feature_value) ? nsplits_final : static_cast<size_t>(lower_bound(split_points.begin(), split_points.end(), feature_value) - split_points.begin());
        if (num_splits_right != 0) {
            split_buckets[num_splits_right - 1].push_back(i);
        }
    }

    num_jumps_daughter.assign(num_unique_event_times * dim, 0);
    num_at_risk_daughter.assign(num_unique_event_times * num_states, 0);
    // the parent's censoring counts have already been materialised and can now be reused by the daughter sweep
    num_censored.assign(num_unique_event_times * num_states, 0);
    vector<double> split_values(nsplits_final, -1);
    size_t current_num_obs_right = 0;

    for (size_t s = nsplits_final; s-- > 0;) {
        for (size_t observation : split_buckets[s]) {
            addObservationToQuantities(observation, num_jumps_daughter, num_at_risk_daughter, num_censored);
            ++current_num_obs_right;
        }
        size_t num_obs_left = current_node_obs.size() - current_num_obs_right;
        if (current_num_obs_right < min_node_size || num_obs_left < min_node_size) {
            continue;
        }

        materialiseAtRisk(num_jumps_daughter, num_censored, num_at_risk_daughter);
        split_values[s] = computeSplitValue(num_jumps_daughter, num_at_risk_daughter);
    }

    // inspect split values in their original ascending order to retain deterministic tie handling
    for (size_t i = 0; i < nsplits_final; ++i) {
        if (split_values[i] > best_split_val) {
            best_split_val = split_values[i];
            best_feature = feature;
            // save the threshold which was actually evaluated, also when only some feature values were sampled
            best_threshold = {split_points[i]};
        }
    }
}

void MultistateTree::bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold) {
    const vector<size_t>& current_node_obs = node_obs[node_index];
    vector<double> feature_values = data->getValues(current_node_obs, feature);
    feature_values.erase(remove_if(feature_values.begin(), feature_values.end(), [](double value) { return std::isnan(value); }), feature_values.end());
    feature_values = uniqueValues(std::move(feature_values));
    size_t num_feature_values = feature_values.size();
    size_t num_partition_values = min(num_feature_values, static_cast<size_t>(63));

    unordered_set<uint64_t> partition_masks;
    // generate partitions (breaks if no possible splits)
    if (generateCategoricalPartitions(feature_values, partition_masks)) {
        return;
    }

    // group observations once, so a partition only visits categories included by its bitmask
    vector<vector<size_t>> category_observations(num_feature_values);
    for (size_t i : current_node_obs) {
        double feature_value = data->get_x(i, feature);
        if (std::isnan(feature_value)) {
            continue;   // as during prediction, missing categorical values belong to the right daughter
        }
        size_t category = static_cast<size_t>(lower_bound(feature_values.begin(), feature_values.end(), feature_value) - feature_values.begin());
        if (category < num_feature_values) {
            category_observations[category].push_back(i);
        }
    }

    // consider each partition (bitmask)
    for (const auto& mask : partition_masks) {
        size_t num_obs_left = 0;
        // partition masks have 63 usable bits; any additional categories always remain in the right daughter
        for (size_t category = 0; category < num_partition_values; ++category) {
            if ((mask >> category) & 1) {
                num_obs_left += category_observations[category].size();
            }
        }
        size_t num_obs_right = current_node_obs.size() - num_obs_left;
        if (num_obs_left < min_node_size || num_obs_right < min_node_size) {
            continue;
        }

        num_jumps_daughter.assign(num_unique_event_times * dim, 0);
        num_at_risk_daughter.assign(num_unique_event_times * num_states, 0);
        // reuse the same censoring workspace which was used for the parent and earlier partitions
        num_censored.assign(num_unique_event_times * num_states, 0);
        for (size_t category = 0; category < num_partition_values; ++category) {
            if ((mask >> category) & 1) {
                for (size_t i : category_observations[category]) {
                    addObservationToQuantities(i, num_jumps_daughter, num_at_risk_daughter, num_censored);
                }
            }
        }
        materialiseAtRisk(num_jumps_daughter, num_censored, num_at_risk_daughter);
        double split_val = computeSplitValue(num_jumps_daughter, num_at_risk_daughter);

        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_feature = feature;
            best_threshold.clear();
            for (size_t category = 0; category < num_partition_values; ++category) {
                if ((mask >> category) & 1) {
                    best_threshold.push_back(feature_values[category]);
                }
            }
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

    /*
      A standalone tree keeps training leaf IDs for its public prediction methods. Forest predictions already have to
      route OOB and external observations and therefore route every row, avoiding an N-entry size_t vector in every tree.
    */
    if (!forest_tree) {
        for (size_t i : node_obs[node_index]) {
            prediction_node_IDs[i] = node_index;
        }
        if (honest) {
            for (size_t i : holdout_node_obs[node_index]) {
                prediction_node_IDs[i] = node_index;
            }
        }
    }

    // response indices are no longer needed once all leaf quantities and training leaf IDs have been saved
    vector<size_t>().swap(node_obs[node_index]);
    if (honest) {
        vector<size_t>().swap(holdout_node_obs[node_index]);
    }
}

// function to create a split for a multi-state tree. returns true if leaf, otherwise false
bool MultistateTree::createSplit(size_t node_index) {
    // initialise response summaries and quantities which are reused throughout the tree only once
    prepareMultistateData();

    const vector<size_t>& current_node_obs = node_obs[node_index];

    // if no split is possible, make the node a leaf
    if (current_node_obs.size() < 2 * min_node_size) {
        const vector<size_t>& estimation_obs = honest ? holdout_node_obs[node_index] : current_node_obs;
        computeMultistateQuantities(estimation_obs, num_jumps, num_at_risk);
        makeLeaf(node_index);
        return true;
    }

    computeMultistateQuantities(current_node_obs, num_jumps, num_at_risk);   // update parent multi-state info
    prepareJumpEventTimeIDs();                                               // empty event times cannot affect most split rules
    // the Peto-Prentice and Fleming-Harrington splitting rules both need occupation probabilities to be computed
    if (splitrule_id == MultistateSplitRule::PetoPrentice || splitrule_id == MultistateSplitRule::FlemingHarrington) {
        prepareSplitOccupationProbabilities();
    }

    double best_split_val = -1.0;
    size_t best_feature = 0;
    vector<double> best_threshold;
    vector<size_t> best_left_indices;
    vector<size_t> best_right_indices;
    
    // only used for honesty
    vector<size_t> holdout_left_indices;
    vector<size_t> holdout_right_indices;

    // sample mtry features from the vector prepared once at the root
    vector<size_t> sampled_features = sampleIndices(response_data->feature_indices, mtry, false, random_number_generator);
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
            computeMultistateQuantities(holdout_node_obs[node_index], num_jumps, num_at_risk);
        }
        // otherwise, use parent survival info already computed earlier
        makeLeaf(node_index);
        return true;
    }

    // construct both daughter index sets only once, after all features and split points have been compared
    bool categorical_split = categorical[best_feature];
    best_left_indices.reserve(current_node_obs.size());
    best_right_indices.reserve(current_node_obs.size());
    for (size_t i : current_node_obs) {
        double feature_value = data->get_x(i, best_feature);
        bool goes_left = categorical_split ? find(best_threshold.begin(), best_threshold.end(), feature_value) != best_threshold.end() : feature_value <= best_threshold[0];
        if (goes_left) {
            best_left_indices.push_back(i);
        } else {
            best_right_indices.push_back(i);
        }
    }

    // update the holdout index sets in exactly the same way if the tree is honest
    if (honest) {
        const vector<size_t>& current_holdout_node_obs = holdout_node_obs[node_index];
        holdout_left_indices.reserve(current_holdout_node_obs.size());
        holdout_right_indices.reserve(current_holdout_node_obs.size());
        for (size_t i : current_holdout_node_obs) {
            double feature_value = data->get_x(i, best_feature);
            bool goes_left = categorical_split ? find(best_threshold.begin(), best_threshold.end(), feature_value) != best_threshold.end() : feature_value <= best_threshold[0];
            if (goes_left) {
                holdout_left_indices.push_back(i);
            } else {
                holdout_right_indices.push_back(i);
            }
        }
    }

    /*
      an honest leaf must contain estimation observations. If the selected split empties one holdout daughter, keep the
      current node as a leaf instead of borrowing growing observations and silently weakening honesty.
    */
    if (honest && (holdout_left_indices.empty() || holdout_right_indices.empty())) {
        computeMultistateQuantities(holdout_node_obs[node_index], num_jumps, num_at_risk);
        makeLeaf(node_index);
        return true;
    }
    
    // a best split was found, update the tree
    node_sizes.push_back(best_left_indices.size());
    node_sizes.push_back(best_right_indices.size());
    node_obs.push_back(std::move(best_left_indices));          // construct left daughter
    node_obs.push_back(std::move(best_right_indices));         // construct right daughter
    feature_IDs.push_back(best_feature);
    thresholds.push_back(std::move(best_threshold));
    na.push_back(vector<double>());
    init_dist.push_back(vector<double>());
    if (save_predictions) {
        KM_censoring_full.push_back(vector<double>());
        if (!forest_tree) {
            KM_censoring.push_back(vector<double>());
        }
    }

    // for honest trees, update the holdout indices
    if (honest) {
        holdout_node_obs.push_back(std::move(holdout_left_indices));
        holdout_node_obs.push_back(std::move(holdout_right_indices));
    }

    // the parent indices are no longer needed after its two daughters have been constructed
    vector<size_t>().swap(node_obs[node_index]);
    if (honest) {
        vector<size_t>().swap(holdout_node_obs[node_index]);
    }
    return false;
}

// computes the initial distribution in a node
void MultistateTree::computeInitialDist(size_t node_index) {
    vector<double> init_dist(num_states, 0);
    size_t num_obs = honest ? holdout_node_obs[node_index].size() : node_obs[node_index].size();
    double inverse_num_obs = 1.0 / static_cast<double>(num_obs);
    for (size_t state = 0; state < num_states; ++state) {
        init_dist[state] = static_cast<double>(num_at_risk[state]) * inverse_num_obs;
    }
    this->init_dist.push_back(std::move(init_dist));
}

// computes the matrix of Nelson--Aalen estimators in node node_index
void MultistateTree::computeNA(size_t node_index) {
    vector<double> na(num_unique_event_times * dim, 0);

    // each cumulative matrix starts as the preceding matrix, after which only observed transitions have to be updated
    for (size_t t = 1; t < num_unique_event_times; ++t) {
        size_t matrix_index = t * dim;
        copy(na.begin() + matrix_index - dim, na.begin() + matrix_index, na.begin() + matrix_index);

        for (const MultistateValidJumpInfo& jump : response_data->valid_jumps) {
            size_t count = num_jumps[matrix_index + jump.matrix_index];
            size_t at_risk = num_at_risk[t * num_states + jump.from_state];
            if (count != 0 && at_risk != 0) {
                double increment = static_cast<double>(count) / static_cast<double>(at_risk);
                na[matrix_index + jump.matrix_index] += increment;
                // the diagonal is minus the sum of the off-diagonal entries in its row
                na[matrix_index + jump.from_state * num_states + jump.from_state] -= increment;
            }
        }
    }
    this->na.push_back(std::move(na));
}

void MultistateTree::computeCensoringKM(size_t node_index) {
    const vector<size_t>& indices = honest ? holdout_node_obs[node_index] : node_obs[node_index];
    computeCensoringKMExternal(indices, node_index);
}

void MultistateTree::computeCensoringKMExternal(const vector<size_t>& indices, size_t node_index) {
    vector<double> KM_full(num_censoring_times, 1);
    vector<size_t> num_events(num_censoring_times, 0);
    vector<size_t> num_censored_endpoints(num_censoring_times, 0);
    const vector<double>& times = data->getTimes();
    const vector<size_t>& last_observed_times = data->getLastObservedTimes();
    const vector<uint8_t>& censoring_states = data->getCensoringStates();

    // every selected observation belongs to one terminal leaf, so looking up endpoints here costs no more searches than
    // precomputing them for the full data and avoids retaining another size_t for every observation while growing
    for (size_t i : indices) {
        size_t endpoint_id;
        if (censoring_endpoint_ids != nullptr) {
            endpoint_id = (*censoring_endpoint_ids)[i];
        } else {
            double endpoint = times[last_observed_times[i]];
            endpoint_id = static_cast<size_t>(
                lower_bound(censoring_times->begin(), censoring_times->end(), endpoint) -
                censoring_times->begin());
        }
        // no censoring or event
        if (endpoint_id >= num_censoring_times) {
            continue;
        }
        // no censoring but an event has occured
        if (censoring_states[i] == 0) {
            ++num_events[endpoint_id];
        } else {
            // otherwise censoring
            ++num_censored_endpoints[endpoint_id];
        }
    }

    // endpoints at a time remain at risk at that time and are removed before the next censoring time
    size_t current_num_at_risk = indices.size();
    for (size_t t = 0; t < num_censoring_times; ++t) {
        double previous_KM = t == 0 ? 1.0 : KM_full[t - 1];
        double denominator = static_cast<double>(current_num_at_risk) - num_events[t];
        if (denominator > 0) {
            KM_full[t] = previous_KM * (1.0 - num_censored_endpoints[t] / denominator);
        } else {
            KM_full[t] = previous_KM;
        }
        current_num_at_risk -= num_events[t] + num_censored_endpoints[t];
    }

    vector<double> KM_event;
    if (!forest_tree) {
        KM_event.assign(num_unique_event_times, 1);
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            if (event_censoring_time_ids[t] < num_censoring_times) {
                KM_event[t] = KM_full[event_censoring_time_ids[t]];
            }
        }
    }
    if (node_index < KM_censoring_full.size()) {
        KM_censoring_full[node_index] = std::move(KM_full);
    } else {
        KM_censoring_full.push_back(std::move(KM_full));
    }
    if (!forest_tree && node_index < KM_censoring.size()) {
        KM_censoring[node_index] = std::move(KM_event);
    } else if (!forest_tree) {
        KM_censoring.push_back(std::move(KM_event));
    }
}

/*
  Populate censoring estimators after fitting from exactly the same estimation sample which was used in the leaves.
  Keeping the original indices is important here: repeated bootstrap indices are repeated observations in the KM risk
  set, while honest trees must use their holdout sample rather than every observation which happens to route to a leaf.
*/
void MultistateTree::computeCensoringKMLazy() {
    if (censoring_km_ready) {
        return;
    }
    if (censoring_indices.empty()) {
        throw runtime_error("Cannot compute multi-state censoring from an empty estimation sample");
    }

    resizeKM();
    vector<vector<size_t>> leaf_groups(num_nodes);
    for (uint32_t observation : censoring_indices) {
        // route directly through the fitted tree; forest-owned trees are allowed to discard prediction_node_IDs
        leaf_groups[predictionLeafID(observation)].push_back(observation);
    }

    for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
        if (!leaf_groups[node_id].empty()) {
            computeCensoringKMExternal(leaf_groups[node_id], node_id);
        }
    }

    // once the leaf estimators exist, neither the bootstrap indices nor the event-to-censoring lookup is needed again
    vector<uint32_t>().swap(censoring_indices);
    vector<size_t>().swap(event_censoring_time_ids);
    censoring_endpoint_ids.reset();
    censoring_km_ready = true;
}

// splitting rules for multi-state trees
//--------------------------------------------------------------------------------------

double MultistateTree::computeSplitValue(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter) {
    switch (splitrule_id) {
        case MultistateSplitRule::LogRank:
            return logRank(num_jumps_daughter, num_at_risk_daughter);
        case MultistateSplitRule::Gehan:
            return Gehan(num_jumps_daughter, num_at_risk_daughter);
        case MultistateSplitRule::TaroneWare:
            return TaroneWare(num_jumps_daughter, num_at_risk_daughter);
        case MultistateSplitRule::Conserve:
            return conserve(num_jumps_daughter, num_at_risk_daughter);
        case MultistateSplitRule::ApproxLogRank:
            return approxLogRank(num_jumps_daughter, num_at_risk_daughter);
        case MultistateSplitRule::PetoPrentice:
            return petoPrentice(num_jumps_daughter, num_at_risk_daughter);
        case MultistateSplitRule::FlemingHarrington:
            return flemingHarrington(num_jumps_daughter, num_at_risk_daughter);
    }
    return -1;
}

double MultistateTree::logRank(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter) {
    double LR = 0;
    for (size_t jump_id = 0; jump_id < response_data->valid_jumps.size(); ++jump_id) {
        const MultistateValidJumpInfo& jump = response_data->valid_jumps[jump_id];
        double sum_num = 0;
        double sum_den = 0;

        // event times without this transition have zero contribution and were collected once for the parent node
        for (size_t t : jump_event_time_ids[jump_id]) {
            const double d = static_cast<double>(num_jumps[t * dim + jump.matrix_index]);
            const double d1 = static_cast<double>(num_jumps_daughter[t * dim + jump.matrix_index]);
            const double Y = static_cast<double>(num_at_risk[t * num_states + jump.from_state]);
            const double Y1 = static_cast<double>(num_at_risk_daughter[t * num_states + jump.from_state]);

            // state-specific risk sets can be replenished later, so an unusable time is skipped rather than ending the loop
            if (Y < 2 || Y1 < 1) {
                continue;
            }
            double at_risk_frac = Y1 / Y;
            sum_num += d1 - d * at_risk_frac;
            sum_den += d * at_risk_frac * (1.0 - at_risk_frac) * (Y - d) / (Y - 1.0);
        }

        if (sum_den != 0) {
            LR += sum_num * sum_num / sum_den;
        }
    }
    return LR > 0 ? LR : -1;
}

double MultistateTree::Gehan(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter) {
    double G = 0;
    for (size_t jump_id = 0; jump_id < response_data->valid_jumps.size(); ++jump_id) {
        const MultistateValidJumpInfo& jump = response_data->valid_jumps[jump_id];
        double sum_num = 0;
        double sum_den = 0;

        for (size_t t : jump_event_time_ids[jump_id]) {
            const double d = static_cast<double>(num_jumps[t * dim + jump.matrix_index]);
            const double d1 = static_cast<double>(num_jumps_daughter[t * dim + jump.matrix_index]);
            const double Y = static_cast<double>(num_at_risk[t * num_states + jump.from_state]);
            const double Y1 = static_cast<double>(num_at_risk_daughter[t * num_states + jump.from_state]);
            if (Y < 2 || Y1 < 1) {
                continue;
            }
            sum_num += Y * d1 - d * Y1;
            sum_den += d * Y1 * (Y - Y1) * (Y - d) / (Y - 1.0);
        }

        if (sum_den != 0) {
            G += sum_num * sum_num / sum_den;
        }
    }
    return G > 0 ? G : -1;
}

double MultistateTree::TaroneWare(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter) {
    double TW = 0;
    for (size_t jump_id = 0; jump_id < response_data->valid_jumps.size(); ++jump_id) {
        const MultistateValidJumpInfo& jump = response_data->valid_jumps[jump_id];
        double sum_num = 0;
        double sum_den = 0;

        for (size_t t : jump_event_time_ids[jump_id]) {
            const double d = static_cast<double>(num_jumps[t * dim + jump.matrix_index]);
            const double d1 = static_cast<double>(num_jumps_daughter[t * dim + jump.matrix_index]);
            const double Y = static_cast<double>(num_at_risk[t * num_states + jump.from_state]);
            const double Y1 = static_cast<double>(num_at_risk_daughter[t * num_states + jump.from_state]);
            if (Y < 2 || Y1 < 1) {
                continue;
            }
            double at_risk_frac = Y1 / Y;
            sum_num += sqrt(Y) * (d1 - d * at_risk_frac);
            sum_den += d * Y1 * (1.0 - at_risk_frac) * (Y - d) / (Y - 1.0);
        }

        if (sum_den != 0) {
            TW += sum_num * sum_num / sum_den;
        }
    }
    return TW > 0 ? TW : -1;
}

// WARNING: This splitting rule may be completely nonsensical for multi-states (but it seems to work with the absolute value fix)
double MultistateTree::conserve(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter) {
    double cons = 0;
    for (const MultistateValidJumpInfo& jump : response_data->valid_jumps) {
        double NAsum1 = 0;
        double NAsum2 = 0;
        double sum1_jump = 0;
        double sum2_jump = 0;

        for (size_t t = 0; t + 1 < num_unique_event_times; ++t) {
            size_t matrix_index = t * dim + jump.matrix_index;
            size_t d1 = num_jumps_daughter[matrix_index];
            size_t d2 = num_jumps[matrix_index] - d1;
            size_t at_risk1 = num_at_risk_daughter[t * num_states + jump.from_state];
            size_t at_risk2 = num_at_risk[t * num_states + jump.from_state] - at_risk1;
            if (d1 != 0 && at_risk1 != 0) {
                NAsum1 += static_cast<double>(d1) / at_risk1;
            }
            if (d2 != 0 && at_risk2 != 0) {
                NAsum2 += static_cast<double>(d2) / at_risk2;
            }

            size_t at_risk1_next = num_at_risk_daughter[(t + 1) * num_states + jump.from_state];
            size_t at_risk2_next = num_at_risk[(t + 1) * num_states + jump.from_state] - at_risk1_next;
            // unlike in survival, a state can gain individuals, hence the absolute differences
            size_t difference1 = at_risk1 > at_risk1_next ? at_risk1 - at_risk1_next : at_risk1_next - at_risk1;
            size_t difference2 = at_risk2 > at_risk2_next ? at_risk2 - at_risk2_next : at_risk2_next - at_risk2;
            sum1_jump += difference1 * at_risk1_next * NAsum1;
            sum2_jump += difference2 * at_risk2_next * NAsum2;
        }

        size_t parent_initial = num_at_risk[jump.from_state];
        if (parent_initial != 0) {
            size_t daughter_initial = num_at_risk_daughter[jump.from_state];
            size_t other_initial = parent_initial - daughter_initial;
            cons += (daughter_initial * sum1_jump + other_initial * sum2_jump) / parent_initial;
        }
    }
    return 1.0 / (1.0 + cons);
}

double MultistateTree::approxLogRank(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter) {
    double aLR = 0;
    for (size_t jump_id = 0; jump_id < response_data->valid_jumps.size(); ++jump_id) {
        const MultistateValidJumpInfo& jump = response_data->valid_jumps[jump_id];
        double D1 = 0;
        double D = 0;
        double sum_num = 0;

        for (size_t t : jump_event_time_ids[jump_id]) {
            const double d = static_cast<double>(num_jumps[t * dim + jump.matrix_index]);
            const double d1 = static_cast<double>(num_jumps_daughter[t * dim + jump.matrix_index]);
            const double Y = static_cast<double>(num_at_risk[t * num_states + jump.from_state]);
            const double Y1 = static_cast<double>(num_at_risk_daughter[t * num_states + jump.from_state]);
            if (Y == 0) {
                continue;
            }
            sum_num += d1 - Y1 * d / Y;
            D1 += d1;
            D += d;
        }

        double denominator_squared = (D1 - sum_num) * (D - D1 + sum_num);
        if (denominator_squared > 0 && D > 0) {
            aLR += abs(sqrt(D) * sum_num / sqrt(denominator_squared));
        }
    }
    return aLR > 0 ? aLR : -1;
}

double MultistateTree::petoPrentice(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter) {
    double PP = 0;
    for (size_t jump_id = 0; jump_id < response_data->valid_jumps.size(); ++jump_id) {
        const MultistateValidJumpInfo& jump = response_data->valid_jumps[jump_id];
        double sum_num = 0;
        double sum_den = 0;

        for (size_t t : jump_event_time_ids[jump_id]) {
            const double d = static_cast<double>(num_jumps[t * dim + jump.matrix_index]);
            const double d1 = static_cast<double>(num_jumps_daughter[t * dim + jump.matrix_index]);
            const double Y = static_cast<double>(num_at_risk[t * num_states + jump.from_state]);
            const double Y1 = static_cast<double>(num_at_risk_daughter[t * num_states + jump.from_state]);
            if (Y < 2 || Y1 < 1) {
                continue;
            }

            const double at_risk_frac = Y1 / Y;
            const double weight = parent_occupation_probs[t * num_states + jump.from_state] * Y / (1.0 + Y);
            sum_num += weight * (d1 - d * at_risk_frac);
            sum_den += weight * weight * d * at_risk_frac * (1.0 - at_risk_frac) * (Y - d) / (Y - 1.0);
        }

        if (sum_den > 0) {
            PP += sum_num / sqrt(sum_den);
        }
    }

    PP = abs(PP);
    return PP > 0 ? PP : -1;
}

double MultistateTree::flemingHarrington(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter) {
    long double FH = 0;
    for (size_t jump_id = 0; jump_id < response_data->valid_jumps.size(); ++jump_id) {
        const MultistateValidJumpInfo& jump = response_data->valid_jumps[jump_id];
        long double max_log_weight = -numeric_limits<long double>::infinity();

        // Find a transition-specific scale without computing potentially underflowing raw weights.
        for (size_t t : jump_event_time_ids[jump_id]) {
            const double d = static_cast<double>(num_jumps[t * dim + jump.matrix_index]);
            const double Y = static_cast<double>(num_at_risk[t * num_states + jump.from_state]);
            const double Y1 = static_cast<double>(num_at_risk_daughter[t * num_states + jump.from_state]);
            if (Y < 2 || Y1 < 1 || Y1 >= Y || d >= Y) {
                continue;
            }

            const double occupation_probability = clamp(parent_occupation_probs[t * num_states + jump.from_state], 0.0, 1.0);
            const long double log_weight = flemingHarringtonLogWeight(occupation_probability, (*fh_weights_a)[jump.from_state], (*fh_weights_b)[jump.from_state]);
            max_log_weight = max(max_log_weight, log_weight);
        }
        if (!isfinite(max_log_weight)) {
            continue;
        }

        long double sum_num = 0;
        long double sum_den = 0;
        for (size_t t : jump_event_time_ids[jump_id]) {
            const double d = static_cast<double>(num_jumps[t * dim + jump.matrix_index]);
            const double d1 = static_cast<double>(num_jumps_daughter[t * dim + jump.matrix_index]);
            const double Y = static_cast<double>(num_at_risk[t * num_states + jump.from_state]);
            const double Y1 = static_cast<double>(num_at_risk_daughter[t * num_states + jump.from_state]);
            if (Y < 2 || Y1 < 1 || Y1 >= Y || d >= Y) {
                continue;
            }

            const double occupation_probability = clamp(parent_occupation_probs[t * num_states + jump.from_state], 0.0, 1.0);
            const long double log_weight = flemingHarringtonLogWeight(occupation_probability, (*fh_weights_a)[jump.from_state], (*fh_weights_b)[jump.from_state]);
            if (!isfinite(log_weight)) {
                continue;
            }
            const long double weight = exp(log_weight - max_log_weight);
            const long double at_risk_frac = static_cast<long double>(Y1 / Y);
            sum_num += weight * (d1 - d * at_risk_frac);
            sum_den += weight * weight * d * at_risk_frac * (1.0 - at_risk_frac) * (Y - d) / (Y - 1.0);
        }

        if (sum_den > 0) {
            FH += sum_num / sqrt(sum_den);
        }
    }

    FH = abs(FH);
    if (!isfinite(FH) || FH <= 0) {
        return -1;
    }
    return static_cast<double>(FH);
}

// prediction for multi-state trees
//--------------------------------------------------------------------------------------

// compute a flattened vector of predictions (the Nelson-Aalen estimators at each event time)
vector<double> MultistateTree::computePredictions(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    size_t prediction_size = num_unique_event_times * dim;
    vector<double> predictions(num_obs * prediction_size);
    for (size_t i = 0; i < num_obs; ++i) {
        size_t leaf_id = predictionLeafID(new_data, i);
        const vector<double>& pred = na[leaf_id];
        copy(pred.begin(), pred.end(), predictions.begin() + i * prediction_size);
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
    bool new_data_provided = new_data.getNumberOfObs() != 0;
    size_t num_obs = new_data_provided ? new_data.getNumberOfObs() : data->getNumberOfObs();
    size_t prediction_size = num_unique_event_times * dim;

    // optional outputs are not allocated unless they were requested
    vector<double> predictions(num_obs * prediction_size);
    vector<double> predictions_init;
    vector<double> censoring;
    if (compute_initial) {
        predictions_init.resize(num_obs * num_states);
    }
    if (compute_censoring) {
        censoring.resize(num_obs * num_censoring_times);
    }

    for (size_t i = 0; i < num_obs; ++i) {
        // route observations here as this method is also available to forest-owned trees, where prediction_node_IDs
        // is deliberately released after growing
        size_t leaf_id = new_data_provided ? predictionLeafID(new_data, i) : predictionLeafID(i);
        const vector<double>& pred = na[leaf_id];
        copy(pred.begin(), pred.end(), predictions.begin() + i * prediction_size);

        if (compute_initial) {
            const vector<double>& init = init_dist[leaf_id];
            copy(init.begin(), init.end(), predictions_init.begin() + i * num_states);
        }
        if (compute_censoring) {
            const vector<double>& cens = KM_censoring_full[leaf_id];
            copy(cens.begin(), cens.end(), censoring.begin() + i * num_censoring_times);
        }
    }

    // initializer-list construction copies every inner vector, so move the potentially large outputs explicitly
    vector<vector<double>> result;
    result.reserve(1 + static_cast<size_t>(compute_initial) + static_cast<size_t>(compute_censoring));
    result.push_back(std::move(predictions));
    if (compute_initial) {
        result.push_back(std::move(predictions_init));
    }
    if (compute_censoring) {
        result.push_back(std::move(censoring));
    }
    return result;
}

/*
  Error computations need occupation probabilities and censoring predictions, not one complete Nelson--Aalen matrix for
  every observation. Compute an occupation curve once for each visited leaf and copy the much smaller curve to its rows.
  This removes a num_states factor from the observation-level prediction memory used during scoring.
*/
vector<vector<double>> MultistateTree::computeErrorPredictions(const Data& new_data) {
    bool new_data_provided = new_data.getNumberOfObs() != 0;
    size_t num_obs = new_data_provided ? new_data.getNumberOfObs() : data->getNumberOfObs();
    size_t occupation_size = num_unique_event_times * num_states;

    vector<double> occupation_probabilities(num_obs * occupation_size);
    vector<double> censoring(num_obs * num_censoring_times);
    vector<vector<double>> occupation_cache(num_nodes);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t leaf_id = new_data_provided ? predictionLeafID(new_data, i) : predictionLeafID(i);
        vector<double>& leaf_occupation = occupation_cache[leaf_id];

        // observations in the same leaf have identical Nelson--Aalen estimators and initial distributions
        if (leaf_occupation.empty()) {
            leaf_occupation = occupationProbabilitiesCpp(na[leaf_id], init_dist[leaf_id], num_states, 1);
        }
        copy(leaf_occupation.begin(), leaf_occupation.end(),
             occupation_probabilities.begin() + i * occupation_size);

        const vector<double>& leaf_censoring = KM_censoring_full[leaf_id];
        copy(leaf_censoring.begin(), leaf_censoring.end(), censoring.begin() + i * num_censoring_times);
    }

    vector<vector<double>> result;
    result.reserve(2);
    result.push_back(std::move(occupation_probabilities));
    result.push_back(std::move(censoring));
    return result;
}

// error estimation for multi-state trees
//--------------------------------------------------------------------------------------

// computes a vector of the Brier score using given IPCW weights for multi-state predictions 
// (states_ind is a flattened vector of boolean indicators of whether observation i at event time t is in state j)
vector<double> computeBrierScoreMultistate(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times,
                                           const List& occupation_probs, const vector<double>& state_weights) {
    size_t num_unique_event_times = unique_event_times.size();
    size_t num_states = state_weights.size();
    size_t num_obs = weights.size() / num_unique_event_times;
    vector<double> brier(num_unique_event_times * num_states, 0);

    for (size_t i = 0; i < num_obs; ++i) {
        const List& occ_probs_obs = as<List>(occupation_probs[i]);
        size_t obs_index = i * num_unique_event_times * num_states;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            NumericVector occ_probs_obs_t = occ_probs_obs[t];
            size_t time_index = obs_index + t * num_states;
            double ipcw = weights[i * num_unique_event_times + t];
            if (ipcw == 0) {
                continue;
            }
            // difference to survival: need to compute a contribution to the score across all states
            for (size_t state = 0; state < num_states; ++state) {
                double residual = occ_probs_obs_t[state] - static_cast<double>(states_ind[time_index + state]);
                brier[t * num_states + state] += ipcw * residual * residual * state_weights[state];
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
    size_t num_states = state_weights.size();
    size_t num_obs = weights.size() / num_unique_event_times;
    vector<double> kl(num_unique_event_times * num_states, 0);

    for (size_t i = 0; i < num_obs; ++i) {
        const List& occ_probs_obs = as<List>(occupation_probs[i]);
        size_t obs_index = i * num_unique_event_times * num_states;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            NumericVector occ_probs_obs_t = occ_probs_obs[t];
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
    size_t num_states = state_weights.size();
    size_t num_obs = weights.size() / num_unique_event_times;
    vector<double> brier(num_unique_event_times * num_states, 0);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t obs_index = i * num_unique_event_times * num_states;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            size_t time_index = obs_index + t * num_states;
            double ipcw = weights[i * num_unique_event_times + t];
            if (ipcw == 0) {
                continue;
            }
            // difference to survival: need to compute a contribution to the score across all states
            for (size_t state = 0; state < num_states; ++state) {
                double residual = occupation_probs[time_index + state] -
                  static_cast<double>(states_ind[time_index + state]);
                brier[t * num_states + state] += ipcw * residual * residual * state_weights[state];
            }
        }
    }
    return brier;
}

// same function as above but where the occupation probabilities are instead given by a flattened vector
vector<double> computeKLScoreCppMultistate(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times,
                                           const vector<double>& occupation_probs, const vector<double>& state_weights) {
    size_t num_unique_event_times = unique_event_times.size();
    size_t num_states = state_weights.size();
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
    result.shrink_to_fit();
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
    // this grid is retained by the fitted tree, so release capacity left by duplicate endpoints
    result.shrink_to_fit();
    return result;
}

/*
  maps each response time to the fitted event-time grid. When the grid is thinned, several successive transitions can
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
    vector<size_t> transition_positions;
    vector<size_t> mapped_transition_ids;
    transition_positions.reserve(max_response_length - 1);
    mapped_transition_ids.reserve(max_response_length - 1);

    for (size_t obs = 0; obs < num_obs; ++obs) {
        const size_t offset = obs * max_response_length;

        // determine the response length of the observation 
        // (the total number of states occupied until absorption)
        size_t response_length = 1;
        while (response_length < max_response_length && states[offset + response_length] != 0) {
            ++response_length;
        }

        // reuse these two small buffers instead of allocating them separately for every observation
        transition_positions.clear();
        mapped_transition_ids.clear();

        for (size_t j = 1; j < response_length; ++j) {
            const size_t index = offset + j;
            // means that censoring has occured
            if (states[index] == states[index - 1]) {
                continue;
            }

            // find the smallest unique event time larger than or equal to the current observation time
            auto it = lower_bound(unique_event_times.begin(), unique_event_times.end(), times[index]);
            // find the number of steps from the first unique event time until the current observation time
            size_t event_time_id = static_cast<size_t>(it - unique_event_times.begin());
            // ensures that we don't choose an event time which is out of bounds
            if (event_time_id >= num_event_times) {
                event_time_id = num_event_times - 1;
            }
            transition_positions.push_back(index);
            mapped_transition_ids.push_back(event_time_id);
        }

        // we compute the number of transitions separately to ensure that the grid is big enough to handle it
        const size_t num_transitions = mapped_transition_ids.size();
        if (num_transitions >= num_event_times) {
            throw invalid_argument("num_event_times is too small to preserve the order of a multi-state response path");
        }

        // prefer the existing lower-bound mapping and only move a transition when
        // thinning would collapse it onto an earlier transition from the same path
        for (size_t j = 0; j < num_transitions; ++j) {
            // ensure that the earliest jump time is never zero and that a new transition always gets its own time
            const size_t earliest_id = j == 0 ? 1 : mapped_transition_ids[j - 1] + 1;
            mapped_transition_ids[j] = max(mapped_transition_ids[j], earliest_id);
        }

        // near the right edge there may not be room to move transitions forward
        // move the affected tail backwards while retaining strict ordering
        if (num_transitions > 0 && mapped_transition_ids.back() >= num_event_times) {
            mapped_transition_ids.back() = num_event_times - 1;
            for (size_t j = num_transitions - 1; j > 0; --j) {
                mapped_transition_ids[j - 1] = min(mapped_transition_ids[j - 1], mapped_transition_ids[j] - 1);
            }
        }

        for (size_t j = 0; j < num_transitions; ++j) {
            response_event_time_ids[transition_positions[j]] = mapped_transition_ids[j];
        }

        if (response_length >= 2) {
            const size_t last_index = offset + response_length - 1;
            // if censoring occurs
            if (states[last_index] == states[last_index - 1]) {
                // determine the first element in unique_event_times larger than the current time
                auto it = upper_bound(unique_event_times.begin(), unique_event_times.end(), times[last_index]);
                // determine number of jumps until the censoring time
                size_t censoring_time_id = static_cast<size_t>(it - unique_event_times.begin());
                // for ensuring that censoring always occurs after the final event has taken place
                if (num_transitions > 0) {
                    censoring_time_id = max(censoring_time_id, mapped_transition_ids.back() + 1);
                }
                response_event_time_ids[last_index] = censoring_time_id;
            }
        }
    }
    return response_event_time_ids;
}

/*
  computes the observed state directly from each response path. The generic Data implementation reconstructs dense jump,
  cumulative-jump, at-risk and censoring arrays for every observation; scoring only needs one true state bit at each time.
*/
vector<bool> computeStateIndicatorsMultistate(const Data& data, const vector<size_t>& response_event_time_ids, size_t num_unique_event_times) {
    const vector<uint8_t>& states = data.getStates();
    const vector<size_t>& last_observed_times = data.getLastObservedTimes();
    const vector<uint8_t>& censoring_states = data.getCensoringStates();
    size_t max_response_length = data.getMaxResponseLength();
    size_t num_states = data.getNumberOfStates();
    size_t num_obs = data.getNumberOfObs();

    if (response_event_time_ids.size() != states.size()) {
        throw invalid_argument("Response event-time IDs do not match the multi-state response data");
    }

    vector<bool> result(num_obs * num_unique_event_times * num_states, false);
    for (size_t i = 0; i < num_obs; ++i) {
        // first determine the actual length of the observation
        size_t response_offset = i * max_response_length;
        size_t response_length = 1;
        while (response_length < max_response_length && states[response_offset + response_length] != 0) {
            ++response_length;
        }

        // determine the id of the censoring time (= num_unique_event_times for no censoring)
        size_t current_state = states[response_offset] - 1;
        size_t next_response_position = 1;
        size_t censoring_time_id = num_unique_event_times;
        if (censoring_states[i] != 0) {
            censoring_time_id = response_event_time_ids[last_observed_times[i]];
        }

        // for every unique event time, determine the state indicator
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            // transitions at t change the observed state at t, while a repeated final state only marks censoring
            while (next_response_position < response_length) {
                size_t response_index = response_offset + next_response_position;
                if (states[response_index] == states[response_index - 1]) {
                    ++next_response_position;
                    continue;
                }
                if (response_event_time_ids[response_index] > t) {
                    break;
                }
                current_state = states[response_index] - 1;
                ++next_response_position;
            }

            // if no censoring has occured, update the state indicator
            if (t < censoring_time_id) {
                result[(i * num_unique_event_times + t) * num_states + current_state] = true;
            }
        }
    }
    return result;
}

// computes occupation probabilities from flattened Nelson-Aalen estimators and one initial distribution per estimator
vector<double> occupationProbabilitiesCpp(const vector<double>& na, const vector<double>& init, size_t num_states, size_t num_estimators) {
    if (num_estimators == 0 || num_states == 0) {
        return {};
    }
    size_t stride_length = na.size() / num_estimators;
    size_t dim = num_states * num_states;
    size_t num_unique_event_times = stride_length / dim;
    vector<double> result(num_estimators * num_unique_event_times * num_states, 0);     // num_estimators corresponds to num_obs
    vector<double> previous(num_states, 0);
    vector<double> current(num_states, 0);

    /*
      we only need init * product integral, not the full product-integral matrix. Update that probability row directly,
      reducing both the working memory and one complete matrix dimension in the computation (rolling matrix approach)
    */
    for (size_t estimator = 0; estimator < num_estimators; ++estimator) {
        size_t init_index = estimator * num_states;
        size_t result_index = estimator * num_unique_event_times * num_states;
        copy(init.begin() + init_index, init.begin() + init_index + num_states, previous.begin());
        copy(previous.begin(), previous.end(), result.begin() + result_index);          // p(0) = init

        size_t na_index = estimator * num_unique_event_times * dim;
        for (size_t t = 1; t < num_unique_event_times; ++t) {
            fill(current.begin(), current.end(), 0);
            size_t current_matrix = na_index + t * dim;
            size_t previous_matrix = current_matrix - dim;
            for (size_t from_state = 0; from_state < num_states; ++from_state) {
                double previous_probability = previous[from_state];
                for (size_t to_state = 0; to_state < num_states; ++to_state) {
                    size_t matrix_index = from_state * num_states + to_state;
                    double contribution = na[current_matrix + matrix_index] - na[previous_matrix + matrix_index];
                    if (from_state == to_state) {
                        ++contribution;
                    }
                    current[to_state] += previous_probability * contribution;
                }
            }
            copy(current.begin(), current.end(), result.begin() + result_index + t * num_states);
            previous.swap(current);
        }
    }
    return result;
}

// same function as before, but now accepts List inputs instead for na and init (each is a List of Lists)
vector<double> occupationProbabilities(const List& na, const List& init, size_t num_states) {
    size_t num_obs = na.size();
    if (num_obs == 0 || num_states == 0) {
        return {};
    }
    size_t num_unique_event_times = as<List>(na[0]).size();
    vector<double> result(num_obs * num_unique_event_times * num_states, 0);
    vector<double> previous(num_states, 0);
    vector<double> current(num_states, 0);

    for (size_t observation = 0; observation < num_obs; ++observation) {
        List na_obs = na[observation];
        NumericVector init_obs = init[observation];
        if (static_cast<size_t>(init_obs.size()) != num_states) {
            throw invalid_argument("Initial distributions must have one entry for every state");
        }
        size_t result_index = observation * num_unique_event_times * num_states;
        copy_n(init_obs.begin(), num_states, previous.begin());
        copy(previous.begin(), previous.end(), result.begin() + result_index);

        for (size_t t = 1; t < num_unique_event_times; ++t) {
            NumericMatrix na_obs_cur = na_obs[t];
            NumericMatrix na_obs_prev = na_obs[t - 1];
            fill(current.begin(), current.end(), 0);
            for (size_t from_state = 0; from_state < num_states; ++from_state) {
                double previous_probability = previous[from_state];
                for (size_t to_state = 0; to_state < num_states; ++to_state) {
                    double contribution = na_obs_cur(from_state, to_state) - na_obs_prev(from_state, to_state);
                    if (from_state == to_state) {
                        ++contribution;
                    }
                    current[to_state] += previous_probability * contribution;
                }
            }
            copy(current.begin(), current.end(), result.begin() + result_index + t * num_states);
            previous.swap(current);
        }
    }
    return result;
}

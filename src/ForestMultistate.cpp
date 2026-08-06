#include "ForestMultistate.h"
#include "TreeSurvival.h"
#include <exception>

namespace {

/*
  Computes init * product integral directly. VIMP only needs one probability row, not the complete state-by-state
  product-integral matrix. Updating that row reduces the work from O(TS^3) to O(TS^2) and the workspace to two S-vectors.
*/
vector<double> occupationProbabilitiesVIMP(const vector<double>& na, const vector<double>& init, size_t num_states) {
    const size_t dim = num_states * num_states;
    const size_t num_event_times = na.size() / dim;
    vector<double> result(num_event_times * num_states, 0);
    vector<double> previous(init);
    vector<double> current(num_states, 0);

    // the occupation probabilities at time zero are exactly the predicted initial distribution
    copy(previous.begin(), previous.end(), result.begin());
    for (size_t t = 1; t < num_event_times; ++t) {
        fill(current.begin(), current.end(), 0);
        const size_t current_matrix = t * dim;
        const size_t previous_matrix = current_matrix - dim;

        for (size_t from_state = 0; from_state < num_states; ++from_state) {
            const double previous_probability = previous[from_state];
            for (size_t to_state = 0; to_state < num_states; ++to_state) {
                const size_t matrix_index = from_state * num_states + to_state;
                double contribution = na[current_matrix + matrix_index] -
                                      na[previous_matrix + matrix_index];
                if (from_state == to_state) {
                    ++contribution;
                }
                current[to_state] += previous_probability * contribution;
            }
        }
        copy(current.begin(), current.end(), result.begin() + t * num_states);
        previous.swap(current);
    }
    return result;
}

double sphericalLossVIMP(const vector<double>& occupation, size_t time_offset, size_t observed_state, const vector<double>& state_weights) {
    double denominator_squared = 0;
    for (size_t j = 0; j < state_weights.size(); ++j) {
        double probability = occupation[time_offset + j];
        denominator_squared += state_weights[j] * probability * probability;
    }
    double reward = denominator_squared > 0 ?
        state_weights[observed_state] * occupation[time_offset + observed_state] / sqrt(denominator_squared) : 0;
    return 1 - reward;
}

/*
  VIMP scoring needs a single observed state ID at each time. Sweep each response path directly instead of first creating
  N*T*S packed indicators and then searching every state entry to recover the one true ID.
*/
vector<uint8_t> observedStateIDsVIMP(const Data& data, const vector<size_t>& response_event_time_ids, size_t num_event_times) {
    const vector<uint8_t>& states = data.getStates();
    const vector<size_t>& last_observed_times = data.getLastObservedTimes();
    const vector<uint8_t>& censoring_states = data.getCensoringStates();
    const size_t max_response_length = data.getMaxResponseLength();
    const size_t num_obs = data.getNumberOfObs();
    const uint8_t missing_state = data.getNumberOfStates();
    vector<uint8_t> result(num_obs * num_event_times, missing_state);   // num_states is the placeholder value when the event is not observed

    // determine the actual response length from the padded data representation
    for (size_t i = 0; i < num_obs; ++i) {
        const size_t response_offset = i * max_response_length;
        size_t response_length = 1;
        while (response_length < max_response_length &&
               states[response_offset + response_length] != 0) {
            ++response_length;
        }

        size_t current_state = states[response_offset] - 1;
        size_t next_response_position = 1;

        // if censoring occurs, determine the id of the censoring time
        size_t censoring_time_id = num_event_times;
        if (censoring_states[i] != 0) {
            censoring_time_id = response_event_time_ids[last_observed_times[i]];
        }

        // now move forward in the padded representation until the time of the event is found (if censoring has not occured)
        for (size_t t = 0; t < num_event_times; ++t) {
            while (next_response_position < response_length) {
                const size_t response_index = response_offset + next_response_position;
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
            if (t < censoring_time_id) {
                result[i * num_event_times + t] = static_cast<uint8_t>(current_state);
            }
        }
    }
    return result;
}

}

// constructor for multi-state forests
//--------------------------------------------------------------------------------------
MultistateForest::MultistateForest(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids,
                                   uint8_t num_states, bool save_predictions,
                                   shared_ptr<const vector<double>> fh_weights_a,
                                   shared_ptr<const vector<double>> fh_weights_b) :
    save_predictions {save_predictions}, unique_event_times {unique_event_times},
    response_event_time_ids {response_event_time_ids}, num_unique_event_times {unique_event_times.size()},
    num_states {num_states}, dim {static_cast<size_t>(num_states) * num_states},
    fh_weights_a {std::move(fh_weights_a)}, fh_weights_b {std::move(fh_weights_b)} {}

// functions for growing multi-state forests
//--------------------------------------------------------------------------------------

// grows a multi-state forest using multithreading via OpenMP
void MultistateForest::grow() {
    if (ntrees == 0) {
        throw runtime_error("The number of trees must be at least one");
    }
    if (min_node_size == 0) {
        throw runtime_error("The minimal node size must be at least one");
    }
    const vector<bool>& categorical = data->getCategorical();
    const vector<size_t> unique_values = data->getUniqueValues();
    for (size_t feature = 0; feature < categorical.size(); ++feature) {
        if (categorical[feature] && unique_values[feature] > 63) {
            throw runtime_error("Categorical features with more than 63 values are not supported");
        }
    }

    clearVIMPCache();
    size_t n = data->getNumberOfObs();
    size_t subsample_size = floor(sample_rate * n);
    if (!honest && subsample_size == 0) {
        throw runtime_error("The sampled multi-state growing set must contain at least one observation");
    }
    if (honest && !double_bootstrap && subsample_size < 2) {
        throw runtime_error("An honest multi-state tree needs at least one growing and one estimation observation");
    }
    if (honest && double_bootstrap) {
        size_t growing_half_size = n / 2;
        size_t estimation_half_size = n - growing_half_size;
        if (floor(sample_rate * growing_half_size) == 0 ||
            floor(sample_rate * estimation_half_size) == 0) {
            throw runtime_error("An honest multi-state tree needs at least one growing and one estimation observation");
        }
    }
    
    // create vector of indices from 1 to n
    vector<size_t> global_indices(n);
    for (size_t i = 0; i < n; ++i) {
        global_indices[i] = i;
    }

    trees.resize(ntrees);
    oob_indices.resize(ntrees);

    // create pointers to construct the trees
    shared_ptr<vector<double>> unique_event_times = make_shared<vector<double>>(this->unique_event_times);
    censoring_times = make_shared<vector<double>>(
        uniqueCensoringTimesMultistate(this->unique_event_times, data->getTimes(), data->getLastObservedTimes()));
    event_censoring_time_ids.assign(num_unique_event_times, censoring_times->size());
    size_t censoring_time_id = 0;
    for (size_t t = 0; t < num_unique_event_times; ++t) {
        while (censoring_time_id + 1 < censoring_times->size() &&
               (*censoring_times)[censoring_time_id + 1] <= this->unique_event_times[t]) {
            ++censoring_time_id;
        }
        if (censoring_time_id < censoring_times->size() &&
            (*censoring_times)[censoring_time_id] <= this->unique_event_times[t]) {
            event_censoring_time_ids[t] = censoring_time_id;
        }
    }
    shared_ptr<const MultistateResponseData> response_data = prepareMultistateResponseData(
        *data, this->response_event_time_ids, num_unique_event_times, num_states);
    auto endpoint_ids = make_shared<vector<size_t>>(n, censoring_times->size());
    const vector<double>& response_times = data->getTimes();
    const vector<size_t>& last_observed_times = data->getLastObservedTimes();
    for (size_t observation = 0; observation < n; ++observation) {
        const double endpoint = response_times[last_observed_times[observation]];
        (*endpoint_ids)[observation] = static_cast<size_t>(
            lower_bound(censoring_times->begin(), censoring_times->end(), endpoint) -
            censoring_times->begin());
    }
    shared_ptr<const vector<size_t>> censoring_endpoint_ids = endpoint_ids;

    size_t n_threads = this->nworkers;
    Rcout << "Growing forest using " << n_threads << " threads" << endl;
    vector<string> tree_errors(ntrees);

    // use OpenMP for parallel tree growing
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
      try {
        // give each thread its own random number generator to prevent races
        mt19937 local_rng(seed + i);
        unique_ptr<MultistateTree> tree;

        vector<size_t> bootstrap_indices;
        
        // bootstrap
        if (!honest) {
            bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
            oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
            tree = make_unique<MultistateTree>(
                unique_event_times, nullptr, std::move(bootstrap_indices), num_states,
                save_predictions, vector<size_t>(), censoring_times, response_data,
                censoring_endpoint_ids, fh_weights_a, fh_weights_b);
        } 
        // for honest trees, we differ between double and single bootstrap
        else {
            if (double_bootstrap) {
                // in the case of double bootstrap, first split into growing and holdout sets and then bootstrap separately
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(global_indices, local_rng);
                size_t grow_size = floor(sample_rate * partition.first.size());
                size_t holdout_size = floor(sample_rate * partition.second.size());
                vector<size_t> grow = sampleIndices(partition.first, grow_size, swr, local_rng);
                vector<size_t> holdout = sampleIndices(partition.second, holdout_size, swr, local_rng);
                oob_indices[i] = computeOOBIndicesDouble(grow, holdout, n);
                tree = make_unique<MultistateTree>(
                    unique_event_times, nullptr, std::move(grow), num_states,
                    save_predictions, std::move(holdout), censoring_times, response_data,
                    censoring_endpoint_ids, fh_weights_a, fh_weights_b);

            } else {
                // if no double bootstrap, bootstrap the whole dataset and then split
                bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(bootstrap_indices, local_rng);
                oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
                tree = make_unique<MultistateTree>(
                    unique_event_times, nullptr, std::move(partition.first), num_states,
                    save_predictions, std::move(partition.second), censoring_times, response_data,
                    censoring_endpoint_ids, fh_weights_a, fh_weights_b);
            }
        }

        uniform_int_distribution<size_t> dist(0, numeric_limits<size_t>::max());
        tree->initialise(data, mtry, min_node_size, max_depth, nsplits, splitrule, honest, dist(local_rng));
        // forest trees route observations directly, so remove the N-entry base allocation before growing any nodes
        tree->releasePredictionNodeIDs();
        tree->setRNG(local_rng);
        tree->grow();
        trees[i] = std::move(tree);
      } catch (const std::exception& error) {
        tree_errors[i] = error.what();
      }
    }

    for (const string& error : tree_errors) {
        if (!error.empty()) {
            throw runtime_error(error);
        }
    }

    // compute all quantities of interest from the vector of trees
    computeForestQuantities();
}

// functions for predicting with multi-state forests
//--------------------------------------------------------------------------------------

vector<double> MultistateForest::predict(const vector<double>& x) {
    const size_t prediction_size = num_unique_event_times * dim;
    vector<double> result(prediction_size, 0);
    for (const auto& base_tree : trees) {
        MultistateTree* tree = static_cast<MultistateTree*>(base_tree.get());
        const vector<double>& prediction = tree->getNA()[tree->predictionLeafID(x)];
        for (size_t k = 0; k < prediction_size; ++k) {
            result[k] += prediction[k];
        }
    }
    for (double& value : result) {
        value /= ntrees;
    }
    return result;
}

/*

Computes all predictions, both in-bag and OOB, result is a vector with two, four or five flattened vectors
depending on the parameters. First two vectors are always in-bag and OOB Nelson-Aalen estimators, and the
same structure applies to the remaining output vectors. If compute_initial = true, the next two vectors are
in-bag and OOB predicted initial distributions, and if compute_censoring = true, the OOB censoring predictions
are added to the result vector.

*/

vector<vector<double>> MultistateForest::computePredictions(bool compute_initial, bool compute_censoring) {
    const size_t num_obs = data->getNumberOfObs();
    const size_t prediction_size = num_unique_event_times * dim;

    // initialise vectors of predictions (in-bag and OOB), predicted initial distributions and censoring distributions
    vector<double> predictions(num_obs * prediction_size, 0);
    vector<double> oob_predictions(num_obs * prediction_size, 0);
    vector<double> predictions_init;
    vector<double> oob_predictions_init;
    vector<double> oob_censoring;
    if (compute_initial) {
        predictions_init.assign(num_obs * num_states, 0);
        oob_predictions_init.assign(num_obs * num_states, 0);
    }
    if (compute_censoring) {
        oob_censoring.assign(num_obs * num_unique_event_times, 0);
    }

    // resolve the derived pointers once, rather than performing a checked cast for every observation and tree
    vector<MultistateTree*> multistate_trees(ntrees);
    for (size_t tree_id = 0; tree_id < ntrees; ++tree_id) {
        multistate_trees[tree_id] = static_cast<MultistateTree*>(trees[tree_id].get());
    }

    #pragma omp parallel for schedule(static) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        const size_t prediction_offset = i * prediction_size;
        const size_t initial_offset = i * num_states;
        const size_t censoring_offset = i * num_unique_event_times;
        size_t num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = multistate_trees[j];
            const size_t leaf_id = tree->predictionLeafID(i);
            const vector<double>& tree_prediction = tree->getNA()[leaf_id];
            const bool is_oob = oob_indices[j][i];

            for (size_t k = 0; k < prediction_size; ++k) {
                predictions[prediction_offset + k] += tree_prediction[k];
            }
            if (is_oob) {
                ++num_oob_trees;
                for (size_t k = 0; k < prediction_size; ++k) {
                    oob_predictions[prediction_offset + k] += tree_prediction[k];
                }
            }

            if (compute_initial) {
                const vector<double>& tree_initial = tree->getInitDist()[leaf_id];
                for (size_t state = 0; state < num_states; ++state) {
                    predictions_init[initial_offset + state] += tree_initial[state];
                    if (is_oob) {
                        oob_predictions_init[initial_offset + state] += tree_initial[state];
                    }
                }
            }

            // as before, censoring predictions are OOB predictions and therefore only use OOB trees
            if (compute_censoring && is_oob) {
                const vector<double>& tree_censoring = tree->getKMCensoringFull()[leaf_id];
                for (size_t t = 0; t < num_unique_event_times; ++t) {
                    const size_t source_id = event_censoring_time_ids[t];
                    oob_censoring[censoring_offset + t] += source_id < tree_censoring.size() ?
                        tree_censoring[source_id] : 1.0;
                }
            }
        }

        // normalise and save NA predictions
        for (size_t k = 0; k < prediction_size; ++k) {
            predictions[prediction_offset + k] /= ntrees;
            if (num_oob_trees > 0) {
                oob_predictions[prediction_offset + k] /= num_oob_trees;
            }
        }

        // normalise and save initial distribution predictions
        if (compute_initial) {
            for (size_t state = 0; state < num_states; ++state) {
                predictions_init[initial_offset + state] /= ntrees;
                if (num_oob_trees > 0) {
                    oob_predictions_init[initial_offset + state] /= num_oob_trees;
                }
            }
        }

        // normalise and save censoring distribution predictions
        if (compute_censoring && num_oob_trees > 0) {
            for (size_t t = 0; t < num_unique_event_times; ++t) {
                oob_censoring[censoring_offset + t] /= num_oob_trees;
            }
        }
    }

    // initializer-list construction copies every large vector, so append the optional results by move
    vector<vector<double>> result;
    result.reserve(2 + 2 * static_cast<size_t>(compute_initial) + static_cast<size_t>(compute_censoring));
    result.push_back(std::move(predictions));
    result.push_back(std::move(oob_predictions));
    if (compute_initial) {
        result.push_back(std::move(predictions_init));
        result.push_back(std::move(oob_predictions_init));
    }
    if (compute_censoring) {
        result.push_back(std::move(oob_censoring));
    }
    return result;
}

/*

Computes all predictions on a new dataset, result is a vector with one, two or three flattened vectors
depending on the parameters. First vector is always a flattened vector of Nelson-Aalen estimators. If
compute_initial = true, the next two vectors are 
in-bag and OOB predicted initial distributions, and if compute_censoring = true, the OOB censoring predictions
are added to the result vector.

*/

vector<vector<double>> MultistateForest::computePredictions(const Data& new_data, bool compute_initial, bool compute_censoring) {
    const size_t num_obs = new_data.getNumberOfObs();
    const size_t prediction_size = num_unique_event_times * dim;
    
    // initialise vectors of predicted NA estimators, initial distributions and censoring
    vector<double> predictions(num_obs * prediction_size, 0);
    vector<double> predictions_init, censoring;
    if (compute_initial) {
        predictions_init.assign(num_obs * num_states, 0);
    }
    if (compute_censoring) {
        censoring.assign(num_obs * num_unique_event_times, 0);
    }

    vector<MultistateTree*> multistate_trees(ntrees);
    for (size_t tree_id = 0; tree_id < ntrees; ++tree_id) {
        multistate_trees[tree_id] = static_cast<MultistateTree*>(trees[tree_id].get());
    }

    #pragma omp parallel for schedule(static) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        const size_t prediction_offset = i * prediction_size;
        const size_t initial_offset = i * num_states;
        const size_t censoring_offset = i * num_unique_event_times;

        // aggregate predictions over all trees for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = multistate_trees[j];
            const size_t leaf_id = tree->predictionLeafID(new_data, i);
            const vector<double>& tree_prediction = tree->getNA()[leaf_id];
            for (size_t k = 0; k < prediction_size; ++k) {
                predictions[prediction_offset + k] += tree_prediction[k];
            }

            if (compute_initial) {
                const vector<double>& tree_initial = tree->getInitDist()[leaf_id];
                for (size_t state = 0; state < num_states; ++state) {
                    predictions_init[initial_offset + state] += tree_initial[state];
                }
            }
            if (compute_censoring) {
                const vector<double>& tree_censoring = tree->getKMCensoringFull()[leaf_id];
                for (size_t t = 0; t < num_unique_event_times; ++t) {
                    const size_t source_id = event_censoring_time_ids[t];
                    censoring[censoring_offset + t] += source_id < tree_censoring.size() ?
                        tree_censoring[source_id] : 1.0;
                }
            }
        }

        // normalise and save predictions
        for (size_t k = 0; k < prediction_size; ++k) {
            predictions[prediction_offset + k] /= ntrees;
        }

        // normalise and save predictions for initial distributions
        if (compute_initial) {
            for (size_t state = 0; state < num_states; ++state) {
                predictions_init[initial_offset + state] /= ntrees;
            }
        }

        // normalise and save predictions for censoring distributions
        if (compute_censoring) {
            for (size_t t = 0; t < num_unique_event_times; ++t) {
                censoring[censoring_offset + t] /= ntrees;
            }
        }
    }

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

// Populate leaf-level censoring estimators when they were not saved during fitting.
void MultistateForest::computePredictionsCensoring() {
    if (save_predictions) {
        return;
    }
    clearVIMPCache();
    vector<string> tree_errors(ntrees);
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t j = 0; j < ntrees; ++j) {
      try {
        MultistateTree* tree = static_cast<MultistateTree*>(trees[j].get());
        tree->computeCensoringKMLazy();
      } catch (const std::exception& error) {
        tree_errors[j] = error.what();
      }
    }
    for (const string& error : tree_errors) {
        if (!error.empty()) {
            throw runtime_error(error);
        }
    }
    save_predictions = true;
}

/*
  Error computation needs occupation probabilities and a censoring curve, not one N*T*S*S Nelson--Aalen output retained
  for the complete dataset. Average one observation at a time, immediately turn its averaged estimator into occupation
  probabilities, and reuse the large T*S*S workspace for the next observation.
*/
vector<vector<double>> MultistateForest::computeErrorPredictions(const Data& new_data) {
    if (!save_predictions) {
        computePredictionsCensoring();
    }
    const bool new_data_provided = new_data.getNumberOfObs() != 0;
    const size_t num_obs = new_data_provided ? new_data.getNumberOfObs() : data->getNumberOfObs();
    const size_t prediction_size = num_unique_event_times * dim;
    const size_t occupation_size = num_unique_event_times * num_states;
    const size_t num_censoring_times = censoring_times->size();
    vector<double> occupation(num_obs * occupation_size, 0);
    vector<double> censoring(num_obs * num_censoring_times, 0);

    vector<MultistateTree*> multistate_trees(ntrees);
    for (size_t tree_id = 0; tree_id < ntrees; ++tree_id) {
        multistate_trees[tree_id] = static_cast<MultistateTree*>(trees[tree_id].get());
    }

    /*
      Allocate the large workspaces before entering OpenMP so allocation failures remain ordinary C++ exceptions. Never
      create more copies than observations, which is particularly important for one-row prediction with many workers.
    */
    size_t num_error_workers = max(static_cast<size_t>(1), min(static_cast<size_t>(this->nworkers), max(static_cast<size_t>(1), num_obs)));
    vector<vector<double>> average_na_workspaces(num_error_workers, vector<double>(prediction_size, 0));
    vector<vector<double>> average_initial_workspaces(num_error_workers, vector<double>(num_states, 0));
    vector<vector<double>> average_censoring_workspaces(num_error_workers, vector<double>(num_censoring_times, 0));
    vector<vector<double>> previous_workspaces(num_error_workers, vector<double>(num_states, 0));
    vector<vector<double>> current_workspaces(num_error_workers, vector<double>(num_states, 0));

    #pragma omp parallel num_threads(num_error_workers)
    {
        size_t worker_id = static_cast<size_t>(omp_get_thread_num());
        vector<double>& average_na = average_na_workspaces[worker_id];
        vector<double>& average_initial = average_initial_workspaces[worker_id];
        vector<double>& average_censoring = average_censoring_workspaces[worker_id];
        vector<double>& previous = previous_workspaces[worker_id];
        vector<double>& current = current_workspaces[worker_id];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < num_obs; ++i) {
            fill(average_na.begin(), average_na.end(), 0);
            fill(average_initial.begin(), average_initial.end(), 0);
            fill(average_censoring.begin(), average_censoring.end(), 0);
            size_t contributing_trees = 0;

            for (size_t j = 0; j < ntrees; ++j) {
                // training errors are OOB errors; errors on new data average every tree
                if (!new_data_provided && !oob_indices[j][i]) {
                    continue;
                }
                MultistateTree* tree = multistate_trees[j];
                const size_t leaf_id = new_data_provided ? tree->predictionLeafID(new_data, i) : tree->predictionLeafID(i);
                const vector<double>& tree_na = tree->getNA()[leaf_id];
                const vector<double>& tree_initial = tree->getInitDist()[leaf_id];
                const vector<double>& tree_censoring = tree->getKMCensoringFull()[leaf_id];
                ++contributing_trees;

                for (size_t k = 0; k < prediction_size; ++k) {
                    average_na[k] += tree_na[k];
                }
                for (size_t state = 0; state < num_states; ++state) {
                    average_initial[state] += tree_initial[state];
                }
                for (size_t t = 0; t < num_censoring_times; ++t) {
                    average_censoring[t] += tree_censoring[t];
                }
            }

            if (contributing_trees == 0) {
                continue;   // preserve the existing all-zero result for an observation with no OOB trees
            }
            for (double& value : average_na) {
                value /= contributing_trees;
            }
            for (size_t state = 0; state < num_states; ++state) {
                average_initial[state] /= contributing_trees;
            }

            const size_t censoring_offset = i * num_censoring_times;
            for (size_t t = 0; t < num_censoring_times; ++t) {
                censoring[censoring_offset + t] = average_censoring[t] / contributing_trees;
            }

            // occupation at zero is the averaged initial distribution
            const size_t occupation_offset = i * occupation_size;
            copy(average_initial.begin(), average_initial.end(), previous.begin());
            copy(previous.begin(), previous.end(), occupation.begin() + occupation_offset);
            for (size_t t = 1; t < num_unique_event_times; ++t) {
                fill(current.begin(), current.end(), 0);
                const size_t current_matrix = t * dim;
                const size_t previous_matrix = current_matrix - dim;
                for (size_t from_state = 0; from_state < num_states; ++from_state) {
                    const double previous_probability = previous[from_state];
                    for (size_t to_state = 0; to_state < num_states; ++to_state) {
                        const size_t matrix_index = from_state * num_states + to_state;
                        double contribution = average_na[current_matrix + matrix_index] -
                                              average_na[previous_matrix + matrix_index];
                        if (from_state == to_state) {
                            ++contribution;
                        }
                        current[to_state] += previous_probability * contribution;
                    }
                }
                copy(current.begin(), current.end(),
                     occupation.begin() + occupation_offset + t * num_states);
                previous.swap(current);
            }
        }
    }

    vector<vector<double>> result;
    result.reserve(2);
    result.push_back(std::move(occupation));
    result.push_back(std::move(censoring));
    return result;
}

// function to clear all quantities related to VIMP calculation only, is called before growing the forest
// and before computing the KM censoring estimators in the leaves
void MultistateForest::clearVIMPCache() {
    vimp_cache_ready = false;
    vimp_event_times.clear();
    vimp_observed_states.clear();
    vimp_uncensored.clear();
    vimp_first_endpoint_event_ids.clear();
    vimp_oob_indices.clear();
    vimp_oob_leaf_ids.clear();
    vimp_tree_uses_feature.clear();
    vimp_leaf_cache_ids.clear();
    vimp_leaf_occupation_probs.clear();
    vimp_leaf_event_ipcw.clear();
    vimp_oob_endpoint_ipcw.clear();
    vimp_baseline_ready = false;
    vimp_baseline_error_type.clear();
    vimp_baseline_state_weights.clear();
    vimp_baseline_scores.clear();
}

/*
  this function computes all indices needed for computing VIMP across any feature, namely the indices
  needed in baseline computations, including populating the leaves with occupation probabilities, but
  only those needed for baseline computations, the rest are computed lazily (only when needed by e.g.
  a shuffled feature / random daughter assignment)
*/
void MultistateForest::prepareVIMPCache() {
    if (vimp_cache_ready) {
        return;
    }

    const size_t num_obs = data->getNumberOfObs();
    const size_t num_states = data->getNumberOfStates();
    const size_t num_vimp_times = num_unique_event_times;
    vimp_event_times = unique_event_times;

    /*
      Use the same complete grid as the ordinary multi-state error. In particular, time zero measures changes in predicted
      initial distributions and the trapezoidal integration range is now identical for ordinary error and VIMP.
    */
    vimp_observed_states = observedStateIDsVIMP(*data, response_event_time_ids, num_vimp_times);

    // vimp_uncensored[obs] is one when the final state was observed and zero when censoring occurred
    const vector<uint8_t>& censoring_states = data->getCensoringStates();
    vimp_uncensored.resize(num_obs);
    for (size_t obs = 0; obs < num_obs; ++obs) {
        vimp_uncensored[obs] = censoring_states[obs] == 0;
    }

    // compute the OOB indices (not as bools) and initialise all temporary VIMP values we need
    OOBNonBoolIndices(vimp_oob_indices, oob_indices);
    vimp_oob_leaf_ids.resize(ntrees);
    vimp_leaf_cache_ids.resize(ntrees);
    vimp_leaf_occupation_probs.resize(ntrees);
    vimp_leaf_event_ipcw.resize(ntrees);
    vimp_oob_endpoint_ipcw.resize(ntrees);
    vimp_tree_uses_feature.assign(ntrees, vector<bool>(data->getNumberOfFeatures(), false));

    // all trees in a forest share the same censoring grid
    // precompute the exact G(t) and G(T-) lookup positions used by IPCW
    const vector<double>& censoring_times = *this->censoring_times;
    const size_t no_censoring_id = censoring_times.size();
    const vector<size_t>& event_censoring_ids = event_censoring_time_ids;

    // for the subject endpoint, lower_bound - 1 gives the largest censoring time strictly before T_i, hence G(T_i-)
    // recall that pure censoring times are not included in the unique event times for multi-states
    const vector<double>& times = data->getTimes();
    vector<size_t> endpoint_censoring_ids(num_obs, no_censoring_id);
    vimp_first_endpoint_event_ids.resize(num_obs);
    for (size_t obs = 0; obs < num_obs; ++obs) {
        double endpoint = times[data->getLastObservedTimes()[obs]];
        // determine the id of the endpoint
        vimp_first_endpoint_event_ids[obs] = static_cast<size_t>(distance(vimp_event_times.begin(), lower_bound(vimp_event_times.begin(), vimp_event_times.end(), endpoint)));
        // now determine where in the censoring time grid the endpoint belongs
        // (recall that this grid is fixed for all trees, so it only needs to be computed once)
        auto it = lower_bound(censoring_times.begin(), censoring_times.end(), endpoint);
        if (it != censoring_times.begin()) {
            endpoint_censoring_ids[obs] = static_cast<size_t>(distance(censoring_times.begin(), it) - 1);
        }
    }

    const size_t no_leaf_cache_id = numeric_limits<size_t>::max();
    vector<string> cache_errors(ntrees);
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t j = 0; j < ntrees; ++j) {
      try {
        // fetch tree and initialise pointers (caches) to the different VIMP-related quantities for the
        // tree such as estimators and tree info
        MultistateTree* tree = static_cast<MultistateTree*>(trees[j].get());
        const vector<vector<double>>& tree_na = tree->getNA();
        const vector<vector<double>>& tree_init = tree->getInitDist();
        vector<size_t>& leaf_cache_ids = vimp_leaf_cache_ids[j];
        leaf_cache_ids.assign(tree_na.size(), no_leaf_cache_id);
        vector<vector<double>>& occupation_cache = vimp_leaf_occupation_probs[j];
        vector<vector<double>>& event_ipcw_cache = vimp_leaf_event_ipcw[j];
        const vector<size_t>& tree_oob = vimp_oob_indices[j];
        const size_t num_oob_obs = tree_oob.size();
        const vector<vector<double>>& censoring_cache = tree->getKMCensoringFull();

        // here we check (in parallel) what nodes actually use the feature since if the feature
        // is never used by a particular tree, the contribution to the VIMP will be zero
        for (size_t node = 0; node < tree->getLeftDaughters().size(); ++node) {
            // ensures that we don't check terminal nodes (such nodes contain the placeholder value 0)
            if (tree->getLeftDaughters()[node] != 0) {
                vimp_tree_uses_feature[j][tree->getFeatureIDs()[node]] = true;
            }
        }

        // compute the leaf ids once and for all for OOB observations for this tree (j)
        vector<size_t>& leaf_ids = vimp_oob_leaf_ids[j];
        leaf_ids.resize(tree_oob.size());
        for (size_t row = 0; row < tree_oob.size(); ++row) {
            leaf_ids[row] = tree->predictionLeafID(tree_oob[row]);
        }

        /*
          baseline predictions and IPCW only use leaves reached by original
          OOB observations. Other occupation curves are populated lazily if
          a later feature permutation reaches them
        */
        vector<unsigned char> original_leaf(tree_na.size(), 0);
        for (size_t leaf_id : leaf_ids) {
            original_leaf[leaf_id] = 1;
        }
        for (size_t node = 0; node < tree_na.size(); ++node) {
            if (!original_leaf[node]) {
                continue;
            }
            const size_t cache_id = occupation_cache.size();
            leaf_cache_ids[node] = cache_id;
            occupation_cache.push_back(
                occupationProbabilitiesVIMP(tree_na[node], tree_init[node], num_states));
            event_ipcw_cache.push_back(vector<double>(num_vimp_times, 0));
            vector<double>& event_ipcw = event_ipcw_cache.back();
            for (size_t t = 0; t < num_vimp_times; ++t) {
                // here the weight is computed; if censoring does not occur (== no_censoring_id),
                // then the weight is 1/n, otherwise 1 / nG(t) (the subject is still at risk)
                size_t censoring_id = event_censoring_ids[t];
                double censoring = censoring_id == no_censoring_id ?
                    1.0 : censoring_cache[node][censoring_id];  // recall censoring_cache is the KM estimator
                if (censoring > 0) {
                    event_ipcw[t] = 1.0 / (static_cast<double>(num_oob_obs) * censoring);
                }
            }
        }

        // repeat the logic but for the endpoint, here the IPCW weight is 1 / nG(T_i-) in the case of censoring
        vector<double>& endpoint_ipcw = vimp_oob_endpoint_ipcw[j];
        endpoint_ipcw.resize(tree_oob.size(), 0);
        for (size_t row = 0; row < tree_oob.size(); ++row) {
            size_t obs_id = tree_oob[row];
            size_t censoring_id = endpoint_censoring_ids[obs_id];
            double censoring = censoring_id == no_censoring_id ?
                1.0 : censoring_cache[leaf_ids[row]][censoring_id];
            if (censoring > 0) {
                endpoint_ipcw[row] = 1.0 / (static_cast<double>(tree_oob.size()) * censoring);
            }
        }
      } catch (const std::exception& error) {
        cache_errors[j] = error.what();
      }
    }

    for (const string& error : cache_errors) {
        if (!error.empty()) {
            clearVIMPCache();
            throw runtime_error(error);
        }
    }

    vimp_cache_ready = true;
}

// computes the baseline score for VIMP, since this is the same across all features (used for all VIMP methods)
void MultistateForest::prepareVIMPBaseline(const string& error_type, const vector<double>& state_weights) {
    // check if the baseline VIMP quantities for this particular choice of error and state_weights have already been computed
    if (vimp_baseline_ready && vimp_baseline_error_type == error_type &&
        vimp_baseline_state_weights == state_weights) {
        return;
    }

    const size_t num_states = data->getNumberOfStates();
    const size_t num_vimp_times = vimp_event_times.size();
    vector<double> baseline_scores(ntrees, 0);
    vector<string> tree_errors(ntrees);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t j = 0; j < ntrees; ++j) {
      try {
        if (vimp_oob_indices[j].empty()) {
            continue;
        }
        // state contributions are consumed together during integration, so keep one accumulated value per time
        vector<double> score(num_vimp_times, 0);

        for (size_t row = 0; row < vimp_oob_indices[j].size(); ++row) {
            // fetch all relevant quantities for this observation
            size_t obs_id = vimp_oob_indices[j][row];
            size_t leaf_id = vimp_oob_leaf_ids[j][row];
            size_t cache_id = vimp_leaf_cache_ids[j][leaf_id];
            const vector<double>& occupation = vimp_leaf_occupation_probs[j][cache_id];
            const vector<double>& event_ipcw = vimp_leaf_event_ipcw[j][cache_id];
            size_t first_endpoint_event = vimp_first_endpoint_event_ids[obs_id];

            for (size_t t = 0; t < num_vimp_times; ++t) {
                // if t is before the endpoint, the IPCW weight is in the vimp_leaf_event_ipcw vector,
                // otherwise in the vimp_oob_endpoint_ipcw vector
                double ipcw = t < first_endpoint_event ? event_ipcw[t] :
                    (vimp_uncensored[obs_id] ? vimp_oob_endpoint_ipcw[j][row] : 0.0);
                if (ipcw == 0) {
                    continue;
                }

                // compute the selected prediction loss
                size_t time_offset = t * num_states;
                size_t observed_state = vimp_observed_states[obs_id * num_vimp_times + t];
                if (error_type == "brier") {
                    double contribution = 0;
                    for (size_t k = 0; k < num_states; ++k) {
                        size_t offset = time_offset + k;
                        double prediction = occupation[offset];
                        double residual = k == observed_state ? 1.0 - prediction : prediction;
                        contribution += residual * residual * state_weights[k];
                    }
                    score[t] += ipcw * contribution;
                } else if (error_type == "spherical" && observed_state < num_states) {
                    score[t] += ipcw * sphericalLossVIMP(occupation, time_offset, observed_state, state_weights);
                } else if (error_type == "kl" && observed_state < num_states && state_weights[observed_state] != 0) {
                    size_t offset = time_offset + observed_state;
                    score[t] -= ipcw * log(probabilityForLogScore(occupation[offset])) * state_weights[observed_state];
                }
            }
        }
        // save the baseline score, which is the same across all features
        baseline_scores[j] = computeIntegratedScore(score, vimp_event_times).second;
      } catch (const std::exception& error) {
        tree_errors[j] = error.what();
      }
    }

    for (const string& error : tree_errors) {
        if (!error.empty()) {
            throw runtime_error(error);
        }
    }

    vimp_baseline_scores = std::move(baseline_scores);
    vimp_baseline_error_type = error_type;
    vimp_baseline_state_weights = state_weights;
    vimp_baseline_ready = true;
}

// simply a wrapper function for the two functions above with some messages
void MultistateForest::prepareVIMPComputation(size_t feature, const string& error_type,
                                              const vector<double>& state_weights) {
    if (error_type != "brier" && error_type != "kl" && error_type != "spherical") {
        throw runtime_error("Unknown choice of loss function. Choose 'brier', 'kl', or 'spherical'");
    }
    const size_t num_states = data->getNumberOfStates();
    if (state_weights.size() != num_states) {
        throw runtime_error("The number of state weights must equal the number of states");
    }
    if (ntrees == 0) {
        throw runtime_error("Cannot compute VIMP for a forest with zero trees");
    }
    if (feature >= data->getNumberOfFeatures()) {
        throw runtime_error("VIMP feature index is out of range");
    }
    if (unique_event_times.size() <= 1) {
        throw runtime_error("Cannot compute multi-state VIMP without observed event times");
    }
    if (!save_predictions) {
        computePredictionsCensoring();
    }
    prepareVIMPCache();
    prepareVIMPBaseline(error_type, state_weights);
}

// computes VIMP contribution for a single tree (tree_id) given the predicted VIMP leafs and the baseline values
// computed earlier
double MultistateForest::computeVIMPForLeafAssignments(size_t tree_id, const vector<size_t>& prediction_leaf_ids,
                                                       const string& error_type,
                                                       const vector<double>& state_weights, vector<double>& score) {
    const vector<size_t>& tree_oob = vimp_oob_indices[tree_id];
    if (prediction_leaf_ids.size() != tree_oob.size()) {
        throw runtime_error("Internal error: VIMP leaf assignments do not match the OOB sample");
    }

    // fetch tree and all relevant tree info
    MultistateTree* tree = static_cast<MultistateTree*>(trees[tree_id].get());
    const size_t num_states = data->getNumberOfStates();
    const size_t num_score_event_times = vimp_event_times.size();
    const size_t no_leaf_cache_id = numeric_limits<size_t>::max();
    vector<size_t>& leaf_cache_ids = vimp_leaf_cache_ids[tree_id];
    vector<vector<double>>& occupation_cache = vimp_leaf_occupation_probs[tree_id];
    vector<vector<double>>& event_ipcw_cache = vimp_leaf_event_ipcw[tree_id];
    const vector<vector<double>>& tree_na = tree->getNA();
    const vector<vector<double>>& tree_init = tree->getInitDist();

    // here we compute occupation probabilities in a leaf if these have not already
    // been computed earlier when we prepared the baseline (in prepareVIMPCache)
    for (size_t leaf_id : prediction_leaf_ids) {
        if (leaf_id >= leaf_cache_ids.size()) {
            throw runtime_error("Internal error: VIMP routed an observation to an invalid leaf");
        }
        if (leaf_cache_ids[leaf_id] == no_leaf_cache_id) {
            leaf_cache_ids[leaf_id] = occupation_cache.size();
            occupation_cache.push_back(
                occupationProbabilitiesVIMP(tree_na[leaf_id], tree_init[leaf_id], num_states));
            // perturbed leaves only need occupation probabilities; keep cache indices aligned with an empty IPCW entry
            event_ipcw_cache.push_back(vector<double>());
        }
    }

    // fetch IPCW values and leaf IDs for the tree
    score.assign(num_score_event_times, 0);
    const vector<size_t>& original_leaf_ids = vimp_oob_leaf_ids[tree_id];

    for (size_t row = 0; row < tree_oob.size(); ++row) {
        const size_t obs_id = tree_oob[row];
        const size_t original_cache_id = leaf_cache_ids[original_leaf_ids[row]];
        const size_t prediction_cache_id = leaf_cache_ids[prediction_leaf_ids[row]];

        for (size_t t = 0; t < num_score_event_times; ++t) {
            /*
              if we are below the endpoint, use the IPCW value from before the event, otherwise, if the
              observation is uncensored, use the endpoint IPCW value, and if the event is censored, the 
              weight is zero, and the observation contributes zero to the score
            */
            double ipcw = t < vimp_first_endpoint_event_ids[obs_id] ?
                event_ipcw_cache[original_cache_id][t] :
                (vimp_uncensored[obs_id] ? vimp_oob_endpoint_ipcw[tree_id][row] : 0.0);
            if (ipcw == 0) {
                continue;
            }
            const size_t time_offset = t * num_states;
            const size_t observed_state = vimp_observed_states[obs_id * num_score_event_times + t];
            if (error_type == "brier") {
                double contribution = 0;
                for (size_t state = 0; state < num_states; ++state) {
                    const size_t offset = time_offset + state;
                    const double prediction = occupation_cache[prediction_cache_id][offset];
                    const double residual = state == observed_state ?
                        1.0 - prediction : prediction;
                    contribution += residual * residual * state_weights[state];
                }
                score[t] += ipcw * contribution;
            } else if (error_type == "spherical" && observed_state < num_states) {
                score[t] += ipcw * sphericalLossVIMP(
                    occupation_cache[prediction_cache_id], time_offset, observed_state, state_weights);
            } else if (error_type == "kl" && observed_state < num_states &&
                       state_weights[observed_state] != 0) {
                const size_t offset = time_offset + observed_state;
                score[t] -= ipcw * log(probabilityForLogScore(
                    occupation_cache[prediction_cache_id][offset])) * state_weights[observed_state];
            }
        }
    }

    // finally, compute VIMP for this tree (we use the normalised version by convention)
    return computeIntegratedScore(score, vimp_event_times).second - vimp_baseline_scores[tree_id];
}

// computes VIMP for multi-state forests using permutation of the feature values across OOB observations
double MultistateForest::computeVIMPPermute(size_t feature, int feature_seed, const string& error_type, const vector<double>& state_weights) {
    prepareVIMPComputation(feature, error_type, state_weights);

    vector<double> tree_vimp(ntrees, 0);
    vector<unsigned char> tree_valid(ntrees, 0);
    vector<string> tree_errors(ntrees);

    #pragma omp parallel num_threads(this->nworkers)
    {
        vector<double> shuffled_oob_values;
        vector<size_t> shuffled_leaf_ids;
        vector<double> score_shuffled;

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < ntrees; ++i) {
          try {
            // fetch tree and the indices for its OOB observations
            MultistateTree* tree = static_cast<MultistateTree*>(trees[i].get());
            const vector<size_t>& tree_oob_indices = vimp_oob_indices[i];
            const size_t num_oob_obs = tree_oob_indices.size();
            if (num_oob_obs == 0) continue;
            tree_valid[i] = 1;
            // if the feature is never split upon, the VIMP contribution is zero, 
            // and the tree can be safely skipped
            if (!vimp_tree_uses_feature[i][feature]) {
                continue;
            }

            // fetch feature values and shuffle these
            shuffled_oob_values.resize(num_oob_obs);
            for (size_t j = 0; j < num_oob_obs; ++j) {
                shuffled_oob_values[j] = data->get_x(tree_oob_indices[j], feature);
            }
            mt19937 local_rng = makeVIMPTreeRNG(feature_seed, i);
            shuffle(shuffled_oob_values.begin(), shuffled_oob_values.end(), local_rng);

            // compute the leaf that the shuffled observation ends up and check whether it is the same
            // as the original one, in which case we may again skip the VIMP computation entirely
            const vector<size_t>& original_leaf_ids = vimp_oob_leaf_ids[i];
            shuffled_leaf_ids.resize(num_oob_obs);
            bool any_leaf_changed = false;
            for (size_t j = 0; j < num_oob_obs; ++j) {
                shuffled_leaf_ids[j] = tree->predictionLeafIDPermuted(tree_oob_indices[j], feature, shuffled_oob_values[j]);
                any_leaf_changed = any_leaf_changed || shuffled_leaf_ids[j] != original_leaf_ids[j];
            }
            if (!any_leaf_changed) {
                continue;
            }

            // this is where the actual tree VIMP computation takes place
            tree_vimp[i] = computeVIMPForLeafAssignments(i, shuffled_leaf_ids, error_type, state_weights, score_shuffled);
          } catch (const std::exception& error) {
            tree_errors[i] = error.what();
          }
        }
    }

    for (const string& error : tree_errors) {
        if (!error.empty()) {
            throw runtime_error(error);
        }
    }

    // aggregate VIMP over the whole forest
    double total_vimp = 0;
    size_t valid_trees = 0;
    for (size_t i = 0; i < ntrees; ++i) {
        if (tree_valid[i]) {
            total_vimp += tree_vimp[i];
            ++valid_trees;
        }
    }
    if (valid_trees == 0) {
        throw runtime_error("Cannot compute VIMP without OOB observations");
    }
    return total_vimp / static_cast<double>(valid_trees);
}

// computes VIMP for multi-state forests using random daughter assignment when the feature of interest
// is encountered across OOB observations
double MultistateForest::computeVIMPRandom(size_t feature, int feature_seed, const string& error_type, const vector<double>& state_weights) {
    prepareVIMPComputation(feature, error_type, state_weights);

    vector<double> tree_vimp(ntrees, 0);
    vector<unsigned char> tree_valid(ntrees, 0);
    vector<string> tree_errors(ntrees);

    #pragma omp parallel num_threads(this->nworkers)
    {
        vector<size_t> random_leaf_ids;
        vector<double> score_random;

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < ntrees; ++i) {
          try {
            // fetch tree and the indices for its OOB observations
            MultistateTree* tree = static_cast<MultistateTree*>(trees[i].get());
            const vector<size_t>& tree_oob = vimp_oob_indices[i];
            const size_t num_oob_obs = tree_oob.size();
            if (num_oob_obs == 0) {
                continue;
            }
            tree_valid[i] = 1;
            // if the feature is never split upon, the VIMP contribution is zero, 
            // and the tree can be safely skipped
            if (!vimp_tree_uses_feature[i][feature]) {
                continue;
            }

            // compute the leaf that the observation ends up after random daughter assignment and check whether
            // it is the same as the original one, in which case we may again skip the VIMP computation entirely
            mt19937 local_rng = makeVIMPTreeRNG(feature_seed, i);
            const vector<size_t>& original_leaf_ids = vimp_oob_leaf_ids[i];
            random_leaf_ids.resize(num_oob_obs);
            bool any_leaf_changed = false;
            for (size_t row = 0; row < num_oob_obs; ++row) {
                random_leaf_ids[row] = tree->predictionLeafIDVIMP(tree_oob[row], feature, local_rng);
                any_leaf_changed = any_leaf_changed || random_leaf_ids[row] != original_leaf_ids[row];
            }
            if (!any_leaf_changed) {
                continue;
            }

            // this is where the actual tree VIMP computation takes place
            tree_vimp[i] = computeVIMPForLeafAssignments(i, random_leaf_ids, error_type, state_weights, score_random);
          } catch (const std::exception& error) {
            tree_errors[i] = error.what();
          }
        }
    }

    for (const string& error : tree_errors) {
        if (!error.empty()) {
            throw runtime_error(error);
        }
    }

    // aggregate VIMP over the whole forest
    double total_vimp = 0;
    size_t valid_trees = 0;
    for (size_t i = 0; i < ntrees; ++i) {
        if (tree_valid[i]) {
            total_vimp += tree_vimp[i];
            ++valid_trees;
        }
    }
    if (valid_trees == 0) {
        throw runtime_error("Cannot compute VIMP without OOB observations");
    }
    return total_vimp / static_cast<double>(valid_trees);
}

#include "ForestMultistate.h"
#include "TreeSurvival.h"
#include <exception>

namespace {

/*
  computes only the nonzero-time occupation rows needed by VIMP. The loop ordering matches occupationProbabilitiesCpp,
  but two rolling product-integral matrices replace its full time-by-state-by-state workspace. The output is the same as for
  occupationProbabilitiesCpp, but the working memory is O(num_states^2) instead of (num_unique_event_times * num_states^2)
*/
vector<double> occupationProbabilitiesVIMP(const vector<double>& na, const vector<double>& init, size_t num_states) {
    const size_t dim = num_states * num_states;
    const size_t num_event_times = na.size() / dim;
    vector<double> result((num_event_times - 1) * num_states, 0);
    vector<double> previous(dim, 0);
    vector<double> current(dim, 0);

    // initialise with the identity matrix
    for (size_t state = 0; state < num_states; ++state) {
        previous[state * num_states + state] = 1;
    }

    for (size_t t = 1; t < num_event_times; ++t) {
        fill(current.begin(), current.end(), 0);
        for (size_t j = 0; j < num_states; ++j) {
            for (size_t k = 0; k < num_states; ++k) {
                double prev = previous[j * num_states + k];
                for (size_t l = 0; l < num_states; ++l) {
                    double contribution = na[t * dim + k * num_states + l] -
                                          na[(t - 1) * dim + k * num_states + l];
                    if (k == l) {
                        ++contribution;
                    }
                    current[j * num_states + l] += prev * contribution;
                }
            }
        }
        size_t output_row = (t - 1) * num_states;
        for (size_t j = 0; j < num_states; ++j) {
            for (size_t k = 0; k < num_states; ++k) {
                result[output_row + k] += init[j] * current[j * num_states + k];
            }
        }
        previous.swap(current);
    }
    return result;
}

}

// constructor for multi-state forests
//--------------------------------------------------------------------------------------
MultistateForest::MultistateForest(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, uint8_t num_states, bool save_predictions) : 
    unique_event_times {unique_event_times}, response_event_time_ids {response_event_time_ids}, save_predictions {save_predictions} {
        this->num_unique_event_times = unique_event_times.size();
}

// functions for growing multi-state forests
//--------------------------------------------------------------------------------------

// grows a multi-state forest using multithreading via OpenMP
void MultistateForest::grow() {
    clearVIMPCache();
    size_t n = data->getNumberOfObs();
    uint8_t num_states = data->getNumberOfStates();
    
    // create vector of indices from 1 to n
    vector<size_t> global_indices(n);
    for (size_t i = 0; i < n; ++i) {
        global_indices[i] = i;
    }

    trees.resize(ntrees);
    oob_indices.resize(ntrees);

    // create pointers to construct the trees
    shared_ptr<vector<double>> unique_event_times = make_shared<vector<double>>(this->unique_event_times);
    shared_ptr<vector<size_t>> response_event_time_ids = make_shared<vector<size_t>>(this->response_event_time_ids);
    shared_ptr<vector<double>> censoring_times = make_shared<vector<double>>(
        uniqueCensoringTimesMultistate(this->unique_event_times, data->getTimes(), data->getLastObservedTimes()));

    size_t n_threads = this->nworkers;
    omp_set_num_threads(n_threads);
    Rcout << "Growing forest using " << n_threads << " threads" << endl;
    size_t progress = 0;

    // use OpenMP for parallel tree growing
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < static_cast<size_t>(ntrees); ++i) {
        size_t current;
        #pragma omp atomic capture
        current = ++progress;
        #pragma omp critical
        {
            cout << "Growing tree " << current << "/" << ntrees << endl;
            //Rcout << "\rGrowing tree " << current << "/" << ntrees << std::flush;
            if (current == static_cast<size_t>(ntrees)) {
                cout << endl;
                //Rcout << endl;
            }
        }
        // give each thread its own random number generator to prevent races
        mt19937 local_rng(seed + i);
        unique_ptr<MultistateTree> tree;

        vector<size_t> bootstrap_indices;
        vector<size_t> holdout_indices;     // only relevant for honest trees
        
        // bootstrap
        if (!honest) {
            size_t subsample_size = floor(sample_rate * n);
            bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
            oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
            tree = make_unique<MultistateTree>(unique_event_times, response_event_time_ids, bootstrap_indices, num_states,
                                               save_predictions, vector<size_t>(), censoring_times);
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
                tree = make_unique<MultistateTree>(unique_event_times, response_event_time_ids, grow, num_states,
                                                   save_predictions, holdout, censoring_times);
                oob_indices[i] = computeOOBIndicesDouble(grow, holdout, n);

            } else {
                // if no double bootstrap, bootstrap the whole dataset and then split
                size_t subsample_size = floor(sample_rate * n);
                bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(bootstrap_indices, local_rng);
                tree = make_unique<MultistateTree>(unique_event_times, response_event_time_ids, partition.first, num_states,
                                                   save_predictions, partition.second, censoring_times);
                oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
            }
        }

        uniform_int_distribution<size_t> dist(0, numeric_limits<size_t>::max());
        tree->initialise(data, mtry, min_node_size, nsplits, splitrule, honest, dist(local_rng));
        tree->setRNG(local_rng);
        tree->grow();
        trees[i] = std::move(tree);
    }

    // compute all quantities of interest from the vector of trees
    computeForestQuantities();
}

// functions for predicting with multi-state forests
//--------------------------------------------------------------------------------------

vector<double> MultistateForest::predict(const vector<double>& x) {
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;

    vector<double> result(num_unique_event_times * dim, 0);
    for (const auto& tree : trees) {
        vector<double> prediction = get<vector<double>>(tree->predict(x));
        sum_vectors(result, prediction);
    }
    for (size_t i = 0; i < num_unique_event_times * dim; ++i) {
        result[i] = result[i]/ntrees;
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
    size_t num_obs = data->getNumberOfObs();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;

    // initialise vectors of predictions (in-bag and OOB), predicted initial distributions and censoring distributions
    vector<double> predictions(num_obs * num_unique_event_times * dim);
    vector<double> oob_predictions(num_obs * num_unique_event_times * dim);
    vector<double> predictions_init(num_obs * num_states);
    vector<double> oob_predictions_init(num_obs * num_states);
    vector<double> oob_censoring(num_obs * num_unique_event_times);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        // initialise predictions for the current observation
        vector<double> pred(num_unique_event_times * dim, 0);
        vector<double> pred_oob(num_unique_event_times * dim, 0);
        vector<double> pred_init, pred_init_oob, pred_cens;
        if (compute_initial) {
            pred_init.assign(num_states, 0);
            pred_init_oob.assign(num_states, 0);
        }
        if (compute_censoring) {
            pred_cens.assign(num_unique_event_times, 0);
        }
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
            vector<double> tree_pred, tree_pred_init, tree_pred_cens;
            size_t leaf_id;
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                leaf_id = tree->predictionLeafID(data->get_x_row(i));    // the leaf id is not saved during fitting for OOB observations
                // compute the tree prediction and aggregate
                tree_pred = tree->getNA()[leaf_id];
                sum_vectors(pred, tree_pred);
                sum_vectors(pred_oob, tree_pred);

                // repeat for initial distribution
                if (compute_initial) { 
                    tree_pred_init = tree->getInitDist()[leaf_id];
                    sum_vectors(pred_init, tree_pred_init);
                    sum_vectors(pred_init_oob, tree_pred_init);
                }

                // repeat for censoring
                if (compute_censoring) {
                    tree_pred_cens = tree->getKMCensoring()[leaf_id];
                    sum_vectors(pred_cens, tree_pred_cens);
                }
            }
            
            else {
                leaf_id = tree->getPredictionNodeIDs()[i];   // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
                tree_pred = tree->getNA()[leaf_id];
                sum_vectors(pred, tree_pred);

                if (compute_initial) {
                    tree_pred_init = tree->getInitDist()[leaf_id];
                    sum_vectors(pred_init, tree_pred_init);
                }
                // we do not compute censoring KM estimators for in-bag data
            }
        }


        // normalise and save NA predictions
        for (size_t k = 0; k < num_unique_event_times * dim; ++k) {
            pred[k] /= ntrees;
            if (num_oob_trees > 0) {
                pred_oob[k] /= num_oob_trees;
            }
            predictions[i * num_unique_event_times * dim + k] = pred[k];
            oob_predictions[i * num_unique_event_times * dim + k] = pred_oob[k];
        }

        // normalise and save initial distribution predictions
        if (compute_initial) {
            for (size_t j = 0; j < num_states; ++j) {
                pred_init[j] /= ntrees;
                if (num_oob_trees > 0) {
                    pred_init_oob[j] /= num_oob_trees;
                }
                predictions_init[i * num_states + j] = pred_init[j];
                oob_predictions_init[i * num_states + j] = pred_init_oob[j];
            }
        }

        // normalise and save censoring distribution predictions
        if (compute_censoring) {
            for (size_t t = 0; t < num_unique_event_times; ++t) {
                if (num_oob_trees > 0) {
                    pred_cens[t] /= num_oob_trees;
                }
                oob_censoring[i * num_unique_event_times + t] = pred_cens[t];
            }
        }
    }

    // finally return the computed predictions
    if (compute_initial && compute_censoring) {
        return {predictions, oob_predictions, predictions_init, oob_predictions_init, oob_censoring};
    } 
    else if (compute_initial) {
        return {predictions, oob_predictions, predictions_init, oob_predictions_init};
    }
    else if (compute_censoring) {
        return {predictions, oob_predictions, oob_censoring};
    }
    else {
        return {predictions, oob_predictions};
    }

}

/*

Computes all predictions on a new dataset, result is a vector with one, two or three flattened vectors
depending on the parameters. First vector is always a flattened vector of Nelson-Aalen estimators. If
compute_initial = true, the next two vectors are 
in-bag and OOB predicted initial distributions, and if compute_censoring = true, the OOB censoring predictions
are added to the result vector.

*/

vector<vector<double>> MultistateForest::computePredictions(const Data& new_data, bool compute_initial, bool compute_censoring) {
    size_t num_obs = new_data.getNumberOfObs();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    
    // initialise vectors of predicted NA estimators, initial distributions and censoring
    vector<double> predictions(num_obs * num_unique_event_times * dim);
    vector<double> predictions_init, censoring;
    if (compute_initial) {
        predictions_init.assign(num_obs * num_states, 0);
    }
    if (compute_censoring) {
        censoring.assign(num_obs * num_unique_event_times, 0);
    }

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times * dim, 0);
        vector<double> pred_init, pred_censoring;
        if (compute_initial) {
            pred_init.assign(num_states, 0);
        }
        if (compute_censoring) {
            pred_censoring.assign(num_unique_event_times, 0);
        }

        // aggregate predictions over all trees for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
            size_t leaf_id = tree->predictionLeafID(new_data.get_x_row(i));
            vector<double> tree_pred = tree->getNA()[leaf_id];
            sum_vectors(pred, tree_pred);

            if (compute_initial) {
                vector<double> tree_pred_init = tree->getInitDist()[leaf_id];
                sum_vectors(pred_init, tree_pred_init);
            }
            if (compute_censoring) {
                vector<double> tree_pred_cens = tree->getKMCensoring()[leaf_id];
                sum_vectors(pred_censoring, tree_pred_cens);
            }
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times * dim; ++k) {
            pred[k] /= ntrees;
            predictions[i * num_unique_event_times * dim + k] = pred[k];
        }

        // normalise and save predictions for initial distributions
        if (compute_initial) {
            for (size_t j = 0; j < num_states; ++j) {
                pred_init[j] /= ntrees;
                predictions_init[i * num_states + j] = pred_init[j];
            }
        }

        // normalise and save predictions for censoring distributions
        if (compute_censoring) {
            for (size_t t = 0; t < num_unique_event_times; ++t) {
                pred_censoring[t] /= ntrees;
                censoring[i * num_unique_event_times + t] = pred_censoring[t];
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
    }
    else {
        return {predictions};
    }
}

// Populate leaf-level censoring estimators when they were not saved during fitting.
void MultistateForest::computePredictionsCensoring() {
    clearVIMPCache();
    size_t num_obs = data->getNumberOfObs();
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t j = 0; j < ntrees; ++j) {
        MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
        size_t num_nodes = tree->getNumberOfNodes();
        vector<vector<size_t>> leaf_groups(num_nodes);

        for (size_t i = 0; i < num_obs; ++i) {
            size_t leaf_id = tree->predictionLeafID(i);
            leaf_groups[leaf_id].push_back(i);
        }

        tree->resizeKM();
        for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
            if (!leaf_groups[node_id].empty()) {
                tree->computeCensoringKMExternal(leaf_groups[node_id], node_id);
            }
        }
    }
    save_predictions = true;
}

void MultistateForest::clearVIMPCache() {
    vimp_cache_ready = false;
    vimp_event_times.clear();
    vimp_observed_states.clear();
    vimp_uncensored.clear();
    vimp_first_endpoint_event_ids.clear();
    vimp_oob_indices.clear();
    vimp_oob_leaf_ids.clear();
    vimp_tree_uses_feature.clear();
    vimp_leaf_occupation_probs.clear();
    vimp_leaf_event_ipcw.clear();
    vimp_oob_endpoint_ipcw.clear();
    vimp_baseline_ready = false;
    vimp_baseline_error_type.clear();
    vimp_baseline_state_weights.clear();
    vimp_baseline_scores.clear();
}

void MultistateForest::prepareVIMPCache() {
    if (vimp_cache_ready) {
        return;
    }

    const size_t num_obs = data->getNumberOfObs();
    const size_t num_states = data->getNumberOfStates();
    const size_t num_vimp_times = num_unique_event_times - 1;
    vimp_event_times.assign(unique_event_times.begin() + 1, unique_event_times.end());

    // state membership does not depend on the permuted feature. Store only
    // the scored (nonzero) event-time rows once for all features
    vimp_observed_states.assign(num_obs * num_vimp_times, static_cast<uint8_t>(num_states));
    {
        vector<bool> full_state_indicators = data->computeStateIndicators(response_event_time_ids, unique_event_times);
        for (size_t obs = 0; obs < num_obs; ++obs) {
            size_t source_row = obs * num_unique_event_times * num_states + num_states;
            size_t target_row = obs * num_vimp_times;
            for (size_t t = 0; t < num_vimp_times; ++t) {
                for (size_t state = 0; state < num_states; ++state) {
                    if (full_state_indicators[source_row + t * num_states + state]) {
                        vimp_observed_states[target_row + t] = static_cast<uint8_t>(state);
                        break;
                    }
                }
            }
        }
    }

    // vimp_uncensored[obs] = 0 means that final state was observed/uncensored
    const vector<uint8_t>& censoring_states = data->getCensoringStates();
    vimp_uncensored.resize(num_obs);
    for (size_t obs = 0; obs < num_obs; ++obs) {
        vimp_uncensored[obs] = censoring_states[obs] == 0;
    }

    OOBNonBoolIndices(vimp_oob_indices, oob_indices);
    vimp_oob_leaf_ids.resize(ntrees);
    vimp_leaf_occupation_probs.resize(ntrees);
    vimp_leaf_event_ipcw.resize(ntrees);
    vimp_oob_endpoint_ipcw.resize(ntrees);
    vimp_tree_uses_feature.assign(ntrees, vector<bool>(data->getNumberOfFeatures(), false));

    // all trees in a forest share the same censoring grid
    // precompute the exact G(t) and G(T-) lookup positions used by IPCW
    MultistateTree* first_tree = dynamic_cast<MultistateTree*>(trees[0].get());
    const vector<double>& censoring_times = first_tree->getCensoringTimes();
    const size_t no_censoring_id = numeric_limits<size_t>::max();

    // use upper_bound to find the largest censoring grid time less than or equal to the time under consideration
    // (in vimp_event_times) to look up the values G(t)
    vector<size_t> event_censoring_ids(num_vimp_times, no_censoring_id);
    for (size_t t = 0; t < num_vimp_times; ++t) {
        auto it = upper_bound(censoring_times.begin(), censoring_times.end(), vimp_event_times[t]);
        if (it != censoring_times.begin()) {
            event_censoring_ids[t] = static_cast<size_t>(distance(censoring_times.begin(), it) - 1);
        }
    }

    // for the subject endpoint, we use lower_bound to find the largest censoring time strictly below T_i, yielding G(T_i-)
    const vector<double>& times = data->getTimes();
    const vector<size_t>& last_observed_times = data->getLastObservedTimes();
    vector<size_t> endpoint_censoring_ids(num_obs, no_censoring_id);
    vimp_first_endpoint_event_ids.resize(num_obs);
    for (size_t obs = 0; obs < num_obs; ++obs) {
        double endpoint = times[last_observed_times[obs]];
        vimp_first_endpoint_event_ids[obs] = static_cast<size_t>(distance(vimp_event_times.begin(), lower_bound(vimp_event_times.begin(), vimp_event_times.end(), endpoint)));
        auto it = lower_bound(censoring_times.begin(), censoring_times.end(), endpoint);
        if (it != censoring_times.begin()) {
            endpoint_censoring_ids[obs] = static_cast<size_t>(distance(censoring_times.begin(), it) - 1);
        }
    }

    vector<string> cache_errors(ntrees);
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t tree_id = 0; tree_id < ntrees; ++tree_id) {
      try {
        // here we check (in parallel) what nodes actually use the feature since if the feature
        // is never used by a particular tree, the contribution to the VIMP will be zero
        MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[tree_id].get());
        const vector<vector<double>>& tree_na = tree->getNA();
        const vector<vector<double>>& tree_init = tree->getInitDist();
        vector<vector<double>>& occupation_cache = vimp_leaf_occupation_probs[tree_id];
        occupation_cache.resize(tree_na.size());
        vector<vector<double>>& event_ipcw_cache = vimp_leaf_event_ipcw[tree_id];
        event_ipcw_cache.resize(tree_na.size());
        const vector<size_t>& tree_oob = vimp_oob_indices[tree_id];
        const size_t num_oob_obs = tree_oob.size();
        const vector<vector<double>>& censoring_cache = tree->getKMCensoringFull();
        const vector<size_t>& left_daughters = tree->getLeftDaughters();
        const vector<size_t>& feature_ids = tree->getFeatureIDs();
        for (size_t node = 0; node < left_daughters.size(); ++node) {
            // ensures that we don't check terminal nodes (such nodes contain the placeholder value 0)
            if (left_daughters[node] != 0) {
                vimp_tree_uses_feature[tree_id][feature_ids[node]] = true;
            }
        }

        vector<size_t>& leaf_ids = vimp_oob_leaf_ids[tree_id];
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
            occupation_cache[node] = occupationProbabilitiesVIMP(tree_na[node], tree_init[node], num_states);
            vector<double>& event_ipcw = event_ipcw_cache[node];
            event_ipcw.resize(num_vimp_times, 0);
            for (size_t t = 0; t < num_vimp_times; ++t) {
                size_t censoring_id = event_censoring_ids[t];
                double censoring = censoring_id == no_censoring_id ?
                    1.0 : censoring_cache[node][censoring_id];
                if (censoring > 0) {
                    event_ipcw[t] = 1.0 / (static_cast<double>(num_oob_obs) * censoring);
                }
            }
        }

        vector<double>& endpoint_ipcw = vimp_oob_endpoint_ipcw[tree_id];
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
        cache_errors[tree_id] = error.what();
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

void MultistateForest::prepareVIMPBaseline(const string& error_type, const vector<double>& state_weights) {
    if (vimp_baseline_ready && vimp_baseline_error_type == error_type &&
        vimp_baseline_state_weights == state_weights) {
        return;
    }

    const size_t num_states = data->getNumberOfStates();
    const size_t num_vimp_times = vimp_event_times.size();
    const size_t score_size = num_vimp_times * num_states;
    vector<double> baseline_scores(ntrees, 0);
    vector<string> tree_errors(ntrees);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t tree_id = 0; tree_id < ntrees; ++tree_id) {
      try {
        const vector<size_t>& tree_oob = vimp_oob_indices[tree_id];
        if (tree_oob.empty()) {
            continue;
        }
        vector<double> score(score_size, 0);
        const vector<size_t>& leaf_ids = vimp_oob_leaf_ids[tree_id];
        const vector<vector<double>>& occupation_cache = vimp_leaf_occupation_probs[tree_id];
        const vector<vector<double>>& event_ipcw_cache = vimp_leaf_event_ipcw[tree_id];
        const vector<double>& endpoint_ipcw_cache = vimp_oob_endpoint_ipcw[tree_id];

        for (size_t row = 0; row < tree_oob.size(); ++row) {
            size_t obs_id = tree_oob[row];
            size_t leaf_id = leaf_ids[row];
            const vector<double>& occupation = occupation_cache[leaf_id];
            const vector<double>& event_ipcw = event_ipcw_cache[leaf_id];
            size_t first_endpoint_event = vimp_first_endpoint_event_ids[obs_id];
            for (size_t t = 0; t < num_vimp_times; ++t) {
                double ipcw = t < first_endpoint_event ? event_ipcw[t] :
                    (vimp_uncensored[obs_id] ? endpoint_ipcw_cache[row] : 0.0);
                if (ipcw == 0) {
                    continue;
                }
                size_t time_offset = t * num_states;
                size_t observed_state = vimp_observed_states[obs_id * num_vimp_times + t];
                if (error_type == "brier") {
                    for (size_t state = 0; state < num_states; ++state) {
                        size_t offset = time_offset + state;
                        double prediction = occupation[offset];
                        double residual = state == observed_state ? 1.0 - prediction : prediction;
                        score[offset] += ipcw * residual * residual * state_weights[state];
                    }
                } else if (observed_state < num_states && state_weights[observed_state] != 0) {
                    size_t offset = time_offset + observed_state;
                    score[offset] -= ipcw * log(probabilityForLogScore(occupation[offset])) *
                                     state_weights[observed_state];
                }
            }
        }
        baseline_scores[tree_id] = computeIntegratedScore(score, vimp_event_times, true).second;
      } catch (const std::exception& error) {
        tree_errors[tree_id] = error.what();
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

void MultistateForest::prepareVIMPComputation(size_t feature, const string& error_type,
                                              const vector<double>& state_weights) {
    if (error_type != "brier" && error_type != "kl") {
        throw runtime_error("Unknown choice of loss function. Choose either 'brier' or 'kl'");
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

double MultistateForest::computeVIMPForLeafAssignments(size_t tree_id,
                                                        const vector<size_t>& prediction_leaf_ids,
                                                        bool use_brier,
                                                        const vector<double>& state_weights,
                                                        vector<double>& score) {
    const vector<size_t>& tree_oob = vimp_oob_indices[tree_id];
    if (prediction_leaf_ids.size() != tree_oob.size()) {
        throw runtime_error("Internal error: VIMP leaf assignments do not match the OOB sample");
    }

    MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[tree_id].get());
    const size_t num_states = data->getNumberOfStates();
    const size_t num_score_event_times = vimp_event_times.size();
    vector<vector<double>>& occupation_cache = vimp_leaf_occupation_probs[tree_id];
    const vector<vector<double>>& tree_na = tree->getNA();
    const vector<vector<double>>& tree_init = tree->getInitDist();

    for (size_t leaf_id : prediction_leaf_ids) {
        if (leaf_id >= occupation_cache.size()) {
            throw runtime_error("Internal error: VIMP routed an observation to an invalid leaf");
        }
        if (occupation_cache[leaf_id].empty()) {
            occupation_cache[leaf_id] = occupationProbabilitiesVIMP(
                tree_na[leaf_id], tree_init[leaf_id], num_states);
        }
    }

    score.assign(num_score_event_times * num_states, 0);
    const vector<size_t>& original_leaf_ids = vimp_oob_leaf_ids[tree_id];
    const vector<vector<double>>& event_ipcw_cache = vimp_leaf_event_ipcw[tree_id];
    const vector<double>& endpoint_ipcw_cache = vimp_oob_endpoint_ipcw[tree_id];

    for (size_t row = 0; row < tree_oob.size(); ++row) {
        const size_t obs_id = tree_oob[row];
        const size_t original_leaf_id = original_leaf_ids[row];
        const vector<double>& occupation = occupation_cache[prediction_leaf_ids[row]];
        const vector<double>& event_ipcw = event_ipcw_cache[original_leaf_id];
        const size_t first_endpoint_event = vimp_first_endpoint_event_ids[obs_id];
        for (size_t t = 0; t < num_score_event_times; ++t) {
            double ipcw = t < first_endpoint_event ? event_ipcw[t] :
                (vimp_uncensored[obs_id] ? endpoint_ipcw_cache[row] : 0.0);
            if (ipcw == 0) {
                continue;
            }
            const size_t time_offset = t * num_states;
            const size_t observed_state =
                vimp_observed_states[obs_id * num_score_event_times + t];
            if (use_brier) {
                for (size_t state = 0; state < num_states; ++state) {
                    const size_t offset = time_offset + state;
                    const double prediction = occupation[offset];
                    const double residual = state == observed_state ?
                        1.0 - prediction : prediction;
                    score[offset] += ipcw * residual * residual * state_weights[state];
                }
            } else if (observed_state < num_states && state_weights[observed_state] != 0) {
                const size_t offset = time_offset + observed_state;
                score[offset] -= ipcw * log(probabilityForLogScore(occupation[offset])) *
                                 state_weights[observed_state];
            }
        }
    }

    return computeIntegratedScore(score, vimp_event_times, true).second -
           vimp_baseline_scores[tree_id];
}

double MultistateForest::computeVIMPPermute(size_t feature, int feature_seed, string error_type,
                                            const vector<double>& state_weights) {
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
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[i].get());
            const vector<size_t>& tree_oob_indices = vimp_oob_indices[i];
            const size_t num_oob_obs = tree_oob_indices.size();
            if (num_oob_obs == 0) continue;
            tree_valid[i] = 1;
            if (!vimp_tree_uses_feature[i][feature]) {
                continue;
            }

            shuffled_oob_values.resize(num_oob_obs);
            for (size_t j = 0; j < num_oob_obs; ++j) {
                shuffled_oob_values[j] = data->get_x(tree_oob_indices[j], feature);
            }
            mt19937 local_rng = makeVIMPTreeRNG(feature_seed, i);
            shuffle(shuffled_oob_values.begin(), shuffled_oob_values.end(), local_rng);

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

            tree_vimp[i] = computeVIMPForLeafAssignments(
                i, shuffled_leaf_ids, error_type == "brier", state_weights, score_shuffled);
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

double MultistateForest::computeVIMPRandom(size_t feature, int feature_seed, string error_type,
                                           const vector<double>& state_weights) {
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
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[i].get());
            const vector<size_t>& tree_oob = vimp_oob_indices[i];
            const size_t num_oob_obs = tree_oob.size();
            if (num_oob_obs == 0) {
                continue;
            }
            tree_valid[i] = 1;
            if (!vimp_tree_uses_feature[i][feature]) {
                continue;
            }

            mt19937 local_rng = makeVIMPTreeRNG(feature_seed, i);
            const vector<size_t>& original_leaf_ids = vimp_oob_leaf_ids[i];
            random_leaf_ids.resize(num_oob_obs);
            bool any_leaf_changed = false;
            for (size_t row = 0; row < num_oob_obs; ++row) {
                random_leaf_ids[row] = tree->predictionLeafIDVIMP(
                    tree_oob[row], feature, local_rng);
                any_leaf_changed = any_leaf_changed ||
                    random_leaf_ids[row] != original_leaf_ids[row];
            }
            if (!any_leaf_changed) {
                continue;
            }

            tree_vimp[i] = computeVIMPForLeafAssignments(
                i, random_leaf_ids, error_type == "brier", state_weights, score_random);
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

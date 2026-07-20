#include "ForestSurvival.h"

#include <cmath>

namespace {
vector<double> eventTimesAtIDs(const vector<double>& event_times, const vector<size_t>& ids) {
    vector<double> selected_event_times;
    selected_event_times.reserve(ids.size());
    for (size_t id : ids) {
        selected_event_times.push_back(event_times[id]);
    }
    return selected_event_times;
}

double computeVIMPProbabilityLoss(const SurvivalTree& tree, const vector<size_t>& prediction_leaf_ids, const vector<size_t>& censoring_leaf_ids, 
                                  const vector<size_t>& observation_ids, const double* times, const double* ind, const vector<double>& score_times,
                                  const vector<size_t>& score_time_ids, const vector<size_t>& censoring_time_ids, string error_type) {
    size_t num_obs = observation_ids.size();
    size_t num_score_times = score_times.size();
    const vector<double>& censoring_times = tree.getCensoringTimes();
    const vector<vector<double>>& survival_probabilities = tree.getSurvivalProbabilities();
    const vector<vector<double>>& KM_censoring = tree.getKMCensoringFull();
    vector<double> score(num_score_times, 0);

    // compute the score directly from the leaf estimators to avoid materialising
    // one prediction and censoring matrix for every tree and feature
    for (size_t i = 0; i < num_obs; ++i) {
        size_t obs_id = observation_ids[i];
        double time = times[obs_id];
        double indicator = ind[obs_id];
        const vector<double>& survival = survival_probabilities[prediction_leaf_ids[i]];
        const vector<double>& censoring = KM_censoring[censoring_leaf_ids[i]];

        double censoring_before_event = 1;
        auto event_it = lower_bound(censoring_times.begin(), censoring_times.end(), time);
        if (event_it != censoring_times.begin()) {
            censoring_before_event = censoring[static_cast<size_t>(distance(censoring_times.begin(), event_it) - 1)];
        }

        for (size_t t = 0; t < num_score_times; ++t) {
            double weight = 0;
            if (time <= score_times[t] && indicator == 1) {
                if (censoring_before_event > 0) {
                    weight = 1 / (num_obs * censoring_before_event);
                }
            } else if (time > score_times[t]) {
                size_t censoring_time_id = censoring_time_ids[t];
                double censoring_at_time = censoring_time_id == censoring_times.size() ? 1 : censoring[censoring_time_id];
                if (censoring_at_time > 0) {
                    weight = 1 / (num_obs * censoring_at_time);
                }
            }

            double probability = survival[score_time_ids[t]];
            if (error_type == "brier") {
                double residual = time <= score_times[t] ? probability : 1 - probability;
                score[t] += residual * residual * weight;
            } else if (weight > 0) {
                probability = probabilityForLogScore(probability);
                score[t] -= (time <= score_times[t] ? log1p(-probability) : log(probability)) * weight;
            }
        }
    }
    return computeIntegratedScore(score, score_times).second;
}
}

// constructor for survival forests
//--------------------------------------------------------------------------------------

SurvivalForest::SurvivalForest(vector<double> unique_event_times, vector<size_t> response_event_time_ids,
                               vector<size_t> true_event_time_ids, vector<double> censoring_times, bool save_predictions) :
    unique_event_times {std::move(unique_event_times)}, response_event_time_ids {std::move(response_event_time_ids)},
    true_event_time_ids {std::move(true_event_time_ids)}, censoring_times {std::move(censoring_times)} {
        this->num_unique_event_times = this->unique_event_times.size();
        this->save_predictions = save_predictions;
}

// functions for growing survival forests
//--------------------------------------------------------------------------------------

// grows a survival forest using multithreading via OpenMP
void SurvivalForest::grow() {
    if (ntrees == 0) {
        throw runtime_error("The number of trees must be at least one");
    }
    if (min_node_size == 0) {
        throw runtime_error("The minimal node size must be at least one");
    }
    const vector<bool>& categorical = data->getCategorical();
    vector<size_t> unique_values = data->getUniqueValues();
    for (size_t i = 0; i < categorical.size(); ++i) {
        if (categorical[i] && unique_values[i] > 63) {
            throw runtime_error("Categorical features with more than 63 values are not supported");
        }
    }

    size_t n = data->getNumberOfObs();
    
    // create vector of indices from 1 to n
    vector<size_t> global_indices(n);
    for (size_t i = 0; i < n; ++i) {
        global_indices[i] = i;
    }

    trees.resize(ntrees);
    oob_indices.resize(ntrees);
    vimp_oob_indices.clear();
    vimp_leaf_ids.clear();
    vimp_tree_uses_feature.clear();
    vimp_tree_concordance.clear();
    vimp_tree_brier.clear();
    vimp_tree_kl.clear();
    vimp_event_times.clear();
    vimp_censoring_time_ids.clear();

    // create pointers to construct the trees
    shared_ptr<vector<double>> unique_event_times = make_shared<vector<double>>(this->unique_event_times);
    shared_ptr<vector<size_t>> response_event_time_ids = make_shared<vector<size_t>>(this->response_event_time_ids);
    shared_ptr<vector<size_t>> true_event_time_ids = make_shared<vector<size_t>>(this->true_event_time_ids);
    shared_ptr<vector<double>> censoring_times = make_shared<vector<double>>(this->censoring_times);
    shared_ptr<vector<size_t>> removal_time_ids = make_shared<vector<size_t>>(
      computeRemovalTimeIDs(*data, this->unique_event_times, this->response_event_time_ids));

    size_t n_threads = this->nworkers;
    Rcout << "Growing forest using " << n_threads << " threads" << endl;

    // use OpenMP for parallel tree growing
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        // give each thread its own random number generator to prevent races
        mt19937 local_rng(seed + i);
        unique_ptr<SurvivalTree> tree;

        vector<size_t> bootstrap_indices;
        // bootstrap
        if (!honest) {
            size_t subsample_size = floor(sample_rate * n);
            bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
            oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
            tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, true_event_time_ids,
              std::move(bootstrap_indices), save_predictions, vector<size_t>(), censoring_times, removal_time_ids);
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
                tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, true_event_time_ids,
                  std::move(grow), save_predictions, std::move(holdout), censoring_times, removal_time_ids);

            } else {
                // if no double bootstrap, bootstrap the whole dataset and then split
                size_t subsample_size = floor(sample_rate * n);
                bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(bootstrap_indices, local_rng);
                oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
                tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, true_event_time_ids,
                  std::move(partition.first), save_predictions, std::move(partition.second), censoring_times, removal_time_ids);
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

// functions for predicting with survival forests
//--------------------------------------------------------------------------------------

vector<double> SurvivalForest::predict(const vector<double>& x) {
    vector<double> result(num_unique_event_times, 0);
    for (const auto& base_tree : trees) {
        SurvivalTree* tree = static_cast<SurvivalTree*>(base_tree.get());
        const vector<double>& prediction = tree->getCHF()[tree->predictionLeafID(x)];
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            result[t] += prediction[t];
        }
    }
    for (size_t i = 0; i < num_unique_event_times; ++i) {
        result[i] /= ntrees;
    }
    return result;
}

/*

Computes all predictions, both in-bag and OOB. The first two flattened vectors are the in-bag and OOB
Nelson-Aalen estimators. If compute_censoring = true, the third vector contains the OOB KM censoring
estimators and the fourth contains the corresponding left limits at the observed event times

*/

vector<vector<double>> SurvivalForest::computePredictions(bool compute_censoring) {
    size_t num_obs = data->getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);
    vector<double> oob_predictions(num_obs * num_unique_event_times);
    vector<double> censoring;
    vector<double> censoring_before_event;
    vector<size_t> censoring_before_ids;
    if (compute_censoring) {
        censoring.assign(num_obs * num_unique_event_times, 0);
        censoring_before_event.assign(num_obs, 0);
        censoring_before_ids.assign(num_obs, censoring_times.size());
        const double* times = data->get_y_col_ptr(0);
        for (size_t i = 0; i < num_obs; ++i) {
            auto it = lower_bound(censoring_times.begin(), censoring_times.end(), times[i]);
            if (it != censoring_times.begin()) {
                censoring_before_ids[i] = static_cast<size_t>(distance(censoring_times.begin(), it) - 1);
            }
        }
    }

    vector<SurvivalTree*> survival_trees(ntrees);
    for (size_t i = 0; i < ntrees; ++i) {
        survival_trees[i] = static_cast<SurvivalTree*>(trees[i].get());
    }

    #pragma omp parallel num_threads(this->nworkers)
    {
        vector<double> pred(num_unique_event_times, 0);
        vector<double> oob_pred(num_unique_event_times, 0);
        vector<double> cens(compute_censoring ? num_unique_event_times : 0, 0);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < num_obs; ++i) {
            fill(pred.begin(), pred.end(), 0);
            fill(oob_pred.begin(), oob_pred.end(), 0);
            if (compute_censoring) {
                fill(cens.begin(), cens.end(), 0);
            }
            double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB
            double cens_before = 0;

            // compute the sum of all predictions for observation i
            for (size_t j = 0; j < ntrees; ++j) {
                SurvivalTree* tree = survival_trees[j];
                size_t leaf_id;
                bool is_oob = oob_indices[j][i];
                if (is_oob) {
                    ++num_oob_trees;
                    leaf_id = tree->predictionLeafID(i);
                } else {
                    // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
                    leaf_id = tree->getPredictionNodeIDs()[i];
                }

                const vector<double>& tree_pred = tree->getCHF()[leaf_id];
                for (size_t t = 0; t < num_unique_event_times; ++t) {
                    pred[t] += tree_pred[t];
                }
                if (is_oob) {
                    for (size_t t = 0; t < num_unique_event_times; ++t) {
                        oob_pred[t] += tree_pred[t];
                    }
                    if (compute_censoring) {
                        const vector<double>& tree_cens = tree->getKMCensoring()[leaf_id];
                        for (size_t t = 0; t < num_unique_event_times; ++t) {
                            cens[t] += tree_cens[t];
                        }
                        size_t censoring_time_id = censoring_before_ids[i];
                        cens_before += censoring_time_id == censoring_times.size() ? 1 :
                          tree->getKMCensoringFull()[leaf_id][censoring_time_id];
                    }
                }
            }

            // normalise and save predictions
            for (size_t t = 0; t < num_unique_event_times; ++t) {
                pred[t] /= ntrees;
                oob_pred[t] = num_oob_trees > 0 ? oob_pred[t] / num_oob_trees : 0;
                size_t index = i * num_unique_event_times + t;
                predictions[index] = pred[t];
                oob_predictions[index] = oob_pred[t];
            }
            if (compute_censoring) {
                for (size_t t = 0; t < num_unique_event_times; ++t) {
                    size_t index = i * num_unique_event_times + t;
                    censoring[index] = num_oob_trees > 0 ? cens[t] / num_oob_trees : 0;
                }
                censoring_before_event[i] = num_oob_trees > 0 ? cens_before / num_oob_trees : 0;
            }
        }
    }
    vector<vector<double>> result;
    result.reserve(compute_censoring ? 4 : 2);
    result.push_back(std::move(predictions));
    result.push_back(std::move(oob_predictions));
    if (compute_censoring) {
        result.push_back(std::move(censoring));
        result.push_back(std::move(censoring_before_event));
    }
    return result;
}

/*

Computes all predictions on a new dataset. The first flattened vector contains the Nelson-Aalen estimators.
If compute_censoring = true, the second contains the KM censoring estimators. When evaluation_times are
provided, the third contains the corresponding left limits of the censoring distribution

*/

vector<vector<double>> SurvivalForest::computePredictions(const Data& new_data, bool compute_censoring, const vector<double>* evaluation_times) {
    if (!compute_censoring && evaluation_times != nullptr) {
        throw runtime_error("Evaluation times require censoring predictions");
    }

    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);
    vector<double> censoring;
    vector<double> censoring_before_event;
    vector<size_t> censoring_before_ids;
    if (compute_censoring) {
        censoring.assign(num_obs * num_unique_event_times, 0);
        if (evaluation_times != nullptr) {
            if (evaluation_times->size() != num_obs) {
                throw runtime_error("The number of evaluation times must match the number of observations");
            }
            censoring_before_event.assign(num_obs, 0);
            censoring_before_ids.assign(num_obs, censoring_times.size());
            for (size_t i = 0; i < num_obs; ++i) {
                auto it = lower_bound(censoring_times.begin(), censoring_times.end(), (*evaluation_times)[i]);
                if (it != censoring_times.begin()) {
                    censoring_before_ids[i] = static_cast<size_t>(distance(censoring_times.begin(), it) - 1);
                }
            }
        }
    }

    vector<SurvivalTree*> survival_trees(ntrees);
    for (size_t i = 0; i < ntrees; ++i) {
        survival_trees[i] = static_cast<SurvivalTree*>(trees[i].get());
    }

    #pragma omp parallel num_threads(this->nworkers)
    {
        vector<double> pred(num_unique_event_times, 0);
        vector<double> cens(compute_censoring ? num_unique_event_times : 0, 0);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < num_obs; ++i) {
            fill(pred.begin(), pred.end(), 0);
            if (compute_censoring) {
                fill(cens.begin(), cens.end(), 0);
            }
            double cens_before = 0;
            for (size_t j = 0; j < ntrees; ++j) {
                SurvivalTree* tree = survival_trees[j];
                size_t leaf_id = tree->predictionLeafID(new_data, i);
                const vector<double>& tree_pred = tree->getCHF()[leaf_id];
                for (size_t t = 0; t < num_unique_event_times; ++t) {
                    pred[t] += tree_pred[t];
                }
                if (compute_censoring) {
                    const vector<double>& tree_cens = tree->getKMCensoring()[leaf_id];
                    for (size_t t = 0; t < num_unique_event_times; ++t) {
                        cens[t] += tree_cens[t];
                    }
                    if (evaluation_times != nullptr) {
                        size_t censoring_time_id = censoring_before_ids[i];
                        cens_before += censoring_time_id == censoring_times.size() ? 1 :
                          tree->getKMCensoringFull()[leaf_id][censoring_time_id];
                    }
                }
            }

            // normalise and save predictions
            for (size_t t = 0; t < num_unique_event_times; ++t) {
                predictions[i * num_unique_event_times + t] = pred[t] / ntrees;
            }
            if (compute_censoring) {
                for (size_t t = 0; t < num_unique_event_times; ++t) {
                    censoring[i * num_unique_event_times + t] = cens[t] / ntrees;
                }
                if (evaluation_times != nullptr) {
                    censoring_before_event[i] = cens_before / ntrees;
                }
            }
        }
    }
    vector<vector<double>> result;
    result.reserve(compute_censoring ? (evaluation_times == nullptr ? 2 : 3) : 1);
    result.push_back(std::move(predictions));
    if (compute_censoring) {
        result.push_back(std::move(censoring));
        if (evaluation_times != nullptr) {
            result.push_back(std::move(censoring_before_event));
        }
    }
    return result;
}

// for populating the leaves with the censoring KM estimators if these have not already been saved during fitting
void SurvivalForest::computePredictionsCensoring() {
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t j = 0; j < ntrees; ++j) {
        // populate the leaf estimators from the original estimation sample
        SurvivalTree* tree = static_cast<SurvivalTree*>(trees[j].get());
        tree->computeCensoringKMLazy();
    }
    // so that predictions are not needlessly recomputed later
    save_predictions = true;
}

void SurvivalForest::prepareVIMPStructure() {
    if (!vimp_oob_indices.empty()) {
        return;
    }

    OOBNonBoolIndices(vimp_oob_indices, oob_indices);
    vimp_leaf_ids.resize(ntrees);
    vimp_tree_uses_feature.assign(ntrees, vector<bool>(data->getNumberOfFeatures(), false));

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        SurvivalTree* tree = static_cast<SurvivalTree*>(trees[i].get());
        const vector<size_t>& left_daughters = tree->getLeftDaughters();
        const vector<size_t>& feature_IDs = tree->getFeatureIDs();

        // leaf nodes have a dummy feature ID, so only consider internal nodes
        for (size_t j = 0; j < left_daughters.size(); ++j) {
            if (left_daughters[j] != 0) {
                vimp_tree_uses_feature[i][feature_IDs[j]] = true;
            }
        }

        const vector<size_t>& tree_oob_indices = vimp_oob_indices[i];
        vimp_leaf_ids[i].resize(tree_oob_indices.size());
        for (size_t j = 0; j < tree_oob_indices.size(); ++j) {
            vimp_leaf_ids[i][j] = tree->predictionLeafID(tree_oob_indices[j]);
        }
    }
}

const vector<double>& SurvivalForest::prepareVIMPBaseline(string error_type) {
    prepareVIMPStructure();

    vector<double>* baseline;
    if (error_type == "concordance") {
        baseline = &vimp_tree_concordance;
    } else if (error_type == "brier") {
        baseline = &vimp_tree_brier;
    } else {
        baseline = &vimp_tree_kl;
    }
    if (!baseline->empty()) {
        return *baseline;
    }

    bool use_concordance = error_type == "concordance";
    if (!use_concordance) {
        if (!save_predictions) {
            computePredictionsCensoring();
        }
        if (vimp_event_times.empty()) {
            vimp_event_times = eventTimesAtIDs(unique_event_times, true_event_time_ids);
            if (vimp_event_times.empty()) {
                throw runtime_error("Cannot compute probability-score VIMP without observed event times");
            }
            vimp_censoring_time_ids.assign(vimp_event_times.size(), censoring_times.size());
            for (size_t t = 0; t < vimp_event_times.size(); ++t) {
                auto it = upper_bound(censoring_times.begin(), censoring_times.end(), vimp_event_times[t]);
                if (it != censoring_times.begin()) {
                    vimp_censoring_time_ids[t] = static_cast<size_t>(distance(censoring_times.begin(), it) - 1);
                }
            }
        }
    }

    baseline->assign(ntrees, std::numeric_limits<double>::quiet_NaN());
    const double* times = data->get_y_col_ptr(0);
    const double* ind = data->get_y_col_ptr(1);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        SurvivalTree* tree = static_cast<SurvivalTree*>(trees[i].get());
        const vector<size_t>& tree_oob_indices = vimp_oob_indices[i];
        const vector<size_t>& tree_leaf_ids = vimp_leaf_ids[i];
        size_t num_oob_obs = tree_oob_indices.size();
        if (num_oob_obs == 0) {
            continue;
        }

        if (use_concordance) {
            vector<double> times_tree(num_oob_obs);
            vector<double> ind_tree(num_oob_obs);
            vector<double> outcomes(num_oob_obs);
            const vector<double>& chf_outcomes = tree->getCHFOutcomes();
            for (size_t j = 0; j < num_oob_obs; ++j) {
                times_tree[j] = times[tree_oob_indices[j]];
                ind_tree[j] = ind[tree_oob_indices[j]];
                outcomes[j] = chf_outcomes[tree_leaf_ids[j]];
            }
            (*baseline)[i] = computeConcordanceIndex(outcomes, times_tree, ind_tree);
            continue;
        }

        tree->prepareSurvivalProbabilities();
        (*baseline)[i] = computeVIMPProbabilityLoss(*tree, tree_leaf_ids, tree_leaf_ids, tree_oob_indices,
          times, ind, vimp_event_times, true_event_time_ids, vimp_censoring_time_ids, "brier");
    }

    return *baseline;
}

double SurvivalForest::computeVIMPPermute(size_t feature, int feature_seed, string error_type) {
    /*
      for each tree, compute the difference in the sums of errors,
      then divide by the number of OOB samples for that tree
      finally, take the average VIMP over all trees
    */
    if (feature >= data->getNumberOfFeatures()) {
        throw runtime_error("Feature index is out of range");
    }
    bool use_concordance = error_type == "concordance";
    if (!use_concordance && error_type != "brier" && error_type != "kl") {
        throw runtime_error("Unknown choice of loss function. Choose 'brier', 'concordance', or 'kl'");
    }

    const vector<double>& baseline = prepareVIMPBaseline(error_type);
    const double* times = data->get_y_col_ptr(0);
    const double* ind = data->get_y_col_ptr(1);
    vector<double> tree_vimp(ntrees, 0);
    vector<unsigned char> tree_valid(ntrees, 0);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        SurvivalTree* tree = static_cast<SurvivalTree*>(trees[i].get());
        const vector<size_t>& tree_oob_indices = vimp_oob_indices[i];
        const vector<size_t>& tree_leaf_ids = vimp_leaf_ids[i];
        size_t num_oob_obs = tree_oob_indices.size();
        if (num_oob_obs == 0) {
            continue;
        }

        // a feature which is not used in the tree has VIMP contribution zero
        if (!vimp_tree_uses_feature[i][feature]) {
            tree_valid[i] = use_concordance ? std::isfinite(baseline[i]) : 1;
            continue;
        }

        vector<double> shuffled_oob_values(num_oob_obs);
        for (size_t j = 0; j < num_oob_obs; ++j) {
            shuffled_oob_values[j] = data->get_x(tree_oob_indices[j], feature);
        }
        mt19937 local_rng = makeVIMPTreeRNG(feature_seed, i);
        shuffle(shuffled_oob_values.begin(), shuffled_oob_values.end(), local_rng);

        vector<size_t> shuffled_leaf_ids(num_oob_obs);
        bool leaf_changed = false;
        for (size_t j = 0; j < num_oob_obs; ++j) {
            shuffled_leaf_ids[j] = tree->predictionLeafIDPermuted(tree_oob_indices[j], feature, shuffled_oob_values[j]);
            leaf_changed = leaf_changed || shuffled_leaf_ids[j] != tree_leaf_ids[j];
        }
        if (!leaf_changed) {
            tree_valid[i] = use_concordance ? std::isfinite(baseline[i]) : 1;
            continue;
        }

        if (use_concordance) {
            vector<double> times_tree(num_oob_obs);
            vector<double> ind_tree(num_oob_obs);
            vector<double> outcomes_shuffled(num_oob_obs);
            const vector<double>& chf_outcomes = tree->getCHFOutcomes();
            for (size_t j = 0; j < num_oob_obs; ++j) {
                times_tree[j] = times[tree_oob_indices[j]];
                ind_tree[j] = ind[tree_oob_indices[j]];
                outcomes_shuffled[j] = chf_outcomes[shuffled_leaf_ids[j]];
            }
            double cindex_shuffled = computeConcordanceIndex(outcomes_shuffled, times_tree, ind_tree);
            if (std::isfinite(baseline[i]) && std::isfinite(cindex_shuffled)) {
                tree_vimp[i] = baseline[i] - cindex_shuffled;
                tree_valid[i] = 1;
            }
            continue;
        }

        double shuffled_loss = computeVIMPProbabilityLoss(*tree, shuffled_leaf_ids, tree_leaf_ids,
          tree_oob_indices, times, ind, vimp_event_times, true_event_time_ids, vimp_censoring_time_ids, "brier");
        tree_vimp[i] = shuffled_loss - baseline[i];
        tree_valid[i] = 1;
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
        throw runtime_error("Cannot compute VIMP without OOB observations and comparable survival pairs");
    }
    return total_vimp / (double) valid_trees;
}

double SurvivalForest::computeVIMPRandom(size_t feature, int feature_seed, string error_type) {
    /*
    Tree-level random-daughter VIMP. For each tree, OOB observations are
    dropped down normally and with random left/right assignments whenever
    the target feature is encountered. The tree-level C-index difference is
    averaged over trees, weighted by each tree's OOB count.
    */
    if (feature >= data->getNumberOfFeatures()) {
        throw runtime_error("Feature index is out of range");
    }
    bool use_concordance = error_type == "concordance";
    if (!use_concordance && error_type != "brier" && error_type != "kl") {
        throw runtime_error("Unknown choice of loss function. Choose 'brier', 'concordance', or 'kl'");
    }

    const vector<double>& baseline = prepareVIMPBaseline(error_type);
    const double* times = data->get_y_col_ptr(0);
    const double* ind = data->get_y_col_ptr(1);
    vector<double> tree_vimp(ntrees, 0);
    vector<double> tree_weight(ntrees, 0);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        SurvivalTree* tree = static_cast<SurvivalTree*>(trees[i].get());
        const vector<size_t>& tree_oob_indices = vimp_oob_indices[i];
        const vector<size_t>& tree_leaf_ids = vimp_leaf_ids[i];
        size_t num_oob_obs = tree_oob_indices.size();
        if (num_oob_obs == 0) {
            continue;
        }

        // a feature which is not used in the tree has VIMP contribution zero
        if (!vimp_tree_uses_feature[i][feature]) {
            tree_weight[i] = use_concordance ?
              (std::isfinite(baseline[i]) ? num_oob_obs : 0) : 1;
            continue;
        }

        mt19937 local_rng = makeVIMPTreeRNG(feature_seed, i);
        vector<size_t> random_leaf_ids(num_oob_obs);
        bool leaf_changed = false;
        for (size_t j = 0; j < num_oob_obs; ++j) {
            random_leaf_ids[j] = tree->predictionLeafIDVIMP(tree_oob_indices[j], feature, local_rng);
            leaf_changed = leaf_changed || random_leaf_ids[j] != tree_leaf_ids[j];
        }
        if (!leaf_changed) {
            tree_weight[i] = use_concordance ?
              (std::isfinite(baseline[i]) ? num_oob_obs : 0) : 1;
            continue;
        }

        if (use_concordance) {
            vector<double> times_tree(num_oob_obs);
            vector<double> ind_tree(num_oob_obs);
            vector<double> outcomes_random(num_oob_obs);
            const vector<double>& chf_outcomes = tree->getCHFOutcomes();
            for (size_t j = 0; j < num_oob_obs; ++j) {
                times_tree[j] = times[tree_oob_indices[j]];
                ind_tree[j] = ind[tree_oob_indices[j]];
                outcomes_random[j] = chf_outcomes[random_leaf_ids[j]];
            }
            double cindex_random = computeConcordanceIndex(outcomes_random, times_tree, ind_tree);
            if (std::isfinite(baseline[i]) && std::isfinite(cindex_random)) {
                tree_vimp[i] = (baseline[i] - cindex_random) * num_oob_obs;
                tree_weight[i] = num_oob_obs;
            }
            continue;
        }

        double random_loss = computeVIMPProbabilityLoss(*tree, random_leaf_ids, tree_leaf_ids,
          tree_oob_indices, times, ind, vimp_event_times, true_event_time_ids, vimp_censoring_time_ids, "brier");
        tree_vimp[i] = random_loss - baseline[i];
        tree_weight[i] = 1;
    }

    double total_vimp = 0;
    double total_weight = 0;
    for (size_t i = 0; i < ntrees; ++i) {
        total_vimp += tree_vimp[i];
        total_weight += tree_weight[i];
    }
    if (total_weight == 0) {
        throw runtime_error("Cannot compute VIMP without OOB observations and a defined loss");
    }
    return total_vimp / total_weight;
}

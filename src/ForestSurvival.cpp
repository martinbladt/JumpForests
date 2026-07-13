#include "ForestSurvival.h"

#include <cmath>

namespace {
mt19937 makeVIMPTreeRNG(int feature_seed, size_t tree_id) {
    seed_seq::result_type seed_data[2] = {
        static_cast<seed_seq::result_type>(feature_seed),
        static_cast<seed_seq::result_type>(tree_id)
    };
    seed_seq tree_seed(seed_data, seed_data + 2);
    return mt19937(tree_seed);
}

vector<double> eventTimesAtIDs(const vector<double>& event_times, const vector<size_t>& ids) {
    vector<double> selected_event_times;
    selected_event_times.reserve(ids.size());
    for (size_t id : ids) {
        selected_event_times.push_back(event_times[id]);
    }
    return selected_event_times;
}
}

// constructor for survival forests
//--------------------------------------------------------------------------------------

SurvivalForest::SurvivalForest(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, const vector<size_t>& true_event_time_ids, bool save_predictions) :
    unique_event_times {unique_event_times}, response_event_time_ids {response_event_time_ids}, true_event_time_ids {true_event_time_ids} {
        this->num_unique_event_times = unique_event_times.size();
        this->save_predictions = save_predictions;
}

// functions for growing survival forests
//--------------------------------------------------------------------------------------

// grows a survival forest using multithreading via OpenMP
void SurvivalForest::grow() {
    size_t n = data->getNumberOfObs();
    
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
    shared_ptr<vector<size_t>> true_event_time_ids = make_shared<vector<size_t>>(this->true_event_time_ids);

    size_t n_threads = this->nworkers;
    omp_set_num_threads(n_threads);
    Rcout << "Growing forest using " << n_threads << " threads" << endl;

    // use OpenMP for parallel tree growing
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < static_cast<int>(ntrees); ++i) {
        // give each thread its own random number generator to prevent races
        mt19937 local_rng(seed + i);
        unique_ptr<SurvivalTree> tree;

        vector<size_t> bootstrap_indices;
        vector<size_t> holdout_indices;     // only relevant for honest trees
        
        // bootstrap
        if (!honest) {
            size_t subsample_size = floor(sample_rate * n);
            bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
            oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
            tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, true_event_time_ids, bootstrap_indices, save_predictions);
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
                tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, true_event_time_ids, grow, save_predictions, holdout);
                oob_indices[i] = computeOOBIndicesDouble(grow, holdout, n);

            } else {
                // if no double bootstrap, bootstrap the whole dataset and then split
                size_t subsample_size = floor(sample_rate * n);
                bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(bootstrap_indices, local_rng);
                tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, true_event_time_ids, partition.first, save_predictions, partition.second);
                oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
            }
        }

        // Create and grow tree
        //auto tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, true_event_time_ids, bootstrap_indices);

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
    for (const auto& tree : trees) {
        vector<double> prediction = get<vector<double>>(tree->predict(x));
        sum_vectors(result, prediction);
    }
    for (int i = 0; i < num_unique_event_times; ++i) {
        result[i] = result[i]/ntrees;
    }
    return result;
}

/*

Computes all predictions, both in-bag and OOB, result is a vector with two or three flattened vectors
depending on the parameters. First two vectors are always in-bag and OOB Nelson-Aalen estimators, and the
third is a flattened vector of OOB KM censoring estimators, computed only if compute_censoring = true

*/

vector<vector<double>> SurvivalForest::computePredictions(bool compute_censoring) {
    size_t num_obs = data->getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);
    vector<double> oob_predictions(num_obs * num_unique_event_times);
    vector<double> censoring;
    if (compute_censoring) {
        censoring.assign(num_obs * num_unique_event_times, 0);
    }

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times, 0);
        vector<double> oob_pred(num_unique_event_times, 0);
        vector<double> cens;
        if (compute_censoring) {
            cens.assign(num_unique_event_times, 0);
        }
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                size_t leaf_id = tree->predictionLeafID(data->get_x_row(i));
                vector<double> tree_pred = tree->getCHF()[leaf_id];
                if (compute_censoring) {
                    sum_vectors(cens, tree->getKMCensoring()[leaf_id]);
                }
                sum_vectors(pred, tree_pred);
                sum_vectors(oob_pred, tree_pred);
            }
            // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
            else {
                vector<double> tree_pred = tree->getCHF()[tree->getPredictionNodeIDs()[i]];
                sum_vectors(pred, tree_pred);
            }
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times; ++k) {
            pred[k] /= ntrees;
            if (num_oob_trees > 0) {
                oob_pred[k] /= num_oob_trees;
                if (compute_censoring) {
                    cens[k] /= num_oob_trees;
                }
            }
            size_t index = i * num_unique_event_times + k;
            predictions[index] = pred[k];
            oob_predictions[index] = oob_pred[k];
            if (compute_censoring) {
                censoring[index] = cens[k];
            }
        }
    }
    if (compute_censoring) {
        return {predictions, oob_predictions, censoring};
    } else {
        return {predictions, oob_predictions};
    }
}

/*

Computes all predictions on a new dataset, result is a vector with one or two flattened vectors
depending on the parameters. The first is a flattened vector of Nelson-Aalen estimators, and the
second is a flattened vector of OOB KM censoring estimators, computed only if compute_censoring = true

*/

vector<vector<double>> SurvivalForest::computePredictions(const Data& new_data, bool compute_censoring) {
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);
    vector<double> censoring(num_obs * num_unique_event_times);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times, 0);
        vector<double> cens;
        if (compute_censoring) {
            cens.assign(num_unique_event_times, 0);
        }

        for (size_t j = 0; j < ntrees; ++j) {
            SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
            size_t leaf_id = tree->predictionLeafID(new_data.get_x_row(i));
            sum_vectors(pred, tree->getCHF()[leaf_id]);
            if (compute_censoring) {
                sum_vectors(cens, tree->getKMCensoring()[leaf_id]);
            }
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times; ++k) {
            pred[k] /= ntrees;
            predictions[i * num_unique_event_times + k] = pred[k];
            if (compute_censoring) {
                cens[k] /= ntrees;
                censoring[i * num_unique_event_times + k] = cens[k];
            }
        }
    }
    if (compute_censoring) {
        return {predictions, censoring};
    } else {
        return {predictions};
    }
}

// for populating the leaves with the censoring KM estimators if these have not already been saved during fitting
void SurvivalForest::computePredictionsCensoring() {
    //#pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t j = 0; j < ntrees; ++j) {
        // fetch tree and group all the observations by leaves
        SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
        vector<size_t> leafIDs = tree->getPredictionNodeIDs();
        size_t num_terminal_nodes = tree->getNumberOfTerminalNodes();
        size_t num_nodes = tree->getNumberOfNodes();
        vector<vector<size_t>> leaf_groups = groupByLeaf(leafIDs, num_nodes, num_terminal_nodes);

        // compute the numbers at risk and the number of deaths in each leaf and the corresponding KM estimator for the survival distribution
        tree->resizeKM();   // to ensure that the vector of Kaplan-Meier estimators for censoring is sufficiently large and initialised
        for (size_t k = 0; k < num_nodes; ++k) {
            if (!leaf_groups[k].empty()) {  // we are in a terminal node
                tree->computeCensoringKMExternal(leaf_groups[k], leafIDs[leaf_groups[k][0]]);
            }
        }
    }
    // so that predictions are not needlessly recomputed later
    save_predictions = true;
}

double SurvivalForest::computeVIMPPermute(size_t feature, int feature_seed, string error_type) {
    /*
      for each tree, compute the difference in the sums of errors, 
      then divide by the number of OOB samples for that tree
      finally, take the average VIMP over all trees
    */

    const vector<double>& times = data->get_y_col(0);
    const vector<double>& ind = data->get_y_col(1);

    vector<double> brier_event_times;
    vector<size_t> brier_response_event_time_ids;
    size_t num_brier_event_times = num_unique_event_times;
    if (error_type == "brier") {
        if (!save_predictions) {
            computePredictionsCensoring();
        }
        brier_event_times = eventTimesAtIDs(unique_event_times, true_event_time_ids);
        if (brier_event_times.empty()) {
            throw runtime_error("Cannot compute Brier VIMP without observed event times");
        }
        brier_response_event_time_ids = computeResponseEventTimeIDs(brier_event_times, times);
        num_brier_event_times = brier_event_times.size();
    }

    double total_vimp = 0;
    size_t valid_trees = 0;
    
    // shuffle feature values for all trees among the oob covariates
    vector<vector<size_t>> oob_indices_non_bool;
    OOBNonBoolIndices(oob_indices_non_bool, oob_indices);
    
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers) reduction(+:total_vimp,valid_trees)
    for (size_t i = 0; i < ntrees; ++i) {
        SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[i].get());
        const vector<size_t>& tree_oob_indices = oob_indices_non_bool[i];
        size_t num_oob_obs = tree_oob_indices.size();
        if (num_oob_obs == 0) {
            continue;
        }

        // shuffle values for this tree
        vector<double> shuffled_oob_values;
        shuffled_oob_values.reserve(num_oob_obs);
        for (size_t obs_id : tree_oob_indices) {
            shuffled_oob_values.push_back(data->get_x(obs_id, feature));
        }
        mt19937 local_rng = makeVIMPTreeRNG(feature_seed, i);
        shuffle(shuffled_oob_values.begin(), shuffled_oob_values.end(), local_rng);

        // initialise vectors of outcomes when the loss is C-index
        vector<double> tree_outcomes;
        vector<double> tree_outcomes_shuffled;
        if (error_type == "concordance") {
            tree_outcomes.reserve(num_oob_obs);
            tree_outcomes_shuffled.reserve(num_oob_obs);
        }

        // initialise vectors of KM estimators
        vector<double> tree_KM_predictions;
        vector<double> tree_KM_censoring;
        vector<double> tree_KM_predictions_shuffled;
        if (error_type == "brier") {
            tree_KM_predictions.assign(num_oob_obs * num_brier_event_times, 0);
            tree_KM_censoring.assign(num_oob_obs * num_brier_event_times, 0);
            tree_KM_predictions_shuffled.assign(num_oob_obs * num_brier_event_times, 0);
        }

        // initialise vectors of times and indicators belonging to the OOB
        // observations of this tree
        vector<double> times_tree;
        vector<double> ind_tree;
        vector<size_t> obs_indices_tree;
        times_tree.reserve(num_oob_obs);
        ind_tree.reserve(num_oob_obs);
        obs_indices_tree.reserve(num_oob_obs);

        for (size_t j = 0; j < num_oob_obs; ++j) {
            size_t obs_id = tree_oob_indices[j];
            times_tree.push_back(times[obs_id]);
            ind_tree.push_back(ind[obs_id]);
            obs_indices_tree.push_back(obs_id);

            // fetch tree predictions without shuffled values for 'feature'
            vector<double> x = data->get_x_row(obs_id);
            size_t leaf_id = tree->predictionLeafID(x);
            vector<double> tree_pred = tree->getCHF()[leaf_id];

            // fetch tree predictions with shuffled values for 'feature'
            x[feature] = shuffled_oob_values[j];
            size_t leaf_id_shuffled = tree->predictionLeafID(x);
            vector<double> tree_pred_shuffled = tree->getCHF()[leaf_id_shuffled];

            if (error_type == "brier") {
                // fetch KM estimators for the CHF and censoring for the tree
                const vector<double> tree_KM_pred = KaplanMeier(tree_pred);
                const vector<double>& tree_KM_cens = tree->getKMCensoring()[leaf_id];
                const vector<double> tree_KM_pred_shuffled = KaplanMeier(tree_pred_shuffled);

                // now save in the flattened vectors
                size_t obs_index = num_brier_event_times * j;
                for (size_t t = 0; t < num_brier_event_times; ++t) {
                    size_t source_time = true_event_time_ids[t];
                    tree_KM_predictions[obs_index + t] = tree_KM_pred[source_time];
                    tree_KM_censoring[obs_index + t] = tree_KM_cens[source_time];
                    tree_KM_predictions_shuffled[obs_index + t] = tree_KM_pred_shuffled[source_time];
                }

            } else if (error_type == "concordance") {
                // update outcomes
                tree_outcomes.push_back(vector_sum(tree_pred));
                tree_outcomes_shuffled.push_back(vector_sum(tree_pred_shuffled));
            } else {
                throw runtime_error("Unknown choice of loss function. Choose either 'brier' or 'concordance'");
            }
        }
        // now compute and save tree VIMP
        if (error_type == "brier") {
            // Keep the loss fixed: only the survival prediction is permuted.
            vector<double> ipcw = computeIPCWCpp(times_tree, ind_tree, brier_event_times, brier_response_event_time_ids, tree_KM_censoring, obs_indices_tree);

            // compute vector of Brier scores
            vector<double> brier = computeBrierScoreCpp(times_tree, ipcw, brier_event_times, tree_KM_predictions);
            vector<double> brier_shuffled = computeBrierScoreCpp(times_tree, ipcw, brier_event_times, tree_KM_predictions_shuffled);

            // we use the normalised integrated Brier score
            double ibs_normalised = computeIBS(brier, brier_event_times).second;
            double ibs_normalised_shuffled = computeIBS(brier_shuffled, brier_event_times).second;

            total_vimp += ibs_normalised_shuffled - ibs_normalised;
            ++valid_trees;
            
        } else if (error_type == "concordance") {
            double cindex = computeConcordanceIndex(tree_outcomes, times_tree, ind_tree);
            double cindex_shuffled = computeConcordanceIndex(tree_outcomes_shuffled, times_tree, ind_tree);
            if (std::isfinite(cindex) && std::isfinite(cindex_shuffled)) {
                total_vimp += cindex - cindex_shuffled;
                ++valid_trees;
            }
        }
        //total_vimp += (computeConcordanceIndex(tree_outcomes, times_tree, ind_tree) - computeConcordanceIndex(tree_outcomes_shuffled, times_tree, ind_tree)) * num_oob_obs;
    }
    
    if (valid_trees == 0) {
        throw runtime_error("Cannot compute VIMP without OOB observations and comparable survival pairs");
    }

    // return forest VIMP
    return total_vimp / (double) valid_trees;
}

double SurvivalForest::computeVIMPRandom(size_t feature, int feature_seed) {
    /*
    Tree-level random-daughter VIMP. For each tree, OOB observations are
    dropped down normally and with random left/right assignments whenever
    the target feature is encountered. The tree-level C-index difference is
    averaged over trees, weighted by each tree's OOB count.
    */

    size_t num_obs = data->getNumberOfObs();
    const vector<double>& times = data->get_y_col(0);
    const vector<double>& ind = data->get_y_col(1);

    double total_vimp = 0;
    double total_oob = 0;

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers) reduction(+:total_vimp,total_oob)
    for (size_t i = 0; i < ntrees; ++i) {
        SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[i].get());
        mt19937 local_rng = makeVIMPTreeRNG(feature_seed, i);
        size_t num_oob_obs = 0;

        vector<double> tree_outcomes;
        vector<double> tree_outcomes_random;
        vector<double> times_tree;
        vector<double> ind_tree;
        tree_outcomes.reserve(num_obs);
        tree_outcomes_random.reserve(num_obs);
        times_tree.reserve(num_obs);
        ind_tree.reserve(num_obs);

        for (size_t j = 0; j < num_obs; ++j) {
            if (!oob_indices[i][j]) {
                continue;
            }

            ++num_oob_obs;
            vector<double> x = data->get_x_row(j);
            vector<double> tree_pred = get<vector<double>>(tree->predict(x));
            vector<double> tree_pred_random = get<vector<double>>(tree->predictVIMP(x, feature, local_rng));

            // Keep the same risk score currently used by permutation VIMP.
            tree_outcomes.push_back(vector_sum(tree_pred));
            tree_outcomes_random.push_back(vector_sum(tree_pred_random));
            times_tree.push_back(times[j]);
            ind_tree.push_back(ind[j]);
        }

        // now compute and save tree VIMP
        total_oob += num_oob_obs;
        total_vimp += (computeConcordanceIndex(tree_outcomes, times_tree, ind_tree) - computeConcordanceIndex(tree_outcomes_random, times_tree, ind_tree)) * num_oob_obs;
    }

    // return forest VIMP
    return total_vimp / total_oob;
}

// computes VIMP predictions by random daughter assignments (this function is no longer used)
vector<double> SurvivalForest::computePredictionsVIMPRandom(size_t feature, int feature_seed) {
    size_t num_obs = data->getNumberOfObs();
    vector<double> vimp_predictions(num_obs * num_unique_event_times, 0);
    vector<size_t> num_oob_trees(num_obs, 0);

    /*
      Use one RNG stream per tree and advance it across that tree's OOB cases.
      Reseeding per observation would give every OOB case in a tree the same random daughter sequence.
    */
    for (size_t j = 0; j < ntrees; ++j) {
        SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
        mt19937 local_rng = makeVIMPTreeRNG(feature_seed, j);

        for (size_t i = 0; i < num_obs; ++i) {
            if (!oob_indices[j][i]) {
                continue;
            }

            ++num_oob_trees[i];
            vector<double> vimp_tree_pred = get<vector<double>>(tree->predictVIMP(data->get_x_row(i), feature, local_rng));
            for (size_t k = 0; k < num_unique_event_times; ++k) {
                vimp_predictions[i * num_unique_event_times + k] += vimp_tree_pred[k];
            }
        }
    }

    // normalise predictions by the number of OOB trees for each observation
    for (size_t i = 0; i < num_obs; ++i) {
        for (size_t k = 0; k < num_unique_event_times; ++k) {
            if (num_oob_trees[i] > 0) {
                vimp_predictions[i * num_unique_event_times + k] /= num_oob_trees[i];
            }
        }
    }
    return vimp_predictions; 
}

// computes VIMP predictions by permuting features
vector<double> SurvivalForest::computePredictionsVIMPPermute(size_t feature, int feature_seed) {
    size_t num_obs = data->getNumberOfObs();
    vector<double> vimp_predictions(num_obs * num_unique_event_times);

    // shuffle feature values for all trees among the oob covariates
    vector<vector<size_t>> oob_indices_non_bool;
    OOBNonBoolIndices(oob_indices_non_bool, oob_indices);
    const vector<vector<double>>& shuffled_values_feature = shuffledFeatureValues(oob_indices_non_bool, feature, feature_seed);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> vimp_pred(num_unique_event_times, 0);
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all oob predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                vector<double> x = data->get_x_row(i);
                x[feature] = shuffled_values_feature[j][i];
                vector<double> vimp_tree_pred = get<vector<double>>(tree->predict(x));
                sum_vectors(vimp_pred, vimp_tree_pred);
            }
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times; ++k) {
            if (num_oob_trees > 0) {
                vimp_pred[k] /= num_oob_trees;
            }
            vimp_predictions[i * num_unique_event_times + k] = vimp_pred[k];
        }
    }
    return vimp_predictions; 
}

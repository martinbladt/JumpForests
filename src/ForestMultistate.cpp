#include "ForestMultistate.h"

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
            tree = make_unique<MultistateTree>(unique_event_times, response_event_time_ids, bootstrap_indices, num_states, save_predictions);
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
                tree = make_unique<MultistateTree>(unique_event_times, response_event_time_ids, grow, num_states, save_predictions, holdout);
                oob_indices[i] = computeOOBIndicesDouble(grow, holdout, n);

            } else {
                // if no double bootstrap, bootstrap the whole dataset and then split
                size_t subsample_size = floor(sample_rate * n);
                bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(bootstrap_indices, local_rng);
                tree = make_unique<MultistateTree>(unique_event_times, response_event_time_ids, partition.first, num_states, save_predictions, partition.second);
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
    size_t num_obs = data->getNumberOfObs();
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t j = 0; j < ntrees; ++j) {
        MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
        size_t num_nodes = tree->getNumberOfNodes();
        vector<vector<size_t>> leaf_groups(num_nodes);

        for (size_t i = 0; i < num_obs; ++i) {
            size_t leaf_id = tree->predictionLeafID(data->get_x_row(i));
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

#include "ForestMultistate.h"

// constructor for multi-state forests
//--------------------------------------------------------------------------------------
MultistateForest::MultistateForest(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, uint8_t num_states) : 
    unique_event_times {unique_event_times}, response_event_time_ids {response_event_time_ids} {
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
            tree = make_unique<MultistateTree>(unique_event_times, response_event_time_ids, bootstrap_indices, num_states);
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
                tree = make_unique<MultistateTree>(unique_event_times, response_event_time_ids, grow, num_states, holdout);
                oob_indices[i] = computeOOBIndicesDouble(grow, holdout, n);

            } else {
                // if no double bootstrap, bootstrap the whole dataset and then split
                size_t subsample_size = floor(sample_rate * n);
                auto global_bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(global_bootstrap_indices, local_rng);
                tree = make_unique<MultistateTree>(unique_event_times, response_event_time_ids, partition.first, num_states, partition.second);
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

pair<vector<double>, vector<double>> MultistateForest::computePredictions() {
    size_t num_obs = data->getNumberOfObs();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    vector<double> predictions(num_obs * num_unique_event_times * dim);
    vector<double> oob_predictions(num_obs * num_unique_event_times * dim);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times * dim, 0);
        vector<double> oob_pred(num_unique_event_times * dim, 0);
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                vector<double> tree_pred = get<vector<double>>(tree->predict(data->get_x_row(i)));
                sum_vectors(pred, tree_pred);
                sum_vectors(oob_pred, tree_pred);
            }
            // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
            else {
                vector<double> tree_pred = tree->getNA()[tree->getPredictionNodeIDs()[i]];
                sum_vectors(pred, tree_pred);
            }
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times * dim; ++k) {
            pred[k] /= ntrees;
            if (num_oob_trees > 0) {
                oob_pred[k] /= num_oob_trees;
            }
            predictions[i * num_unique_event_times * dim + k] = pred[k];
            oob_predictions[i * num_unique_event_times * dim + k] = oob_pred[k];
        }
    }
    return {predictions, oob_predictions};
}

pair<vector<double>, vector<double>> MultistateForest::computePredictedInitialDistributions() {
    size_t num_obs = data->getNumberOfObs();
    uint8_t num_states = data->getNumberOfStates();
    vector<double> predictions(num_obs * num_states);
    vector<double> oob_predictions(num_obs * num_states);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_states, 0);
        vector<double> oob_pred(num_states, 0);
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                vector<double> tree_pred = tree->predictInitDist(data->get_x_row(i));
                sum_vectors(pred, tree_pred);
                sum_vectors(oob_pred, tree_pred);
            }
            // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
            else {
                vector<double> tree_pred = tree->getInitDist()[tree->getPredictionNodeIDs()[i]];
                sum_vectors(pred, tree_pred);
            }
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_states; ++k) {
            pred[k] /= ntrees;
            if (num_oob_trees > 0) {
                oob_pred[k] /= num_oob_trees;
            }
            predictions[i * num_states + k] = pred[k];
            oob_predictions[i * num_states + k] = oob_pred[k];
        }
    }
    return {predictions, oob_predictions};
}

vector<double> MultistateForest::computePredictions(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    vector<double> predictions(num_obs * num_unique_event_times * dim);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times * dim, 0);

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
            vector<double> tree_pred = get<vector<double>>(tree->predict(new_data.get_x_row(i)));
            sum_vectors(pred, tree_pred);
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times * dim; ++k) {
            pred[k] /= ntrees;
            predictions[i * num_unique_event_times * dim + k] = pred[k];
        }
    }
    return predictions;
}

vector<double> MultistateForest::computePredictedInitialDistributions(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    uint8_t num_states = data->getNumberOfStates();
    vector<double> predictions(num_obs * num_states);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_states, 0);

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
            vector<double> tree_pred = tree->predictInitDist(new_data.get_x_row(i));
            sum_vectors(pred, tree_pred);
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_states; ++k) {
            pred[k] /= ntrees;
            predictions[i * num_states + k] = pred[k];
        }
    }
    return predictions;
}
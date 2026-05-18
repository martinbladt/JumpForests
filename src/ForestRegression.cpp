#include "ForestRegression.h"

// functions for growing regression forests
//--------------------------------------------------------------------------------------

RegressionForest::RegressionForest() {
    
}

// grows a regression forest using multithreading via OpenMP
void RegressionForest::grow() {
    int n = data->getNumberOfObs();
    
    // create vector of indices from 1 to n
    vector<size_t> global_indices(n);
    for (size_t i = 0; i < n; ++i) {
        global_indices[i] = i;
    }

    trees.resize(ntrees);
    oob_indices.resize(ntrees);

    int n_threads = this->nworkers;
    omp_set_num_threads(n_threads);
    Rcout << "Growing forest using " << n_threads << " threads" << endl;

    // use OpenMP for parallel tree growing
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (int i = 0; i < static_cast<int>(ntrees); ++i) {
        // give each thread its own random number generator to prevent races
        mt19937 local_rng(seed + i);
        unique_ptr<RegressionTree> tree;

        vector<size_t> bootstrap_indices;
        vector<size_t> holdout_indices;     // only relevant for honest trees

        // bootstrap
        if (!honest) {
            size_t subsample_size = floor(sample_rate * n);
            bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
            oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
            tree = make_unique<RegressionTree>(bootstrap_indices);
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
                tree = make_unique<RegressionTree>(grow, holdout);
                oob_indices[i] = computeOOBIndicesDouble(grow, holdout, n);

            } else {
                // if no double bootstrap, bootstrap the whole dataset and then split
                size_t subsample_size = floor(sample_rate * n);
                bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(bootstrap_indices, local_rng);
                tree = make_unique<RegressionTree>(partition.first, partition.second);
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

// functions for predicting with survival forests
//--------------------------------------------------------------------------------------

double RegressionForest::predict(const vector<double>& x) {
    double result = 0;
    for (const auto& tree : trees) {
        double prediction = get<double>(tree->predict(x));
        result += prediction;
    }
    return result / ntrees;
}

pair<vector<double>, vector<double>> RegressionForest::computePredictions() {
    size_t num_obs = data->getNumberOfObs();
    vector<double> predictions(num_obs);
    vector<double> oob_predictions(num_obs);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        double pred = 0;
        double pred_oob = 0;
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            RegressionTree* tree = dynamic_cast<RegressionTree*>(trees[j].get());
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                double tree_pred = get<double>(tree->predict(data->get_x_row(i)));
                pred += tree_pred;
                pred_oob += tree_pred;
            }
            // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
            else {
                double tree_pred = tree->getMeans()[tree->getPredictionNodeIDs()[i]];
                pred += tree_pred;
            }
        }

        // normalise and save predictions
        predictions[i] = pred / ntrees;
        oob_predictions[i] = pred_oob / num_oob_trees;
    }

    return {predictions, oob_predictions};
}

vector<double> RegressionForest::computePredictions(const Data& new_data) {
    size_t num_features = new_data.getNumberOfFeatures();
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        double pred = 0;

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            RegressionTree* tree = dynamic_cast<RegressionTree*>(trees[j].get());
            pred += get<double>(tree->predict(new_data.get_x_row(i)));
        }
        predictions[i] = pred / ntrees;
    }
    return predictions;
}

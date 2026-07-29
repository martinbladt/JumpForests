#include "ForestRegression.h"

// functions for growing regression forests
//--------------------------------------------------------------------------------------

RegressionForest::RegressionForest() {
    
}

// grows a regression forest using multithreading via OpenMP
void RegressionForest::grow() {
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
    
    // create vector of indices from 0 to n - 1
    vector<size_t> global_indices(n);
    for (size_t i = 0; i < n; ++i) {
        global_indices[i] = i;
    }

    trees.resize(ntrees);
    oob_indices.resize(ntrees);
    vimp_oob_indices.clear();
    vimp_tree_uses_feature.clear();
    vimp_tree_errors.clear();

    size_t n_threads = this->nworkers;
    Rcout << "Growing forest using " << n_threads << " threads" << endl;

    // use OpenMP for parallel tree growing
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        // give each thread its own random number generator to prevent races
        mt19937 local_rng(seed + i);
        unique_ptr<RegressionTree> tree;

        vector<size_t> bootstrap_indices;

        // bootstrap
        if (!honest) {
            size_t subsample_size = floor(sample_rate * n);
            bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
            oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
            tree = make_unique<RegressionTree>(std::move(bootstrap_indices));
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
                tree = make_unique<RegressionTree>(std::move(grow), std::move(holdout));

            } else {
                // if no double bootstrap, bootstrap the whole dataset and then split
                size_t subsample_size = floor(sample_rate * n);
                bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(bootstrap_indices, local_rng);
                oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
                tree = make_unique<RegressionTree>(std::move(partition.first), std::move(partition.second));
            }
        }

        uniform_int_distribution<size_t> dist(0, numeric_limits<size_t>::max());
        tree->initialise(data, mtry, min_node_size, nsplits, splitrule, honest,
                         dist(local_rng), splitrule_par);
        tree->setRNG(local_rng);
        tree->grow();
        trees[i] = std::move(tree);
    }

    // compute all quantities of interest from the vector of trees
    computeForestQuantities();
}

// functions for predicting with regression forests
//--------------------------------------------------------------------------------------

double RegressionForest::predict(const vector<double>& x) {
    long double result = 0;
    for (const auto& tree : trees) {
        RegressionTree* regression_tree = static_cast<RegressionTree*>(tree.get());
        result += regression_tree->predictValue(x);
    }
    return static_cast<double>(result / ntrees);
}

pair<vector<double>, vector<double>> RegressionForest::computePredictions() {
    size_t num_obs = data->getNumberOfObs();
    vector<double> predictions(num_obs);
    vector<double> oob_predictions(num_obs);

    vector<RegressionTree*> regression_trees(ntrees);
    for (size_t i = 0; i < ntrees; ++i) {
        regression_trees[i] = static_cast<RegressionTree*>(trees[i].get());
    }

    #pragma omp parallel for schedule(static) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        long double pred = 0;
        long double pred_oob = 0;
        size_t num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            RegressionTree* tree = regression_trees[j];
            const vector<double>& tree_means = tree->getMeans();
            const vector<size_t>& prediction_node_IDs = tree->getPredictionNodeIDs();
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                double tree_pred = tree->predictValue(i);
                pred += tree_pred;
                pred_oob += tree_pred;
            }
            // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
            else {
                double tree_pred = tree_means[prediction_node_IDs[i]];
                pred += tree_pred;
            }
        }

        // normalise and save predictions
        predictions[i] = static_cast<double>(pred / ntrees);
        oob_predictions[i] = num_oob_trees > 0 ?
          static_cast<double>(pred_oob / num_oob_trees) : NA_REAL;
    }

    return {predictions, oob_predictions};
}

vector<double> RegressionForest::computePredictions(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs);

    vector<RegressionTree*> regression_trees(ntrees);
    for (size_t i = 0; i < ntrees; ++i) {
        regression_trees[i] = static_cast<RegressionTree*>(trees[i].get());
    }

    #pragma omp parallel for schedule(static) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        long double pred = 0;

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            pred += regression_trees[j]->predictValue(new_data, i);
        }
        predictions[i] = static_cast<double>(pred / ntrees);
    }
    return predictions;
}

void RegressionForest::prepareVIMPCache() {
    if (!vimp_oob_indices.empty()) {
        return;
    }

    // quantities below are the same for every feature and only have to be computed once
    OOBNonBoolIndices(vimp_oob_indices, oob_indices);
    vimp_tree_uses_feature.assign(ntrees, vector<bool>(data->getNumberOfFeatures(), false));
    vimp_tree_errors.assign(ntrees, 0);
    const vector<double>& y = data->get_y();

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        RegressionTree* tree = static_cast<RegressionTree*>(trees[i].get());
        const vector<size_t>& left_daughters = tree->getLeftDaughters();
        const vector<size_t>& feature_IDs = tree->getFeatureIDs();

        // leaf nodes have a dummy feature ID, so only consider internal nodes
        for (size_t j = 0; j < left_daughters.size(); ++j) {
            if (left_daughters[j] != 0) {
                vimp_tree_uses_feature[i][feature_IDs[j]] = true;
            }
        }

        const vector<size_t>& tree_oob_indices = vimp_oob_indices[i];
        if (tree_oob_indices.empty()) {
            continue;
        }

        const vector<double>& tree_means = tree->getMeans();
        double tree_oob_error = 0;
        for (size_t obs_id : tree_oob_indices) {
            double tree_pred = tree_means[tree->predictionLeafID(obs_id)];
            double residual = y[obs_id] - tree_pred;
            tree_oob_error += residual * residual;
        }
        vimp_tree_errors[i] = tree_oob_error;
    }
}

/*
    For each tree, compute the sum across all OOB observations for that tree
    of the difference in MSE for the covariate vector where the values in column
    'feature' has been shuffled and left unchanged, respectively. Then take the
    average over the forest. This is Breiman-Cutler feature importance.
*/

double RegressionForest::computeVIMPPermute(size_t feature, int feature_seed) {
    if (feature >= data->getNumberOfFeatures()) {
        throw runtime_error("Feature index is out of range");
    }
    prepareVIMPCache();
    const vector<double>& y = data->get_y();
    vector<double> tree_vimp(ntrees, 0);
    vector<unsigned char> tree_has_oob(ntrees, 0);

    #pragma omp parallel num_threads(this->nworkers)
    {
        vector<double> shuffled_oob_values;

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < ntrees; ++i) {
            RegressionTree* tree = static_cast<RegressionTree*>(trees[i].get());
            const vector<size_t>& tree_oob_indices = vimp_oob_indices[i];
            size_t num_oob_obs = tree_oob_indices.size();
            if (num_oob_obs == 0) {
                continue;
            }
            tree_has_oob[i] = 1;

            // a feature which is not used in the tree has VIMP contribution zero
            if (!vimp_tree_uses_feature[i][feature]) {
                continue;
            }

            shuffled_oob_values.resize(num_oob_obs);
            for (size_t j = 0; j < num_oob_obs; ++j) {
                shuffled_oob_values[j] = data->get_x(tree_oob_indices[j], feature);
            }
            mt19937 local_rng = makeVIMPTreeRNG(feature_seed, i);
            shuffle(shuffled_oob_values.begin(), shuffled_oob_values.end(), local_rng);

            const vector<double>& tree_means = tree->getMeans();
            double tree_oob_error_shuffled = 0;
            for (size_t j = 0; j < num_oob_obs; ++j) {
                size_t obs_id = tree_oob_indices[j];
                size_t leaf_id = tree->predictionLeafIDPermuted(obs_id, feature, shuffled_oob_values[j]);
                double residual = y[obs_id] - tree_means[leaf_id];
                tree_oob_error_shuffled += residual * residual;
            }
            tree_vimp[i] = (tree_oob_error_shuffled - vimp_tree_errors[i]) / (double) num_oob_obs;
        }
    }

    // average the VIMP over trees with OOB observations
    double result = 0;
    size_t valid_trees = 0;
    for (size_t i = 0; i < ntrees; ++i) {
        if (tree_has_oob[i]) {
            result += tree_vimp[i];
            ++valid_trees;
        }
    }
    if (valid_trees == 0) {
        throw runtime_error("Cannot compute VIMP without OOB observations");
    }
    return result / (double) valid_trees;
}

/*
    For each tree, compute the sum across all OOB observations for that tree
    of the difference in MSE for the predicted value with and without random
    daughter assignment
*/

double RegressionForest::computeVIMPRandom(size_t feature, int feature_seed) {
    if (feature >= data->getNumberOfFeatures()) {
        throw runtime_error("Feature index is out of range");
    }
    prepareVIMPCache();
    const vector<double>& y = data->get_y();
    vector<double> tree_vimp(ntrees, 0);
    vector<unsigned char> tree_has_oob(ntrees, 0);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        RegressionTree* tree = static_cast<RegressionTree*>(trees[i].get());
        const vector<size_t>& tree_oob_indices = vimp_oob_indices[i];
        size_t num_oob_obs = tree_oob_indices.size();
        if (num_oob_obs == 0) {
            continue;
        }
        tree_has_oob[i] = 1;

        // a feature which is not used in the tree has VIMP contribution zero
        if (!vimp_tree_uses_feature[i][feature]) {
            continue;
        }

        mt19937 local_rng = makeVIMPTreeRNG(feature_seed, i);
        const vector<double>& tree_means = tree->getMeans();
        double tree_oob_error_random = 0;
        
        for (size_t obs_id : tree_oob_indices) {
            size_t leaf_id = tree->predictionLeafIDVIMP(obs_id, feature, local_rng);
            double residual = y[obs_id] - tree_means[leaf_id];
            tree_oob_error_random += residual * residual;
        }
        tree_vimp[i] = (tree_oob_error_random - vimp_tree_errors[i]) / (double) num_oob_obs;
    }

    // average the VIMP over trees with OOB observations
    double result = 0;
    size_t valid_trees = 0;
    for (size_t i = 0; i < ntrees; ++i) {
        if (tree_has_oob[i]) {
            result += tree_vimp[i];
            ++valid_trees;
        }
    }
    if (valid_trees == 0) {
        throw runtime_error("Cannot compute VIMP without OOB observations");
    }
    return result / (double) valid_trees;
}

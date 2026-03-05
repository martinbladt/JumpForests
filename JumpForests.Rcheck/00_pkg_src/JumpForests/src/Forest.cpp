#include "Forest.h"

void Forest::initialise(shared_ptr<Data> data, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, string splitrule,
                        unsigned int ntrees, bool honest, bool swr, double sample_rate, unsigned int seed, unsigned int nworkers) {
    // initialise with the chosen hyperparameters
    this->data = data;
    this->mtry = mtry;
    this->min_node_size = min_node_size;
    this->nsplits = nsplits;
    this->ntrees = ntrees;
    this->trees.reserve(ntrees);
    this->seed = seed;
    this->splitrule = splitrule;
    this->honest = honest;
    this->swr = swr;
    this->sample_rate = sample_rate;
    
    // set number of threads to use during fitting and predicting
    int max_workers = thread::hardware_concurrency();
    if (nworkers <= max_workers && nworkers >= 1) {
        this->nworkers = nworkers;
    } else {
        this->nworkers = max_workers;
    }
    random_number_generator.seed(seed);
}

void Forest::computeForestQuantities() {
    double total_num_nodes = 0;
    double total_num_terminal_nodes = 0;
    double sum_tree_depth = 0;
    
    // update tree info
    for (const auto& tree : trees) {
        total_num_nodes += tree->getNumberOfNodes();
        total_num_terminal_nodes += tree->getNumberOfTerminalNodes();
        sum_tree_depth += tree->getTreeDepth();
    }
    
    double num_trees = static_cast<double>(ntrees);
    this->avg_num_nodes = total_num_nodes/num_trees;
    this->avg_num_terminal_nodes = total_num_terminal_nodes/num_trees;
    this->avg_tree_depth = sum_tree_depth/num_trees;
}

// shuffles the value of one particular feature among all oob covariates for all trees
vector<vector<double>> Forest::shuffledFeatureValues(const vector<vector<size_t>>& oob_indices_non_bool, size_t feature, int feature_seed) {
    size_t num_obs = data->getNumberOfObs();
    
    // now compute shuffled values for all trees and observations
    vector<vector<double>> result(ntrees);
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        vector<double> result_tree = data->get_x_col(feature);

        // extract all OOB values for the feature
        vector<double> oob_values;
        oob_values.reserve(oob_indices_non_bool[i].size());
        for (size_t obs_id : oob_indices_non_bool[i]) {
            oob_values.push_back(result_tree[obs_id]);
        }

        // thread-safe local random number generator
        mt19937 local_rng(feature_seed + i);

        // shuffle OOB values
        shuffle(oob_values.begin(), oob_values.end(), local_rng);
        
        // shuffle indices
        //vector<size_t> tree_oob_indices = oob_indices_non_bool[i];
        //shuffle(tree_oob_indices.begin(), tree_oob_indices.end(), local_rng);

        // now select the permuted feature values
        for (size_t j = 0; j < oob_indices_non_bool[i].size(); ++j) {
            result_tree[oob_indices_non_bool[i][j]] = oob_values[j];
        }
        result[i] = std::move(result_tree);
    }
    return result;
}

// translates the matrix of booleans into a matrix of actual indices
void OOBNonBoolIndices(vector<vector<size_t>>& oob_indices_non_bool, const vector<vector<bool>>& oob_indices) {
    size_t ntrees = oob_indices.size();
    size_t num_obs = oob_indices[0].size();
    for (size_t i = 0; i < ntrees; ++i) {
        vector<size_t> indices;
        for (size_t j = 0; j < num_obs; ++j) {
            if (oob_indices[i][j]) {
                indices.push_back(j);
            }
        }
        oob_indices_non_bool.push_back(indices);
    }
}

// translates a NumericMatrix into a flattened 2D vector in C++ (not used)
vector<double> matrixRtoCpp(const NumericMatrix& feature_matrix) {
    size_t num_obs = feature_matrix.nrow();
    size_t num_features = feature_matrix.ncol();
    vector<double> result(num_obs * num_features);

    // fill the flattened array observation by observation
    for (size_t i = 0; i < num_obs; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            result[i * num_features + j] = feature_matrix[i, j];
        }
    }
    return result;
}

#include "Forest.h"

mt19937 makeVIMPTreeRNG(int feature_seed, size_t tree_id) {
    seed_seq::result_type seed_data[2] = {
        static_cast<seed_seq::result_type>(feature_seed),
        static_cast<seed_seq::result_type>(tree_id)
    };
    seed_seq tree_seed(seed_data, seed_data + 2);
    return mt19937(tree_seed);
}

void Forest::initialise(shared_ptr<Data> data, unsigned int mtry, unsigned int min_node_size, double max_depth, unsigned int nsplits,
                        string splitrule, unsigned int ntrees, bool honest, bool swr, double sample_rate, bool double_bootstrap,
                        unsigned int seed, unsigned int nworkers, double splitrule_par) {
    // initialise with the chosen hyperparameters
    this->data = data;
    this->mtry = mtry;
    this->min_node_size = min_node_size;
    this->nsplits = nsplits;
    this->ntrees = ntrees;
    this->trees.reserve(ntrees);
    this->seed = seed;
    this->splitrule = splitrule;
    this->splitrule_par = splitrule_par;
    this->honest = honest;
    this->swr = swr;
    this->sample_rate = sample_rate;
    this->double_bootstrap = double_bootstrap;
    this->max_depth = max_depth;
    
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

// translates the matrix of booleans into a matrix of actual indices
void OOBNonBoolIndices(vector<vector<size_t>>& oob_indices_non_bool, const vector<vector<bool>>& oob_indices) {
    size_t ntrees = oob_indices.size();
    size_t num_obs = oob_indices[0].size();
    oob_indices_non_bool.clear();
    oob_indices_non_bool.reserve(ntrees);
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

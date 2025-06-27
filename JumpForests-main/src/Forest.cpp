#include "Forest.h"

void Forest::initialise(shared_ptr<Data> data, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, unsigned int ntrees, unsigned int seed) {
    // initialise with the chosen hyperparameters
    this->data = data;
    this->mtry = mtry;
    this->min_node_size = min_node_size;
    this->nsplits = nsplits;
    this->ntrees = ntrees;

    random_number_generator.seed(seed);
}

void Forest::computeForestQuantities() {
    double total_num_nodes = 0;
    double total_num_terminal_nodes = 0;
    double sum_tree_depth = 0;
    
    for (const auto& tree : trees) {
        this->left_daughters.push_back(tree->getLeftDaughters());
        this->feature_IDs.push_back(tree->getFeatureIDs());
        this->thresholds.push_back(tree->getThresholds());
        this->depths.push_back(tree->getDepths());
        this->num_nodes.push_back(tree->getNumberOfNodes());
        this->num_terminal_nodes.push_back(tree->getNumberOfTerminalNodes());
        this->tree_depths.push_back(tree->getTreeDepth());

        // update tree info
        total_num_nodes += tree->getNumberOfNodes();
        total_num_terminal_nodes += tree->getNumberOfTerminalNodes();
        sum_tree_depth += tree->getTreeDepth();
    }
    
    double num_trees = static_cast<double>(ntrees);
    this->avg_num_nodes = total_num_nodes/num_trees;
    this->avg_num_terminal_nodes = total_num_terminal_nodes/num_trees;
    this->avg_tree_depth = sum_tree_depth/num_trees;
}

void Forest::cleanUp() {
    vector<vector<bool>>().swap(oob_indices);
}
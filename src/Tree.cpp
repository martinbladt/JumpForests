#include "Tree.h"

void Tree::initialise(shared_ptr<Data> data, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, string splitrule, bool honest, unsigned int seed) {
    // initialise with the chosen hyperparameters
    this->data = data;
    this->mtry = mtry;
    this->min_node_size = min_node_size;
    this->nsplits = nsplits;
    this->splitrule = splitrule;
    this->honest = honest;

    // initialise tree info
    num_nodes = 1;
    num_terminal_nodes = 0;
    prediction_node_IDs = vector<size_t>((*data).getNumberOfObs(), 0);  // should this be a choice from the user?

    // set seed
    random_number_generator.seed(seed);
}

void Tree::setRNG(mt19937 rng) {
    random_number_generator = rng;
}

size_t Tree::predictionLeafID(const vector<double>& x) {
    size_t current_node = 0;
    while (left_daughters[current_node] != 0) { // while not yet in a terminal node
        // if the feature is categorical, check whether the coordinate of x belongs to the left or right subset
        if (data->getCategorical()[feature_IDs[current_node]]) {
            const vector<double>& left_subset = thresholds[current_node];
            if (find(left_subset.begin(), left_subset.end(), x[feature_IDs[current_node]]) != left_subset.end()) {
                current_node = left_daughters[current_node];
            }
            else {
                // right daughter is always the left plus one
                current_node = left_daughters[current_node] + 1;
            }
        }
        // if the feature is continuous, check whether the coordinate of x is below the threshold
        else {
            if (x[feature_IDs[current_node]] <= thresholds[current_node][0]) {
                current_node = left_daughters[current_node];
            }
            else {
                // right daughter is always the left plus one
                current_node = left_daughters[current_node] + 1;
            }
        }
    }
    return(current_node);
}

size_t Tree::predictionLeafIDVIMP(const vector<double>& x, size_t feature, mt19937& rng) {
    uniform_int_distribution<size_t> daughter_id(0, 1);
    size_t current_node = 0;
    while (left_daughters[current_node] != 0) { // while not yet in a terminal node
        // if the feature in the current node equals the chosen feature, make daughter assignment random
        if (feature_IDs[current_node] == feature) {
            //size_t left_daughter_size = node_sizes[left_daughters[current_node]];
            //size_t right_daughter_size = node_sizes[left_daughters[current_node] + 1];
            //discrete_distribution<size_t> daughter_id({left_daughter_size, right_daughter_size});
            size_t daughter = daughter_id(rng);
            //cout << "daughter:" << daughter << endl;
            current_node = left_daughters[current_node] + daughter;
        }
        // if not, do prediction as normal
        else {
            // if the feature is categorical, check whether the coordinate of x belongs to the left or right subset
            if (data->getCategorical()[feature_IDs[current_node]]) {
                const vector<double>& left_subset = thresholds[current_node];
                if (find(left_subset.begin(), left_subset.end(), x[feature_IDs[current_node]]) != left_subset.end()) {
                    current_node = left_daughters[current_node];
                }
                else {
                    // right daughter is always the left plus one
                    current_node = left_daughters[current_node] + 1;
                }
            }
            // if the feature is continuous, check whether the coordinate of x is below the threshold
            else {
                if (x[feature_IDs[current_node]] <= thresholds[current_node][0]) {
                    current_node = left_daughters[current_node];
                }
                else {
                    // right daughter is always the left plus one
                    current_node = left_daughters[current_node] + 1;
                }
            }
        }
    }
    return current_node;
}
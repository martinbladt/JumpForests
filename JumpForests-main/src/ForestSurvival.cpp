#include "ForestSurvival.h"

// constructor for survival forests
//--------------------------------------------------------------------------------------

SurvivalForest::SurvivalForest(const vector<double> unique_event_times) :
    unique_event_times {unique_event_times} {
        this->num_unique_event_times = unique_event_times.size();
}

// functions for growing survival forests
//--------------------------------------------------------------------------------------

void SurvivalForest::grow() {
    int n = (*data).getNumberOfObs();
    for (int i = 0; i < ntrees; ++i) {
        // for now, use classical (Efron) bootstrap, later subsampling without replacement should be implemented
        vector<size_t> bootstrap_indices = sampleIndices(n, n, true, random_number_generator);
        vector<bool> oob_indices_tree = computeOOBIndices(bootstrap_indices, n);
        oob_indices.push_back(oob_indices_tree);

        // grow each survival tree
        unique_ptr<SurvivalTree> tree = make_unique<SurvivalTree>(unique_event_times, bootstrap_indices);
        uniform_int_distribution<size_t> dist(0, numeric_limits<size_t>::max());
        tree->initialise(data, mtry, min_node_size, nsplits, dist(random_number_generator));
        tree->setRNG(random_number_generator);
        tree->grow();
        trees.push_back(move(tree));
    }

    // compute all quantities of interest from the vector of trees
    computeForestQuantities();
}

// functions for predicting with survival forests
//--------------------------------------------------------------------------------------

// should later be extended to use multithreading and include OOB
vector<double> SurvivalForest::predict(vector<double> x) {
    vector<double> result(num_unique_event_times, 0);
    for (const auto& tree : trees) {
        vector<double> prediction = get<vector<double>>(tree->predict(x));
        sum_vectors(result, prediction);
    }
    for (int i = 0; i < num_unique_event_times; ++i) {
        result[i] = result[i]/ntrees;
    }
    return(result);
}
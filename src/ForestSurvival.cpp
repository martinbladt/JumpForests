#include "ForestSurvival.h"

// constructor for survival forests
//--------------------------------------------------------------------------------------

SurvivalForest::SurvivalForest(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, const vector<size_t>& true_event_time_ids) :
    unique_event_times {unique_event_times}, response_event_time_ids {response_event_time_ids}, true_event_time_ids {true_event_time_ids} {
        this->num_unique_event_times = unique_event_times.size();
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
            tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, true_event_time_ids, bootstrap_indices);
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
                tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, true_event_time_ids, grow, holdout);
                oob_indices[i] = computeOOBIndicesDouble(grow, holdout, n);

            } else {
                // if no double bootstrap, bootstrap the whole dataset and then split
                size_t subsample_size = floor(sample_rate * n);
                auto global_bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(global_bootstrap_indices, local_rng);
                tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, true_event_time_ids, partition.first, partition.second);
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

        // for debugging
        //#pragma omp critical
        //Rcout << "Finished growing tree " << i 
        //          << " on thread " << omp_get_thread_num() << endl;
    }

    // compute all quantities of interest from the vector of trees
    computeForestQuantities();
}

/*
// old grow function that does not use multithreading
void SurvivalForest::grow() {
    int n = (*data).getNumberOfObs();
    for (size_t i = 0; i < ntrees; ++i) {
        // for now, use classical (Efron) bootstrap, later subsampling without replacement should be implemented
        // optimisation idea: no need to allocate {0, 1, ..., n - 1} many times when n is fixed (write a separate bootstrap function)
        vector<size_t> bootstrap_indices = sampleIndices(n, n, true, random_number_generator);
        vector<bool> oob_indices_tree = computeOOBIndices(bootstrap_indices, n);
        oob_indices.push_back(oob_indices_tree);

        // grow each survival tree
        Rcout << "Growing tree " << i << endl;
        unique_ptr<SurvivalTree> tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, bootstrap_indices);
        uniform_int_distribution<size_t> dist(0, numeric_limits<size_t>::max());
        tree->initialise(data, mtry, min_node_size, nsplits, dist(random_number_generator));
        tree->setRNG(random_number_generator);
        tree->grow();
        trees.push_back(std::move(tree));
    }

    // compute all quantities of interest from the vector of trees
    computeForestQuantities();
}
*/



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

pair<vector<double>, vector<double>> SurvivalForest::computePredictions() {
    size_t num_obs = data->getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);
    vector<double> oob_predictions(num_obs * num_unique_event_times);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times, 0);
        vector<double> oob_pred(num_unique_event_times, 0);
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                vector<double> tree_pred = get<vector<double>>(tree->predict(data->get_x_row(i)));
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
            }
            predictions[i * num_unique_event_times + k] = pred[k];
            oob_predictions[i * num_unique_event_times + k] = oob_pred[k];
        }
    }

    return {predictions, oob_predictions};
}

vector<double> SurvivalForest::computePredictions(const Data& new_data) {
    size_t num_features = new_data.getNumberOfFeatures();
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times, 0);

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
            vector<double> tree_pred = get<vector<double>>(tree->predict(new_data.get_x_row(i)));
            sum_vectors(pred, tree_pred);
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times; ++k) {
            pred[k] /= ntrees;
            predictions[i * num_unique_event_times + k] = pred[k];
        }
    }
    return predictions;
}

double SurvivalForest::computeVIMPPermute(size_t feature, int feature_seed) {
    /*
    for each tree, compute the difference in the sums of errors, 
    then divide by the number of OOB samples for that tree
    finally, take the average VIMP over all trees
     */

    size_t num_obs = data->getNumberOfObs();
    const vector<double>& times = data->get_y_col(0);
    const vector<double>& ind = data->get_y_col(1);

    double total_vimp = 0;
    double total_oob = 0;   // for proper weighting by number of OOB observations
    // shuffle feature values for all trees among the oob covariates
    vector<vector<size_t>> oob_indices_non_bool;
    OOBNonBoolIndices(oob_indices_non_bool, oob_indices);
    const vector<vector<double>>& shuffled_values_feature = shuffledFeatureValues(oob_indices_non_bool, feature, feature_seed);
    
    // #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[i].get());
        size_t num_oob_obs = oob_indices_non_bool[i].size();
        total_oob += num_oob_obs;
        
        // initialise vectors of outcomes
        vector<double> tree_outcomes;
        vector<double> tree_outcomes_shuffled;
        tree_outcomes.reserve(num_oob_obs);
        tree_outcomes_shuffled.reserve(num_oob_obs);

        // initialise vectors of times and indicators belonging to the OOB
        // observations of this tree
        vector<double> times_tree;
        vector<double> ind_tree;
        times_tree.reserve(num_oob_obs);
        ind_tree.reserve(num_oob_obs);

        for (size_t j = 0; j < num_obs; ++j) {
            if (oob_indices[i][j]) {
                vector<double> x = data->get_x_row(j);
                vector<double> tree_pred = get<vector<double>>(tree->predict(x));
                x[feature] = shuffled_values_feature[i][j];
                vector<double> tree_pred_shuffled = get<vector<double>>(tree->predict(x));

                // update outcomes
                //tree_outcomes.push_back(vector_sum(tree_pred));
                //tree_outcomes_shuffled.push_back(vector_sum(tree_pred_shuffled));

                // test using the final chf value instead of the sum
                tree_outcomes.push_back(tree_pred[num_unique_event_times - 1]);
                tree_outcomes_shuffled.push_back(tree_pred_shuffled[num_unique_event_times - 1]);

                // update times
                times_tree.push_back(times[j]);
                ind_tree.push_back(ind[j]);
            }
        }

        // now compute and save tree VIMP
        total_vimp += (computeConcordanceIndex(tree_outcomes, times_tree, ind_tree) - computeConcordanceIndex(tree_outcomes_shuffled, times_tree, ind_tree)) * num_oob_obs;
    }
    
    // return forest VIMP
    return total_vimp / total_oob;
}

// computes VIMP predictions by random daughter assignments
// (this method does not work properly, I don't know why)
vector<double> SurvivalForest::computePredictionsVIMPRandom(size_t feature, int feature_seed) {
    size_t num_obs = data->getNumberOfObs();
    vector<double> vimp_predictions(num_obs * num_unique_event_times);

    // for reproducibility, use the random number generator of the forest but with a feature dependent seed
    //random_number_generator.seed(seed + feature);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> vimp_pred(num_unique_event_times, 0);
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all oob predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
            
            if (oob_indices[j][i]) {
                // local random number generator to avoid races
                mt19937 local_rng(feature_seed + j);
                ++num_oob_trees;
                vector<double> vimp_tree_pred = get<vector<double>>(tree->predictVIMP(data->get_x_row(i), feature, local_rng));
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
                // local random number generator to avoid races
                mt19937 local_rng(feature_seed + j);
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
#include "ForestClassification.h"

namespace {
mt19937 makeVIMPTreeRNG(int feature_seed, size_t tree_id) {
    seed_seq::result_type seed_data[2] = {
        static_cast<seed_seq::result_type>(feature_seed),
        static_cast<seed_seq::result_type>(tree_id)
    };
    seed_seq tree_seed(seed_data, seed_data + 2);
    return mt19937(tree_seed);
    }
}

// functions for growing classification forests
//--------------------------------------------------------------------------------------

ClassificationForest::ClassificationForest() {

}

// grows a classification forest using multithreading via OpenMP
void ClassificationForest::grow() {
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
    for (size_t i = 0; i < static_cast<size_t>(ntrees); ++i) {
        // give each thread its own random number generator to prevent races
        mt19937 local_rng(seed + i);
        unique_ptr<ClassificationTree> tree;

        vector<size_t> bootstrap_indices;
        vector<size_t> holdout_indices;     // only relevant for honest trees

        // bootstrap
        if (!honest) {
            size_t subsample_size = floor(sample_rate * n);
            bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
            oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
            tree = make_unique<ClassificationTree>(bootstrap_indices);
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
                tree = make_unique<ClassificationTree>(grow, holdout);
                oob_indices[i] = computeOOBIndicesDouble(grow, holdout, n);

            } else {
                // if no double bootstrap, bootstrap the whole dataset and then split
                size_t subsample_size = floor(sample_rate * n);
                bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(bootstrap_indices, local_rng);
                tree = make_unique<ClassificationTree>(partition.first, partition.second);
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

// functions for predicting with classification forests
//--------------------------------------------------------------------------------------

double ClassificationForest::predict(const vector<double>& x) {
    double result = 0;
    for (const auto& tree : trees) {
        double prediction = get<double>(tree->predict(x));
        result += prediction;
    }
    return result / ntrees;
}

vector<vector<double>> ClassificationForest::computePredictions() {
    size_t num_obs = data->getNumberOfObs();
    size_t num_classes = data->getNumClasses();
    vector<double> predictions(num_obs);
    vector<double> oob_predictions(num_obs);
    vector<double> predictions_prob(num_obs * num_classes);
    vector<double> oob_predictions_prob(num_obs * num_classes);

    // temporary quantities to determine the final predicted class (by majority rule) and their probabilities for each observation
    vector<double> class_counts_obs;
    vector<double> oob_class_counts_obs;
    vector<double> class_probs_obs;
    vector<double> oob_class_probs_obs;

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        class_counts_obs.assign(num_classes, 0);
        oob_class_counts_obs.assign(num_classes, 0);
        class_probs_obs.assign(num_classes, 0);
        oob_class_probs_obs.assign(num_classes, 0);

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            ClassificationTree* tree = dynamic_cast<ClassificationTree*>(trees[j].get());
            
            if (oob_indices[j][i]) {
                // update number of OOB trees and fetch id of the leaf belonging to the current observation
                ++num_oob_trees;
                size_t leaf_id = tree->predictionLeafID(data->get_x_row(i));

                // fetch predicted class and update class counts
                size_t tree_pred_class = static_cast<size_t>(tree->getClasses()[leaf_id]);
                ++class_counts_obs[tree_pred_class];
                ++oob_class_counts_obs[tree_pred_class];

                // fetch predicted class probabilities and increment
                const vector<double>& tree_pred_probs = tree->getClassProportions()[leaf_id];
                class_probs_obs[tree_pred_class] += tree_pred_probs[tree_pred_class];
                oob_class_probs_obs[tree_pred_class] += tree_pred_probs[tree_pred_class];
            }
            // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
            else {
                // repeat the above but only for the inbag-observation
                size_t tree_pred_class = static_cast<size_t>(tree->getClasses()[tree->getPredictionNodeIDs()[i]]);
                ++class_counts_obs[tree_pred_class];
                const vector<double>& tree_pred_probs = tree->getClassProportions()[tree->getPredictionNodeIDs()[i]];
                class_probs_obs[tree_pred_class] += tree_pred_probs[tree_pred_class];
            }
        }

        // normalise and save probability predictions
        size_t obs_index = i * num_classes;
        for (size_t c = 0; c < num_classes; ++c) {
            predictions_prob[obs_index + c] = class_probs_obs[c] / ntrees;
            oob_predictions_prob[obs_index + c] = oob_class_probs_obs[c] / num_oob_trees;
        }
        // determine the final class prediction by majority rule
        predictions[i] = mostFrequentClass(class_counts_obs);
        oob_predictions[i] = mostFrequentClass(oob_class_counts_obs);
    }

    return {predictions, oob_predictions, predictions_prob, oob_predictions_prob};
}

/*
    For each tree, compute the sum across all OOB observations for that tree
    of the difference in error for the covariate vector where the values in column
    'feature' has been shuffled and left unchanged, respectively. Then take the
    average over the forest. This is Breiman-Cutler feature importance. Returns a
    vector of both class-wise VIMP and aggregated VIMP.
*/

vector<double> ClassificationForest::computeVIMPPermute(size_t feature, int feature_seed, string error_type) {
    size_t num_obs = data->getNumberOfObs();
    size_t num_classes = data->getNumClasses();
    const vector<double> y = data->get_y();
    vector<double> result(num_classes + 1, 0);  // one VIMP value for each class plus one aggregated value
    vector<double> tree_vimp(ntrees * (num_classes + 1), 0);

    // shuffle feature values for all trees among the oob covariates
    vector<vector<size_t>> oob_indices_non_bool;
    OOBNonBoolIndices(oob_indices_non_bool, oob_indices);
    const vector<vector<double>>& shuffled_values_feature = shuffledFeatureValues(oob_indices_non_bool, feature, feature_seed);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        ClassificationTree* tree = dynamic_cast<ClassificationTree*>(trees[i].get());
        size_t num_oob_obs = oob_indices_non_bool[i].size();
        vector<double> tree_oob_error(num_classes + 1, 0);
        vector<double> tree_oob_error_shuffled(num_classes + 1, 0);

        for (size_t j = 0; j < num_obs; ++j) {
            if (oob_indices[i][j]) {
                // fetch tree predictions without shuffled values for 'feature'
                vector<double> x = data->get_x_row(j);
                size_t leaf_id = tree->predictionLeafID(x);
                double tree_pred_class = tree->getClasses()[leaf_id];
                vector<double> tree_pred_probs = tree->getClassProportions()[leaf_id];

                // fetch tree predictions with shuffled values for 'feature'
                x[feature] = shuffled_values_feature[i][j];
                size_t leaf_id_shuffled = tree->predictionLeafID(x);
                double tree_pred_class_shuffled = tree->getClasses()[leaf_id_shuffled];
                vector<double> tree_pred_probs_shuffled = tree->getClassProportions()[leaf_id_shuffled];

                if (error_type == "brier") {                        // Brier score
                    for (double c = 0; c < num_classes; ++c) {
                        if (c == y[j]) {
                            tree_oob_error[c] += (1 - tree_pred_probs[c]) * (1 - tree_pred_probs[c]);
                            tree_oob_error[num_classes] += tree_oob_error[c];
                        } else {
                            tree_oob_error[c] += tree_pred_probs[c] * tree_pred_probs[c];
                            tree_oob_error[num_classes] += tree_oob_error[c];
                        }
                        if (c == y[j]) {
                            tree_oob_error_shuffled[c] += (1 - tree_pred_probs_shuffled[c]) * (1 - tree_pred_probs_shuffled[c]);
                            tree_oob_error_shuffled[num_classes] += tree_oob_error_shuffled[c];
                        } else {
                            tree_oob_error_shuffled[c] += tree_pred_probs_shuffled[c] * tree_pred_probs_shuffled[c];
                            tree_oob_error_shuffled[num_classes] += tree_oob_error_shuffled[c];
                        }
                    }
                } else if (error_type == "misc") {                   // misclassification error
                    for (double c = 0; c < num_classes; ++c) {
                        if (c == tree_pred_class && tree_pred_class != y[j]) {
                            ++tree_oob_error[c];
                            ++tree_oob_error[num_classes];
                        }
                        if (c == tree_pred_class_shuffled && tree_pred_class_shuffled != y[j]) {
                            ++tree_oob_error_shuffled[c];
                            ++tree_oob_error_shuffled[num_classes];
                        }
                    }
                } else {
                    throw("Unknown choice of loss function. Choose either 'brier' or 'misc' (misclassification)");
                }
            }
        }
        // add VIMP contribution from the tree for each class and in total
        size_t index = i * (num_classes + 1);
        for (size_t c = 0; c < num_classes + 1; ++c) {
            tree_vimp[index + c] = (tree_oob_error_shuffled[c] - tree_oob_error[c]) / (double) num_oob_obs;
        }
    }
    for (size_t i = 0; i < ntrees; ++i) {
        size_t index = i * (num_classes + 1);
        for (size_t c = 0; c < num_classes + 1; ++c) {
            result[c] += tree_vimp[index + c];
        }
    }
    // average VIMP over all trees and return final forest VIMP
    for (size_t c = 0; c < num_classes + 1; ++c) {
        result[c] /= (double) ntrees;
    }
    if (error_type == "brier") {    // use the adjusted Brier score as in the vignette https://www.randomforestsrc.org/articles/rfsrc-subsample.html
        result[num_classes] *= (double) num_classes / (double) (num_classes - 1);
    }
    return result;
}

/*
    For each tree, compute the sum across all OOB observations for that tree
    of the difference in error for the predicted value with and without random
    daughter assignment. Returns a vector of both class-wise VIMP and aggregated VIMP.
*/

vector<double> ClassificationForest::computeVIMPRandom(size_t feature, int feature_seed, string error_type) {
    size_t num_obs = data->getNumberOfObs();
    size_t num_classes = data->getNumClasses();
    const vector<double> y = data->get_y();
    vector<double> result(num_classes + 1, 0);  // one VIMP value for each class plus one aggregated value
    vector<double> tree_vimp(ntrees * (num_classes + 1), 0);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        ClassificationTree* tree = dynamic_cast<ClassificationTree*>(trees[i].get());
        mt19937 local_rng = makeVIMPTreeRNG(feature_seed, i);
        size_t num_oob_obs = 0;
        vector<double> tree_oob_error(num_classes + 1, 0);
        vector<double> tree_oob_error_random(num_classes + 1, 0);

        for (size_t j = 0; j < num_obs; ++j) {
            if (oob_indices[i][j]) {
                ++num_oob_obs;

                // fetch tree predictions without random daughter assignment
                vector<double> x = data->get_x_row(j);
                size_t leaf_id = tree->predictionLeafID(x);
                double tree_pred_class = tree->getClasses()[leaf_id];
                vector<double> tree_pred_probs = tree->getClassProportions()[leaf_id];

                // fetch predictions with random daughter assignment
                size_t leaf_id_vimp = tree->predictionLeafIDVIMP(x, feature, local_rng);
                double tree_pred_class_random = tree->getClasses()[leaf_id_vimp];
                vector<double> tree_pred_probs_random = tree->getClassProportions()[leaf_id_vimp];

                if (error_type == "brier") {                        // Brier score
                    for (double c = 0; c < num_classes; ++c) {
                        if (c == y[j]) {
                            tree_oob_error[c] += (1 - tree_pred_probs[c]) * (1 - tree_pred_probs[c]);
                            tree_oob_error[num_classes] += tree_oob_error[c];
                        } else {
                            tree_oob_error[c] += tree_pred_probs[c] * tree_pred_probs[c];
                            tree_oob_error[num_classes] += tree_oob_error[c];
                        }
                        if (c == y[j]) {
                            tree_oob_error_random[c] += (1 - tree_pred_probs_random[c]) * (1 - tree_pred_probs_random[c]);
                            tree_oob_error_random[num_classes] += tree_oob_error_random[c];
                        } else {
                            tree_oob_error_random[c] += tree_pred_probs_random[c] * tree_pred_probs_random[c];
                            tree_oob_error_random[num_classes] += tree_oob_error_random[c];
                        }
                    }
                } else if (error_type == "misc") {                   // misclassification error
                    for (double c = 0; c < num_classes; ++c) {
                        if (c == tree_pred_class && tree_pred_class != y[j]) {
                            ++tree_oob_error[c];
                            ++tree_oob_error[num_classes];
                        }
                        if (c == tree_pred_class_random && tree_pred_class_random != y[j]) {
                            ++tree_oob_error_random[c];
                            ++tree_oob_error_random[num_classes];
                        }
                    }
                } else {
                    throw("Unknown choice of loss function. Choose either 'brier' or 'misc' (misclassification)");
                }
            }
        }
        // add VIMP contribution from the tree for each class and in total
        size_t index = i * (num_classes + 1);
        for (size_t c = 0; c < num_classes + 1; ++c) {
            tree_vimp[index + c] = (tree_oob_error_random[c] - tree_oob_error[c]) / (double) num_oob_obs;
        }
    }
    for (size_t i = 0; i < ntrees; ++i) {
        size_t index = i * (num_classes + 1);
        for (size_t c = 0; c < num_classes + 1; ++c) {
            result[c] += tree_vimp[index + c];
        }
    }
    // average VIMP over all trees and return final forest VIMP
    for (size_t c = 0; c < num_classes + 1; ++c) {
        result[c] /= (double) ntrees;
    }
    if (error_type == "brier") {    // use the adjusted Brier score as in the vignette https://www.randomforestsrc.org/articles/rfsrc-subsample.html
        result[num_classes] *= (double) num_classes / (double) (num_classes - 1);
    }
    return result;
}

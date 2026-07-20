#include "ForestClassification.h"

#include <cmath>

namespace {
size_t encodedResponseToClassIndex(double response, size_t num_classes) {
    double rounded_response = round(response);
    if (!isfinite(response) || abs(response - rounded_response) > 1e-8 ||
        rounded_response < 1 || rounded_response > static_cast<double>(num_classes)) {
        throw runtime_error("Classification response values must be encoded as 1, ..., num_classes");
    }
    return static_cast<size_t>(rounded_response) - 1;
}
}

// functions for growing classification forests
//--------------------------------------------------------------------------------------

ClassificationForest::ClassificationForest() {

}

// grows a classification forest using multithreading via OpenMP
void ClassificationForest::grow() {
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
    
    // create vector of indices from 1 to n
    vector<size_t> global_indices(n);
    for (size_t i = 0; i < n; ++i) {
        global_indices[i] = i;
    }

    trees.resize(ntrees);
    oob_indices.resize(ntrees);
    vimp_oob_indices.clear();
    vimp_tree_uses_feature.clear();
    vimp_tree_errors_misc.clear();
    vimp_tree_errors_brier.clear();

    size_t n_threads = this->nworkers;
    Rcout << "Growing forest using " << n_threads << " threads" << endl;

    // use OpenMP for parallel tree growing
    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
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
            tree = make_unique<ClassificationTree>(std::move(bootstrap_indices));
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
                tree = make_unique<ClassificationTree>(std::move(grow), std::move(holdout));

            } else {
                // if no double bootstrap, bootstrap the whole dataset and then split
                size_t subsample_size = floor(sample_rate * n);
                bootstrap_indices = sampleIndices(global_indices, subsample_size, swr, local_rng);
                pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(bootstrap_indices, local_rng);
                oob_indices[i] = computeOOBIndices(bootstrap_indices, n);
                tree = make_unique<ClassificationTree>(std::move(partition.first), std::move(partition.second));
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

vector<vector<double>> ClassificationForest::computePredictions() {
    size_t num_obs = data->getNumberOfObs();
    size_t num_classes = data->getNumClasses();
    vector<double> predictions(num_obs);
    vector<double> oob_predictions(num_obs);
    vector<double> predictions_prob(num_obs * num_classes);
    vector<double> oob_predictions_prob(num_obs * num_classes);

    vector<ClassificationTree*> classification_trees(ntrees);
    for (size_t i = 0; i < ntrees; ++i) {
        classification_trees[i] = static_cast<ClassificationTree*>(trees[i].get());
    }

    #pragma omp parallel num_threads(this->nworkers)
    {
        vector<double> class_counts_obs(num_classes, 0);
        vector<double> oob_class_counts_obs(num_classes, 0);
        vector<double> class_probs_obs(num_classes, 0);
        vector<double> oob_class_probs_obs(num_classes, 0);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < num_obs; ++i) {
            double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB
            fill(class_counts_obs.begin(), class_counts_obs.end(), 0);
            fill(oob_class_counts_obs.begin(), oob_class_counts_obs.end(), 0);
            fill(class_probs_obs.begin(), class_probs_obs.end(), 0);
            fill(oob_class_probs_obs.begin(), oob_class_probs_obs.end(), 0);

            // compute the sum of all predictions for observation i
            for (size_t j = 0; j < ntrees; ++j) {
                ClassificationTree* tree = classification_trees[j];
                const vector<double>& tree_classes = tree->getClasses();
                const vector<vector<double>>& tree_class_proportions = tree->getClassProportions();
                size_t leaf_id;

                if (oob_indices[j][i]) {
                    // update number of OOB trees and fetch id of the leaf belonging to the current observation
                    ++num_oob_trees;
                    leaf_id = tree->predictionLeafID(i);

                    // fetch predicted class and update OOB quantities
                    size_t tree_pred_class = static_cast<size_t>(tree_classes[leaf_id]) - 1;
                    ++oob_class_counts_obs[tree_pred_class];
                    const vector<double>& tree_pred_probs = tree_class_proportions[leaf_id];
                    for (size_t c = 0; c < num_classes; ++c) {
                        oob_class_probs_obs[c] += tree_pred_probs[c];
                    }
                }
                // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
                else {
                    leaf_id = tree->getPredictionNodeIDs()[i];
                }

                // update quantities based on all trees
                size_t tree_pred_class = static_cast<size_t>(tree_classes[leaf_id]) - 1;
                ++class_counts_obs[tree_pred_class];
                const vector<double>& tree_pred_probs = tree_class_proportions[leaf_id];
                for (size_t c = 0; c < num_classes; ++c) {
                    class_probs_obs[c] += tree_pred_probs[c];
                }
            }

            // normalise and save probability predictions
            size_t obs_index = i * num_classes;
            for (size_t c = 0; c < num_classes; ++c) {
                predictions_prob[obs_index + c] = class_probs_obs[c] / ntrees;
                oob_predictions_prob[obs_index + c] = num_oob_trees > 0 ? oob_class_probs_obs[c] / num_oob_trees : NA_REAL;
            }
            // determine the final class prediction by majority rule
            predictions[i] = mostFrequentClass(class_counts_obs);
            oob_predictions[i] = num_oob_trees > 0 ? mostFrequentClass(oob_class_counts_obs) : NA_REAL;
        }
    }

    return {predictions, oob_predictions, predictions_prob, oob_predictions_prob};
}

pair<vector<double>, vector<double>> ClassificationForest::computePredictions(const Data& new_data, bool compute_probs) {
    size_t num_obs = new_data.getNumberOfObs();
    size_t num_classes = data->getNumClasses();
    vector<double> predictions(num_obs);
    vector<double> predictions_prob(num_obs * num_classes);

    vector<ClassificationTree*> classification_trees(ntrees);
    for (size_t i = 0; i < ntrees; ++i) {
        classification_trees[i] = static_cast<ClassificationTree*>(trees[i].get());
    }

    #pragma omp parallel num_threads(this->nworkers)
    {
        vector<double> class_counts_obs(num_classes, 0);
        vector<double> class_probs_obs(compute_probs ? num_classes : 0, 0);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < num_obs; ++i) {
            fill(class_counts_obs.begin(), class_counts_obs.end(), 0);
            if (compute_probs) {
                fill(class_probs_obs.begin(), class_probs_obs.end(), 0);
            }

            for (size_t j = 0; j < ntrees; ++j) {
                ClassificationTree* tree = classification_trees[j];
                size_t leaf_id = tree->predictionLeafID(new_data, i);

                // fetch predicted class and update class counts
                size_t tree_pred_class = static_cast<size_t>(tree->getClasses()[leaf_id]) - 1;
                ++class_counts_obs[tree_pred_class];

                // fetch predicted class probabilities if compute_probs == true and increment
                if (compute_probs) {
                    const vector<double>& tree_pred_probs = tree->getClassProportions()[leaf_id];
                    for (size_t c = 0; c < num_classes; ++c) {
                        class_probs_obs[c] += tree_pred_probs[c];
                    }
                }
            }

            // normalise and save probability predictions if compute_probs == true
            if (compute_probs) {
                size_t obs_index = i * num_classes;
                for (size_t c = 0; c < num_classes; ++c) {
                    predictions_prob[obs_index + c] = class_probs_obs[c] / ntrees;
                }
            }
            // determine the final class prediction by majority rule
            predictions[i] = mostFrequentClass(class_counts_obs);
        }
    }
    return {predictions, predictions_prob};
}

void ClassificationForest::prepareVIMPCache() {
    if (!vimp_oob_indices.empty()) {
        return;
    }

    size_t num_classes = data->getNumClasses();
    size_t error_size = num_classes + 1;

    // quantities below are the same for every feature and only have to be computed once
    OOBNonBoolIndices(vimp_oob_indices, oob_indices);
    vimp_tree_uses_feature.assign(ntrees, vector<bool>(data->getNumberOfFeatures(), false));
    vimp_tree_errors_misc.assign(ntrees * error_size, 0);
    vimp_tree_errors_brier.assign(ntrees * error_size, 0);
    const vector<double>& y = data->get_y();

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        ClassificationTree* tree = static_cast<ClassificationTree*>(trees[i].get());
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

        const vector<double>& tree_classes = tree->getClasses();
        const vector<vector<double>>& tree_class_proportions = tree->getClassProportions();
        size_t error_index = i * error_size;
        for (size_t obs_id : tree_oob_indices) {
            size_t response_class = encodedResponseToClassIndex(y[obs_id], num_classes);
            size_t leaf_id = tree->predictionLeafID(obs_id);
            size_t tree_pred_class = encodedResponseToClassIndex(tree_classes[leaf_id], num_classes);
            const vector<double>& tree_pred_probs = tree_class_proportions[leaf_id];

            if (tree_pred_class != response_class) {
                ++vimp_tree_errors_misc[error_index + response_class];
                ++vimp_tree_errors_misc[error_index + num_classes];
            }
            for (size_t c = 0; c < num_classes; ++c) {
                double residual = tree_pred_probs[c] - static_cast<double>(c == response_class);
                double error = residual * residual;
                vimp_tree_errors_brier[error_index + c] += error;
                vimp_tree_errors_brier[error_index + num_classes] += error;
            }
        }
    }
}

/*
    For each tree, compute the sum across all OOB observations for that tree
    of the difference in error for the covariate vector where the values in column
    'feature' has been shuffled and left unchanged, respectively. Then take the
    average over the forest. This is Breiman-Cutler feature importance. Returns a
    vector of both class-wise VIMP and aggregated VIMP.
*/

vector<double> ClassificationForest::computeVIMPPermute(size_t feature, int feature_seed, string error_type) {
    if (feature >= data->getNumberOfFeatures()) {
        throw runtime_error("Feature index is out of range");
    }
    bool use_brier = error_type == "brier";
    if (!use_brier && error_type != "misc") {
        throw runtime_error("Unknown choice of loss function. Choose either 'brier' or 'misc' (misclassification)");
    }
    prepareVIMPCache();

    size_t num_classes = data->getNumClasses();
    size_t error_size = num_classes + 1;
    const vector<double>& y = data->get_y();
    vector<double> result(num_classes + 1, 0);  // one VIMP value for each class plus one aggregated value
    vector<double> tree_vimp(ntrees * error_size, 0);
    vector<unsigned char> tree_has_oob(ntrees, 0);
    const vector<double>& baseline_errors = use_brier ? vimp_tree_errors_brier : vimp_tree_errors_misc;

    #pragma omp parallel num_threads(this->nworkers)
    {
        vector<double> shuffled_oob_values;
        vector<double> tree_oob_error_shuffled(error_size, 0);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < ntrees; ++i) {
            ClassificationTree* tree = static_cast<ClassificationTree*>(trees[i].get());
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

            fill(tree_oob_error_shuffled.begin(), tree_oob_error_shuffled.end(), 0);
            const vector<double>& tree_classes = tree->getClasses();
            const vector<vector<double>>& tree_class_proportions = tree->getClassProportions();
            for (size_t j = 0; j < num_oob_obs; ++j) {
                size_t obs_id = tree_oob_indices[j];
                size_t response_class = encodedResponseToClassIndex(y[obs_id], num_classes);
                size_t leaf_id = tree->predictionLeafIDPermuted(obs_id, feature, shuffled_oob_values[j]);

                if (use_brier) {                                // Brier score
                    const vector<double>& tree_pred_probs = tree_class_proportions[leaf_id];
                    for (size_t c = 0; c < num_classes; ++c) {
                        double residual = tree_pred_probs[c] - static_cast<double>(c == response_class);
                        double error = residual * residual;
                        tree_oob_error_shuffled[c] += error;
                        tree_oob_error_shuffled[num_classes] += error;
                    }
                } else if (encodedResponseToClassIndex(tree_classes[leaf_id], num_classes) != response_class) {
                    ++tree_oob_error_shuffled[response_class];
                    ++tree_oob_error_shuffled[num_classes];
                }
            }

            // add VIMP contribution from the tree for each class and in total
            size_t index = i * error_size;
            for (size_t c = 0; c < error_size; ++c) {
                tree_vimp[index + c] = (tree_oob_error_shuffled[c] - baseline_errors[index + c]) / (double) num_oob_obs;
            }
        }
    }

    size_t valid_trees = 0;
    for (size_t i = 0; i < ntrees; ++i) {
        if (tree_has_oob[i] == 0) {
            continue;
        }
        ++valid_trees;
        size_t index = i * error_size;
        for (size_t c = 0; c < error_size; ++c) {
            result[c] += tree_vimp[index + c];
        }
    }
    if (valid_trees == 0) {
        throw runtime_error("Cannot compute VIMP without OOB observations");
    }
    // average VIMP over all trees and return final forest VIMP
    for (size_t c = 0; c < error_size; ++c) {
        result[c] /= (double) valid_trees;
    }
    if (use_brier) {    // use the adjusted Brier score as in the vignette https://www.randomforestsrc.org/articles/rfsrc-subsample.html
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
    if (feature >= data->getNumberOfFeatures()) {
        throw runtime_error("Feature index is out of range");
    }
    bool use_brier = error_type == "brier";
    if (!use_brier && error_type != "misc") {
        throw runtime_error("Unknown choice of loss function. Choose either 'brier' or 'misc' (misclassification)");
    }
    prepareVIMPCache();

    size_t num_classes = data->getNumClasses();
    size_t error_size = num_classes + 1;
    const vector<double>& y = data->get_y();
    vector<double> result(num_classes + 1, 0);  // one VIMP value for each class plus one aggregated value
    vector<double> tree_vimp(ntrees * error_size, 0);
    vector<unsigned char> tree_has_oob(ntrees, 0);
    const vector<double>& baseline_errors = use_brier ? vimp_tree_errors_brier : vimp_tree_errors_misc;

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < ntrees; ++i) {
        ClassificationTree* tree = static_cast<ClassificationTree*>(trees[i].get());
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
        vector<double> tree_oob_error_random(error_size, 0);
        const vector<double>& tree_classes = tree->getClasses();
        const vector<vector<double>>& tree_class_proportions = tree->getClassProportions();

        for (size_t obs_id : tree_oob_indices) {
            size_t response_class = encodedResponseToClassIndex(y[obs_id], num_classes);
            // fetch predictions with random daughter assignment
            size_t leaf_id = tree->predictionLeafIDVIMP(obs_id, feature, local_rng);

            if (use_brier) {                                    // Brier score
                const vector<double>& tree_pred_probs = tree_class_proportions[leaf_id];
                for (size_t c = 0; c < num_classes; ++c) {
                    double residual = tree_pred_probs[c] - static_cast<double>(c == response_class);
                    double error = residual * residual;
                    tree_oob_error_random[c] += error;
                    tree_oob_error_random[num_classes] += error;
                }
            } else if (encodedResponseToClassIndex(tree_classes[leaf_id], num_classes) != response_class) {
                ++tree_oob_error_random[response_class];
                ++tree_oob_error_random[num_classes];
            }
        }

        // add VIMP contribution from the tree for each class and in total
        size_t index = i * error_size;
        for (size_t c = 0; c < error_size; ++c) {
            tree_vimp[index + c] = (tree_oob_error_random[c] - baseline_errors[index + c]) / (double) num_oob_obs;
        }
    }

    size_t valid_trees = 0;
    for (size_t i = 0; i < ntrees; ++i) {
        if (tree_has_oob[i] == 0) {
            continue;
        }
        ++valid_trees;
        size_t index = i * error_size;
        for (size_t c = 0; c < error_size; ++c) {
            result[c] += tree_vimp[index + c];
        }
    }
    if (valid_trees == 0) {
        throw runtime_error("Cannot compute VIMP without OOB observations");
    }
    // average VIMP over all trees and return final forest VIMP
    for (size_t c = 0; c < error_size; ++c) {
        result[c] /= (double) valid_trees;
    }
    if (use_brier) {    // use the adjusted Brier score as in the vignette https://www.randomforestsrc.org/articles/rfsrc-subsample.html
        result[num_classes] *= (double) num_classes / (double) (num_classes - 1);
    }
    return result;
}

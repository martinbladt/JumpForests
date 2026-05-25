/*

Functions for classification trees

*/

#include "TreeClassification.h"

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

// constructor for ClassificationTree
//--------------------------------------------------------------------------------------

ClassificationTree::ClassificationTree(const vector<size_t>& subset_indices, const vector<size_t>& estimation_indices) {
  this->node_obs.push_back(subset_indices);
  this->holdout_node_obs.push_back(estimation_indices);
  this->node_sizes.push_back(subset_indices.size());
}

// functions for growing classification trees
//--------------------------------------------------------------------------------------

// given the node observations, computes the number of each class
vector<double> ClassificationTree::computeClassCounts(const vector<size_t>& node_obs) {
  size_t num_classes = data->getNumClasses();
  vector<double> result(num_classes, 0);
  for (size_t i = 1; i < num_classes + 1; ++i) {
    for (size_t j : node_obs) {
      if (data->get_y(j) == i) {
        ++result[i - 1];
      }
    }
  }
  return result;
}

// given the class counts, returns a pair consisting of the class with the highest count (the predicted class)
// and a vector with the computed class proportions
pair<double, vector<double>> ClassificationTree::computePredictedClass(const vector<double>& class_counts, double node_size) {
  double predicted_class = mostFrequentClass(class_counts);
  
  // now compute the class proportions
  size_t num_classes = data->getNumClasses();
  vector<double> proportions(num_classes);
  for (size_t i = 0; i < num_classes; ++i) {
    proportions[i] = class_counts[i] / node_size;
  }

  return {predicted_class, proportions};
}

void ClassificationTree::makeLeaf(size_t node_index) {
  // update tree info
  feature_IDs.push_back(0);
  thresholds.push_back({});

  // since we are in a terminal node, we save the indices for the observations
  for (size_t i : node_obs[node_index]) {
    prediction_node_IDs[i] = node_index;
  }
  if (honest) {
    for (size_t i : holdout_node_obs[node_index]) {
      prediction_node_IDs[i] = node_index;
    }
  }
}

void ClassificationTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold, 
                                             vector<double>& best_class_counts_left, vector<double>& best_class_counts_right) {
  const vector<size_t>& current_node_obs = node_obs[node_index];
  size_t num_obs_parent = current_node_obs.size();
  size_t num_classes = data->getNumClasses();

  // samples split points
  vector<double> split_points;
  size_t nsplits_final = sampleSplitPoints(split_points, current_node_obs, feature);

  // no possible splits
  if (nsplits_final == 0) {
    return;
  }

  // do initial sweep to compute the number of observations and class counts in right node
  num_obs_right.assign(nsplits_final, 0);
  class_counts_right.assign(nsplits_final * num_classes, 0);
  for (size_t i : current_node_obs) {
    double feature_val = data->get_x(i, feature);
    double response_val = data->get_y(i);
    for (size_t s = 0; s < nsplits_final; ++s) {
      if (feature_val > split_points[s]) {
        // add one to the number of observations in right node for split s
        ++num_obs_right[s];
        // update the class counts in right node for split s
        for (size_t c = 0; c < num_classes; ++c) {
          if (response_val == c + 1) {
            ++class_counts_right[s * num_classes + c];
          }
        }
      } else {
        break;
      }
    }
  }
  // now find the best split
  for (size_t s = 0; s < nsplits_final; ++s) {
    size_t index = s * num_classes;
    // if one of the daughter nodes are too small, skip the computation for that split
    size_t num_obs_left = num_obs_parent - num_obs_right[s];
    if (num_obs_right[s] < min_node_size || num_obs_left < min_node_size) {
      continue;
    }

    // compute class counts in the left node residually
    vector<double> class_counts_left(num_classes, 0);
    const vector<double>& class_counts_parent = class_counts_node[node_index];
    for (size_t c = 0; c < num_classes; ++c) {
      class_counts_left[c] = class_counts_parent[c] - class_counts_right[index + c];
    }

    const vector<double>& class_prop_left = classCountsToProportions(class_counts_left, num_classes, num_obs_left);
    const vector<double>& class_prop_right = classCountsToProportions(class_counts_right, num_classes, num_obs_right[s], s);

    // since the proportions are computed before calling the splitting rule, we don't need the additional split_index option,
    // but I keep it for now in case I am going to change it
    double split_val;
    if (splitrule == "gini") {
      split_val = Gini(class_prop_left, class_prop_right, num_obs_left, num_obs_right[s]);
    }
    if (splitrule == "entropy") {
      const vector<double>& class_prop_parent = classCountsToProportions(class_counts_parent, num_classes, num_obs_parent);
      split_val = Entropy(class_prop_left, class_prop_right, num_obs_left, num_obs_right[s], class_prop_parent);
    }
    if (splitrule == "misc") {
      split_val = Misclassification(class_prop_left, class_prop_right, num_obs_left, num_obs_right[s]);
    }
    if (splitrule == "twoing") {
      split_val = Twoing(class_prop_left, class_prop_right, num_obs_left, num_obs_right[s]);
    }
    if (splitrule == "hellinger") {
      split_val = Hellinger(class_prop_left, class_prop_right);
    }

    if (split_val > best_split_val) {
      best_split_val = split_val;
      best_feature = feature;
      best_class_counts_left = class_counts_left;
      for (size_t c = 0; c < num_classes; ++c) {
        best_class_counts_right[c] = class_counts_right[index + c];
      }
      best_threshold = {split_points[s]};
    }
  }
}

void ClassificationTree::bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold, 
                                              vector<size_t>& best_left_indices, vector<size_t>& best_right_indices, vector<double>& best_class_counts_left, vector<double>& best_class_counts_right) {
  const vector<double>& feature_values = uniqueValues(data->getValues(node_obs[node_index], feature));
  const vector<double>& class_counts_parent = class_counts_node[node_index];
  size_t num_classes = data->getNumClasses();
  size_t num_feature_values = feature_values.size();

  unordered_set<uint64_t> partition_masks;
  // generate partitions (breaks if no possible splits)
  if (generateCategoricalPartitions(feature_values, partition_masks)) {
      return;
  }

  // consider each partition (bitmask)
  for (const auto& mask : partition_masks) {
    unordered_set<double> left_values;
    for (size_t i = 0; i < num_feature_values; ++i) {
        if ((mask >> i) & 1) {
            left_values.insert(feature_values[i]);
        }
    }

    vector<size_t> current_left_indices;
    vector<size_t> current_right_indices;
    for (size_t obs_id : node_obs[node_index]) {
        if (left_values.count(data->get_x(obs_id, feature))) {
            current_left_indices.push_back(obs_id);
        } else {
            current_right_indices.push_back(obs_id);
        }
    }

    size_t n_left = current_left_indices.size();
    size_t n_right = current_right_indices.size();

    if (n_left < min_node_size || n_right < min_node_size) {
        continue;
    }

    // here we have to compute the class counts in one of the daughters from scratch
    vector<double> class_counts_left = computeClassCounts(current_left_indices);
    // compute the class counts in the other node residually
    vector<double> class_counts_right(num_classes, 0);
    for (size_t c = 0; c < num_classes; ++c) {
      class_counts_right[c] = class_counts_parent[c] - class_counts_left[c];
    }

    size_t num_obs_left = current_left_indices.size();
    size_t num_obs_right = current_right_indices.size();
    const vector<double>& class_prop_left = classCountsToProportions(class_counts_left, num_classes, num_obs_left);
    const vector<double>& class_prop_right = classCountsToProportions(class_counts_right, num_classes, num_obs_right);

    // since the proportions are computed before calling the splitting rule, we don't need the additional split_index option,
    // but I keep it for now in case I am going to change it
    double split_val;
    if (splitrule == "gini") {
      split_val = Gini(class_prop_left, class_prop_right, num_obs_left, num_obs_right);
    }
    if (splitrule == "entropy") {
      const vector<double>& class_prop_parent = classCountsToProportions(class_counts_parent, num_classes, num_obs_left + num_obs_right);
      split_val = Entropy(class_prop_left, class_prop_right, num_obs_left, num_obs_right, class_prop_parent);
    }
    if (splitrule == "misc") {
      split_val = Misclassification(class_prop_left, class_prop_right, num_obs_left, num_obs_right);
    }
    if (splitrule == "twoing") {
      split_val = Twoing(class_prop_left, class_prop_right, num_obs_left, num_obs_right);
    }
    if (splitrule == "hellinger") {
      split_val = Hellinger(class_prop_left, class_prop_right);
    }

    if (split_val > best_split_val) {
      best_split_val = split_val;
      best_left_indices = std::move(current_left_indices);
      best_right_indices = std::move(current_right_indices);
      best_feature = feature;
      best_threshold.assign(left_values.begin(), left_values.end());
      best_class_counts_left = std::move(class_counts_left);
      best_class_counts_right = std::move(class_counts_right);
    }
  }
}

// function to create a split for a classification tree. returns true if leaf, otherwise false
bool ClassificationTree::createSplit(size_t node_index) {
    const vector<size_t>& current_node_obs = node_obs[node_index];
    // if we are in the root node, the class counts need to be computed
    if (class_counts_node.empty()) {
      class_counts_node.push_back(computeClassCounts(current_node_obs));
    }

    // if no split is possible, make the node a leaf
    if (current_node_obs.size() < 2 * min_node_size) {
      if (!honest) {
        // for dishonest trees, use the earlier computed class counts
        pair<double, vector<double>> predictions = computePredictedClass(class_counts_node[node_index], node_sizes[node_index]);
        classes.push_back(predictions.first);
        class_proportions.push_back(predictions.second);
      } else {
        // for honest trees, we have to compute the class counts for the holdout observations
        const vector<size_t>& holdout_obs = holdout_node_obs[node_index];
        if (holdout_obs.empty()) {
          // if we have no holdout observations, use the predicted class and class probabilities from the parent node
          pair<double, vector<double>> predictions = computePredictedClass(class_counts_node[node_index], node_sizes[node_index]);
          classes.push_back(predictions.first);
          class_proportions.push_back(predictions.second);
        } else {
          // for honest trees, use the holdout set for computing class counts and proportions
          vector<double> class_counts_holdout = computeClassCounts(holdout_obs);
          pair<double, vector<double>> predictions = computePredictedClass(class_counts_holdout, holdout_obs.size());
          classes.push_back(predictions.first);
          class_proportions.push_back(predictions.second);
        }
      }
      makeLeaf(node_index);
      return true;
    }

    double best_split_val = -1.0;
    size_t best_feature = 0;
    vector<double> best_threshold;
    vector<size_t> best_left_indices;
    vector<size_t> best_right_indices;
    size_t num_classes = data->getNumClasses();
    vector<double> best_class_counts_left(num_classes, 0);
    vector<double> best_class_counts_right(num_classes, 0);

    // only used for honesty
    vector<size_t> holdout_left_indices;
    vector<size_t> holdout_right_indices;

    // sample mtry features
    size_t num_features = data->getNumberOfFeatures();
    vector<size_t> feature_indices(num_features);
    for (size_t i = 0; i < num_features; ++i) {
        feature_indices[i] = i;
    }
    vector<size_t> sampled_features = sampleIndices(feature_indices, mtry, false, random_number_generator);

    // now consider each of the sampled features
    for (size_t i : sampled_features) {
        if (data->getCategorical()[i]) {
            // finds the best split and constructs the indices of the best left and right node
            bestSplitCategorical(node_index, i, best_split_val, best_feature, best_threshold,
                                 best_left_indices, best_right_indices,
                                 best_class_counts_left, best_class_counts_right);
        }
        else {
            // does not return the best indices, so this has to be done later
            bestSplitContinuous(node_index, i, best_split_val, best_feature, best_threshold, best_class_counts_left, best_class_counts_right);
        }
    }

    // if no best split is found, make the node a leaf
    if (best_split_val < 0) {
      if (!honest) {
        // for dishonest trees, use the earlier computed class counts
        pair<double, vector<double>> predictions = computePredictedClass(class_counts_node[node_index], node_sizes[node_index]);
        classes.push_back(predictions.first);
        class_proportions.push_back(predictions.second);
      } else {
        // for honest trees, we have to compute the class counts for the holdout observations
        const vector<size_t>& holdout_obs = holdout_node_obs[node_index];
        if (holdout_obs.empty()) {
          // if we have no holdout observations, use the predicted class and class probabilities from the parent node
          pair<double, vector<double>> predictions = computePredictedClass(class_counts_node[node_index], node_sizes[node_index]);
          classes.push_back(predictions.first);
          class_proportions.push_back(predictions.second);
        } else {
          // for honest trees, use the holdout set for computing class counts and proportions
          vector<double> class_counts_holdout = computeClassCounts(holdout_obs);
          pair<double, vector<double>> predictions = computePredictedClass(class_counts_holdout, holdout_obs.size());
          classes.push_back(predictions.first);
          class_proportions.push_back(predictions.second);
        }
      }
      makeLeaf(node_index);
      return true;
    }

    // for a categorical feature, the best indices are already saved, but if the feature is 
    // continuous, they should be recomputed from scratch (and only once)
    if (!(data->getCategorical()[best_feature])) {
        best_left_indices.clear();
        best_right_indices.clear();
        for (size_t i : current_node_obs) {
            if (data->get_x(i, best_feature) <= best_threshold[0]) {
                best_left_indices.push_back(i);
            } else {
                best_right_indices.push_back(i);
            }
        }
        // update the holdout index sets if the tree is honest
        if (honest) {
            const vector<size_t>& current_holdout_node_obs = holdout_node_obs[node_index];
            for (size_t i : current_holdout_node_obs) {
                if (data->get_x(i, best_feature) <= best_threshold[0]) {
                    holdout_left_indices.push_back(i);
                } else {
                    holdout_right_indices.push_back(i);
                }
            }
        }
    }

    // the best holdout index sets also need to be constructed if the split is categorical
    if (honest && data->getCategorical()[best_feature]) {
        const vector<size_t>& current_holdout_node_obs = holdout_node_obs[node_index];
        for (size_t i : current_holdout_node_obs) {
            if (find(best_threshold.begin(), best_threshold.end(), data->get_x(i, best_feature)) != best_threshold.end()) {
                holdout_left_indices.push_back(i);
            } else {
                holdout_right_indices.push_back(i);
            }
        }
    }

    // a best split was found, update the tree
    node_obs.push_back(best_left_indices);                // construct left daughter
    node_obs.push_back(best_right_indices);               // construct right daughter
    class_counts_node.push_back(best_class_counts_left);  // save class counts
    class_counts_node.push_back(best_class_counts_right);
    node_sizes.push_back(best_left_indices.size());
    node_sizes.push_back(best_right_indices.size());
    feature_IDs.push_back(best_feature);
    thresholds.push_back(best_threshold);
    classes.push_back(0);
    class_proportions.push_back({});

    // for honest trees, update the holdout indices
    if (honest) {
        holdout_node_obs.push_back(holdout_left_indices);
        holdout_node_obs.push_back(holdout_right_indices);
    }
    return false;
}

// splitting rules for classification trees
//--------------------------------------------------------------------------------------

// split_id = 0 for categorical splits

double ClassificationTree::Gini(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, size_t split_id) {
  size_t num_classes = data->getNumClasses();
  double sum_left = 0;
  double sum_right = 0;
  size_t index = split_id * num_classes;

  for (size_t c = 0; c < num_classes; ++c) {
    sum_left += class_prop_left[c] * class_prop_left[c];
    sum_right += class_prop_right[index + c] * class_prop_right[index + c];
  }
  return num_obs_left * sum_left + num_obs_right * sum_right;
}

// there may be an optimisation of entropy where the split value is always positive without including the parent term
double ClassificationTree::Entropy(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, const vector<double>& class_prop_parent, size_t split_id) {
  size_t num_classes = data->getNumClasses();
  double sum_left = 0;
  double sum_right = 0;
  double sum_parent = 0;
  size_t index = split_id * num_classes;

  for (size_t c = 0; c < num_classes; ++c) {
    sum_left += class_prop_left[c] * log2(class_prop_left[c]);
    sum_right += class_prop_right[index + c] * log2(class_prop_right[index + c]);
    sum_parent += class_prop_parent[c] * log2(class_prop_parent[c]);
  }
  return sum_parent - (num_obs_left * sum_left + num_obs_right * sum_right) / (num_obs_left + num_obs_right);
}

double ClassificationTree::Misclassification(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, size_t split_id) {
  size_t num_classes = data->getNumClasses();
  size_t index = split_id * num_classes;
  double max_prop_left = *max_element(class_prop_left.begin(), class_prop_left.end());
  double max_prop_right = *max_element(class_prop_right.begin() + index, class_prop_right.begin() + index + num_classes); // make sure that this makes sense

  return num_obs_left * max_prop_left + num_obs_right * max_prop_right;
}

double ClassificationTree::Twoing(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, size_t split_id) {
  size_t num_classes = data->getNumClasses();
  size_t index = split_id * num_classes;
  double sum = 0;

  for (size_t c = 0; c < num_classes; ++c) {
    sum += abs(class_prop_left[c] - class_prop_right[index + c]);
  }
  return num_obs_left * num_obs_right * sum * sum;
}

double ClassificationTree::Hellinger(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t split_id) {
  size_t num_classes = data->getNumClasses();
  size_t index = split_id * num_classes;
  double sum = 0;

  for (size_t c = 0; c < num_classes; ++c) {
    double term = (sqrt(class_prop_left[c]) - sqrt(class_prop_right[index + c]));
    sum += term * term;
  }

  return sum;
}

// prediction for classification trees
//--------------------------------------------------------------------------------------

// for computing predictions on the training data
pair<vector<double>, vector<double>> ClassificationTree::computePredictions() {
  size_t num_obs = data->getNumberOfObs();
  size_t num_classes = data->getNumClasses();
  vector<double> predictions_class(num_obs);
  vector<double> predictions_prob(num_obs * num_classes);

  // add multithreading here?
  for (size_t i = 0; i < num_obs; ++i) {
    size_t leaf_id = predictionLeafID(data->get_x_row(i));
    predictions_class[i] = classes[leaf_id];
    size_t index = i * num_classes;
    for (size_t c = 0; c < num_classes; ++c) {
      predictions_prob[index + c] = class_proportions[leaf_id][c];
    }
  }
  return {predictions_class, predictions_prob};
}

// for computing predictions on new data
pair<vector<double>, vector<double>> ClassificationTree::computePredictions(const Data& new_data) {
  size_t num_obs = new_data.getNumberOfObs();
  size_t num_classes = data->getNumClasses();
  vector<double> predictions_class(num_obs);
  vector<double> predictions_prob(num_obs * num_classes);

  // add multithreading here?
  for (size_t i = 0; i < num_obs; ++i) {
    size_t leaf_id = predictionLeafID(new_data.get_x_row(i));
    predictions_class[i] = classes[leaf_id];
    size_t index = i * num_classes;
    for (size_t c = 0; c < num_classes; ++c) {
      predictions_prob[index + c] = class_proportions[leaf_id][c];
    }
  }
  return {predictions_class, predictions_prob};
}

// error estimation for classification trees
//--------------------------------------------------------------------------------------

// computes the misclassification error for each class and in total based on a vector of predictions and a test vector response
vector<double> computeMisclassificationError(const vector<double>& class_predictions, const vector<double>& response, size_t num_classes) {
  if (class_predictions.size() != response.size()) {
    throw runtime_error("Number of classification predictions does not match number of responses");
  }
  if (response.empty()) {
    throw runtime_error("Cannot compute classification error on an empty response vector");
  }

  vector<double> result(num_classes + 1, 0);    // last entry is the aggregate misclassification error
  vector<double> class_counts(num_classes, 0);

  for (size_t i = 0; i < response.size(); ++i) {
    size_t response_class = encodedResponseToClassIndex(response[i], num_classes);
    size_t predicted_class = encodedResponseToClassIndex(class_predictions[i], num_classes);
    ++class_counts[response_class];
    if (response_class != predicted_class) {
      ++result[response_class];
      ++result[num_classes];
    }
  }
  // normalise the results
  for (size_t c = 0; c < num_classes; ++c) {
    if (class_counts[c] > 0) {
      result[c] /= class_counts[c];
    } else {
      result[c] = NA_REAL;
    }
  }
  result[num_classes] /= response.size();
  return result;
}

// computes the Brier score error for each class and in total based on a flattened vector of predictions of the class probabilities
// and a vector of predicted classes response
double computeBrierScoreError(const vector<double>& prob_predictions, const vector<double>& response, size_t num_classes) {
  if (num_classes < 2) {
    throw runtime_error("Brier score requires at least two classes");
  }
  if (response.empty()) {
    throw runtime_error("Cannot compute Brier score on an empty response vector");
  }
  if (prob_predictions.size() != response.size() * num_classes) {
    throw runtime_error("Number of class-probability predictions does not match response/classes");
  }

  double result = 0;
  
  for (size_t i = 0; i < response.size(); ++i) {
    size_t index = i * num_classes;
    size_t response_class = encodedResponseToClassIndex(response[i], num_classes);
    for (size_t c = 0; c < num_classes; ++c) {
      if (c == response_class) {
        result += (1 - prob_predictions[index + c]) * (1 - prob_predictions[index + c]);
      } else {
        result += prob_predictions[index + c] * prob_predictions[index + c];
      }
    }
  }

  return result / ((double) response.size() * (double) num_classes);
}

double computeNormalizedBrierScoreError(const vector<double>& prob_predictions, const vector<double>& response, size_t num_classes) {
  return computeBrierScoreError(prob_predictions, response, num_classes) *
         (double) num_classes * (double) num_classes / (double) (num_classes - 1);
}

// computes the confusion matrix based as a flattened vector of length num_classes^2
vector<size_t> computeConfusionMatrix(const vector<double>& class_predictions, const vector<double>& response, size_t num_classes) {
  if (class_predictions.size() != response.size()) {
    throw runtime_error("Number of classification predictions does not match number of responses");
  }

  vector<size_t> result(num_classes * num_classes, 0);
  for (size_t i = 0; i < response.size(); ++i) {
    size_t row = encodedResponseToClassIndex(response[i], num_classes);
    size_t col = encodedResponseToClassIndex(class_predictions[i], num_classes);
    ++result[row * num_classes + col];
  }
  return result;
}

// miscellaneous functions related to classification trees
//--------------------------------------------------------------------------------------

vector<double> classCountsToProportions(const vector<double>& class_counts, size_t num_classes, size_t num_obs, size_t split_id) {
  vector<double> result(num_classes, 0);
  size_t index = split_id * num_classes;
  for (size_t c = 0; c < num_classes; ++c) {
    result[c] = class_counts[index + c] / (double) num_obs;
  }
  return result;
}

// determines the most frequent class by majority rule given a vector of class counts
double mostFrequentClass(const vector<double>& class_counts) {
  double highest_count = 0;
  double result = 0;
  for (size_t i = 0; i < class_counts.size(); ++i) {
    if (class_counts[i] > highest_count) {
      highest_count = class_counts[i];
      result = i + 1;
    }
  }
  return result;
}

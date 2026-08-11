/*

Functions for classification trees

*/

#include "TreeClassification.h"

#include <cmath>
#include <limits>

namespace {
size_t encodedResponseToClassIndex(double response, size_t num_classes) {
  double rounded_response = round(response);
  if (!isfinite(response) || abs(response - rounded_response) > 1e-8 ||
      rounded_response < 1 || rounded_response > static_cast<double>(num_classes)) {
    throw runtime_error("Classification response values must be encoded as 1, ..., num_classes");
  }
  return static_cast<size_t>(rounded_response) - 1;
}

double proportionLog2(double proportion) {
  return proportion > 0 ? proportion * log2(proportion) : 0;
}
}

// constructor for ClassificationTree
//--------------------------------------------------------------------------------------

ClassificationTree::ClassificationTree(vector<size_t> subset_indices, vector<size_t> estimation_indices) {
  this->node_sizes.push_back(subset_indices.size());
  this->node_obs.push_back(std::move(subset_indices));
  this->holdout_node_obs.push_back(std::move(estimation_indices));
}

// functions for growing classification trees
//--------------------------------------------------------------------------------------

// given the node observations, computes the number of each class
vector<double> ClassificationTree::computeClassCounts(const vector<size_t>& node_obs) {
  size_t num_classes = data->getNumClasses();
  vector<double> result(num_classes, 0);
  for (size_t i : node_obs) {
    size_t class_id = static_cast<size_t>(data->get_y(i)) - 1;
    ++result[class_id];
  }
  return result;
}

void ClassificationTree::reserveTreeMemory(size_t num_obs) {
  // the minimal node size gives an upper bound for the number of nodes in the tree
  size_t max_terminal_nodes = min_node_size == 0 ? max(static_cast<size_t>(1), num_obs) :
    max(static_cast<size_t>(1), num_obs / min_node_size);
  size_t max_num_nodes = 2 * max_terminal_nodes - 1;
  // avoid excessive reservation for unusually large shallow trees
  max_num_nodes = min(max_num_nodes, static_cast<size_t>(1024));

  node_obs.reserve(max_num_nodes);
  if (honest) {
    holdout_node_obs.reserve(max_num_nodes);
  }
  node_sizes.reserve(max_num_nodes);
  left_daughters.reserve(max_num_nodes);
  feature_IDs.reserve(max_num_nodes);
  thresholds.reserve(max_num_nodes);
  depths.reserve(max_num_nodes);
  classes.reserve(max_num_nodes);
  class_proportions.reserve(max_num_nodes);
  class_counts_node.reserve(max_num_nodes);
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

double ClassificationTree::computeSplitValue(const vector<double>& class_prop_left, const vector<double>& class_prop_right,
                                             size_t num_obs_left, size_t num_obs_right) {
  switch (splitrule_id) {
    case ClassificationSplitRule::Gini:
      return Gini(class_prop_left, class_prop_right, num_obs_left, num_obs_right);
    case ClassificationSplitRule::Entropy:
      return Entropy(class_prop_left, class_prop_right, num_obs_left, num_obs_right);
    case ClassificationSplitRule::Misclassification:
      return Misclassification(class_prop_left, class_prop_right, num_obs_left, num_obs_right);
    case ClassificationSplitRule::Twoing:
      return Twoing(class_prop_left, class_prop_right, num_obs_left, num_obs_right);
    case ClassificationSplitRule::Hellinger:
      return Hellinger(class_prop_left, class_prop_right);
  }
  return 0;
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

  // the observation indices are no longer needed after the node is made terminal
  vector<size_t>().swap(node_obs[node_index]);
  if (honest) {
    vector<size_t>().swap(holdout_node_obs[node_index]);
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

  // first place each observation at the last split point where it belongs to the right node
  num_obs_right.assign(nsplits_final, 0);
  class_counts_right.assign(nsplits_final * num_classes, 0);
  for (size_t i : current_node_obs) {
    double feature_val = data->get_x(i, feature);
    // missing feature values are sent to the right daughter during prediction
    size_t num_splits_right = std::isnan(feature_val) ? nsplits_final :
      lower_bound(split_points.begin(), split_points.end(), feature_val) - split_points.begin();
    if (num_splits_right > 0) {
      size_t split_id = num_splits_right - 1;
      size_t class_id = static_cast<size_t>(data->get_y(i)) - 1;
      ++num_obs_right[split_id];
      ++class_counts_right[split_id * num_classes + class_id];
    }
  }

  // accumulate backwards since an observation to the right of one split point
  // also belongs to the right of every smaller split point
  for (size_t s = nsplits_final - 1; s > 0; --s) {
    num_obs_right[s - 1] += num_obs_right[s];
    size_t index = s * num_classes;
    size_t previous_index = index - num_classes;
    for (size_t c = 0; c < num_classes; ++c) {
      class_counts_right[previous_index + c] += class_counts_right[index + c];
    }
  }

  vector<double> class_counts_left(num_classes, 0);
  vector<double> class_prop_left(num_classes, 0);
  vector<double> class_prop_right(num_classes, 0);
  const vector<double>& class_counts_parent = class_counts_node[node_index];

  // now find the best split
  for (size_t s = 0; s < nsplits_final; ++s) {
    size_t index = s * num_classes;
    // if one of the daughter nodes are too small, skip the computation for that split
    size_t num_obs_left = num_obs_parent - num_obs_right[s];
    if (num_obs_right[s] < min_node_size || num_obs_left < min_node_size) {
      continue;
    }

    // compute class counts in the left node residually
    for (size_t c = 0; c < num_classes; ++c) {
      class_counts_left[c] = class_counts_parent[c] - class_counts_right[index + c];
      class_prop_left[c] = class_counts_left[c] / (double) num_obs_left;
      class_prop_right[c] = class_counts_right[index + c] / (double) num_obs_right[s];
    }

    double split_val = computeSplitValue(class_prop_left, class_prop_right, num_obs_left, num_obs_right[s]);

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
                                              vector<double>& best_class_counts_left, vector<double>& best_class_counts_right) {
  const vector<size_t>& current_node_obs = node_obs[node_index];
  vector<double> feature_values = data->getValues(current_node_obs, feature);
  bool has_missing_values = any_of(
    feature_values.begin(), feature_values.end(),
    [](double value) { return std::isnan(value); });
  feature_values.erase(remove_if(feature_values.begin(), feature_values.end(),
    [](double value) { return std::isnan(value); }), feature_values.end());
  feature_values = uniqueValues(std::move(feature_values));
  const vector<double>& class_counts_parent = class_counts_node[node_index];
  size_t num_classes = data->getNumClasses();
  size_t num_feature_values = feature_values.size();

  unordered_set<uint64_t> partition_masks;
  // generate partitions (breaks if no possible splits)
  if (generateCategoricalPartitions(
        feature_values, partition_masks, has_missing_values)) {
      return;
  }

  // compute the number of observations and class counts for each category once
  vector<size_t> category_counts(num_feature_values, 0);
  vector<double> category_class_counts(num_feature_values * num_classes, 0);
  for (size_t i = 0; i < current_node_obs.size(); ++i) {
    size_t obs_id = current_node_obs[i];
    double feature_value = data->get_x(obs_id, feature);
    // missing feature values are sent to the right daughter during prediction
    size_t category = std::isnan(feature_value) ? num_feature_values :
      lower_bound(feature_values.begin(), feature_values.end(), feature_value) - feature_values.begin();
    if (category < num_feature_values) {
      size_t class_id = static_cast<size_t>(data->get_y(obs_id)) - 1;
      ++category_counts[category];
      ++category_class_counts[category * num_classes + class_id];
    }
  }

  vector<double> class_counts_left(num_classes, 0);
  vector<double> class_counts_right(num_classes, 0);
  vector<double> class_prop_left(num_classes, 0);
  vector<double> class_prop_right(num_classes, 0);

  // consider each partition (bitmask)
  for (const auto& mask : partition_masks) {
    size_t n_left = 0;
    fill(class_counts_left.begin(), class_counts_left.end(), 0);
    for (size_t i = 0; i < num_feature_values; ++i) {
        if (i < 63 && ((mask >> i) & 1)) {
            n_left += category_counts[i];
            size_t index = i * num_classes;
            for (size_t c = 0; c < num_classes; ++c) {
              class_counts_left[c] += category_class_counts[index + c];
            }
        }
    }

    size_t n_right = current_node_obs.size() - n_left;

    if (n_left < min_node_size || n_right < min_node_size) {
        continue;
    }

    // compute the class counts in the right node residually
    for (size_t c = 0; c < num_classes; ++c) {
      class_counts_right[c] = class_counts_parent[c] - class_counts_left[c];
      class_prop_left[c] = class_counts_left[c] / (double) n_left;
      class_prop_right[c] = class_counts_right[c] / (double) n_right;
    }

    double split_val = computeSplitValue(class_prop_left, class_prop_right, n_left, n_right);

    if (split_val > best_split_val) {
      best_split_val = split_val;
      best_feature = feature;
      unordered_set<double> left_values;
      for (size_t i = 0; i < num_feature_values; ++i) {
        if (i < 63 && ((mask >> i) & 1)) {
          left_values.insert(feature_values[i]);
        }
      }
      best_threshold.assign(left_values.begin(), left_values.end());
      best_class_counts_left = class_counts_left;
      best_class_counts_right = class_counts_right;
    }
  }
}

// function to create a split for a classification tree. returns true if leaf, otherwise false
bool ClassificationTree::createSplit(size_t node_index) {
    // initialise quantities which are reused throughout the tree
    if (class_counts_node.empty()) {
      reserveTreeMemory(node_obs[node_index].size());
    }
    const vector<size_t>& current_node_obs = node_obs[node_index];
    // if we are in the root node, the class counts need to be computed
    if (class_counts_node.empty()) {
      class_counts_node.push_back(computeClassCounts(current_node_obs));

      size_t num_features = data->getNumberOfFeatures();
      feature_indices.resize(num_features);
      for (size_t i = 0; i < num_features; ++i) {
        feature_indices[i] = i;
      }

      if (splitrule == "gini") {
        splitrule_id = ClassificationSplitRule::Gini;
      } else if (splitrule == "entropy") {
        splitrule_id = ClassificationSplitRule::Entropy;
      } else if (splitrule == "misc") {
        splitrule_id = ClassificationSplitRule::Misclassification;
      } else if (splitrule == "twoing") {
        splitrule_id = ClassificationSplitRule::Twoing;
      } else {
        splitrule_id = ClassificationSplitRule::Hellinger;
      }
    }

    // if no split is possible, make the node a leaf
    if (current_node_obs.size() < 2 * min_node_size || depths[node_index] >= max_depth) {
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

    double best_split_val = -std::numeric_limits<double>::infinity();
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
    vector<size_t> sampled_features = sampleIndices(feature_indices, mtry, false, random_number_generator);
    const vector<bool>& categorical = data->getCategorical();

    // now consider each of the sampled features
    for (size_t i : sampled_features) {
        if (categorical[i]) {
            bestSplitCategorical(node_index, i, best_split_val, best_feature, best_threshold,
                                 best_class_counts_left, best_class_counts_right);
        }
        else {
            // does not return the best indices, so this has to be done later
            bestSplitContinuous(node_index, i, best_split_val, best_feature, best_threshold, best_class_counts_left, best_class_counts_right);
        }
    }

    // if no best split is found, make the node a leaf
    if (!std::isfinite(best_split_val)) {
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

    // construct the indices of the two daughters once for the best split
    bool categorical_split = categorical[best_feature];
    best_left_indices.reserve(current_node_obs.size());
    best_right_indices.reserve(current_node_obs.size());
    for (size_t i : current_node_obs) {
        double feature_value = data->get_x(i, best_feature);
        bool goes_left = categorical_split ?
          find(best_threshold.begin(), best_threshold.end(), feature_value) != best_threshold.end() :
          feature_value <= best_threshold[0];
        if (goes_left) {
            best_left_indices.push_back(i);
        } else {
            best_right_indices.push_back(i);
        }
    }

    // update the holdout index sets if the tree is honest
    if (honest) {
      const vector<size_t>& current_holdout_node_obs = holdout_node_obs[node_index];
      holdout_left_indices.reserve(current_holdout_node_obs.size());
      holdout_right_indices.reserve(current_holdout_node_obs.size());
      for (size_t i : current_holdout_node_obs) {
        double feature_value = data->get_x(i, best_feature);
        bool goes_left = categorical_split ?
          find(best_threshold.begin(), best_threshold.end(), feature_value) != best_threshold.end() :
          feature_value <= best_threshold[0];
        if (goes_left) {
          holdout_left_indices.push_back(i);
        } else {
          holdout_right_indices.push_back(i);
        }
      }
    }

    // a best split was found, update the tree
    size_t best_left_size = best_left_indices.size();
    size_t best_right_size = best_right_indices.size();
    node_obs.push_back(std::move(best_left_indices));                // construct left daughter
    node_obs.push_back(std::move(best_right_indices));               // construct right daughter
    class_counts_node.push_back(std::move(best_class_counts_left));  // save class counts
    class_counts_node.push_back(std::move(best_class_counts_right));
    node_sizes.push_back(best_left_size);
    node_sizes.push_back(best_right_size);
    feature_IDs.push_back(best_feature);
    thresholds.push_back(std::move(best_threshold));
    classes.push_back(0);
    class_proportions.push_back({});

    // for honest trees, update the holdout indices
    if (honest) {
        holdout_node_obs.push_back(std::move(holdout_left_indices));
        holdout_node_obs.push_back(std::move(holdout_right_indices));
    }

    // the parent indices are no longer needed after its daughters have been constructed
    vector<size_t>().swap(node_obs[node_index]);
    if (honest) {
      vector<size_t>().swap(holdout_node_obs[node_index]);
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

double ClassificationTree::Entropy(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, size_t split_id) {
  size_t num_classes = data->getNumClasses();
  double sum_left = 0;
  double sum_right = 0;
  size_t index = split_id * num_classes;

  for (size_t c = 0; c < num_classes; ++c) {
    sum_left += proportionLog2(class_prop_left[c]);
    sum_right += proportionLog2(class_prop_right[index + c]);
  }
  // Parent entropy is constant within a node, so this has the same argmax as information gain.
  return num_obs_left * sum_left + num_obs_right * sum_right;
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
    size_t leaf_id = predictionLeafID(i);
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
    size_t leaf_id = predictionLeafID(new_data, i);
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

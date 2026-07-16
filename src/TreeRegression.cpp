/*

Functions for regression trees

*/

#include "TreeRegression.h"

#include <cmath>

// constructor for RegressionTree
//--------------------------------------------------------------------------------------

RegressionTree::RegressionTree(vector<size_t> subset_indices, vector<size_t> estimation_indices) {
  this->node_sizes.push_back(subset_indices.size());
  this->node_obs.push_back(std::move(subset_indices));
  this->holdout_node_obs.push_back(std::move(estimation_indices));
}

// functions for growing regression trees
//--------------------------------------------------------------------------------------

double RegressionTree::computeSum(const vector<size_t>& indices) {
  double sum = 0;
  for (size_t i : indices) {
    sum += data->get_y(i);
  }
  return sum;
}

double RegressionTree::computeAbsoluteDeviation(const vector<size_t>& indices) {
  vector<double> response_values;
  response_values.reserve(indices.size());
  for (size_t i : indices) {
    response_values.push_back(data->get_y(i));
  }
  return computeAbsoluteDeviation(response_values);
}

double RegressionTree::computeAbsoluteDeviation(vector<double>& response_values) {
  if (response_values.empty()) {
    return 0;
  }

  size_t n = response_values.size();
  auto middle = response_values.begin() + n / 2;
  nth_element(response_values.begin(), middle, response_values.end());
  double median = *middle;
  if (n % 2 == 0) {
    // nth_element only sorts around the upper median, so find the lower median in the first half
    median = (*max_element(response_values.begin(), middle) + median) / 2.0;
  }

  double absolute_deviation = 0;
  for (double response : response_values) {
    absolute_deviation += std::abs(response - median);
  }
  return absolute_deviation;
}

double RegressionTree::computeMSESplitValue(size_t n_left, double sum_left, size_t n_right, double sum_right) {
  return sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;
}

double RegressionTree::computeMAESplitValue(const vector<size_t>& left_indices, const vector<size_t>& right_indices,
                                            double parent_absolute_deviation) {
  return parent_absolute_deviation - computeAbsoluteDeviation(left_indices) - computeAbsoluteDeviation(right_indices);
}

void RegressionTree::reserveTreeMemory(size_t num_obs) {
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
  means.reserve(max_num_nodes);
  sum_node.reserve(max_num_nodes);
}

void RegressionTree::bestSplitContinuous(size_t node_index, size_t feature, bool use_mae, double parent_sum, double parent_absolute_deviation,
                                         double& best_split_val, size_t& best_feature, vector<double>& best_threshold, double& best_sum_left) {
  const vector<size_t>& current_node_obs = node_obs[node_index];
  size_t num_obs_parent = current_node_obs.size();

  // samples split points
  vector<double> split_points;
  size_t nsplits_final = sampleSplitPoints(split_points, current_node_obs, feature);

  // no possible splits
  if (nsplits_final == 0) {
    return;
  }

  // do initial sweep to compute number of observations and sums of responses in right node
  num_obs_right.assign(nsplits_final, 0);
  sums_right.assign(nsplits_final, 0);
  for (size_t i : current_node_obs) {
    double feature_val = data->get_x(i, feature);
    double response_val = data->get_y(i);
    // missing feature values are sent to the right daughter during prediction
    size_t num_splits_right = std::isnan(feature_val) ? nsplits_final :
      lower_bound(split_points.begin(), split_points.end(), feature_val) - split_points.begin();
    for (size_t s = 0; s < num_splits_right; ++s) {
      // add one to the number of observations in right node for split s
      ++num_obs_right[s];
      sums_right[s] += response_val;
    }
  }

  vector<double> left_responses;
  vector<double> right_responses;
  if (use_mae) {
    left_responses.reserve(num_obs_parent);
    right_responses.reserve(num_obs_parent);
  }

  // now find the best split
  for (size_t s = 0; s < nsplits_final; ++s) {
    // if one of the daughter nodes are too small, skip the computation for that split
    size_t num_obs_left = num_obs_parent - num_obs_right[s];
    if (num_obs_right[s] < min_node_size || num_obs_left < min_node_size) {
      continue;
    }

    double sum_left = parent_sum - sums_right[s];
    double split_val;
    if (use_mae) {
      left_responses.clear();
      right_responses.clear();
      for (size_t i : current_node_obs) {
        if (data->get_x(i, feature) <= split_points[s]) {
          left_responses.push_back(data->get_y(i));
        } else {
          right_responses.push_back(data->get_y(i));
        }
      }
      split_val = parent_absolute_deviation - computeAbsoluteDeviation(left_responses) - computeAbsoluteDeviation(right_responses);
    } else {
      split_val = computeMSESplitValue(num_obs_left, sum_left, num_obs_right[s], sums_right[s]);
    }

    if (split_val > best_split_val) {
      best_split_val = split_val;
      best_feature = feature;
      best_sum_left = sum_left;
      best_threshold = {split_points[s]};
    }
  }
}

void RegressionTree::bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature,
                           vector<double>& best_threshold, double& best_sum_left, bool use_mae, double parent_absolute_deviation) {
  const vector<size_t>& current_node_obs = node_obs[node_index];
  vector<double> feature_values = data->getValues(current_node_obs, feature);
  feature_values.erase(remove_if(feature_values.begin(), feature_values.end(),
    [](double value) { return std::isnan(value); }), feature_values.end());
  feature_values = uniqueValues(std::move(feature_values));
  double parent_sum = sum_node[node_index];
  size_t num_feature_values = feature_values.size();

  unordered_set<uint64_t> partition_masks;
  // generate partitions (breaks if no possible splits)
  if (generateCategoricalPartitions(feature_values, partition_masks)) {
      return;
  }

  // compute the number of observations and sum of responses for each category once
  vector<size_t> observation_categories(current_node_obs.size());
  vector<size_t> category_counts(num_feature_values, 0);
  vector<double> category_sums(num_feature_values, 0);
  for (size_t i = 0; i < current_node_obs.size(); ++i) {
    size_t obs_id = current_node_obs[i];
    double feature_value = data->get_x(obs_id, feature);
    // missing feature values are sent to the right daughter during prediction
    size_t category = std::isnan(feature_value) ? num_feature_values :
      lower_bound(feature_values.begin(), feature_values.end(), feature_value) - feature_values.begin();
    observation_categories[i] = category;
    if (category < num_feature_values) {
      ++category_counts[category];
      category_sums[category] += data->get_y(obs_id);
    }
  }

  vector<size_t> current_left_indices;
  vector<size_t> current_right_indices;
  if (use_mae) {
    current_left_indices.reserve(current_node_obs.size());
    current_right_indices.reserve(current_node_obs.size());
  }

  // consider each partition (bitmask)
  for (const auto& mask : partition_masks) {
    size_t n_left = 0;
    double sum_left = 0;
    for (size_t i = 0; i < num_feature_values; ++i) {
        if ((mask >> i) & 1) {
            n_left += category_counts[i];
            sum_left += category_sums[i];
        }
    }

    size_t n_right = current_node_obs.size() - n_left;

    if (n_left < min_node_size || n_right < min_node_size) {
        continue;
    }

    double sum_right = parent_sum - sum_left;
    double split_val;
    if (use_mae) {
      current_left_indices.clear();
      current_right_indices.clear();
      for (size_t i = 0; i < current_node_obs.size(); ++i) {
        if (observation_categories[i] < num_feature_values && ((mask >> observation_categories[i]) & 1)) {
          current_left_indices.push_back(current_node_obs[i]);
        } else {
          current_right_indices.push_back(current_node_obs[i]);
        }
      }
      split_val = computeMAESplitValue(current_left_indices, current_right_indices, parent_absolute_deviation);
    } else {
      split_val = computeMSESplitValue(n_left, sum_left, n_right, sum_right);
    }

    if (split_val > best_split_val) {
        best_split_val = split_val;
        best_feature = feature;
        best_threshold.clear();
        for (size_t i = 0; i < num_feature_values; ++i) {
          if ((mask >> i) & 1) {
            best_threshold.push_back(feature_values[i]);
          }
        }
        best_sum_left = sum_left;
    }
  }
}

void RegressionTree::makeLeaf(size_t node_index) {
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

// function to create a split for a regression tree. returns true if leaf, otherwise false
bool RegressionTree::createSplit(size_t node_index) {
    // if we are in the root node, the sum of the responses needs to be computed
    if (sum_node.empty()) {
      reserveTreeMemory(node_obs[node_index].size());
    }
    const vector<size_t>& current_node_obs = node_obs[node_index];
    if (sum_node.empty()) {
      sum_node.push_back(computeSum(current_node_obs));

      size_t num_features = data->getNumberOfFeatures();
      feature_indices.resize(num_features);
      for (size_t i = 0; i < num_features; ++i) {
        feature_indices[i] = i;
      }
    }
    double parent_sum = sum_node[node_index];

    // if no split is possible, make the node a leaf
    if (current_node_obs.size() < 2 * min_node_size) {
        if (!honest) {
            means.push_back(parent_sum / (double) node_sizes[node_index]);  // for dishonest trees, we may simply reuse the computed sum
        } else {
            const vector<size_t>& holdout_obs = holdout_node_obs[node_index];
            if (holdout_obs.empty()) {
                means.push_back(parent_sum / (double) node_sizes[node_index]);
            } else {
                means.push_back(computeSum(holdout_obs) / (double) holdout_obs.size());  // for honest trees, compute the mean from scratch for the holdout indices
            }
        }
        makeLeaf(node_index);
        return true;
    }

    // sample mtry features
    vector<size_t> sampled_features = sampleIndices(feature_indices, mtry, false, random_number_generator);
    const vector<bool>& categorical = data->getCategorical();

    // compute the parent sum once for all sampled continuous features
    double continuous_parent_sum = parent_sum;
    for (size_t feature : sampled_features) {
      if (!categorical[feature]) {
        continuous_parent_sum = computeSum(current_node_obs);
        break;
      }
    }
    bool use_mae = splitrule == "mae";
    double parent_absolute_deviation = 0;
    if (use_mae) {
      parent_absolute_deviation = computeAbsoluteDeviation(current_node_obs);
    }

    double best_split_val = -1.0;
    size_t best_feature = 0;
    vector<double> best_threshold;
    vector<size_t> best_left_indices;
    vector<size_t> best_right_indices;
    double best_sum_left;
    
    // only used for honesty
    vector<size_t> holdout_left_indices;
    vector<size_t> holdout_right_indices;

    // now consider each of the sampled features
    for (size_t i : sampled_features) {
        if (categorical[i]) {
            bestSplitCategorical(node_index, i, best_split_val, best_feature, best_threshold,
                                 best_sum_left, use_mae, parent_absolute_deviation);
        }
        else {
            // does not return the best indices, so this has to be done later
            bestSplitContinuous(node_index, i, use_mae, continuous_parent_sum, parent_absolute_deviation,
                                best_split_val, best_feature, best_threshold, best_sum_left);
        }
    }

    // if no best split is found, make the node a leaf
    if (best_split_val < 0) {
        if (!honest) {
          means.push_back(parent_sum / (double) node_sizes[node_index]);  // for dishonest trees, use parent info already computed earlier
        } else {
          const vector<size_t>& holdout_obs = holdout_node_obs[node_index];
          if (holdout_obs.empty()) {
            means.push_back(parent_sum / (double) node_sizes[node_index]);
          } else {
            means.push_back(computeSum(holdout_obs) / (double) holdout_obs.size()); // for honest trees, use the holdout set for computing the mean
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
    node_obs.push_back(std::move(best_left_indices));          // construct left daughter
    node_obs.push_back(std::move(best_right_indices));         // construct right daughter
    sum_node.push_back(best_sum_left);                         // save sums of responses
    sum_node.push_back(parent_sum - best_sum_left);
    node_sizes.push_back(best_left_size);
    node_sizes.push_back(best_right_size);
    feature_IDs.push_back(best_feature);
    thresholds.push_back(std::move(best_threshold));
    means.push_back(0);

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

// splitting rules for regression trees
//--------------------------------------------------------------------------------------

// functions related to splitting rules should be moved to here eventually, possibly refactored and with new names

// prediction for regression trees
//--------------------------------------------------------------------------------------

vector<double> RegressionTree::computePredictions(const Data& new_data) {
  size_t num_obs = new_data.getNumberOfObs();
  vector<double> predictions(num_obs);
  for (size_t i = 0; i < num_obs; ++i) {
    predictions[i] = predictValue(new_data, i);
  }
  return predictions;
}

// error estimation for regression trees
//--------------------------------------------------------------------------------------

// computes the mean squared error based on a vector of predictions and a test vector response
double computeMSE(const vector<double>& predictions, const vector<double>& response) {
  size_t n = predictions.size();
  double ssq = 0;
  for (size_t i = 0; i < n; ++i) {
    ssq += (predictions[i] - response[i]) * (predictions[i] - response[i]);
  }
  return ssq / (double) n;
}

// computes the R^2 error based on MSE
double computeR2(double mse, const vector<double>& response) {
  size_t n = response.size();

  // compute the mse for the pure intercept model
  double response_mean = vector_sum(response) / (double) n;
  double null_ssq = 0;
  for (size_t i = 0; i < n; ++i) {
    null_ssq += (response_mean - response[i]) * (response_mean - response[i]);
  }
  double null_mse = null_ssq / (double) n;
  return 1 - mse / null_mse;
}

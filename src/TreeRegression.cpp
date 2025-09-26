/*

Functions for regression trees

*/

#include "TreeRegression.h"

// constructor for RegressionTree
//--------------------------------------------------------------------------------------

RegressionTree::RegressionTree(const vector<size_t>& subset_indices, const vector<size_t>& estimation_indices) {
  this->node_obs.push_back(subset_indices);
  this->holdout_node_obs.push_back(estimation_indices);
  this->node_sizes.push_back(subset_indices.size());
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

void RegressionTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                                         vector<double>& best_threshold, double& best_sum_left) {
  const vector<size_t>& current_node_obs = node_obs[node_index];
  double parent_sum = sum_node[node_index];

  // samples split points
  vector<double> split_points;
  size_t nsplits_final = sampleSplitPoints(split_points, current_node_obs, feature);

  // no possible splits
  if (nsplits_final == 0) {
    return;
  }

  // compute number of observations and the sums of responses at each split value
  number_obs_split.assign(nsplits_final, 0);
  sums_split.assign(nsplits_final, 0);
  for (size_t i : current_node_obs) {
    size_t idx = lower_bound(split_points.begin(), split_points.end(), data->get_x(i, feature)) - split_points.begin();
    
    sums_split[idx] += data->get_y(i);
    ++number_obs_split[idx];
  }

  // now compute the decrease of impurity for each split
  size_t n_left = 0;
  double sum_left = 0;

  for (size_t i = 0; i < nsplits_final; ++i) {
    // skip the split if identical to the previous one (or if no observations)
    if (number_obs_split[i] == 0) {
      continue;
    }

    n_left += number_obs_split[i];
    sum_left += sums_split[i];
    size_t n_right = node_sizes[node_index] - n_left;

    // stop if right child is too small (break since the split points are sorted)
    if (n_right < min_node_size) {
      break;
    }

    // stop if minimal node size is reached
    if (n_left < min_node_size) {
      continue;
    }

    double sum_right = parent_sum - sum_left;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;

    if (decrease > best_split_val) {
      best_split_val = decrease;
      best_feature = feature;
      best_sum_left = sum_left;
      // use average of split points unless it is the final split value
      if (i == nsplits_final - 1) {
        best_threshold = {split_points[i]};
      } else {
        best_threshold = {(split_points[i] + split_points[i + 1]) / 2.0};
      }
    }
  }
}

void RegressionTree::bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices, double& best_sum_left) {
  const vector<double>& feature_values = uniqueValues(data->getValues(node_obs[node_index], feature));
  double parent_sum = sum_node[node_index];
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

    // here we have to compute the means in one of the daughters from scratch
    double sum_left = computeSum(current_left_indices);
    double sum_right = parent_sum - sum_left;

    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;

    if (decrease > best_split_val) {
        best_split_val = decrease;
        best_left_indices = move(current_left_indices);
        best_right_indices = move(current_right_indices);
        best_feature = feature;
        best_threshold.assign(left_values.begin(), left_values.end());
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
}

// function to create a split for a regression tree. returns true if leaf, otherwise false
bool RegressionTree::createSplit(size_t node_index) {
    const vector<size_t>& current_node_obs = node_obs[node_index];
    // if we are in the root node, the sum of the responses needs to be computed
    if (sum_node.empty()) {
      sum_node.push_back(computeSum(current_node_obs));
    }
    double parent_sum = sum_node[node_index];

    // if no split is possible, make the node a leaf
    if (current_node_obs.size() < 2 * min_node_size) {
        if (!honest) {
            means.push_back(parent_sum / (double) node_sizes[node_index]);  // for dishonest trees, we may simply reuse the computed sum
        } else {
            means.push_back(computeSum(holdout_node_obs[node_index]) / (double) holdout_node_obs[node_index].size());  // for honest trees, compute the mean from scratch for the holdout indices
        }
        makeLeaf(node_index);
        return true;
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
            bestSplitCategorical(node_index, i, best_split_val, best_feature, best_threshold, best_left_indices, best_right_indices, best_sum_left);
        }
        else {
            // does not return the best indices, so this has to be done later
            bestSplitContinuous(node_index, i, best_split_val, best_feature, best_threshold, best_sum_left);
        }
    }

    // if no best split is found, make the node a leaf
    if (best_split_val < 0) {
        if (!honest) {
          means.push_back(parent_sum / (double) node_sizes[node_index]);  // for dishonest trees, use parent info already computed earlier
        } else {
          means.push_back(computeSum(holdout_node_obs[node_index]) / (double) holdout_node_obs[node_index].size()); // for honest trees, use the holdout set for computing the mean
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
    node_obs.push_back(best_left_indices);          // construct left daughter
    node_obs.push_back(best_right_indices);         // construct right daughter
    sum_node.push_back(best_sum_left);              // save sums of responses
    sum_node.push_back(parent_sum - best_sum_left);
    node_sizes.push_back(best_left_indices.size());
    node_sizes.push_back(best_right_indices.size());
    feature_IDs.push_back(best_feature);
    thresholds.push_back(best_threshold);
    means.push_back(0);

    // for honest trees, update the holdout indices
    if (honest) {
        holdout_node_obs.push_back(holdout_left_indices);
        holdout_node_obs.push_back(holdout_right_indices);
    }

    return false;
}

// splitting rules for regression trees
//--------------------------------------------------------------------------------------


// prediction for regression trees
//--------------------------------------------------------------------------------------

vector<double> RegressionTree::computePredictions(const Data& new_data) {
  size_t num_obs = new_data.getNumberOfObs();
  vector<double> predictions(num_obs);
  for (size_t i = 0; i < num_obs; ++i) {
    predictions[i] = get<double>(predict(new_data.get_x_row(i)));
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
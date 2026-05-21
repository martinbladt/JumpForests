#ifndef TREE_REGRESSION_H
#define TREE_REGRESSION_H

#include "Tree.h"

class RegressionTree : public Tree {
public:
  RegressionTree(const vector<size_t>& subset_indices, const vector<size_t>& estimation_indices = {});

  const vector<double> getMeans() const {
    return means;
  }

  // prediction for regression trees
  ValueType predict(const vector<double>& x) override {
    return means[predictionLeafID(x)];
  }
  vector<double> computePredictions(const Data& new_data);
  // VIMP prediction for regression trees
  ValueType predictVIMP(const vector<double>& x, size_t feature, mt19937& rng) {
    return means[predictionLeafIDVIMP(x, feature, rng)];
  }

private:
  vector<double> means;     // the means in each terminal node

  // temporary quantities used in growing regression trees
  vector<size_t> num_obs_right;    // number of observations at the splitting points in the right node
  vector<double> sums_right;       // sums of responses at the splitting points in the right node
  vector<double> sum_node;         // sums of responses in each node

  // growing regression trees
  double computeSum(const vector<size_t>& indices);
  double computeAbsoluteDeviation(const vector<size_t>& indices);
  double computeMSESplitValue(size_t n_left, double sum_left, size_t n_right, double sum_right);
  double computeMAESplitValue(const vector<size_t>& left_indices, const vector<size_t>& right_indices,
                              double parent_absolute_deviation);
  void makeLeaf(size_t node_index);
  bool createSplit(size_t node_index) override;
  void bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold, double& best_sum_left);
  void bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices, double& best_sum_left);

  // frees memory from temporary quantities used in growing the tree
  void cleanUpTree() override {
    vector<vector<size_t>>().swap(node_obs);
    vector<size_t>().swap(num_obs_right);
    vector<double>().swap(sums_right);
    vector<double>().swap(sum_node);
  }
};

double computeMSE(const vector<double>& predictions, const vector<double>& response);
double computeR2(double mse, const vector<double>& response);

#endif // TREE_REGRESSION_H

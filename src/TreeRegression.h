#ifndef TREE_REGRESSION_H
#define TREE_REGRESSION_H

#include "Tree.h"

class RegressionTree : public Tree {
public:
  RegressionTree(const vector<size_t>& subset_indices, const vector<size_t>& estimation_indices = {});

  void grow();  // grow the regression tree

  const vector<double> getMeans() const {
    return means;
  }

  // prediction for regression trees
  ValueType predict(const vector<double>& x) override {
    return means[predictionLeafID(x)];
  }

private:
  vector<double> means;     // the means in each terminal node

  // temporary quantities used in growing regression trees
  vector<size_t> number_obs_split;    // number of observations at the splitting points
  vector<double> sums_split;          // sums of responses at the splitting points
  vector<double> sum_node;            // sums of responses in each node

  // growing regression trees
  double computeSum(const vector<size_t>& indices);
  void makeLeaf(size_t node_index);
  bool createSplit(size_t node_index);
  void bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold, double& best_sum_left);
  void bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices, double& best_sum_left);

  // frees memory from temporary quantities used in growing the tree
  void cleanUpTree() override {
    vector<vector<size_t>>().swap(node_obs);
    vector<size_t>().swap(number_obs_split);
    vector<double>().swap(sums_split);
    vector<double>().swap(sum_node);
  }
};

#endif // TREE_REGRESSION_H
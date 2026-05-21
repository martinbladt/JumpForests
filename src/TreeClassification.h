#ifndef TREE_CLASSIFICATION_H
#define TREE_CLASSIFICATION_H

#include "Tree.h"

class ClassificationTree : public Tree {
public:
  ClassificationTree(const vector<size_t>& subset_indices, const vector<size_t>& estimation_indices = {});

  const vector<size_t> getClasses() const {
    return classes;
  }

  // prediction for classification trees
  ValueType predict(const vector<double>& x) override {
    return classes[predictionLeafID(x)];
  }
  //vector<size_t> computePredictions(const Data& new_data) override;
  // VIMP prediction for classification trees
  ValueType predictVIMP(const vector<double>& x, size_t feature, mt19937& rng) {
    return classes[predictionLeafIDVIMP(x, feature, rng)];
  }

private:
  vector<size_t> classes;                          // the predicted class in each terminal node
  vector<vector<double>> class_proportions;        // the predicted class probabilities in each terminal node

  // temporary quantities used in growing classification trees
  vector<size_t> num_obs_right;                    // number of observations at the splitting points in the right node
  vector<double> class_counts_right;               // number of each class in the right node (flattened vector)
  vector<vector<double>> class_counts_node;        // number of each class in every node

  // growing clasification trees
  //double computeSum(const vector<size_t>& indices);
  void makeLeaf(size_t node_index);
  bool createSplit(size_t node_index) override;
  vector<double> computeClassCounts(const vector<size_t>& node_obs);
  pair<double, vector<double>> computePredictedClass(const vector<double>& class_counts, double node_size);
  void ClassificationTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature,
                                         vector<double>& best_threshold, vector<double>& best_class_counts_left, vector<double>& best_class_counts_right);
  void bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices);
  
  // splitting rules
  double Gini(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, size_t split_id = 0);
  double Entropy(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, const vector<double>& class_prop_parent, size_t split_id = 0);
  double Misclassification(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, size_t split_id = 0);
  double Twoing(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, size_t split_id = 0);
  double Hellinger(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t split_id = 0);

  // frees memory from temporary quantities used in growing the tree
  void cleanUpTree() override {
    vector<vector<size_t>>().swap(node_obs);
    vector<size_t>().swap(num_obs_right);
    vector<vector<double>>().swap(class_counts_right);
    vector<vector<double>>().swap(class_counts_node);
  }
};

vector<double> classCountsToProportions(const vector<double>& class_counts, size_t num_classes, size_t num_obs, size_t split_id = 0);

#endif // TREE_CLASSIFICATION_H
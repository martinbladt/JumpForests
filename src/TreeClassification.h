#ifndef TREE_CLASSIFICATION_H
#define TREE_CLASSIFICATION_H

#include "Tree.h"

class ClassificationTree : public Tree {
public:
  ClassificationTree(vector<size_t> subset_indices, vector<size_t> estimation_indices = {});

  const vector<double>& getClasses() const {
    return classes;
  }
  const vector<vector<double>>& getClassProportions() const {
    return class_proportions;
  }

  // prediction for classification trees
  ValueType predict(const vector<double>& x) override {
    return classes[predictionLeafID(x)];
  }
  // VIMP prediction for classification trees
  ValueType predictVIMP(const vector<double>& x, size_t feature, mt19937& rng) {
    return classes[predictionLeafIDVIMP(x, feature, rng)];
  }
  pair<vector<double>, vector<double>> computePredictions();
  pair<vector<double>, vector<double>> computePredictions(const Data& new_data);

private:
  vector<double> classes;                          // the predicted class in each terminal node
  vector<vector<double>> class_proportions;        // the predicted class probabilities in each terminal node

  // temporary quantities used in growing classification trees
  vector<size_t> num_obs_right;                    // number of observations at the splitting points in the right node
  vector<double> class_counts_right;               // number of each class in the right node (flattened vector)
  vector<vector<double>> class_counts_node;        // number of each class in every node
  vector<size_t> feature_indices;                  // feature indices sampled at each node

  enum class ClassificationSplitRule : uint8_t {
    Gini,
    Entropy,
    Misclassification,
    Twoing,
    Hellinger
  };
  ClassificationSplitRule splitrule_id;

  // growing clasification trees
  void reserveTreeMemory(size_t num_obs);
  void makeLeaf(size_t node_index);
  bool createSplit(size_t node_index) override;
  vector<double> computeClassCounts(const vector<size_t>& node_obs);
  pair<double, vector<double>> computePredictedClass(const vector<double>& class_counts, double node_size);
  double computeSplitValue(const vector<double>& class_prop_left, const vector<double>& class_prop_right,
                           size_t num_obs_left, size_t num_obs_right);
  void bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature,
                                         vector<double>& best_threshold, vector<double>& best_class_counts_left, vector<double>& best_class_counts_right);
  void bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold, 
                                              vector<double>& best_class_counts_left, vector<double>& best_class_counts_right);
  
  // splitting rules
  double Gini(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, size_t split_id = 0);
  double Entropy(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, size_t split_id = 0);
  double Misclassification(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, size_t split_id = 0);
  double Twoing(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t num_obs_left, size_t num_obs_right, size_t split_id = 0);
  double Hellinger(const vector<double>& class_prop_left, const vector<double>& class_prop_right, size_t split_id = 0);

  // frees memory from temporary quantities used in growing the tree
  void cleanUpTree() override {
    vector<vector<size_t>>().swap(node_obs);
    vector<vector<size_t>>().swap(holdout_node_obs);
    vector<size_t>().swap(num_obs_right);
    vector<double>().swap(class_counts_right);
    vector<vector<double>>().swap(class_counts_node);
    vector<size_t>().swap(feature_indices);
    left_daughters.shrink_to_fit();
    feature_IDs.shrink_to_fit();
    thresholds.shrink_to_fit();
    depths.shrink_to_fit();
    node_sizes.shrink_to_fit();
    classes.shrink_to_fit();
    class_proportions.shrink_to_fit();
  }
};

vector<double> computeMisclassificationError(const vector<double>& class_predictions, const vector<double>& response, size_t num_classes);
double computeBrierScoreError(const vector<double>& prob_predictions, const vector<double>& response, size_t num_classes);
double computeNormalizedBrierScoreError(const vector<double>& prob_predictions, const vector<double>& response, size_t num_classes);
vector<size_t> computeConfusionMatrix(const vector<double>& class_predictions, const vector<double>& response, size_t num_classes);

vector<double> classCountsToProportions(const vector<double>& class_counts, size_t num_classes, size_t num_obs, size_t split_id = 0);
double mostFrequentClass(const vector<double>& class_counts);

#endif // TREE_CLASSIFICATION_H

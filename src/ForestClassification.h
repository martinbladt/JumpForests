#ifndef FORESTCLASSIFICATION_H
#define FORESTCLASSIFICATION_H

#include "Forest.h"
#include "TreeClassification.h"

class ClassificationForest : public Forest {
public: 
  ClassificationForest();

  // grows a classification forest with multi-threading
  void grow();

  const vector<double> getClasses() {
    return classes;
  }

  /*
    for computing predictions after the forest is grown
    first vector is in-bag predictions of classes, the other oob predictions of classes
    third and fourth vectors are flattened vectors of class probabilities
  */
  vector<vector<double>> computePredictions();
  // computing predictions on a new dataset
  pair<vector<double>, vector<double>> computePredictions(const Data& new_data, bool compute_probs = false);
  // for computing VIMP based on OOB predictions
  vector<double> computeVIMPPermute(size_t feature, int feature_seed, string error_type);
  vector<double> computeVIMPRandom(size_t feature, int feature_seed, string error_type);

private:
  vector<double> classes;

  // quantities reused when computing VIMP for several features
  vector<vector<size_t>> vimp_oob_indices;
  vector<vector<bool>> vimp_tree_uses_feature;
  vector<double> vimp_tree_errors_misc;
  vector<double> vimp_tree_errors_brier;
  void prepareVIMPCache();
};

#endif // FORESTCLASSIFICATION_H

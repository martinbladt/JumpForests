#ifndef FORESTREGRESSION_H
#define FORESTREGRESSION_H

#include "Forest.h"
#include "TreeRegression.h"

class RegressionForest : public Forest {
public: 
  RegressionForest();

  // grows a regression forest with multi-threading
  void grow();
  
  // for computing predictions after the forest is grown
  // first vector is in-bag predictions, the other oob predictions
  pair<vector<double>, vector<double>> computePredictions();
  // computing predictions on a new dataset
  vector<double> computePredictions(const Data& new_data);
  // for computing VIMP based on OOB predictions
  double computeVIMPPermute(size_t feature, int feature_seed);
  double computeVIMPRandom(size_t feature, int feature_seed);

private:
  // quantities reused when computing VIMP for several features
  vector<vector<size_t>> vimp_oob_indices;
  vector<vector<bool>> vimp_tree_uses_feature;
  vector<double> vimp_tree_errors;
  void prepareVIMPCache();
};

#endif // FORESTREGRESSION_H

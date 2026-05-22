#ifndef FORESTCLASSIFICATION_H
#define FORESTCLASSIFICATION_H

#include "Forest.h"
#include "TreeClassification.h"

class ClassificationForest : public Forest {
public: 
  ClassificationForest();

  // grows a regression forest with multi-threading
  void grow();
  
  // computes the prediction for observation x
  double predict(const vector<double>& x);

  const vector<double> getClasses() {
    return classes;
  }

  // for computing predictions after the forest is grown
  // first vector is in-bag predictions of classes, the other oob predictions of classes
  // third and fourth vectors are flattened vectors of class probabilities
  vector<vector<double>> computePredictions();
  // computing predictions on a new dataset
  vector<double> computePredictions(const Data& new_data);
  // for computing VIMP based on OOB predictions
  vector<double> computeVIMPPermute(size_t feature, int feature_seed, string error_type);
  vector<double> computeVIMPRandom(size_t feature, int feature_seed, string error_type);

private:
  vector<double> classes;
};

#endif // FORESTCLASSIFICATION_H
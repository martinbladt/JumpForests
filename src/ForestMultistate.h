#ifndef FORESTMULTISTATE_H
#define FORESTMULTISTATE_H

#include "Forest.h"
#include "TreeMultistate.h"

/*
class RegressionForest : public Forest {
public: 
  RegressionForest();

  // grows a regression forest with multi-threading
  void grow();
  
  // computes the prediction for observation x
  double predict(const vector<double>& x);

  const vector<double> getMeans() {
    return means;
  }

  // for computing predictions after the forest is grown
  // first vector is in-bag predictions, the other oob predictions
  pair<vector<double>, vector<double>> computePredictions();
  // computing predictions on a new dataset
  vector<double> computePredictions(const Data& new_data);

private:
  vector<double> means;
};
*/

#endif // FORESTMULTISTATE_H
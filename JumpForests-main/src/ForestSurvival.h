#ifndef FORESTSURVIVAL_H
#define FORESTSURVIVAL_H

#include "Forest.h"
#include "TreeSurvival.h"

class SurvivalForest : public Forest {
public:
  SurvivalForest(const vector<double> unique_event_times);
  
  // grows a survival forest without multi-threading (mostly for testing purposes)
  void grow();

  // grows a survival forest with multi-threading
  void growThreads();
  
  // predicts the chf for observation x
  vector<double> predict(vector<double> x);

  // get info
  const vector<double> getEventTimes() const {
    return unique_event_times;
  }

  const vector<vector<double>> getCHF() const {
    return chf;
  }
  /*
  const vector<unique_ptr<SurvivalTree>>& getTrees() const {
    return trees;
  }
  */

private:
  // the trees in the forest
  //vector<unique_ptr<SurvivalTree>> trees;

  // quantities of interest specific to survival forests
  const vector<double> unique_event_times;  // vector of ordered unique event times for all data
  size_t num_unique_event_times;            // number of unique event times
  vector<vector<double>> chf;               // the cumulative hazard at the unique_event_times for the forest

  // to compute the chf for the whole forest after the forest is grown
  void computePredictions();
};

#endif // FORESTSURVIVAL_H
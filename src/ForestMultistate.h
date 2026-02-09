#ifndef FORESTMULTISTATE_H
#define FORESTMULTISTATE_H

#include "Forest.h"
#include "TreeMultistate.h"

class MultistateForest : public Forest {
public:
  MultistateForest(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, uint8_t num_states);

  // grows a multi-state forest with multi-threading
  void grow();

  // predicts the Nelson-Aalen estimator for observation x
  vector<double> predict(const vector<double>& x);

  // get info
  const vector<double> getEventTimes() const {
    return unique_event_times;
  }
  const vector<size_t> getResponseEventTimeIDs() const {
    return response_event_time_ids;
  }
  const size_t getNumUniqueEventTimes() const {
    return num_unique_event_times;
  }

  const vector<vector<double>> getNA() const {
    return na;
  }
  // for computing predictions after the forest is grown
  // first vector is a flattened 2D array with in-bag predictions, the other with oob predictions
  pair<vector<double>, vector<double>> computePredictions();
  // computing predictions on a new dataset (output is a flattened array)
  vector<double> computePredictions(const Data& new_data);
private:
  // quantities of interest specific to multi-state forests
  const vector<double> unique_event_times;      // vector of ordered unique event times for all observations across all jumps
  const vector<size_t> response_event_time_ids; // the indices of unique_event_times corresponding to the response times
  size_t num_unique_event_times;                // number of unique event times
  vector<vector<double>> na;                    // the Nelson--Aalen estimator at the unique event times for the forest (vector of flattened matrices)
};

#endif // FORESTMULTISTATE_H
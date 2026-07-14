#ifndef FORESTMULTISTATE_H
#define FORESTMULTISTATE_H

#include "Forest.h"
#include "TreeMultistate.h"

class MultistateForest : public Forest {
public:
  MultistateForest(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, uint8_t num_states, bool save_predictions);

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
  const vector<vector<double>> getInitDist() const {
    return init_dist;
  }
  bool predictionsSaved() {
    return save_predictions;
  }
  void computePredictionsCensoring();
  // for computing predictions after the forest is grown
  vector<vector<double>> computePredictions(bool compute_initial, bool compute_censoring);
  vector<vector<double>> computePredictions(const Data& new_data, bool compute_initial, bool compute_censoring);
private:
  // quantities of interest specific to multi-state forests
  bool save_predictions;
  const vector<double> unique_event_times;      // vector of ordered unique event times for all observations across all jumps
  const vector<size_t> response_event_time_ids; // the indices of unique_event_times corresponding to the response times
  size_t num_unique_event_times;                // number of unique event times
  vector<vector<double>> na;                    // the Nelson--Aalen estimator at the unique event times for the forest (vector of flattened matrices)
  vector<vector<double>> init_dist;             // the estimated initial distribution for the forest
};

#endif // FORESTMULTISTATE_H

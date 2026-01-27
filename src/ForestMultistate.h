#ifndef FORESTMULTISTATE_H
#define FORESTMULTISTATE_H

#include "Forest.h"
#include "TreeMultistate.h"

class MultistateForest : public Forest {
public:
  MultistateForest(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids);

  // grows a multi-state forest with multi-threading
  void grow();

  // get info
  const vector<double> getEventTimes() const {
    return unique_event_times;
  }
  const vector<size_t> getResponseEventTimeIDs() const {
    return response_event_time_ids;
  }

  const vector<vector<double>> getNA() const {
    return na;
  }
private:
  // quantities of interest specific to multi-state forests
  const vector<double> unique_event_times;      // vector of ordered unique event times for all observations across all jumps
  const vector<size_t> response_event_time_ids; // the indices of unique_event_times corresponding to the response times
  size_t num_unique_event_times;                // number of unique event times
  vector<vector<double>> na;                    // the Nelson--Aalen estimator at the unique event times for the forest (vector of flattened matrices)
};

#endif // FORESTMULTISTATE_H
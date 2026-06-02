#ifndef FORESTSURVIVAL_H
#define FORESTSURVIVAL_H

#include "Forest.h"
#include "TreeSurvival.h"

class SurvivalForest : public Forest {
public:
  SurvivalForest(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, const vector<size_t>& true_event_time_ids, bool save_predictions);
  
  // grows a survival forest with multi-threading
  void grow();
  
  // predicts the chf for observation x
  vector<double> predict(const vector<double>& x);

  // get info
  const vector<double> getEventTimes() const {
    return unique_event_times;
  }
  const vector<size_t> getTrueEventTimeIDs() const {
    return true_event_time_ids;
  }
  const size_t getNumUniqueEventTimes() const {
    return num_unique_event_times;
  }
  const vector<size_t> getResponseEventTimeIDs() const {
    return response_event_time_ids;
  }

  const vector<vector<double>> getCHF() const {
    return chf;
  }
  bool predictionsSaved() {
    return save_predictions;
  }
  // for computing predictions after the forest is grown
  // first vector is a flattened 2D array with in-bag predictions, the other with oob predictions
  //pair<vector<double>, vector<double>> computePredictions();
  vector<vector<double>> computePredictions(bool compute_censoring);
  vector<vector<double>> computePredictions(const Data& new_data, bool compute_censoring);
  // computes OOB censoring predictions after these are saved in the terminal nodes
  //vector<double> computePredictionsCensoringOOB();
  // computing predictions on a new dataset (output is a flattened array)
  //vector<double> computePredictions(const Data& new_data);
  //pair<vector<double>, vector<double>> computePredictionsCensoring(const Data& new_data);
  void computePredictionsCensoring();
  // for computing OOB predictions for VIMP
  vector<double> computePredictionsVIMPRandom(size_t feature, int feature_seed);
  vector<double> computePredictionsVIMPPermute(size_t feature, int feature_seed);
  double computeVIMPPermute(size_t feature, int feature_seed, string error_type);
  double computeVIMPRandom(size_t feature, int feature_seed);

private:
  // the trees in the forest
  //vector<unique_ptr<SurvivalTree>> trees;
  bool save_predictions;

  // quantities of interest specific to survival forests
  const vector<double> unique_event_times;      // vector of ordered unique event times for all data
  const vector<size_t> response_event_time_ids; // the indices of unique_event_times corresponding to the response times
  const vector<size_t> true_event_time_ids;     // the indices of unique_event_times for uncensored times
  size_t num_unique_event_times;                // number of unique event times
  vector<vector<double>> chf;                   // the cumulative hazard at the unique_event_times for the forest
};

#endif // FORESTSURVIVAL_H

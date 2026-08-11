#ifndef FORESTSURVIVAL_H
#define FORESTSURVIVAL_H

#include "Forest.h"
#include "TreeSurvival.h"

class SurvivalForest : public Forest {
public:
  SurvivalForest(vector<double> unique_event_times, vector<size_t> response_event_time_ids,
                 vector<size_t> true_event_time_ids, vector<double> censoring_times, bool save_predictions);
  
  // grows a survival forest with multi-threading
  void grow();
  
  // get info
  const vector<double>& getEventTimes() const {
    return unique_event_times;
  }
  const vector<size_t>& getTrueEventTimeIDs() const {
    return true_event_time_ids;
  }
  const size_t getNumUniqueEventTimes() const {
    return num_unique_event_times;
  }
  bool predictionsSaved() {
    return save_predictions;
  }
  // for computing predictions after the forest is grown
  vector<vector<double>> computePredictions(bool compute_censoring);
  vector<vector<double>> computePredictions(const Data& new_data, bool compute_censoring,
                                            const vector<double>* evaluation_times = nullptr);
  // populates terminal nodes with censoring estimators after fitting
  void computePredictionsCensoring();
  // VIMP functions
  double computeVIMPPermute(size_t feature, int feature_seed, string error_type);
  double computeVIMPRandom(size_t feature, int feature_seed, string error_type);

private:
  bool save_predictions;

  // quantities of interest specific to survival forests
  const vector<double> unique_event_times;      // vector of ordered, possibly thinned response times
  const vector<size_t> response_event_time_ids; // the indices of unique_event_times corresponding to the response times
  const vector<size_t> true_event_time_ids;     // the indices of unique_event_times for uncensored times
  const vector<double> censoring_times;         // full response-time grid used for the censoring distribution
  size_t num_unique_event_times;                // number of unique event times
  // quantities reused when computing VIMP for several features
  vector<vector<size_t>> vimp_oob_indices;
  vector<vector<size_t>> vimp_leaf_ids;
  vector<vector<bool>> vimp_tree_uses_feature;
  vector<double> vimp_tree_concordance;
  vector<double> vimp_tree_brier;
  vector<double> vimp_tree_kl;
  vector<double> vimp_event_times;
  vector<size_t> vimp_censoring_time_ids;
  void prepareVIMPStructure();
  const vector<double>& prepareVIMPBaseline(string error_type);
};

#endif // FORESTSURVIVAL_H

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
  // VIMP functions
  double computeVIMPPermute(size_t feature, int feature_seed, string error_type, const vector<double>& state_weights);
  double computeVIMPRandom(size_t feature, int feature_seed);
  
private:
  void clearVIMPCache();
  void prepareVIMPCache();
  void prepareVIMPBaseline(const string& error_type, const vector<double>& state_weights);

  // quantities of interest specific to multi-state forests
  bool save_predictions;
  const vector<double> unique_event_times;      // vector of ordered unique event times for all observations across all jumps
  const vector<size_t> response_event_time_ids; // the indices of unique_event_times corresponding to the response times
  size_t num_unique_event_times;                // number of unique event times
  vector<vector<double>> na;                    // the Nelson--Aalen estimator at the unique event times for the forest (vector of flattened matrices)
  vector<vector<double>> init_dist;             // the estimated initial distribution for the forest

  // lazily populated, response- and tree-level quantities shared by every
  // feature-specific permutation VIMP computation.
  bool vimp_cache_ready = false;
  vector<double> vimp_event_times;                            // the times used for scoring (excludes zero) (length = T - 1)
  vector<uint8_t> vimp_observed_states;                       // observed states (length = (T - 1) * n)
  vector<unsigned char> vimp_uncensored;                      // is the observation censored? no = 0 (length = n)
  vector<size_t> vimp_first_endpoint_event_ids;               // first scoring time at or after the endpoint (length = n)
  vector<vector<size_t>> vimp_oob_indices;                    // original observation IDs
  vector<vector<size_t>> vimp_oob_leaf_ids;                   // original terminal node IDs
  vector<vector<bool>> vimp_tree_uses_feature;                // whether a feature appears in an internal split (for skipping unnecessary paths)
  vector<vector<vector<double>>> vimp_leaf_occupation_probs;  // VIMP occupation probabilities in the leaf for each time and state (one vector for each tree)
  vector<vector<vector<double>>> vimp_leaf_event_ipcw;        // pre-endpoint IPCW at each time (one vector for each tree)
  vector<vector<double>> vimp_oob_endpoint_ipcw;              // IPCW at the subject endpoint at each time (one vector for each tree)
  bool vimp_baseline_ready = false;
  string vimp_baseline_error_type;
  vector<double> vimp_baseline_state_weights;
  vector<double> vimp_baseline_scores;
};

#endif // FORESTMULTISTATE_H

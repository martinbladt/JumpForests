#ifndef FORESTMULTISTATE_H
#define FORESTMULTISTATE_H

#include "Forest.h"
#include "TreeMultistate.h"

class MultistateForest : public Forest {
public:
  MultistateForest(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids,
                   uint8_t num_states, bool save_predictions,
                   shared_ptr<const vector<double>> fh_weights_a = nullptr,
                   shared_ptr<const vector<double>> fh_weights_b = nullptr);

  // grows a multi-state forest with multi-threading
  void grow();

  // predicts the Nelson-Aalen estimator for observation x
  vector<double> predict(const vector<double>& x);

  // get info
  const vector<double>& getEventTimes() const {
    return unique_event_times;
  }
  const vector<size_t>& getResponseEventTimeIDs() const {
    return response_event_time_ids;
  }
  size_t getNumUniqueEventTimes() const {
    return num_unique_event_times;
  }
  const vector<double>& getCensoringTimes() const {
    return *censoring_times;
  }
  bool predictionsSaved() const {
    return save_predictions;
  }
  void computePredictionsCensoring();
  // for computing predictions after the forest is grown
  vector<vector<double>> computePredictions(bool compute_initial, bool compute_censoring);
  vector<vector<double>> computePredictions(const Data& new_data, bool compute_initial, bool compute_censoring);
  vector<vector<double>> computeErrorPredictions(const Data& new_data = Data());
  // VIMP functions
  double computeVIMPPermute(size_t feature, int feature_seed, const string& error_type,
                            const vector<double>& state_weights);
  double computeVIMPRandom(size_t feature, int feature_seed, const string& error_type,
                           const vector<double>& state_weights);
  
private:
  void clearVIMPCache();
  void prepareVIMPCache();
  void prepareVIMPBaseline(const string& error_type, const vector<double>& state_weights);
  void prepareVIMPComputation(size_t feature, const string& error_type, const vector<double>& state_weights);
  double computeVIMPForLeafAssignments(size_t tree_id, const vector<size_t>& prediction_leaf_ids, const string& error_type, 
                                       const vector<double>& state_weights, vector<double>& score);

  // quantities of interest specific to multi-state forests
  bool save_predictions;
  const vector<double> unique_event_times;      // vector of ordered unique event times for all observations across all jumps
  const vector<size_t> response_event_time_ids; // the indices of unique_event_times corresponding to the response times
  size_t num_unique_event_times;                // number of unique event times
  size_t num_states;                            // number of states, kept as size_t for safe flattened dimensions
  size_t dim;                                   // number of entries in one num_states x num_states matrix
  shared_ptr<vector<double>> censoring_times;   // common censoring grid shared by all trees
  vector<size_t> event_censoring_time_ids;      // common event-grid positions in the full censoring grid
  shared_ptr<const vector<double>> fh_weights_a; // source-state-specific Fleming--Harrington a-exponents
  shared_ptr<const vector<double>> fh_weights_b; // source-state-specific Fleming--Harrington b-exponents

  // lazily populated, response- and tree-level quantities shared by every
  // feature-specific VIMP computation
  bool vimp_cache_ready = false;
  vector<double> vimp_event_times;                            // the complete time grid used by the ordinary errors (length = T)
  vector<uint8_t> vimp_observed_states;                       // observed state IDs used for scoring (length = T * n)
  vector<unsigned char> vimp_uncensored;                      // one when the final state was observed, zero when censored
  vector<size_t> vimp_first_endpoint_event_ids;               // id for the first scoring time at or after the endpoint (length = n)
  vector<vector<size_t>> vimp_oob_indices;                    // original observation IDs
  vector<vector<size_t>> vimp_oob_leaf_ids;                   // original terminal node IDs
  vector<vector<bool>> vimp_tree_uses_feature;                // whether a feature appears in an internal split (for skipping unnecessary paths)
  vector<vector<size_t>> vimp_leaf_cache_ids;                 // terminal node ID to compact cache entry for each tree
  vector<vector<vector<double>>> vimp_leaf_occupation_probs;  // compact reached-leaf occupation curves for each tree
  vector<vector<vector<double>>> vimp_leaf_event_ipcw;        // compact original-leaf pre-endpoint IPCW curves
  vector<vector<double>> vimp_oob_endpoint_ipcw;              // IPCW at the subject endpoint at each time (one vector for each tree)
  bool vimp_baseline_ready = false;
  string vimp_baseline_error_type;
  vector<double> vimp_baseline_state_weights;
  vector<double> vimp_baseline_scores;
};

#endif // FORESTMULTISTATE_H

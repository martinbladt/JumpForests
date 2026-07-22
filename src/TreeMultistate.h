#ifndef TREE_MULTISTATE_H
#define TREE_MULTISTATE_H

#include "Tree.h"

class MultistateTree : public Tree {
public:
  MultistateTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids,
            const vector<size_t>& subset_indices, uint8_t num_states, bool save_predictions,
            const vector<size_t>& estimation_indices = {}, shared_ptr<vector<double>> censoring_times = nullptr);

  const vector<double>& getEventTimes() const {
    return *unique_event_times;
  }
  const size_t getNumberOfUniqueEventTimes() const {
    return num_unique_event_times;
  }
  const vector<vector<double>>& getNA() const {
    return na;
  }
  const vector<vector<double>>& getKMCensoring() const {
    return KM_censoring;
  }
  const vector<vector<double>>& getKMCensoringFull() const {
    return KM_censoring_full;
  }
  const vector<double>& getCensoringTimes() const {
    return *censoring_times;
  }
  const vector<vector<double>>& getInitDist() const {
    return init_dist;
  }

  // prediction for multi-state trees
  ValueType predict(const vector<double>& x) override {
    return na[predictionLeafID(x)];
  }
  vector<double> predictInitDist(const vector<double>& x) {
    return init_dist[predictionLeafID(x)];
  }

  vector<double> computePredictions(const Data& new_data);   // not needed (so remove in Tree)
  vector<vector<double>> computePredictions(bool compute_initial, bool compute_censoring, const Data& new_data = Data());
  vector<vector<double>> computeErrorPredictions(const Data& new_data = Data()); // computes only occupation and censoring predictions needed for scoring

  // VIMP prediction for multi-state trees (to be investigated)
  ValueType predictVIMP(const vector<double>& x, size_t feature, mt19937& rng) {
    return na[predictionLeafIDVIMP(x, feature, rng)];
  }

  // when the censoring Kaplan-Meier estimators have to be populated after fitting
  void resizeKM() {
    KM_censoring.assign(num_nodes, vector<double>());
    KM_censoring_full.assign(num_nodes, vector<double>());
  }
  void computeCensoringKMExternal(const vector<size_t>& indices, size_t node_index);
private:
  shared_ptr<vector<double>> unique_event_times;        // vector of ordered unique event times across (pooled across all jumps)
  size_t num_unique_event_times;                        // number of unique event times
  shared_ptr<vector<size_t>> response_event_time_ids;   // the indices of unique_event_times corresponding to the response times (flattened array)
  shared_ptr<vector<double>> censoring_times;           // vector of ordered unique times for the censoring distribution
  size_t num_censoring_times;                           // number of unique censoring times
  size_t num_states;                                    // number of states (saved as size_t to avoid overflow in flattened indices)
  size_t dim;                                           // number of entries in a single num_states x num_states matrix
  bool save_predictions;                                // should predictions (including KM estimators for censoring) be saved during fitting?
  vector<vector<double>> na;                            // the Nelson--Aalen estimator in each terminal node with jumps at the unique_event_times,
                                                        // each vector being a flattened array of length num_states^2
  vector<vector<double>> KM_censoring;                  // the Kaplan-Meier estimate at the unique_event_times for the censoring distribution 
  vector<vector<double>> KM_censoring_full;             // the Kaplan-Meier estimate at the censoring_times for the censoring distribution
  vector<vector<double>> init_dist;                     // estimated initial distribution in each node

  /*
    the following information is computed once before growing the tree. A response path is then represented by its
    initial state, a range in transition_ids and at most one censoring entry. This avoids scanning the padded response
    arrays again for every node, feature and split point.
  */
  struct ValidJumpInfo {
    size_t from_state;
    size_t to_state;
    size_t matrix_index;
  };
  vector<uint8_t> initial_state_ids;              // zero-indexed initial state for every observation
  vector<size_t> transition_offsets;              // offsets into transition_ids (one additional entry marks the final end)
                                                  // example: 0: 2 jumps, 1: 1 jump, 2: 0 jumps, 3: 2 jumps: transition_offsets = {0, 2, 3, 3, 5}
  vector<size_t> transition_ids;                  // flattened event-time and transition-matrix indices
  vector<size_t> censoring_risk_ids;              // flattened event-time and state index, or the risk-array size if uncensored
  vector<size_t> event_censoring_time_ids;        // maps event times to the corresponding position in the censoring grid
  vector<ValidJumpInfo> valid_jumps;              // valid transitions with zero-indexed and flattened indices
  vector<vector<size_t>> jump_event_time_ids;     // parent-node times with a nonzero count for each valid transition
  vector<size_t> feature_indices;                 // feature indices sampled repeatedly while growing the tree
  bool multistate_data_prepared = false;

  // temporary quantities used in growing multi-state trees
  vector<size_t> num_jumps;                       // number of jumps at each unique event time (flattened matrices)
  vector<size_t> num_at_risk;                     // number of individuals residing in each state at each unique event time
  vector<size_t> num_censored;                    // exact censoring counts at each unique event time and state
  vector<size_t> num_jumps_daughter;              // reusable jump counts for one possible daughter node
  vector<size_t> num_at_risk_daughter;            // reusable at-risk counts for one possible daughter node
  vector<size_t> current_num_at_risk;             // rolling state counts used when materialising an at-risk array

  enum class MultistateSplitRule : uint8_t {
    LogRank,
    Gehan,
    TaroneWare,
    Conserve,
    ApproxLogRank
  };
  MultistateSplitRule splitrule_id = MultistateSplitRule::LogRank;

  // growing multi-state trees
  void reserveTreeMemory(size_t num_obs);                         // reserves the vectors which receive one entry per tree node
  void prepareMultistateData();                                   // constructs the compact response representation above
  void addObservationToQuantities(size_t observation, vector<size_t>& jumps, vector<size_t>& at_risk,
                                  vector<size_t>& censored);      // adds one response path to a set of sufficient statistics
  void materialiseAtRisk(const vector<size_t>& jumps, const vector<size_t>& censored, vector<size_t>& at_risk);
  void prepareJumpEventTimeIDs();                                 // skips empty event times while scoring candidate splits
  void computeMultistateQuantities(const vector<size_t>& indices, vector<size_t>& jumps, vector<size_t>& at_risk); // computes the number at risk and the number of jumps at the unique_event_times
  void makeLeaf(size_t node_index);                               // helper function for making a node a leaf
  bool createSplit(size_t node_index) override;                   // returns true if leaf, computes best split
  void computeInitialDist(size_t node_index);                     // computes the initial distribution in a terminal node
  void computeNA(size_t node_index);                              // computes the Nelson--Aalen estimator in a terminal node
  void computeCensoringKM(size_t node_index);                     // computes the KM estimator for the censoring distribution in a terminal node
  void bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold);
  void bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature,
                           vector<double>& best_threshold); // computes the best split for a chosen categorical feature
  double computeSplitValue(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter);
  
  // splitting rules (see notes for more ideas)
  double logRank(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter);
  double Gehan(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter);
  double TaroneWare(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter);
  double conserve(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter);
  double approxLogRank(const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter);

  // frees memory from temporary quantities used in growing the tree
  void cleanUpTree() override {
    vector<vector<size_t>>().swap(node_obs);
    vector<vector<size_t>>().swap(holdout_node_obs);
    vector<size_t>().swap(num_jumps);
    vector<size_t>().swap(num_at_risk);
    vector<size_t>().swap(num_censored);
    vector<size_t>().swap(num_jumps_daughter);
    vector<size_t>().swap(num_at_risk_daughter);
    vector<size_t>().swap(current_num_at_risk);
    vector<uint8_t>().swap(initial_state_ids);
    vector<size_t>().swap(transition_offsets);
    vector<size_t>().swap(transition_ids);
    vector<size_t>().swap(censoring_risk_ids);
    vector<ValidJumpInfo>().swap(valid_jumps);
    vector<vector<size_t>>().swap(jump_event_time_ids);
    vector<size_t>().swap(feature_indices);
    response_event_time_ids.reset();

    // the vectors below remain part of the fitted tree, but their spare capacity is no longer needed after growing
    left_daughters.shrink_to_fit();
    feature_IDs.shrink_to_fit();
    thresholds.shrink_to_fit();
    depths.shrink_to_fit();
    node_sizes.shrink_to_fit();
    prediction_node_IDs.shrink_to_fit();
    na.shrink_to_fit();
    init_dist.shrink_to_fit();
    KM_censoring.shrink_to_fit();
    KM_censoring_full.shrink_to_fit();
  }
};

vector<double> uniqueEventTimesMultistate(const vector<double>& times, const vector<uint8_t>& states);
vector<double> uniqueCensoringTimesMultistate(const vector<double>& unique_event_times, const vector<double>& times, const vector<size_t>& last_observed_times);
vector<size_t> computeResponseEventTimeIDsMultistate(const vector<double>& unique_event_times, const vector<double>& times,
                                                     const vector<uint8_t>& states, uint8_t max_response_length);
vector<bool> computeStateIndicatorsMultistate(const Data& data, const vector<size_t>& response_event_time_ids,
                                              size_t num_unique_event_times);

// error computations for multi-states
vector<double> computeBrierScoreMultistate(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times,
                                   const List& occupation_probs, const vector<double>& state_weights);
vector<double> computeKLScoreMultistate(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times,
                            const List& occupation_probs, const vector<double>& state_weights);
vector<double> computeKLScoreCppMultistate(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times,
                                   const vector<double>& occupation_probs, const vector<double>& state_weights);
vector<double> computeBrierScoreCppMultistate(const vector<bool>& states_ind, const vector<double>& weights, const vector<double>& unique_event_times,
                                      const vector<double>& occupation_probs, const vector<double>& state_weights);

// miscellaneous functions related to multi-states
vector<double> AalenJohansen(const vector<double>& na, uint8_t num_states);
vector<double> occupationProbabilitiesCpp(const vector<double>& na, const vector<double>& init, size_t num_states, size_t num_estimators);
vector<double> occupationProbabilities(const List& na, const List& init, size_t num_states);

#endif // TREE_MULTISTATE_H

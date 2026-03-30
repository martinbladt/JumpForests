#ifndef TREE_MULTISTATE_H
#define TREE_MULTISTATE_H

#include "Tree.h"
//#include <RcppArmadillo.h>

class MultistateTree : public Tree {
public:
  MultistateTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids,
            const vector<size_t>& subset_indices, uint8_t num_states, bool save_predictions,
            const vector<size_t>& estimation_indices = {});

  const vector<double> getEventTimes() const {
    return *unique_event_times;
  }
  const size_t getNumberOfUniqueEventTimes() const {
    return num_unique_event_times;
  }
  const vector<vector<double>> getNA() const {
    return na;
  }
  const vector<vector<double>> getKMCensoring() const {
    return KM_censoring;
  }
  const vector<vector<double>> getInitDist() const {
    return init_dist;
  }

  // prediction for multi-state trees
  ValueType predict(const vector<double>& x) override {
    return na[predictionLeafID(x)];
  }
  vector<double> predictInitDist(const vector<double>& x) {
    return init_dist[predictionLeafID(x)];
  }

  vector<double> computePredictions(const Data& new_data) override;
  //vector<double> computePredictedInitialDistributions(const Data& new_data);
  pair<vector<double>, vector<double>> computePredictedInitialDistributions(const Data& new_data);
  pair<vector<double>, vector<double>> computePredictionsCensoring(const Data& new_data);
  vector<vector<double>> computeAllPredictions(const Data& new_data);
  // VIMP prediction for multi-state trees (to be investigated)
  ValueType predictVIMP(const vector<double>& x, size_t feature, mt19937 rng);

  // when the censoring Kaplan-Meier estimators have to be populated after fitting
  void resizeKM() {
    KM_censoring.assign(num_nodes, vector<double>());
  }
private:
  shared_ptr<vector<double>> unique_event_times;        // vector of ordered unique event times across (pooled across all jumps)
  size_t num_unique_event_times;                        // number of unique event times
  shared_ptr<vector<size_t>> response_event_time_ids;   // the indices of unique_event_times corresponding to the response times (flattened array)
  bool save_predictions;                                // should predictions (including KM estimators for censoring) be saved
  vector<vector<double>> na;                            // the Nelson--Aalen estimator in each terminal node with jumps at the unique_event_times,
                                                        // each vector being a flattened array of length num_states^2
  vector<vector<double>> KM_censoring;                  // the Kaplan-Meier estimate at the unique_event_times for the censoring distribution 
  vector<vector<double>> init_dist;                     // estimated initial distribution in each node

  // temporary quantities used in growing multi-state trees
  vector<size_t> num_jumps;                     // number of jumps at each unique event time (flattened matrix)
  vector<size_t> num_at_risk;                   // number of individuals residing in each state at each unique event time
  vector<size_t> censoring_contribution;        // censoring contribution for each state at each unique event time (used in key decomposition)

  // growing multi-state trees
  void computeMultistateQuantities(const vector<size_t>& indices, vector<size_t>& at_risk, vector<size_t>& jumps); // computes the number at risk and the number of jumps at the unique_event_times
  void makeLeaf(size_t node_index);                               // helper function for making a node a leaf
  bool createSplit(size_t node_index) override;                   // returns true if leaf, computes best split
  void computeInitialDist(size_t node_index);                     // computes the initial distribution in a terminal node
  void computeNA(size_t node_index);                              // computes the Nelson--Aalen estimator in a terminal node
  void computeCensoringKM();                                      // computes the KM estimator for the censoring distribution in a terminal node
  void computeMultistateQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                         vector<size_t>& num_at_risk_right, vector<size_t>& num_jumps_right, size_t nsplits_final);
  void bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold);
  void bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices); // computes the best split for a chosen categorical feature
  
  // splitting rules (see notes for more ideas)
  double logRank(const vector<size_t>& num_jumps, const vector<size_t>& num_at_risk, const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id = 0);
  double Gehan(const vector<size_t>& num_jumps, const vector<size_t>& num_at_risk, const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id = 0);
  double TaroneWare(const vector<size_t>& num_jumps, const vector<size_t>& num_at_risk, const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id = 0);
  double conserve(const vector<size_t>& num_jumps, const vector<size_t>& num_at_risk, const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id = 0);
  double approxLogRank(const vector<size_t>& num_jumps, const vector<size_t>& num_at_risk, const vector<size_t>& num_jumps_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id = 0);

  // frees memory from temporary quantities used in growing the tree
  void cleanUpTree() override {
    vector<vector<size_t>>().swap(node_obs);
    vector<vector<size_t>>().swap(holdout_node_obs);
    vector<size_t>().swap(num_jumps);
    vector<size_t>().swap(num_at_risk);
    vector<size_t>().swap(censoring_contribution);
    // if more temporary quantities are added, they should go here
  }
};

vector<double> uniqueEventTimesMultistate(const vector<double>& times, const vector<uint8_t>& states);
vector<size_t> computeResponseEventTimeIDsMultistate(const vector<double>& unique_event_times, const vector<double>& times, const vector<uint8_t>& states);
// maybe the function below will never be used (the corresponding function for survival is deprecated)
//vector<double> computeUniqueEventTimes(const vector<double>& times);

// need functions for error computation

vector<double> AalenJohansen(const vector<double>& na, uint8_t num_states);

#endif // TREE_MULTISTATE_H
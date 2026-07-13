#ifndef TREE_SURVIVAL_H
#define TREE_SURVIVAL_H

#include "Tree.h"

// create struct to hold all necessary info for each observation when splitting
/*
struct ObsInfo {
    size_t original_index;
    double feature_value;
    double time;
    size_t indicator;
};
*/

class SurvivalTree : public Tree {
public:
  SurvivalTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids,
               shared_ptr<vector<size_t>> true_event_time_ids, const vector<size_t>& subset_indices, bool save_predictions, 
               const vector<size_t>& estimation_indices = {}, shared_ptr<vector<double>> censoring_times = nullptr);

  const vector<double> getEventTimes() const {
    return *unique_event_times;
  }
  const vector<size_t> getTrueEventTimeIDs() const {
    return *true_event_time_ids;
  }
  const vector<size_t> getResponseEventTimeIDs() const {
    return *response_event_time_ids;
  }

  const vector<vector<double>>& getCHF() const {
    return chf;
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

  // prediction for survival trees
  ValueType predict(const vector<double>& x) override {
    return chf[predictionLeafID(x)];
  }

  vector<double> computePredictions(const Data& new_data);
  pair<vector<double>, vector<double>> computePredictionsCensoring(const Data& new_data);
  void computeCensoringKMExternal(const vector<size_t>& indices, size_t node_index);
  // VIMP prediction for survival trees
  ValueType predictVIMP(const vector<double>& x, size_t feature, mt19937& rng) {
    return chf[predictionLeafIDVIMP(x, feature, rng)];
  }

  // when the censoring Kaplan-Meier estimators have to be populated after fitting
  void resizeKM() {
    KM_censoring.assign(num_nodes, vector<double>());
    KM_censoring_full.assign(num_nodes, vector<double>());
  }

private:
  // quantities of interest to survival trees
  //const vector<double> unique_event_times;      // vector of ordered unique event times for all data
  shared_ptr<vector<double>> unique_event_times;
  size_t num_unique_event_times;                  // number of unique event times
  //vector<size_t> response_event_time_ids;       // the indices of unique_event_times corresponding to the response times
  shared_ptr<vector<size_t>> response_event_time_ids;
  //const vector<size_t> true_event_time_ids;     // the indices of unique_event_times for uncensored times
  shared_ptr<vector<size_t>> true_event_time_ids;
  shared_ptr<vector<double>> censoring_times;
  size_t num_censoring_times;
  bool save_predictions;                    // should predictions (including KM estimators for censoring) be saved during fitting?
  vector<vector<double>> chf;               // the cumulative hazard at the unique_event_times
  vector<vector<double>> KM_censoring;      // the Kaplan-Meier estimate at the unique_event_times for the censoring distribution
  vector<vector<double>> KM_censoring_full; // the Kaplan-Meier estimate at the censoring_times for the censoring distribution
  //const vector<size_t> subset_indices;    // indices for the data (bootstrap) (unnecessary!)

  // temporary quantities used in growing survival trees
  vector<size_t> num_deaths;                // the number of deaths at each event time in current node
  vector<size_t> num_at_risk;               // the number at risk at each event time in a parent (current node)
  /*
  unordered_map<size_t, vector<size_t>> cache_num_deaths;   
  unordered_map<size_t, vector<size_t>> cache_num_at_risk;  
  vector<size_t> num_deaths_left;                    // the number of deaths at each event time in left daughter
  vector<size_t> num_at_risk_left;                   // the number at risk at each event time in left daughter
  vector<size_t> num_deaths_right;                   // ditto for right daughter
  vector<size_t> num_at_risk_right;
  */
  

  // growing survival trees
  void computeSurvivalQuantities(const vector<size_t>& indices, vector<size_t>& deaths, vector<size_t>& at_risk);  // computes the number at risk and the number of deaths at the unique_event_times
  void makeLeaf(size_t node_index);                               // helper function for making a node a leaf
  bool createSplit(size_t node_index) override;                   // returns true if leaf, computes best split
  void computeChf(size_t node_index);                             // computes the cumulative hazard in a terminal node
  void computeCensoringKM(size_t node_index);                     // computes the KM estimator for the censoring distribution in a terminal node
  //void updateSurvivalStats(vector<size_t>& deaths, vector<size_t>& at_risk, const ObsInfo& obs, int sign);
  void computeSurvivalQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                         vector<size_t>& num_at_risk_right, vector<size_t>& num_deaths_right, size_t nsplits_final);
  void bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold);
  /*
  void bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold,
                           vector<size_t>& best_left_indices, vector<size_t>& best_right_indices, vector<size_t>& best_num_deaths_left, 
                           vector<size_t>& best_num_at_risk_left, vector<size_t>& best_num_deaths_right, vector<size_t>& best_num_at_risk_right);  // computes the best split for a chosen continuous feature
  void bestSplitContinuous(size_t node_index, size_t feature,
                           double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, 
                           vector<size_t>& best_right_indices); // computes the best split for a chosen continuous feature
  */
  void bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices); // computes the best split for a chosen categorical feature

  // splitting rules
  double logRank(const vector<size_t>& num_deaths, const vector<size_t>& num_at_risk, 
                 const vector<size_t>& num_deaths_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id = 0);
  double conserve(const vector<size_t>& num_deaths, const vector<size_t>& num_at_risk, 
                  const vector<size_t>& num_deaths_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id = 0);
  double approxLogRank(const vector<size_t>& num_deaths, const vector<size_t>& num_at_risk, 
                       const vector<size_t>& num_deaths_daughter, const vector<size_t>& num_at_risk_daughter, size_t split_id = 0);

  // frees memory from temporary quantities used in growing the tree
  void cleanUpTree() override {
    vector<vector<size_t>>().swap(node_obs);
    vector<vector<size_t>>().swap(holdout_node_obs);
    vector<size_t>().swap(num_deaths);
    vector<size_t>().swap(num_at_risk);
  }
};

vector<size_t> computeResponseEventTimeIDs(const vector<double>& unique_event_times, const vector<double>& times);
vector<size_t> computeTrueEventTimeIDs(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, const vector<double>& ind);
vector<double> computeOutcomes(const NumericMatrix& predictions); // for error computation
vector<double> computeOutcomes(const vector<double>& predictions, size_t num_unique_event_times);
vector<double> computeIPCW(const vector<double>& ind, const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids,
                           const NumericMatrix& KM_cens, const vector<double>& times, const vector<size_t>& last_observed_time_ids = {}, const vector<double>& censoring_times = {});
vector<double> computeBrierScore(const vector<double>& times, const vector<double>& weights, const vector<double>& unique_event_times, const NumericMatrix& KM_pred);
vector<double> computeIPCWCpp(const vector<double>& times, const vector<double>& ind, const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, const vector<double>& KM_cens, vector<size_t> obs_indices = {}, const vector<size_t>& last_observed_time_ids = {}, const vector<double>& censoring_times = {});
vector<double> computeBrierScoreCpp(const vector<double>& times, const vector<double>& weights, const vector<double>& unique_event_times, const vector<double>& KM_pred);
pair<double, double> computeIBS(const vector<double>& bs, const vector<double>& unique_event_times, bool multi_state = false);
vector<double> computeUniqueEventTimes(const vector<double>& times, const vector<size_t>& ind);
vector<double> KaplanMeier(const vector<double>& na, size_t num_estimators = 1);
NumericMatrix KaplanMeier(const NumericMatrix& na);
double computeConcordanceIndex(const vector<double>& outcomes, const vector<double>& times, const vector<double>& ind);
vector<double> computeCensoringKMFromEndpoints(const vector<double>& times, const vector<double>& ind, const vector<double>& censoring_times, const vector<size_t>& indices);
double censoringValueAtTime(const vector<double>& KM_cens, const vector<double>& censoring_times, size_t row, size_t row_length, double time, bool left_limit);
vector<double> selectCensoringAtTimes(const vector<double>& KM_cens, const vector<double>& censoring_times, const vector<double>& output_times, size_t num_obs);

#endif // TREE_SURVIVAL_H

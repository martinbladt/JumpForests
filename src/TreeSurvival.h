#ifndef TREE_SURVIVAL_H
#define TREE_SURVIVAL_H

#include "Tree.h"

class SurvivalTree : public Tree {
public:
  SurvivalTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids,
               shared_ptr<vector<size_t>> true_event_time_ids, vector<size_t> subset_indices, bool save_predictions,
               vector<size_t> estimation_indices = {}, shared_ptr<vector<double>> censoring_times = nullptr,
               shared_ptr<vector<size_t>> removal_time_ids = nullptr);

  const vector<double>& getEventTimes() const {
    return *unique_event_times;
  }
  const vector<size_t>& getTrueEventTimeIDs() const {
    return *true_event_time_ids;
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
  const vector<double>& getCHFOutcomes() const {
    return chf_outcomes;
  }
  const vector<vector<double>>& getSurvivalProbabilities() const {
    return survival_probabilities;
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
  void computeCensoringKMLazy();
  void prepareSurvivalProbabilities();
private:
  // quantities of interest to survival trees
  shared_ptr<vector<double>> unique_event_times;
  size_t num_unique_event_times;                  // number of unique event times
  shared_ptr<vector<size_t>> response_event_time_ids;
  shared_ptr<vector<size_t>> true_event_time_ids;
  shared_ptr<vector<double>> censoring_times;
  size_t num_censoring_times;
  bool save_predictions;                    // should predictions (including KM estimators for censoring) be saved during fitting?
  vector<vector<double>> chf;               // the cumulative hazard at the unique_event_times
  vector<double> chf_outcomes;              // sums of the cumulative hazards used in concordance calculations
  vector<vector<double>> survival_probabilities; // survival probabilities computed lazily for VIMP
  vector<vector<double>> KM_censoring;      // the Kaplan-Meier estimate at the unique_event_times for the censoring distribution
  vector<vector<double>> KM_censoring_full; // the Kaplan-Meier estimate at the censoring_times for the censoring distribution
  vector<size_t> censoring_indices;         // estimation sample retained when censoring estimators are computed lazily

  // temporary quantities used in growing survival trees
  vector<size_t> num_deaths;                    // the number of deaths at each event time in current node
  vector<size_t> num_at_risk;                   // the number at risk at each event time in a parent (current node)
  vector<size_t> num_removed;                   // removals from the risk set in the current node
  vector<size_t> num_obs_right;                 // number of observations in the right node for each split point
  vector<size_t> num_removed_right;             // removals from the risk set in the right node
  vector<size_t> num_deaths_right;              // deaths in the right node
  vector<size_t> num_at_risk_right;             // number at risk in the right node
  vector<size_t> feature_indices;               // feature indices sampled at each node
  shared_ptr<vector<size_t>> removal_time_ids;  // the ID for the first event time where each observation is no longer at risk

  enum class SurvivalSplitRule : uint8_t {
    LogRank,
    Conserve,
    ApproxLogRank
  };
  SurvivalSplitRule splitrule_id = SurvivalSplitRule::LogRank;

  // growing survival trees
  void reserveTreeMemory(size_t num_obs);
  void prepareSurvivalData();
  void computeSurvivalQuantities(const vector<size_t>& indices, vector<size_t>& deaths, vector<size_t>& at_risk);  // computes the number at risk and the number of deaths at the unique_event_times
  void makeLeaf(size_t node_index);                               // helper function for making a node a leaf
  bool createSplit(size_t node_index) override;                   // returns true if leaf, computes best split
  void computeChf(size_t node_index);                             // computes the cumulative hazard in a terminal node
  void computeCensoringKM(size_t node_index);                     // computes the KM estimator for the censoring distribution in a terminal node
  void computeSurvivalQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, size_t nsplits_final);
  double computeSplitValue(const vector<size_t>& deaths_daughter, const vector<size_t>& at_risk_daughter, size_t split_id = 0);
  void bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold);
  void bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold); // computes the best split for a chosen categorical feature

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
    vector<size_t>().swap(num_removed);
    vector<size_t>().swap(num_obs_right);
    vector<size_t>().swap(num_removed_right);
    vector<size_t>().swap(num_deaths_right);
    vector<size_t>().swap(num_at_risk_right);
    vector<size_t>().swap(feature_indices);
    removal_time_ids.reset();
    left_daughters.shrink_to_fit();
    feature_IDs.shrink_to_fit();
    thresholds.shrink_to_fit();
    depths.shrink_to_fit();
    node_sizes.shrink_to_fit();
    chf.shrink_to_fit();
    chf_outcomes.shrink_to_fit();
    KM_censoring.shrink_to_fit();
    KM_censoring_full.shrink_to_fit();
  }
};

vector<size_t> computeRemovalTimeIDs(const Data& data, const vector<double>& unique_event_times,
                                     const vector<size_t>& response_event_time_ids);
vector<size_t> computeResponseEventTimeIDs(const vector<double>& unique_event_times, const vector<double>& times);
vector<size_t> computeTrueEventTimeIDs(const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, const vector<double>& ind);
vector<double> computeOutcomes(const NumericMatrix& predictions); // for error computation
vector<double> computeOutcomes(const vector<double>& predictions, size_t num_unique_event_times);
vector<double> computeIPCW(const vector<double>& ind, const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids,
                           const NumericMatrix& KM_cens, const vector<double>& times, const vector<size_t>& last_observed_time_ids = {},
                           const vector<double>& censoring_times = {}, const vector<double>& censoring_before_event = {});
vector<double> computeBrierScore(const vector<double>& times, const vector<double>& weights, const vector<double>& unique_event_times, const NumericMatrix& KM_pred);
vector<double> computeKLScore(const vector<double>& times, const vector<double>& weights, const vector<double>& unique_event_times, const NumericMatrix& KM_pred);
vector<double> computeIPCWCpp(const vector<double>& times, const vector<double>& ind, const vector<double>& unique_event_times,
                              const vector<size_t>& response_event_time_ids, const vector<double>& KM_cens,
                              vector<size_t> obs_indices = {}, const vector<size_t>& last_observed_time_ids = {},
                              const vector<double>& censoring_times = {}, const vector<double>& censoring_before_event = {});
vector<double> computeBrierScoreCpp(const vector<double>& times, const vector<double>& weights, const vector<double>& unique_event_times, const vector<double>& KM_pred);
vector<double> computeKLScoreCpp(const vector<double>& times, const vector<double>& weights, const vector<double>& unique_event_times, const vector<double>& KM_pred);
double probabilityForLogScore(double probability);
pair<double, double> computeIntegratedScore(const vector<double>& score, const vector<double>& unique_event_times, bool multi_state = false);
vector<double> KaplanMeier(const vector<double>& na, size_t num_estimators = 1);
NumericMatrix KaplanMeier(const NumericMatrix& na);
double computeConcordanceIndex(const vector<double>& outcomes, const vector<double>& times, const vector<double>& ind);
vector<double> computeCensoringKMFromEndpoints(const double* times, const double* ind, const vector<double>& censoring_times, const vector<size_t>& indices);
vector<double> selectCensoringAtTimes(const vector<double>& KM_cens, const vector<double>& censoring_times, const vector<double>& output_times, size_t num_obs);

#endif // TREE_SURVIVAL_H

#ifndef TREE_SURVIVAL_H
#define TREE_SURVIVAL_H

#include "Tree.h"

class SurvivalTree : public Tree {
public:
  SurvivalTree(const vector<double> unique_event_times, const vector<size_t> subset_indices);

  void grow();  // grows the survival tree

  const vector<double> getEventTimes() const {
    return unique_event_times;
  }

  const vector<vector<double>> getCHF() const {
    return chf;
  }

  // prediction for survival trees
  ValueType predict(vector<double> x) override {
    return(chf[predictionLeafID(x)]);
  }

  //ValueType predict(Data* data) override;

private:
  // quantities of interest to survival trees
  const vector<double> unique_event_times;  // vector of ordered unique event times for all data
  size_t num_unique_event_times;            // number of unique event times
  vector<vector<double>> chf;               // the cumulative hazard at the unique_event_times
  const vector<size_t> subset_indices;      // indices for the data (bootstrap)

  // temporary quantities used in growing survival trees
  vector<size_t> num_deaths;          // the number of deaths at each event time
  vector<size_t> num_at_risk;         // the number at risk at each event time

  // growing survival trees
  void computeSurvivalQuantities(vector<size_t> indices);       // computes the number at risk and the number of deaths at the unique_event_times
  bool createSplit(size_t node_index);                          // returns true if leaf, computes best split
  void computeChf(size_t node_index);                           // computes the cumulative hazard in a terminal node
  void bestSplitContinuous(size_t node_index, size_t feature,
                           double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, 
                           vector<size_t>& best_right_indices); // computes the best split for a chosen continuous feature
  void bestSplitCategorical(size_t node_index, size_t feature,
                           double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, 
                           vector<size_t>& best_right_indices); // computes the best split for a chosen categorical feature

  // splitting rules
  double log_rank(vector<size_t> left_indices, vector<size_t> right_indices);  // log-rank splitting
  // add more splitting rules later
};

vector<double> computeUniqueEventTimes(const vector<double>& times, const vector<size_t>& ind);
vector<double> KaplanMeyer(vector<double> na);
double computeConcordanceIndex(const vector<double>& outcomes, const vector<double>& times, const vector<double>& ind);

#endif // TREE_SURVIVAL_H
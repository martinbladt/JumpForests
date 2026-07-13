#ifndef JUMP_FORESTS_H
#define JUMP_FORESTS_H

#include "Data.h"
#include "ForestRegression.h"
#include "ForestClassification.h"
#include "ForestSurvival.h"
#include "ForestMultistate.h"

using namespace std;
using namespace Rcpp;

// growing a tree
List JFCppTree(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, CharacterVector splitrule, 
               bool honest, NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical, NumericVector unique, 
               unsigned int seed, size_t num_event_times);
List JFCppTreeMM(List jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame df_features, unsigned int mtry, unsigned int min_node_size, 
               unsigned int nsplits, CharacterVector splitrule, bool honest, NumericVector feature_indices, LogicalVector categorical, NumericVector unique, 
               unsigned int seed, NumericVector state_weights = {}, size_t num_event_times = 0);

// prediction with trees
void JFCppTreePredict(List& JFTree);
//List JFCppTreePredict(const List& JFTree, DataFrame df);
NumericMatrix JFCppTreePredict(const List& JFTree, DataFrame df, NumericVector feature_indices, 
                               LogicalVector categorical, NumericVector unique);
List JFCppTreePredictCensoring(const List& JFTree, DataFrame df, NumericVector feature_indices, 
                               LogicalVector categorical, NumericVector unique);
List JFCppTreePredictMM(const List& JFTree, DataFrame df, NumericVector feature_indices, 
                                 LogicalVector categorical, NumericVector unique, bool compute_initial, bool compute_censoring);

// error computation with trees
void JFCppTreeErrorRegression(List& JFTree, const vector<double>& response);
void JFCppTreeErrorClassification(List& JFTree, const vector<double>& response);
void JFCppTreeErrorSurvival(List& JFTree, const vector<double>& times, const vector<double>& ind, const vector<double>& unique_event_times, const vector<size_t>& unique_event_time_ids);
//double JFCppErrorSurvival(const NumericMatrix& predictions, NumericVector times, NumericVector ind);
double JFCppErrorSurvival(const NumericMatrix& predictions, const vector<double>& times, const vector<double>& ind);
void JFCppTreeErrorMultistate(List& JFTree, const vector<double>& times, const vector<size_t>& last_observed_time_ids, const vector<double>& ind, 
                              const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, const vector<double>& state_weights);
List JFCppTreeError(const List& JFTree, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique, NumericVector response_indices);
List JFCppTreeErrorMultistate(const List& JFTree, uint8_t max_response_length, uint8_t num_states, List jump_data, DataFrame df_features,
                      NumericVector feature_indices, LogicalVector categorical, NumericVector unique, NumericVector state_weights = {});
// growing a forest
//List JFCppForest(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, CharacterVector splitrule,
//    unsigned int ntrees, bool honest, NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical,
//    NumericVector unique, unsigned int seed, unsigned int nworkers);
List JFCppForest(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, CharacterVector splitrule,
    unsigned int ntrees, bool honest, bool swr, double sample_rate, bool double_bootstrap, NumericVector response_indices, NumericVector feature_indices, 
    LogicalVector categorical, NumericVector unique, unsigned int seed, unsigned int nworkers, bool save_predictions, size_t num_event_times);
List JFCppForestMM(List jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame df_features, unsigned int mtry, unsigned int min_node_size, 
  unsigned int nsplits, CharacterVector splitrule, unsigned int ntrees, bool honest, bool swr, double sample_rate, bool double_bootstrap, NumericVector feature_indices, 
  LogicalVector categorical, NumericVector unique, unsigned int seed, unsigned int nworkers, bool save_predictions, size_t num_event_times);

// predicting with forests
void JFCppForestPredict(List& JFForest, bool compute_censoring = true);
void JFCppForestPredictTraining(List& JFForest);
//NumericVector JFCppForestPredictSingle(const List& JFForest, const NumericVector& x);
NumericMatrix JFCppForestPredict(const List& JFForest, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique);
List JFCppForestPredictCensoring(const List& JFForest, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique);
List JFCppForestPredictMM(const List& JFForest, DataFrame df, NumericVector feature_indices, LogicalVector categorical,
                          NumericVector unique, bool compute_initial);

// error computation with forests
void JFCppForestErrorRegression(List& JFForest, const vector<double>& response);
List JFCppForestErrorRegression(const vector<double>& predictions, const vector<double>& response);
void JFCppForestErrorClassification(List& JFForest, const vector<double>& response);
void JFCppForestErrorSurvival(List& JFForest, const vector<double>& times, const vector<double>& ind, const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids);
List JFCppForestErrorSurvivalExternal(List& JFForest);
List JFCppForestError(const List& JFForest, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique, NumericVector response_indices);

// VIMP for forests
double JFCppForestVIMPFeature(const List& JFForest, CharacterVector feature_name, int feature_seed, CharacterVector method, CharacterVector loss);
List JFCppForestVIMP(List& JFForest, int seed, CharacterVector method, CharacterVector loss);

// misc. functions
List getTreeTable(SEXP tree_sexp);

#endif // JUMP_FORESTS_H

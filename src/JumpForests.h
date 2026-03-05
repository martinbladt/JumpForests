#ifndef JUMP_FORESTS_H
#define JUMP_FORESTS_H

#include "Data.h"
#include "ForestRegression.h"
#include "ForestSurvival.h"
#include "ForestMultistate.h"

using namespace std;
using namespace Rcpp;

// growing a tree
List JFCppTree(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, CharacterVector splitrule, bool honest,
    NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical, NumericVector unique);
List JFCppTreeMM(List jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame df_features, unsigned int mtry, unsigned int min_node_size, 
  unsigned int nsplits, CharacterVector splitrule, bool honest, NumericVector feature_indices, LogicalVector categorical, NumericVector unique, unsigned int seed);

// prediction with trees
void JFCppTreePredict(List& JFTree);
//List JFCppTreePredict(const List& JFTree, DataFrame df);
NumericMatrix JFCppTreePredict(const List& JFTree, DataFrame df, NumericVector feature_indices, 
                               LogicalVector categorical, NumericVector unique);
List JFCppTreePredictMM(const List& JFTree, DataFrame df, NumericVector feature_indices, 
                                 LogicalVector categorical, NumericVector unique);

// error computation with trees
void JFCppTreeErrorRegression(List& JFTree, const vector<double>& response);
void JFCppTreeErrorSurvival(List& JFTree, const vector<double>& times, const vector<double>& ind);
//double JFCppErrorSurvival(const NumericMatrix& predictions, NumericVector times, NumericVector ind);
double JFCppErrorSurvival(const NumericMatrix& predictions, const vector<double>& times, const vector<double>& ind);
List JFCppTreeError(const List& JFTree, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique, NumericVector response_indices);
// growing a forest
//List JFCppForest(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, CharacterVector splitrule,
//    unsigned int ntrees, bool honest, NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical,
//    NumericVector unique, unsigned int seed, unsigned int nworkers);
List JFCppForest(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, CharacterVector splitrule,
    unsigned int ntrees, bool honest, bool swr, double sample_rate, NumericVector response_indices, NumericVector feature_indices, 
    LogicalVector categorical, NumericVector unique, unsigned int seed, unsigned int nworkers);
List JFCppForestMM(List jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame df_features, unsigned int mtry, unsigned int min_node_size, 
  unsigned int nsplits, CharacterVector splitrule, unsigned int ntrees, bool honest, bool swr, double sample_rate, NumericVector feature_indices, 
  LogicalVector categorical, NumericVector unique, unsigned int seed, unsigned int nworkers, bool save_predictions);

// predicting with forests
void JFCppForestPredict(List& JFForest);
//NumericVector JFCppForestPredictSingle(const List& JFForest, const NumericVector& x);
NumericMatrix JFCppForestPredict(const List& JFForest, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique);
List JFCppForestPredictMM(const List& JFForest, DataFrame df, NumericVector feature_indices,
                          LogicalVector categorical, NumericVector unique);

// error computation with forests
void JFCppForestErrorRegression(List& JFForest, const vector<double>& response);
List JFCppForestErrorRegression(const vector<double>& predictions, const vector<double>& response);
void JFCppForestErrorSurvival(List& JFForest, const vector<double>& times, const vector<double>& ind);
List JFCppForestError(const List& JFForest, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique, NumericVector response_indices);

// VIMP for forests
double JFCppForestVIMPFeature(const List& JFForest, CharacterVector feature_name, int feature_seed, CharacterVector method);
List JFCppForestVIMP(List& JFForest, int seed, CharacterVector method);

// misc. functions
List getTreeTable(SEXP tree_sexp);

#endif // JUMP_FORESTS_H

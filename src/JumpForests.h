#ifndef JUMP_FORESTS_H
#define JUMP_FORESTS_H

#include "Data.h"
#include "ForestSurvival.h"

using namespace std;
using namespace Rcpp;

// growing a tree
List JFCppTree(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, CharacterVector splitrule, bool honest,
    NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical, NumericVector unique);

// prediction with trees
void JFCppTreePredict(List& JFTree);
List JFCppTreePredict(const List& JFTree, DataFrame df);

// error computation with trees
void JFCppTreeError(List& JFtree);
void JFCppTreeErrorSurvival(List& JFTree, const vector<double>& times, const vector<double>& ind);
double JFCppErrorSurvival(const NumericMatrix& predictions, NumericVector times, NumericVector ind);
// growing a forest
List JFCppForest(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, CharacterVector splitrule,
    unsigned int ntrees, bool honest, NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical,
    NumericVector unique, unsigned int seed, unsigned int nworkers);

// predicting with forests
void JFCppForestPredict(List& JFForest);
NumericVector JFCppForestPredictSingle(const List& JFForest, const NumericVector& x);
NumericMatrix JFCppForestPredict(const List& JFForest, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique);

// error computation with forests
void JFCppForestErrorSurvival(List& JFForest, const vector<double>& times, const vector<double>& ind);

// VIMP for forests
double JFCppForestVIMPFeature(const List& JFForest, CharacterVector feature_name, int feature_seed, CharacterVector method);
List JFCppForestVIMP(List& JFForest, int seed, CharacterVector method);

// misc. functions
List getTreeTable(SEXP tree_sexp);

#endif // JUMP_FORESTS_H
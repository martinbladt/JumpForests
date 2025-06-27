#ifndef JUMP_FORESTS_H
#define JUMP_FORESTS_H

#include "Data.h"
#include "ForestSurvival.h"

using namespace std;
using namespace Rcpp;

List JFCppTree(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, 
    NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical,
    NumericVector unique);
void JFCppTreePredict(List& JFTree);
List JFCppTreePredict(const List& JFTree, DataFrame df);
List JFCppForest(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, 
    unsigned int ntrees, NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical,
    NumericVector unique, NumericVector subset_indices);
void JFCppForestPredict(List& JFForest);

#endif // JUMP_FORESTS_H
#ifndef HELPER_H
#define HELPER_H

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <iterator>
#include <string>
#include <stdexcept>
#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

// helper functions for pure C++
vector<double> uniqueValues(vector<double> input);
vector<vector<double>> compute2Partitions(const vector<double>& feature_values);
vector<size_t> sampleIndices(const vector<size_t>& global_indices, size_t k, bool with_replacement, mt19937 rng);
double vector_sum(const vector<double>& vec);
size_t vector_sum(const vector<size_t>& vec);
void sum_vectors(vector<double>& result, const vector<double>& add);
void printVector(const vector<double>& vec);
void printVector(const vector<size_t>& vec);
void printVector(const vector<bool>& vec);
vector<bool> computeOOBIndices(const vector<size_t>& indices, unsigned int n);
vector<bool> computeOOBIndicesDouble(const vector<size_t>& grow, const vector<size_t>& holdout, unsigned int n);
pair<vector<size_t>, vector<size_t>> partitionHonesty(const vector<size_t>& indices, mt19937 rng);

// helper functions for Rcpp
NumericMatrix selectColumns(const NumericMatrix& matrix, const vector<size_t>& cols);
vector<double> selectColumns(const vector<double>& matrix, const vector<size_t>& cols, size_t row_length);

#endif // HELPER_H
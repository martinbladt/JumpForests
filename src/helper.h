#ifndef HELPER_H
#define HELPER_H

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <iterator>
#include <string>
#include <stdexcept>
#include <queue>
#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

// the following structs are for thinning out unique event times
struct Node {
    double time;
    int prev = -1;
    int next = -1;
    bool active = true;
    size_t version = 0; // Incremented every time the node's neighbors or value change
};

struct Gap {
    double delta;
    int left_idx;
    int right_idx;
    size_t left_ver;
    size_t right_ver;

    // Min-priority queue: smallest gap at the top
    bool operator>(const Gap& other) const {
        return delta > other.delta;
    }
};

// helper functions for growing trees
vector<double> uniqueValues(vector<double> input);
vector<double> thinUniqueEventTimes(const vector<double>& unique_event_times, double proportion_to_remove);
vector<vector<double>> compute2Partitions(const vector<double>& feature_values);
vector<size_t> sampleIndices(const vector<size_t>& global_indices, size_t k, bool with_replacement, mt19937 rng);

// vector operations
double vector_sum(const vector<double>& vec);
size_t vector_sum(const vector<size_t>& vec);
void sum_vectors(vector<double>& result, const vector<double>& add);

// flattened matrix operations
vector<int> columnSums(const vector<int>& matrix, size_t d);
vector<size_t> transpose(const vector<size_t>& matrix, size_t d);
vector<size_t> addMatrices(const vector<size_t>& matrix1, const vector<size_t>& matrix2, size_t d);
vector<int> subtractMatrices(const vector<size_t>& matrix1, const vector<size_t>& matrix2, size_t d);
void cumulativeMatrixSums(vector<size_t>& acc_matrix, const vector<size_t>& matrix, size_t d);
void cumulativeMatrixSums(vector<size_t>& acc_matrix, const vector<size_t>& matrix, size_t d, size_t num_vectors);

// flattened matrix operations for flattened vectors
vector<size_t> columnSums(const vector<size_t>& matrix, size_t begin, size_t end);
vector<size_t> transpose(const vector<size_t>& matrix, size_t begin, size_t end);
vector<int> subtractMatrices(const vector<size_t>& matrix1, size_t begin, size_t end, const vector<size_t>& matrix2);

// printing functions
void printVector(const vector<double>& vec);
void printVector(const vector<size_t>& vec, size_t stride_length = 0);
void printVector(const vector<uint8_t>& vec);
void printVector(const vector<bool>& vec);

// helper functions for forests
vector<bool> computeOOBIndices(const vector<size_t>& indices, size_t n);
vector<bool> computeOOBIndicesDouble(const vector<size_t>& grow, const vector<size_t>& holdout, size_t n);
pair<vector<size_t>, vector<size_t>> partitionHonesty(const vector<size_t>& indices, mt19937 rng);

// helper functions for Rcpp
NumericMatrix selectColumns(const NumericMatrix& matrix, const vector<size_t>& cols);
vector<double> selectColumns(const vector<double>& matrix, const vector<size_t>& cols, size_t row_length);

#endif // HELPER_H
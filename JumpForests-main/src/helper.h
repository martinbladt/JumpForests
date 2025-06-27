#ifndef HELPER_H
#define HELPER_H

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <iterator>
#include <string>
#include <stdexcept>

using namespace std;

vector<double> uniqueValues(vector<double> input);
vector<vector<double>> compute2Partitions(const vector<double>& feature_values);
vector<size_t> sampleIndices(size_t n, size_t k, bool with_replacement, mt19937 generator);
void sum_vectors(vector<double>& result, const vector<double>& add);
void printVector(const vector<double>& vec);
void printVector(const vector<size_t>& vec);
void printVector(const vector<bool>& vec);
vector<bool> computeOOBIndices(const vector<size_t>& indices, unsigned int n);

#endif // HELPER_H
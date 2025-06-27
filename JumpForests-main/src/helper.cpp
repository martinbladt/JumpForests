/*

general helper functions

*/

#include "helper.h"

// returns a vector of the unique values from the vector of doubles 'input'
vector<double> uniqueValues(vector<double> input) {
    sort(input.begin(), input.end());
    auto last = unique(input.begin(), input.end());
    input.erase(last, input.end());   // save memory
    return(input);
}

// computes all 2-partitions of the vector of doubles feature_values and puts all subsets in one vector
// used for determining splits on a categorical variable
vector<vector<double>> compute2Partitions(const vector<double>& feature_values) {
    vector<vector<double>> result;
    size_t n = feature_values.size();

    // Loop over all subsets except the empty set and the whole set
    for (size_t i = 1; i < (1 << n) - 1; ++i) {
        vector<double> subset1, subset2;
        for (size_t j = 0; j < n; ++j) {
            if (i & (1 << j)) {
                subset1.push_back(feature_values[j]);
            } else {
                subset2.push_back(feature_values[j]);
            }
        }

        // To avoid duplicate partitions like {A,B} and {B,A}, ensure subset1 < subset2
        if (subset1 < subset2) {
            result.emplace_back(subset1);
            result.emplace_back(subset2);
        }
    }

    return result;
}

// samples k indices from {0, 1, ..., n - 1} with or without replacement
vector<size_t> sampleIndices(size_t n, size_t k, bool with_replacement, mt19937 generator) {
    if (k > n && with_replacement == false) {
        throw invalid_argument("Cannot sample without replacement when k > n");
    }

    // make vector of indices
    vector<size_t> vec(n);
    for (int i = 0; i < n; ++i) {
        vec[i] = i;
    }

    vector<size_t> result;

    if (with_replacement) {
        uniform_int_distribution<> dist(0, n - 1);
        for (int i = 0; i < k; ++i) {
            int index = dist(generator);
            result.push_back(vec[index]);
        }
    }
    else {
        sample(vec.begin(), vec.end(), back_inserter(result), k, generator);
    }

    return(result);
}

// adds the vector add to the vector result and modifies it (result must be at least as large as add)
void sum_vectors(vector<double>& result, const vector<double>& add) {
    for (int i = 0; i < add.size(); ++i) {
        result[i] += add[i];
    }
}

void printVector(const vector<double>& vec) {
    for (int i = 0; i < vec.size() - 1; ++i) {
        cout << vec[i] << ", ";
    }
    cout << vec[vec.size() - 1] << endl;
}

void printVector(const vector<size_t>& vec) {
    for (int i = 0; i < vec.size() - 1; ++i) {
        cout << vec[i] << ", ";
    }
    cout << vec[vec.size() - 1] << endl;
}

void printVector(const vector<bool>& vec) {
    for (int i = 0; i < vec.size() - 1; ++i) {
        cout << vec[i] << ", ";
    }
    cout << vec[vec.size() - 1] << endl;
}

// indices: which observations are used to fit, n: the total number of observations
vector<bool> computeOOBIndices(const vector<size_t>& indices, unsigned int n) {
    vector<bool> oob(n, true);
    for (size_t i : indices) {
        oob[i] = false;
    }
    return(oob);
}
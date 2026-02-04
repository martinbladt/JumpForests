/*

general helper functions

*/

#include "helper.h"

// returns a vector of the sorted unique values from the vector of doubles 'input'
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

// samples k indices from global_indices with or without replacement
vector<size_t> sampleIndices(const vector<size_t>& global_indices, size_t k, bool with_replacement, mt19937 rng) {    
    size_t n = global_indices.size();
    if (k > n && with_replacement == false) {
        return(global_indices);
    }

    vector<size_t> result;
    if (with_replacement) {
        uniform_int_distribution<> dist(0, n - 1);
        for (int i = 0; i < k; ++i) {
            int index = dist(rng);
            result.push_back(global_indices[index]);
        }
    }
    else {
        sample(global_indices.begin(), global_indices.end(), back_inserter(result), k, rng);
    }

    return(result);
}

// returns the sum of elements in a vector
double vector_sum(const vector<double>& vec) {
    double sum = 0;
    for (double d : vec) {
        sum += d;
    }
    return sum;
}

size_t vector_sum(const vector<size_t>& vec) {
    size_t sum = 0;
    for (size_t d : vec) {
        sum += d;
    }
    return sum;
}

// adds the vector add to the vector result and modifies it (result must be at least as large as add)
void sum_vectors(vector<double>& result, const vector<double>& add) {
    for (int i = 0; i < add.size(); ++i) {
        result[i] += add[i];
    }
}

// computes the vector of column sums of a flattened d x d matrix (by row)
vector<int> columnSums(const vector<int>& matrix, size_t d) {
    vector<int> result(d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            result[i] += matrix[i + d * j];
        }
    }
    return result;
}

// computes the transpose of a flattened d x d matrix (by row)
vector<size_t> transpose(const vector<size_t>& matrix, size_t d) {
    vector<size_t> result(d * d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            result[j * d + i] = matrix[i * d + j];
        }
    }
    return result;
}

// adds two flattened d x d matrices
vector<size_t> addMatrices(const vector<size_t>& matrix1, const vector<size_t>& matrix2, size_t d) {
    vector<size_t> result(d * d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            result[i * d + j] = matrix1[i * d + j] + matrix2[i * d + j];
        }
    }
    return result;
}

// subtracts two flattened d x d matrices
vector<int> subtractMatrices(const vector<size_t>& matrix1, const vector<size_t>& matrix2, size_t d) {
    vector<int> result(d * d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            result[i * d + j] = matrix1[i * d + j] - matrix2[i * d + j];
        }
    }
    return result;
}

// computes the vector of column sums of a flattened matrix in a flattened vector of matrices
vector<size_t> columnSums(const vector<size_t>& matrix, size_t begin, size_t end) {
    size_t dim = end - begin + 1;
    size_t d = static_cast<size_t>(sqrt(dim));
    vector<size_t> result(d, 0);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            result[i] += matrix[begin + i + d * j];
        }
    }
    return result;
}

// computes the transpose of a flattened matrix in a flattened vector of matrices
vector<size_t> transpose(const vector<size_t>& matrix, size_t begin, size_t end) {
    size_t dim = end - begin + 1;
    size_t d = static_cast<size_t>(sqrt(dim));
    vector<size_t> result(dim);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            result[j * d + i] = matrix[begin + i * d + j];
        }
    }
    return result;
}

// subtracts two flattened d x d matrices where the first matrix is selected from a flattened vector of matrices
vector<int> subtractMatrices(const vector<size_t>& matrix1, size_t begin, size_t end, const vector<size_t>& matrix2) {
    size_t dim = end - begin + 1;
    size_t d = static_cast<size_t>(sqrt(dim));
    vector<int> result(dim);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            result[i * d + j] = matrix1[begin + i * d + j] - matrix2[i * d + j];
        }
    }
    return result;
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

void printVector(const vector<uint8_t>& vec) {
    for (int i = 0; i < vec.size() - 1; ++i) {
        cout << static_cast<size_t>(vec[i]) << ", ";
    }
    cout << static_cast<size_t>(vec[vec.size() - 1]) << endl;
}

void printVector(const vector<bool>& vec) {
    for (int i = 0; i < vec.size() - 1; ++i) {
        cout << vec[i] << ", ";
    }
    cout << vec[vec.size() - 1] << endl;
}

void printVector(const vector<string>& vec) {
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
    return oob;
}

// computes OOB indices when data has been partitioned as for honest trees
vector<bool> computeOOBIndicesDouble(const vector<size_t>& grow, const vector<size_t>& holdout, unsigned int n) {
    vector<bool> oob(n, true);
    for (size_t i : grow) {
        oob[i] = false;
    }
    for (size_t i : holdout) {
        oob[i] = true;
    }
    return oob;
}

// used for honest trees when partitioning into a grow subset and a holdout subset
pair<vector<size_t>, vector<size_t>> partitionHonesty(const vector<size_t>& indices, mt19937 rng) {
    size_t n = indices.size();

    // shuffle indices
    vector<size_t> shuffled = indices;
    shuffle(shuffled.begin(), shuffled.end(), rng);


    vector<size_t> grow(shuffled.begin(), shuffled.begin() + n/2);
    vector<size_t> holdout(shuffled.begin() + n/2, shuffled.end());

    return {grow, holdout};
}

// for selecting specific columns from a NumericMatrix (used to filter)
NumericMatrix selectColumns(const NumericMatrix& matrix, const vector<size_t>& cols) {
    size_t n_rows = matrix.nrow();
    size_t n_cols = cols.size();

    NumericMatrix res(n_rows, n_cols);
    for (size_t i = 0; i < n_cols; ++i) {
        res(_, i) = matrix(_, cols[i]);
    }
    return res;
}

// for selecting specific columns from a matrix in flattened vector form (with row_length entries per row)
vector<double> selectColumns(const vector<double>& matrix, const vector<size_t>& cols, size_t row_length) {
    size_t n_rows = matrix.size() / row_length;
    size_t n_cols = cols.size();

    vector<double> res(n_rows * n_cols);
    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = 0; j < n_cols; ++j) {
            res[i * n_cols + j] = matrix[i * row_length + cols[j]];
        }
    }
    return res;
}
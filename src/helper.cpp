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

// used to thin out the UniqueEventTimes for survival and multi-states by a proportion to be removed (implemented by Gemini)
vector<double> thinUniqueEventTimes(const vector<double>& unique_event_times, size_t target_size = 0) {
    size_t n = unique_event_times.size();
    if (target_size >= n) return unique_event_times;

    vector<Node> nodes(n);
    priority_queue<Gap, std::vector<Gap>, std::greater<Gap>> pq;

    // initialise nodes and initial gaps
    for (size_t i = 0; i < n; ++i) {
        nodes[i].time = unique_event_times[i];
        nodes[i].prev = (i == 0) ? -1 : (int)i - 1;
        nodes[i].next = (i == n - 1) ? -1 : (int)i + 1;
        nodes[i].version = 0;
        
        if (i > 0) {
            pq.push({unique_event_times[i] - unique_event_times[i-1], (int)i-1, (int)i, 0, 0});
        }
    }

    size_t current_count = n;

    // merge until target size is reached
    while (current_count > target_size && !pq.empty()) {
        Gap top = pq.top();
        pq.pop();

        Node& left = nodes[top.left_idx];
        Node& right = nodes[top.right_idx];

        // if the nodes were updated/removed since this gap was queued, skip
        if (!left.active || !right.active || left.version != top.left_ver || right.version != top.right_ver) {
            continue;
        }

        // update left node to the average (old)
        left.time = (left.time + right.time) / 2.0;
        // update left node to the largest time (for testing so far)
        left.version++;

        // remove right node from the chain
        right.active = false;
        int r_next_idx = right.next;
        left.next = r_next_idx;
        
        if (r_next_idx != -1) {
            nodes[r_next_idx].prev = top.left_idx;
            nodes[r_next_idx].version++;
        }

        // update the left node's previous neighbor to increment version 
        // (because its gap with 'left' has changed)
        if (left.prev != -1) {
            nodes[left.prev].version++;
            pq.push({left.time - nodes[left.prev].time, 
                     left.prev, top.left_idx, 
                     nodes[left.prev].version, left.version});
        }

        // push the new gap formed to the right
        if (left.next != -1) {
            pq.push({nodes[left.next].time - left.time, 
                     top.left_idx, left.next, 
                     left.version, nodes[left.next].version});
        }

        current_count--;
    }

    // collect remaining points
    std::vector<double> result;
    result.reserve(current_count);
    for (const auto& node : nodes) {
        if (node.active) {
            result.push_back(node.time);
        }
    }

    // temporary to ensure 0 is always included
    result[0] = 0;
    
    result.shrink_to_fit();
    return result;
}

// samples k indices from global_indices with or without replacement
vector<size_t> sampleIndices(const vector<size_t>& global_indices, size_t k, bool with_replacement, mt19937& rng) {    
    size_t n = global_indices.size();
    if (k > n && with_replacement == false) {
        return(global_indices);
    }

    vector<size_t> result;
    result.reserve(k);
    if (with_replacement) {
        uniform_int_distribution<> dist(0, n - 1);
        for (size_t i = 0; i < k; ++i) {
            size_t index = dist(rng);
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

// takes a vector of flattened d x d matrices and computes the cumulative sums of these (used for error computation, not fitting)
// difference is that computations take place in t and not t-
void cumulativeMatrixSumsNoDelay(vector<size_t>& acc_matrix, const vector<size_t>& matrix, size_t d) {
    size_t dim = d * d;
    size_t num_matrices = matrix.size() / dim;

    // initialise the first matrix
    for (size_t j = 0; j < d; ++j) {
        for (size_t k = 0; k < d; ++k) {
            acc_matrix[j * d + k] =  matrix[j * d + k]; // = 0 originally
        }
    }
    // now update the matrix
    for (size_t i = 1; i < num_matrices; ++i) {
        for (size_t j = 0; j < d; ++j) {
            for (size_t k = 0; k < d; ++k) {
                size_t index = i * dim + j * d + k;
                acc_matrix[index] = acc_matrix[index - dim] + matrix[index];
                //acc_matrix[i * dim + j * d + k] = acc_matrix[(i - 1) * dim + j * d + k] + matrix[i * dim + j * d + k];
            }
        }
    }
}

void printVector(const vector<double>& vec) {
    for (size_t i = 0; i < vec.size() - 1; ++i) {
        Rcout << vec[i] << ", ";
    }
    Rcout << vec[vec.size() - 1] << endl;
}

void printVector(const vector<size_t>& vec, size_t stride_length) {
    if (stride_length == 0) {
        for (size_t i = 0; i < vec.size() - 1; ++i) {
            Rcout << vec[i] << ", ";
        }
        Rcout << vec[vec.size() - 1] << endl;
    } else {
        size_t num_vectors = vec.size() / stride_length;
        for (size_t i = 0; i < num_vectors; ++i) {
            Rcout << i << ": " "{";
            for (size_t j = 0; j < stride_length - 1; ++j) {
                Rcout << vec[i * stride_length + j] << ", ";
            }
            Rcout << vec[(i + 1) * stride_length - 1] << "}" << ", ";
        }
        Rcout << endl;
    }
}

void printVector(const vector<uint8_t>& vec) {
    for (size_t i = 0; i < vec.size() - 1; ++i) {
        Rcout << static_cast<size_t>(vec[i]) << ", ";
    }
    Rcout << static_cast<size_t>(vec[vec.size() - 1]) << endl;
}

void printVector(const vector<bool>& vec) {
    for (size_t i = 0; i < vec.size() - 1; ++i) {
        Rcout << vec[i] << ", ";
    }
    Rcout << vec[vec.size() - 1] << endl;
}

// indices: which observations are used to fit, n: the total number of observations
vector<bool> computeOOBIndices(const vector<size_t>& indices, size_t n) {
    vector<bool> oob(n, true);
    for (size_t i : indices) {
        oob[i] = false;
    }
    return oob;
}

// computes OOB indices when data has been partitioned as for honest trees
vector<bool> computeOOBIndicesDouble(const vector<size_t>& grow, const vector<size_t>& holdout, size_t n) {
    vector<bool> oob(n, true);
    for (size_t i : grow) {
        oob[i] = false;
    }
    for (size_t i : holdout) {
        oob[i] = false;
    }
    return oob;
}

// used for honest trees when partitioning into a grow subset and a holdout subset
pair<vector<size_t>, vector<size_t>> partitionHonesty(const vector<size_t>& indices, mt19937& rng) {
    size_t n = indices.size();

    // shuffle indices
    vector<size_t> shuffled = indices;
    shuffle(shuffled.begin(), shuffled.end(), rng);


    vector<size_t> grow(shuffled.begin(), shuffled.begin() + n/2);
    vector<size_t> holdout(shuffled.begin() + n/2, shuffled.end());

    return {grow, holdout};
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

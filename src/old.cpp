#include "Data.h"
#include "TreeSurvival.h"

using namespace std;
using namespace Rcpp;

// all code below is strictly for testing and do not go into the final library
//-------------------------------------------------------------------------------------

// [[Rcpp::export]]
List testList() {
  List result;
  vector<double> test {1, 2, 3.0};
  result.push_back(2, "number");
  result.push_back(test, "vector<double>");
  return result;
}

// [[Rcpp::export]]
int extract_column(DataFrame data, size_t index) {
  NumericVector col = data[index];
  Rcout << "The given vector is: " << col << endl;

  // the line below is not an issue
  vector<double> output(col.begin(), col.end());

  return output.size();
}

// [[Rcpp::export]]
List rcpp_hello_world() {

    CharacterVector x = CharacterVector::create( "foo", "bar" )  ;
    NumericVector y   = NumericVector::create( 0.0, 1.0 ) ;
    List z            = List::create( x, y ) ;

    return z ;
}


// [[Rcpp::export]]
double square_cpp(double x) {
    return x * x;
}

// [[Rcpp::export]]
NumericMatrix timesTwoMatrix(NumericMatrix mat) {
  int nrow = mat.nrow();
  int ncol = mat.ncol();

  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      mat(i, j) = 2 * mat(i, j);
    }
  }

  return mat;
}

// [[Rcpp::export]]
void sampler(int n, int k) {
  vector<size_t> vec(n);
  vector<size_t> out;
  for (int i = 0; i < n; ++i) {
    vec[i] = i;
  }
  sample(vec.begin(), vec.end(), back_inserter(out), k,
                std::mt19937 {random_device{}()});
  
  for (int i = 0; i < k; ++i) {
    Rcout << out[i] << ", ";
  }
  Rcout << endl;
}

// [[Rcpp::export]]
void silly_sampler(int k) {
  vector<double> vec {1, 0.13, 0.145, -0.16, 0.15};
  vector<double> out;
  sample(vec.begin(), vec.end(), back_inserter(out), k,
                std::mt19937 {random_device{}()});
  for (int i = 0; i < k; ++i) {
    Rcout << out[i] << ", ";
  }
  Rcout << endl;
}

// [[Rcpp::export]]
void sampler_wrapper(int n, int k, bool wr) {
  mt19937 generator = mt19937 {random_device{}()};
  vector<size_t> global_indices(n);
  for (size_t i = 0; i < n; ++i) {
    global_indices[i] = i;
  }
  vector<size_t> output = sampleIndices(global_indices, k, wr, generator);
  for (size_t o: output) {
    Rcout << o << ", ";
  }
  Rcout << endl;
}

/*

Below are old versions of different functions related to splitting

*/

/*
// consider optimising by sorting the thresholds (this works but is slow!)
void SurvivalTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices) {
    // sample split points
    vector<double> split_points;
    vector<double> potential_split_points = (*data).getValues(node_obs[node_index], feature);

    sample(potential_split_points.begin(), potential_split_points.end(), back_inserter(split_points), nsplits,
            random_number_generator);

    // now consider each possible threshold
    for (double c : split_points) {
        vector<size_t> left_indices;
        vector<size_t> right_indices;
        // perform the split
        for (int j : node_obs[node_index]) {
            if ((*data).get_x(j, feature) <= c) {
                left_indices.push_back(j);
            } else {
                right_indices.push_back(j);
            }
        }

        // jump to next threshold if split is illegal
        if (left_indices.size() < min_node_size || right_indices.size() < min_node_size) {
            continue;
        }
            
        double split_val = log_rank(left_indices, right_indices);
        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_left_indices = std::move(left_indices);
            best_right_indices = std::move(right_indices);
            best_feature = feature;
            // should maybe be changed to an average to improve stability
            best_threshold = {c};
        }
    }
}
*/

/* old SurvivalTree::computeSurvivalQuantities
void SurvivalTree::computeSurvivalQuantities(vector<size_t> indices) {
    // start by choosing the relevant observations
    vector<double> times(indices.size());
    vector<size_t> indicators(indices.size());
    for (int i = 0; i < indices.size(); ++i) {
        times[i] = (*data).get_y(indices[i], 0);
        indicators[i] = (*data).get_y(indices[i], 1);
    }

    // compute the number at risk and the number of deaths at the unique event times
    for (int i = 0; i < num_unique_event_times; ++i) {
        for (int j = 0; j < times.size(); ++j) {
            if (times[j] >= unique_event_times[i]) {
                num_at_risk[i] += 1;
            }
            if (times[j] == unique_event_times[i] && indicators[j] == 1) {
                num_deaths[i] += 1;
            }
        }
    }
}
*/

/* second attempt (also works and is faster)
void SurvivalTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices) {
    
    vector<size_t> left_indices;
    vector<size_t> right_indices;
    vector<size_t> current_node_obs = node_obs[node_index];
    vector<double> x = data->get_x_col(feature);
    // to prevent many reallocations
    left_indices.reserve(current_node_obs.size());
    right_indices.reserve(current_node_obs.size());

    // sample split points
    vector<double> split_points;
    vector<double> potential_split_points = (*data).getValues(current_node_obs, feature);

    sample(potential_split_points.begin(), potential_split_points.end(), back_inserter(split_points), nsplits,
            random_number_generator);
    
    // sort split points
    sort(split_points.begin(), split_points.end());

    // initial sweep for the first threshold
    const double first_split_point = split_points[0];
    for (size_t i : current_node_obs) {
        if (x[i] <= first_split_point) {
            left_indices.push_back(i);
        } else {
            right_indices.push_back(i);
        }
    }

    // if the initial split is legal, compute the split values
    if (left_indices.size() >= min_node_size && right_indices.size() >= min_node_size) {
        double split_val = log_rank(left_indices, right_indices);
        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_left_indices = left_indices;
            best_right_indices = right_indices;
            best_feature = feature;
            best_threshold = {first_split_point};
        }
    }
    // the number of observations in the right node is decreasing, so if one split is illegal
    // so is every subsequent split
    if (right_indices.size() < min_node_size) {
        return;
    }

    for (size_t t_idx = 1; t_idx < split_points.size(); ++t_idx) {
        const double current_split_point = split_points[t_idx];
        // Use the "erase-remove idiom". It's efficient as it avoids new allocations
        // and moves elements in-place.
        auto new_end = std::remove_if(right_indices.begin(), right_indices.end(),
            [&](size_t data_idx) {
                // Check if this index should now move from right to left
                if (x[data_idx] <= current_split_point) {
                    left_indices.push_back(data_idx); // Add to the left group
                    return true; // Return true to signal it should be "removed" from right
                }
                return false; // Keep it in the right group
            });

        // Erase the "removed" elements from the end of the right_indices vector
        right_indices.erase(new_end, right_indices.end());

        // check whether split is legal. if the right node is too small,
        // every subsequent split is illegal
        if (left_indices.size() < min_node_size) {
            continue;
        }
        if (right_indices.size() < min_node_size) {
            break;
        }

        double split_val = log_rank(left_indices, right_indices);
        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_left_indices = left_indices;
            best_right_indices = right_indices;
            best_feature = feature;
            best_threshold = {(current_split_point + split_points[t_idx - 1])/2};
        }
    }
}
*/

/* another attempt at bestSplitContinuous
void SurvivalTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices) {
    // sample split points
    vector<double> split_points;
    vector<double> potential_split_points = (*data).getValues(node_obs[node_index], feature);
    sample(potential_split_points.begin(), potential_split_points.end(), back_inserter(split_points), nsplits,
            random_number_generator);

    // sort sampled split points
    sort(split_points.begin(), split_points.end());

    vector<size_t> left_counts(nsplits, 0);
    vector<size_t> right_counts(nsplits, 0);
    const vector<size_t>& obs = node_obs[node_index];

    for (int i : obs) {
        double val = data->get_x(i, feature);

        // use binary search to determine the thresholds where the current observation is <=
        auto it = upper_bound(split_points.begin(), split_points.end(), value);
        size_t pos = static_cast<size_t>(distance(split_points.begin(), it));

        for (int t = 0; t < pos; ++t) left_counts[t]++;
        for (int t = pos; t < nsplits; ++t) right_counts[t]++;
    }
}
*/

/* old bestSplitCategorical
void SurvivalTree::bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices) {
    // create all 2-partitions of the values of the given feature
    vector<double> feature_values = uniqueValues((*data).getValues(node_obs[node_index], feature));

    // if only one unique value of the feature, skip the split
    if (feature_values.size() < 2) {
        return;
    }

    // compute all 2-partitions and consider nsplits of them at random
    vector<vector<double>> partitions = compute2Partitions(feature_values);
    vector<size_t> partition_IDs = sampleIndices(partitions.size(), nsplits, false, random_number_generator);

    // now consider the chosen subsets
    for (size_t i : partition_IDs) {
        vector<size_t> left_indices;
        vector<size_t> right_indices;

        // for O(1) expected time lookups in the following loop
        unordered_set<double> s(partitions[i].begin(), partitions[i].end());

        // perform the split
        for (size_t j : node_obs[node_index]) {
            if (s.count((*data).get_x(j, feature)) > 0) {
                left_indices.push_back(j);
            } else {
                right_indices.push_back(j);
            }
        }
        // jump to next threshold if split is illegal
        if (left_indices.size() < min_node_size || right_indices.size() < min_node_size) {
            continue;
        }
            
        double split_val = log_rank(left_indices, right_indices);
        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_left_indices = std::move(left_indices);
            best_right_indices = std::move(right_indices);
            best_feature = feature;
            best_threshold = partitions[i];
        }
    }
}
*/

/*
// helper function for updating survival info between different splits on the same feature
void SurvivalTree::updateSurvivalStats(vector<size_t>& deaths, vector<size_t>& at_risk, const ObsInfo& obs, int sign) {

    // update death counts
    if (obs.indicator == 1) {
        // find the position of this observation's event time.
        auto it = lower_bound(unique_event_times.begin(), unique_event_times.end(), obs.time);
        if (it != unique_event_times.end() && *it == obs.time) {
            size_t pos = distance(unique_event_times.begin(), it);
            deaths[pos] += sign;
        }
    }
    // update at risk counts
    // an individual is at risk for all event times <= their own observation time.
    // find where obs.time fits in the unique_event_times.
    auto it = upper_bound(unique_event_times.begin(), unique_event_times.end(), obs.time);
    size_t end_pos = distance(unique_event_times.begin(), it);
    
    // This observation contributes to the at-risk count for all event times
    // from the beginning up to (but not including) end_pos.
    for (size_t i = 0; i < end_pos; ++i) {
        at_risk[i] += sign;
    }
}
*/

/*
// yet another attempt at bestSplitContinuous using a lot of input from Gemini
// comments: 1) there seems to be a bug somewhere. tree complexity suddenly fell too much. 2) it is actually of the same order in terms of speed as
// my own somewhat naive implementations. not sure why, may be because of too many vector copies and inefficient mix of fetching and recomputing

void SurvivalTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold,
                           vector<size_t>& best_left_indices, vector<size_t>& best_right_indices, vector<size_t>& best_num_deaths_left, 
                           vector<size_t>& best_num_at_risk_left, vector<size_t>& best_num_deaths_right, vector<size_t>& best_num_at_risk_right) {
    vector<size_t> current_node_obs = node_obs[node_index];
    size_t n = current_node_obs.size();
    //double best_split_val_old = best_split_val;
    
    // sort observations based on the feature value for efficient updating of survival information
    vector<ObsInfo> obs_data(n);
    for (size_t i = 0; i < n; ++i) {
        size_t obs_id = current_node_obs[i];
        obs_data[i] = {obs_id, data->get_x(obs_id, feature), data->get_y(obs_id, 0), (size_t)data->get_y(obs_id, 1)};
    }
    sort(obs_data.begin(), obs_data.end(), 
         [](const ObsInfo& a, const ObsInfo& b) {
             return a.feature_value < b.feature_value;
         });

    // choose valid candidate split points
    vector<size_t> split_indices_candidates;
    for (size_t i = min_node_size - 1; i < n - min_node_size; ++i) {
        if (obs_data[i].feature_value < obs_data[i + 1].feature_value) {
            split_indices_candidates.push_back(i);
        }
    }
    // should it occur that no split point is valid, stop early
    if (split_indices_candidates.empty()) {
        return;
    }

    // sample split points among the candidate split points
    vector<size_t> split_indices;
    if (split_indices_candidates.size() <= nsplits) {
        split_indices = split_indices_candidates;
    } else {
        sample(split_indices_candidates.begin(), split_indices_candidates.end(), back_inserter(split_indices), nsplits, random_number_generator);
    }
    // sort the sampled indices to ensure a single forward pass in the main loop
    sort(split_indices.begin(), split_indices.end());
    
    // the left node stats start by being zero
    fill(num_at_risk_left.begin(), num_at_risk_left.end(), 0);
    fill(num_deaths_left.begin(), num_deaths_left.end(), 0);
    size_t n_left = 0;

    // the right node stats are initially set to that of the parent's
    num_at_risk_right = num_at_risk;
    num_deaths_right = num_deaths;
    size_t n_right = n;

    size_t loop_end = split_indices.empty() ? 0 : split_indices.back() + 1; // we need only loop up to the last chosen split point
    size_t eval_ptr = 0;                                                    // pointer to next index in split_points
    double best_local_split_val = -1.0;
    size_t best_local_split_id = 0;

    for (size_t i = 0; i < loop_end; ++i) {
        // update survival information incrementally instead of computing everything from scratch
        const ObsInfo& obs_to_move = obs_data[i];
        updateSurvivalStats(num_deaths_left, num_at_risk_left, obs_to_move, 1);
        n_left++;
        updateSurvivalStats(num_deaths_right, num_at_risk_right, obs_to_move, -1);
        n_right--;

        if (n_left < min_node_size) {
            continue;
        }
        // right_n is decreasing, so if too few observations in the right node, all subsequent splits are illegal
        if (n_right < min_node_size) {
            break;
        }

        // in case of ties, move as a block to avoid redundant split computations
        if (obs_data[i].feature_value == obs_data[i + 1].feature_value) {
            continue;
        }

        if (eval_ptr < split_indices.size() && i == split_indices[eval_ptr]) {
            double split_val = log_rank();
            if (split_val > best_local_split_val) {
                best_local_split_val = split_val;
                best_local_split_id = i;
                // update the final survival quantities for the nodes
                best_num_deaths_left = num_deaths_left;
                best_num_at_risk_left = num_at_risk_left;
                best_num_deaths_right = num_deaths_right;
                best_num_at_risk_right = num_at_risk_right;
            }
            eval_ptr++;
        }
    }

    // if we have found a better split than previously, update best split and reconstruct the best indices
    if (best_local_split_val > best_split_val) {
        best_split_val = best_local_split_val;
        best_feature = feature;
        // choose the threshold in the middle for stability
        best_threshold = {(obs_data[best_local_split_id].feature_value + obs_data[best_local_split_id + 1].feature_value) / 2.0};

        best_left_indices.resize(best_local_split_id + 1);
        best_right_indices.resize(n - (best_local_split_id + 1));

        for (size_t i = 0; i <= best_local_split_id; ++i) {
            best_left_indices[i] = obs_data[i].original_index;
        }
        for (size_t i = best_local_split_id + 1; i < n; ++i) {
            best_right_indices[i - (best_local_split_id + 1)] = obs_data[i].original_index;
        }

        // Re-compute the survival stats for the children just once.
        // This is much faster than copying in the loop - Gemini
        //(I am very sceptical that this should work better, but let's see)
        //computeSurvivalQuantities(best_left_indices, best_num_deaths_left, best_num_at_risk_left);
        //computeSurvivalQuantities(best_right_indices, best_num_deaths_right, best_num_at_risk_right);
        
        //best_left_indices.clear();
        //best_right_indices.clear();
        //double final_threshold = best_threshold[0];
        //for (const auto& obs : obs_data) {
        //    if (obs.feature_value <= final_threshold) {
        //        best_left_indices.push_back(obs.original_index);
        //    } else {
        //        best_right_indices.push_back(obs.original_index);
        //    }
        //}
        
        
    }
}

*/

/* old createSplit function
bool SurvivalTree::createSplit(size_t node_index) {
    vector<size_t> current_obs = node_obs[node_index];
    // if we are in the root node, compute survival quantities from scratch
    if (node_index == 0) {
        
    } 
    // if not in the root node, fetch results of earlier computations
    else {
        this->num_at_risk = cache_num_at_risk[node_index];
        this->num_deaths = cache_num_deaths[node_index];
    }

    // if no split is possible, make the node a leaf
    if (current_obs.size() < 2 * min_node_size) {
        makeLeaf(node_index);
        return true;
    }
    
    double best_split_val = -1.0;
    size_t best_feature = 0;
    vector<double> best_threshold;
    vector<size_t> best_left_indices;
    vector<size_t> best_right_indices;

    // create vectors to hold survival quantities of the two children
    vector<size_t> best_num_at_risk_left, best_num_deaths_left;
    vector<size_t> best_num_at_risk_right, best_num_deaths_right;

    // sample mtry features
    size_t num_features =  data->getNumberOfFeatures();
    vector<size_t> sampled_features = sampleIndices(num_features, mtry, false, random_number_generator);

    // now consider each of the sampled features
    for (size_t i : sampled_features) {
        if (data->getCategorical()[i]) {
            bestSplitCategorical(node_index, i, best_split_val, best_feature, best_threshold, best_left_indices, best_right_indices);
        }
        else {
            bestSplitContinuous(node_index, i, best_split_val, best_feature, best_threshold, best_left_indices, best_right_indices,
                                best_num_deaths_left, best_num_at_risk_left, best_num_deaths_right, best_num_at_risk_right);
        }
    }

    if (best_split_val < 0) {
        makeLeaf(node_index);
        return true;
    }

    // update the survival quantities
    size_t left_child_id = node_obs.size();
    size_t right_child_id = node_obs.size() + 1;
    cache_num_deaths[left_child_id] = std::move(best_num_deaths_left);
    cache_num_at_risk[left_child_id] = std::move(best_num_at_risk_left);
    cache_num_deaths[right_child_id] = std::move(best_num_deaths_right);
    cache_num_at_risk[right_child_id] = std::move(best_num_at_risk_right);

    // update tree
    node_obs.push_back(best_left_indices);
    node_obs.push_back(best_right_indices);
    feature_IDs.push_back(best_feature);
    thresholds.push_back(best_threshold);
    chf.push_back(vector<double>());
    return false;
}
*/

/* first attempt using async (does not work, causes all sorts of thread-related errors)
void SurvivalForest::grow() {
    int n = (*data).getNumberOfObs();
    size_t n_threads = std::thread::hardware_concurrency();  // auto-detect number of threads (should later be set by user)
    mutex mutex_oob, mutex_trees; // protect shared vectors

    // launch tasks asynchronously
    vector<future<void>> futures;

    for (size_t i = 0; i < ntrees; ++i) {
        futures.push_back(async(launch::async, [&, i]() {
            // Each thread gets its own RNG to avoid race conditions
            mt19937 local_rng(random_number_generator());

            // use standard (Efron) bootstrap with replacement for now
            auto bootstrap_indices = sampleIndices(n, n, true, local_rng);
            auto oob_indices_tree = computeOOBIndices(bootstrap_indices, n);

            // store OOB indices safely
            {
                lock_guard<mutex> lock(mutex_oob);
                oob_indices.push_back(oob_indices_tree);
            }

            // grow survival tree
            Rcout << "Growing tree " << i << " on thread " 
                      << this_thread::get_id() << endl;

            auto tree = make_unique<SurvivalTree>(unique_event_times, response_event_time_ids, bootstrap_indices);

            uniform_int_distribution<size_t> dist(0, numeric_limits<size_t>::max());    // set local seed for tree
            tree->initialise(data, mtry, min_node_size, nsplits, dist(local_rng));      // use local random number generator
            tree->setRNG(local_rng);
            tree->grow();

            // store tree safely
            {
                lock_guard<mutex> lock(mutex_trees);
                trees.push_back(std::move(tree));
            }
        }));
    }

    // wait for all tasks to finish
    for (auto& future : futures) {
        future.get();
    }
}
*/

/* old constructor made for NumericMatrix x and NumericMatrix y
Data::Data(DataFrame data, const vector<size_t>& response_indices, const vector<size_t>& feature_indices,
           const vector<bool>& categorical, const vector<size_t>& unique) {
    this->num_obs = data.nrows();
    this->num_features = feature_indices.size();
    this->response_indices = response_indices;

    NumericMatrix x_res(num_obs, num_features);

    // fill the response matrix if response variables are supplied
    if (!response_indices.empty()) {
        NumericMatrix y_res(num_obs, response_indices.size());
        for (size_t i = 0; i < response_indices.size(); ++i) {
            NumericVector col = data[response_indices[i]];
            for (size_t j = 0; j < num_obs; ++j) {
                y_res(j, i) = col[j];
            }
        }
        vector<string> response_names(response_indices.size());
        for (size_t i = 0; i < response_indices.size(); ++i) {
            response_names[i] = as<vector<string>>(data.names())[response_indices[i]];
        }
        this->y = y_res;
        this->response_names = response_names;
    }

    vector<bool> categorical_features(num_features);
    vector<size_t> unique_values_features(num_features);
    vector<string> feature_names(num_features);

    for (size_t i = 0; i < num_features; ++i) {
        // update the feature type (categorical or continuous)
        if (categorical[feature_indices[i]] == true) {
            categorical_features[i] = true;
        } else {
            categorical_features[i] = false;
        }
        
        // update names of features and response(s)
        feature_names[i] = as<vector<string>>(data.names())[feature_indices[i]];

        // update the number of unique values
        unique_values_features[i] = unique[feature_indices[i]];
        
        // fill the feature matrix
        NumericVector col = data[feature_indices[i]];
        for (size_t j = 0; j < num_obs; ++j) {
            x_res(j, i) = col[j];
        }
    }

    this->x = x_res;
    this->categorical = categorical_features;
    this->unique_values = unique_values_features;
    this->feature_names = feature_names;
}
*/


/*
double SurvivalTree::log_rank() {
    double sum_num = 0;
    double sum_den = 0;
    for (size_t i = 0; i < num_unique_event_times; ++i) {
        const double d = (double) num_deaths[i];
        const double d1 = (double) num_deaths_left[i];
        const double Y = (double) num_at_risk[i];
        const double Y1 = (double) num_at_risk_left[i];

        // prevent division by zero in the log-rank test
        if (Y < 2 || Y1 < 1) {
            break;  // since the event times are ordered, all subsequent #at risk will also be too small
        }
        if (d > 0) {
            sum_num += d1 - Y1 * d / Y;
            sum_den += d * (Y1 / Y) * (1.0 - Y1/Y) * (Y - d) / (Y - 1.0);
        }
    }

    if (sum_den > 1e-9) {
        // return the squared log-rank test
        return(sum_num * sum_num / sum_den);
    }
    else {
        return(-1);
    }
}
*/

/*  old version of log_rank (works with the old bestSplitContinuous)
double SurvivalTree::log_rank(const vector<size_t>& left_indices, const vector<size_t>& right_indices) {
    // start by computing the number at risk and the number of deaths in each node
    // (I actually don't like the lines below this, it should be changed at some point)
    computeSurvivalQuantities(left_indices);
    vector<double> Y1(num_at_risk.begin(), num_at_risk.end());
    vector<double> d1(num_deaths.begin(), num_deaths.end());

    // clean num_at_risk and num_deaths
    num_at_risk = vector<size_t>(num_unique_event_times, 0);
    num_deaths = vector<size_t>(num_unique_event_times, 0);

    computeSurvivalQuantities(right_indices);
    vector<double> Y2(num_at_risk.begin(), num_at_risk.end());
    vector<double> d2(num_deaths.begin(), num_deaths.end());

    // clean num_at_risk and num_deaths
    num_at_risk = vector<size_t>(num_unique_event_times, 0);
    num_deaths = vector<size_t>(num_unique_event_times, 0);

    // compute number at risk and number of deaths in the parent node
    vector<double> Y = vector<double>(num_unique_event_times, 0);
    vector<double> d = vector<double>(num_unique_event_times, 0);
    for (size_t i = 0; i < num_unique_event_times; ++i) {
        Y[i] = Y1[i] + Y2[i];
        d[i] = d1[i] + d2[i];
    }

    // compute the log-rank test
    double sum_num = 0;
    double sum_den = 0;
    for (size_t i = 0; i < num_unique_event_times; ++i) {
        // prevent division by zero in the log-rank test
        if (Y[i] < 2 || Y1[i] < 1) {
            break;
        }
        if (d[i] > 0) {
            sum_num += d1[i] - Y1[i] * d[i] / Y[i];
            sum_den += d[i] * (Y1[i] / Y[i]) * (1 - Y1[i]/Y[i]) * (Y[i] - d[i]) / (Y[i] - 1);
        }
    }

    if (sum_den != 0) {
        // return the squared log-rank test
        return(sum_num * sum_num / sum_den);
    }
    else {
        return(-1);
    }
}
*/

/*
double SurvivalTree::logRankCategorical(const vector<size_t>& num_deaths, const vector<size_t>& num_at_risk, 
                                        const vector<size_t>& num_deaths_daughter, const vector<size_t>& num_at_risk_daughter) {
    double sum_num = 0;
    double sum_den = 0;
    for (size_t i = 0; i < num_unique_event_times; ++i) {
        const double d = (double) num_deaths[i];
        const double d1 = (double) num_deaths_daughter[i];
        const double Y = (double) num_at_risk[i];
        const double Y1 = (double) num_at_risk_daughter[i];

        // prevent division by zero in the log-rank test
        if (Y < 2 || Y1 < 1) {
            break;  // since the event times are ordered, all subsequent numbers at risk will also be too small
        }
        if (d > 0) {
            double at_risk_frac = Y1 / Y;
            sum_num += d1 - d * at_risk_frac;
            sum_den += d * at_risk_frac * (1.0 - at_risk_frac) * (Y - d) / (Y - 1.0);
        }
    }

    if (sum_den > 1e-9) {
        // return the squared log-rank test
        return(sum_num * sum_num / sum_den);
    } else {
        return(-1);
    }
}
*/

// the following function is for computing a single prediction (not used)

/*

// [[Rcpp::export]]
NumericVector JFCppForestPredictSingle(const List& JFForest, const NumericVector& x) {
  string type = as<string>(JFForest["tree.type"]);
  vector<double> x_cpp = as<vector<double>>(x);

  if (type == "Regression") {

  }
  if (type == "Classification") {

  }
  if (type == "Survival") {
    SurvivalForest* forest = ((XPtr<SurvivalForest>) JFForest["Forest"]).get();
    NumericVector prediction(forest->getEventTimes().size());
    const vector<double>& pred = forest->predict(x_cpp);
    copy(pred.begin(), pred.end(), prediction.begin());
    return prediction;
    
  }
  if (type == "Multi-state") {

  }
}

*/

// old implementation which doesn't handle at risk calculations correctly (may also be problems with computing the jumps)
/*

void MultistateTree::computeMultistateQuantities(const vector<size_t>& indices, vector<size_t>& at_risk, vector<size_t>& jumps) {
    size_t n = indices.size();
    // fetch states and other relevant data quantities
    const vector<uint8_t>& states = data->getStates();
    uint8_t max_response_length = data->getMaxResponseLength();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    
    num_jumps.assign(num_unique_event_times * dim, 0);
    num_at_risk.assign(num_unique_event_times * num_states, 0);

    for (size_t i : indices) {
        size_t j = 0;
        size_t index = i * max_response_length;
        // a state is 0 if and only if it is not valid e.g. a dead entry in the flattened array of observations
        while (j < max_response_length && states[index] != 0) {
            size_t id = (*response_time_event_ids)[index];
            int current_state_index = states[index] - 1;     // states are always indexed by 1, 2, ... with 0 reserved for 'dead' entries in the flattened array
            int next_state_index = states[index + 1] - 1;
            // find jumps (we assume that max_response_length > 1 i.e. at least one jump occurs in the dataset)
            if (j < max_response_length - 1 && next_state_index != -1 && current_state_index != next_state_index) {
                ++num_jumps[id * dim + current_state_index * num_states + i] - next_state_index];
            }
            // find numbers at risk
            // j == 0 means that we are at the first state of an observation/path
            if (j == 0 || current_state_index != states[index - 1] - 1) {
                ++num_at_risk[id * num_states + current_state_index];
            }
            ++j;
            index = i * max_response_length + j;
        }
    }
}

*/

/*

Gemini's attempt. I do not trust this implementation. It doesn't even handle the j == 0 case

void MultistateTree::computeMultistateQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                         vector<size_t>& num_at_risk_left, vector<size_t>& num_jumps_left, size_t nsplits_final) {
    const vector<size_t>& states = data->getStates();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    uint8_t max_response_length = data->getMaxResponseLength();
    //vector<size_t> delta_num_at_risk_right(nsplits_final * num_unique_event_times * num_states);

    // initialise the number of jumps and at risk to be ones for the parent
    for (size_t s = 0; s < nsplits_final; ++s) {
        copy(num_at_risk.begin(), num_at_risk.end(), num_at_risk_left.begin() + s * num_unique_event_times * num_states);
        copy(num_jumps.begin(), num_jumps.end(), num_jumps_left.begin() + s * num_unique_event_times * dim);
    }

    // sort indices of observations by feature value
    vector<size_t> sorted_indices = node_obs[node_index];
    sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t a, size_t b) {
        return data->get_x(a, feature) < data->get_x(b, feature);
    });

    size_t current_obs_idx = 0;
    for (size_t s = 0; s < nsplits_final; ++s) {
        // carry over changes from the previous split point
        if (s > 0) {
            copy(num_at_risk_left.begin() + (s - 1) * num_unique_event_times * num_states,
                      num_at_risk_left.begin() + s * num_unique_event_times * num_states,
                      num_at_risk_left.begin() + s * num_unique_event_times * num_states);
            copy(num_jumps_left.begin() + (s - 1) * num_unique_event_times * dim,
                      num_jumps_left.begin() + s * num_unique_event_times * dim,
                      num_jumps_left.begin() + s * num_unique_event_times * dim);
        }

        // subtract observations that are to the right of split_points[s]
        while (current_obs_idx < sorted_indices.size() && data->get_x(sorted_indices[current_obs_idx], feature) > split_points[s]) {
            size_t idx = i = sorted_indices[current_obs_idx];
            ++num_obs_right[idx];

            for (size_t j = 0; j < max_response_length; ++j) {
                size_t idx = i * max_response_length + j;
                if (states[idx] == 0) break;    // end of path

                size_t id = (*response_time_event_ids)[idx];
                size_t curr_s = states[idx] - 1;

                // handle jumps
                if (j + 1 < max_response_length && states[idx + 1] != 0 && states[idx + 1] != states[idx]) {
                    size_t next_state = states[idx + 1] - 1;
                    size_t jump_idx = s * num_times * dim + id * dim + curr_s * num_states + next_state;
                    --num_jumps_left[jump_idx];
                }
                // handle at risk
                size_t risk_idx = s * num_times * num_states + id * num_states + curr_s;
                --num_at_risk_left[risk_idx];
            }
            ++current_obs_idx;
        }
    }
}

*/

/*

// old prediction functions for multi-state trees

// compute a flattened vector of predictions (the Nelson-Aalen estimators at each event time)
vector<double> MultistateTree::computePredictions(const Data& new_data) {
    uint8_t num_states = data->getNumberOfStates();
    size_t dim = num_states * num_states;
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times * dim);
    for (size_t i = 0; i < num_obs; ++i) {
        const vector<double>& pred = get<vector<double>>(predict(new_data.get_x_row(i)));   // the na vector has for unknown reasons been destroyed...
        //cout << "Prediction for observation " << i << ":" << endl;
        //printVector(pred);
        //cout << endl;
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            for (size_t j = 0; j < num_states; ++j) {
                for (size_t k = 0; k < num_states; ++k) {
                    predictions[i * num_unique_event_times * dim + t * dim + j * num_states + k] = pred[t * dim + j * num_states + k];
                }
            }
        }
    }
    return predictions;
}

// compute a flattened vector of predictions and of the predicted initial distributions
pair<vector<double>, vector<double>> MultistateTree::computePredictedInitialDistributions(const Data& new_data) {
    uint8_t num_states = data->getNumberOfStates();
    size_t dim = num_states * num_states;
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times * num_states * num_states);
    vector<double> predictions_init(num_obs * num_states);
    for (size_t i = 0; i < num_obs; ++i) {
        size_t leaf_id = predictionLeafID(new_data.get_x_row(i));
        const vector<double>& pred = na[leaf_id];
        const vector<double>& init = init_dist[leaf_id];
        // first save initial distributions
        for (size_t j = 0; j < num_states; ++j) {
            predictions_init[i * num_states + j] = init[j];
        }

        // now save the predicted NA-estimators
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            for (size_t j = 0; j < num_states; ++j) {
                for (size_t k = 0; k < num_states; ++k) {
                    predictions[i * num_unique_event_times * dim + t * dim + j * num_states + k] = pred[t * dim + j * num_states + k];
                }
            }
        }
    }
    return {predictions, predictions_init};
}

// computes a pair of flattened vectors, the first predictions and the second the censoring KM estimators
pair<vector<double>, vector<double>> MultistateTree::computePredictionsCensoring(const Data& new_data) {
    uint8_t num_states = data->getNumberOfStates();
    size_t dim = num_states * num_states;
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times * num_states * num_states);
    vector<double> censoring(num_obs * num_unique_event_times);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t leaf_id = predictionLeafID(new_data.get_x_row(i));
        const vector<double>& pred = na[leaf_id];
        const vector<double>& cens = KM_censoring[leaf_id];
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            // first save the predicted Nelson-Aalen estimator
            for (size_t j = 0; j < num_states; ++j) {
                for (size_t k = 0; k < num_states; ++k) {
                    predictions[i * num_unique_event_times * dim + t * dim + j * num_states + k] = pred[t * dim + j * num_states + k];
                    //predictions[i * num_obs + t * num_unique_event_times + j * num_states + k] = pred[t * num_unique_event_times + j * num_states + k];
                }
            }
            // now save the censoring Kaplan-Meier estimator
            censoring[i * num_unique_event_times + t] = cens[t];
        }
    }
    return {predictions, censoring};
}

// computes three flattened vectors of all predictions (NA estimators, initial distributions and censoring KM estimators)
vector<vector<double>> MultistateTree::computeAllPredictions(const Data& new_data) {
    uint8_t num_states = data->getNumberOfStates();
    size_t dim = num_states * num_states;
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times * num_states * num_states);
    vector<double> predictions_init(num_obs * num_unique_event_times * num_states);
    vector<double> censoring(num_obs * num_unique_event_times);

    for (size_t i = 0; i < num_obs; ++i) {
        size_t leaf_id = predictionLeafID(new_data.get_x_row(i));
        const vector<double>& pred = na[leaf_id];
        const vector<double>& cens = KM_censoring[leaf_id];
        const vector<double>& init = init_dist[leaf_id];
        for (size_t t = 0; t < num_unique_event_times; ++t) {
            // first save predicted NA estimators
            for (size_t j = 0; j < num_states; ++j) {
                for (size_t k = 0; k < num_states; ++k) {
                    predictions[i * num_unique_event_times * dim + t * dim + j * num_states + k] = pred[t * dim + j * num_states + k];
                }
            }
            // save censoring distribution
            censoring[i * num_unique_event_times + t] = cens[t];
        }
        // save initial distribution
        for (size_t j = 0; j < num_states; ++j) {
            predictions_init[i * num_states + j] = init[j];
        }
    }
    return {predictions, predictions_init, censoring};
}

// old prediction functions for multi-state forests

pair<vector<double>, vector<double>> MultistateForest::computePredictions() {
    size_t num_obs = data->getNumberOfObs();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    vector<double> predictions(num_obs * num_unique_event_times * dim);
    vector<double> oob_predictions(num_obs * num_unique_event_times * dim);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times * dim, 0);
        vector<double> oob_pred(num_unique_event_times * dim, 0);
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                vector<double> tree_pred = get<vector<double>>(tree->predict(data->get_x_row(i)));
                sum_vectors(pred, tree_pred);
                sum_vectors(oob_pred, tree_pred);
            }
            // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
            else {
                vector<double> tree_pred = tree->getNA()[tree->getPredictionNodeIDs()[i]];
                sum_vectors(pred, tree_pred);
            }
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times * dim; ++k) {
            pred[k] /= ntrees;
            if (num_oob_trees > 0) {
                oob_pred[k] /= num_oob_trees;
            }
            predictions[i * num_unique_event_times * dim + k] = pred[k];
            oob_predictions[i * num_unique_event_times * dim + k] = oob_pred[k];
        }
    }
    return {predictions, oob_predictions};
}

pair<vector<double>, vector<double>> MultistateForest::computePredictedInitialDistributions() {
    size_t num_obs = data->getNumberOfObs();
    uint8_t num_states = data->getNumberOfStates();
    vector<double> predictions(num_obs * num_states);
    vector<double> oob_predictions(num_obs * num_states);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_states, 0);
        vector<double> oob_pred(num_states, 0);
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                vector<double> tree_pred = tree->predictInitDist(data->get_x_row(i));
                sum_vectors(pred, tree_pred);
                sum_vectors(oob_pred, tree_pred);
            }
            // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
            else {
                vector<double> tree_pred = tree->getInitDist()[tree->getPredictionNodeIDs()[i]];
                sum_vectors(pred, tree_pred);
            }
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_states; ++k) {
            pred[k] /= ntrees;
            if (num_oob_trees > 0) {
                oob_pred[k] /= num_oob_trees;
            }
            predictions[i * num_states + k] = pred[k];
            oob_predictions[i * num_states + k] = oob_pred[k];
        }
    }
    return {predictions, oob_predictions};
}


vector<double> MultistateForest::computePredictions(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    uint8_t num_states = data->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    vector<double> predictions(num_obs * num_unique_event_times * dim);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times * dim, 0);

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
            vector<double> tree_pred = get<vector<double>>(tree->predict(new_data.get_x_row(i)));
            sum_vectors(pred, tree_pred);
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times * dim; ++k) {
            pred[k] /= ntrees;
            predictions[i * num_unique_event_times * dim + k] = pred[k];
        }
    }
    return predictions;
}

vector<double> MultistateForest::computePredictedInitialDistributions(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    uint8_t num_states = data->getNumberOfStates();
    vector<double> predictions(num_obs * num_states);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_states, 0);

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            MultistateTree* tree = dynamic_cast<MultistateTree*>(trees[j].get());
            vector<double> tree_pred = tree->predictInitDist(new_data.get_x_row(i));
            sum_vectors(pred, tree_pred);
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_states; ++k) {
            pred[k] /= ntrees;
            predictions[i * num_states + k] = pred[k];
        }
    }
    return predictions;
}

// old prediction functions for survival forests

// computes predictions, both in-bag and out-of-bag
pair<vector<double>, vector<double>> SurvivalForest::computePredictions() {
    size_t num_obs = data->getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);
    vector<double> oob_predictions(num_obs * num_unique_event_times);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times, 0);
        vector<double> oob_pred(num_unique_event_times, 0);
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                vector<double> tree_pred = get<vector<double>>(tree->predict(data->get_x_row(i)));
                sum_vectors(pred, tree_pred);
                sum_vectors(oob_pred, tree_pred);
            }
            // no need to predict from scratch for in-bag observations since we save the terminal node ID during fitting
            else {
                vector<double> tree_pred = tree->getCHF()[tree->getPredictionNodeIDs()[i]];
                sum_vectors(pred, tree_pred);
            }
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times; ++k) {
            pred[k] /= ntrees;
            if (num_oob_trees > 0) {
                oob_pred[k] /= num_oob_trees;
            }
            predictions[i * num_unique_event_times + k] = pred[k];
            oob_predictions[i * num_unique_event_times + k] = oob_pred[k];
        }
    }

    return {predictions, oob_predictions};
}

// computes out-of-bag censoring predictions
vector<double> SurvivalForest::computePredictionsCensoringOOB() {
    size_t num_obs = data->getNumberOfObs();
    vector<double> oob_censoring(num_obs * num_unique_event_times);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> oob_cens(num_unique_event_times, 0);
        double num_oob_trees = 0;   // for keeping track of the number of trees where observation i is OOB

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
            vector<vector<double>> tree_cens = tree->getKMCensoring();
            
            if (oob_indices[j][i]) {
                ++num_oob_trees;
                size_t leaf_id = tree->predictionLeafID(data->get_x_row(i));
                vector<double> tree_cens_obs = tree_cens[leaf_id];
                sum_vectors(oob_cens, tree_cens_obs);
            }
        }
        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times; ++k) {
            if (num_oob_trees > 0) {
                oob_cens[k] /= num_oob_trees;
            }
            oob_censoring[i * num_unique_event_times + k] = oob_cens[k];
        }
    }
    return oob_censoring;
}

vector<double> SurvivalForest::computePredictions(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times, 0);

        // compute the sum of all predictions for observation i
        for (size_t j = 0; j < ntrees; ++j) {
            SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
            vector<double> tree_pred = get<vector<double>>(tree->predict(new_data.get_x_row(i)));
            sum_vectors(pred, tree_pred);
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times; ++k) {
            pred[k] /= ntrees;
            predictions[i * num_unique_event_times + k] = pred[k];
        }
    }
    return predictions;
}

// computes a pair of flattened vectors, the first predictions and the second the censoring KM estimators
// (when censoring KM estimators are already saved during fitting)
pair<vector<double>, vector<double>> SurvivalForest::computePredictionsCensoring(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);
    vector<double> censoring(num_obs * num_unique_event_times);

    #pragma omp parallel for schedule(dynamic) num_threads(this->nworkers)
    for (size_t i = 0; i < num_obs; ++i) {
        vector<double> pred(num_unique_event_times, 0);
        vector<double> cens(num_unique_event_times, 0);

        for (size_t j = 0; j < ntrees; ++j) {
            SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
            const vector<vector<double>>& tree_predictions = tree->getCHF();
            const vector<vector<double>>& tree_censoring = tree->getKMCensoring();

             // fetch predictions from the tree
            size_t leaf_id = tree->predictionLeafID(new_data.get_x_row(i));
            vector<double> tree_pred = tree_predictions[leaf_id];               // important: will this still work with const? speedup potential? test!
            vector<double> tree_cens = tree_censoring[leaf_id];

            // compute the sum of all predictions for observation i
            sum_vectors(pred, tree_pred);
            sum_vectors(cens, tree_cens);
        }

        // normalise and save predictions
        for (size_t k = 0; k < num_unique_event_times; ++k) {
            pred[k] /= ntrees;
            cens[k] /= ntrees;
            predictions[i * num_unique_event_times + k] = pred[k];
            censoring[i * num_unique_event_times + k] = cens[k];
        }
    }
    return {predictions, censoring};
}

// computes a pair of flattened vectors, the first predictions and the second the censoring KM estimators
pair<vector<double>, vector<double>> SurvivalForest::computePredictionsExternal(const Data& new_data) {
    size_t num_obs = new_data.getNumberOfObs();
    vector<double> predictions(num_obs * num_unique_event_times);
    vector<double> censoring(num_obs * num_unique_event_times);

    for (size_t j = 0; j < ntrees; ++j) {
        SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
        const vector<double>& tree_predictions = tree->getCHF();

        // determine the leaf that each observation belongs to
        vector<size_t> leaf_ids(num_obs);
        for (size_t i = 0; i < num_obs; ++i) {
            leaf_ids[i] = tree->predictionLeafID[new_data.get_x_row(i)];
        }
        
        // now group the observations by leaf, compute the survial quantities and 
        const vector<vector<size_t>>& leaf_obs = groupByLeaf(leaf_ids, tree->getNumberOfTerminalNodes());

    }
}

*/

/* old version

void RegressionTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                                         vector<double>& best_threshold, double& best_sum_left) {
  const vector<size_t>& current_node_obs = node_obs[node_index];
  double parent_sum = sum_node[node_index];

  // samples split points
  vector<double> split_points;
  size_t nsplits_final = sampleSplitPoints(split_points, current_node_obs, feature);

  // no possible splits
  if (nsplits_final == 0) {
    return;
  }

  // compute number of observations and the sums of responses at each split value
  number_obs_split.assign(nsplits_final, 0);
  sums_split.assign(nsplits_final, 0);
  for (size_t i : current_node_obs) {
    size_t idx = lower_bound(split_points.begin(), split_points.end(), data->get_x(i, feature)) - split_points.begin();
    
    sums_split[idx] += data->get_y(i);
    ++number_obs_split[idx];
  }

  // now compute the decrease of impurity for each split
  size_t n_left = 0;
  double sum_left = 0;
  double sum_squared_left = 0;

  for (size_t i = 0; i < nsplits_final; ++i) {
    // skip the split if identical to the previous one (or if no observations)
    if (number_obs_split[i] == 0) {
      continue;
    }

    n_left += number_obs_split[i];
    sum_left += sums_split[i];
    size_t n_right = node_sizes[node_index] - n_left;

    // stop if right child is too small (break since the split points are sorted)
    if (n_right < min_node_size) {
      break;
    }

    // stop if minimal node size is reached
    if (n_left < min_node_size) {
      continue;
    }

    // this is completely nonsensical, do it from scratch
    double sum_right = parent_sum - sum_left;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;
    double split_val = parent_sum * parent_sum / (double) node_sizes[node_index] - decrease;

    if (split_val > best_split_val) {
      best_split_val = split_val;
      best_feature = feature;
      best_sum_left = sum_left;
      // use average of split points unless it is the final split value
      if (i == nsplits_final - 1) {
        best_threshold = {split_points[i]};
      } else {
        best_threshold = {(split_points[i] + split_points[i + 1]) / 2.0};
      }
    }
  }
}

*/
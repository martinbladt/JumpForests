/*

Functions for survival trees

*/

#include "TreeSurvival.h"

// constructor for SurvivalTree
//--------------------------------------------------------------------------------------

SurvivalTree::SurvivalTree(const vector<double> unique_event_times, const vector<size_t> subset_indices) :
    unique_event_times {unique_event_times}, subset_indices {subset_indices} {
        this->node_obs.push_back(subset_indices);
        this->num_unique_event_times = unique_event_times.size();

        // initialise the vector of deaths and individuals at risk
        this->num_at_risk = vector<size_t>(num_unique_event_times, 0);
        this->num_deaths = vector<size_t>(num_unique_event_times, 0);
}

// functions for growing survival trees
//--------------------------------------------------------------------------------------

// computes the number of deaths and the number at risk in each event time
// for the observations given by indices
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

void SurvivalTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices) {
    // sample split points
    vector<double> split_points;
    vector<double> potential_split_points = (*data).getValues(node_obs[node_index], feature);
    /*
    cout << "Line 101: potential_split_points = ";
    for (double x : potential_split_points) {
       cout << x << ", ";
    }
    cout << endl;
    */
    sample(potential_split_points.begin(), potential_split_points.end(), back_inserter(split_points), nsplits,
            random_number_generator);

    /*
    cout << "Line 109: split_points = ";
    for (double s : split_points) {
        cout << s << ", ";
    }
    cout << endl << "No problems until now" << endl;
    */

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
        
        /*
        cout << "Line 127: Left indices = ";
        for (int i : left_indices) {
            cout << i << ", ";
        }
        cout << endl << "Line 131: Right indices = ";
        for (int i : right_indices) {
            cout << i << ", ";
        }
        cout << endl;
        */  

        // jump to next threshold if split is illegal
        if (left_indices.size() < min_node_size || right_indices.size() < min_node_size) {
            continue;
        }
            
        double split_val = log_rank(left_indices, right_indices);
        //cout << "Line 116: split_val = " << split_val << endl;
        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_left_indices = left_indices;
            best_right_indices = right_indices;
            best_feature = feature;
            // should maybe be changed to an average to improve stability
            best_threshold = {c};
        }
    }
}

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
    for (int i : partition_IDs) {
        vector<size_t> left_indices;
        vector<size_t> right_indices;

        // for O(1) expected time lookups in the following loop
        unordered_set<double> s(partitions[i].begin(), partitions[i].end());

        // perform the split
        for (int j : node_obs[node_index]) {
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
        //cout << "Line 116: split_val = " << split_val << endl;
        if (split_val > best_split_val) {
            best_split_val = split_val;
            best_left_indices = left_indices;
            best_right_indices = right_indices;
            best_feature = feature;
            best_threshold = partitions[i];
        }
    }
}

// function to create a split for a survival tree. returns true if leaf, otherwise false
// NB: at the moment only supports continuous covariates, but should be extended to categorical ones ASAP
bool SurvivalTree::createSplit(size_t node_index) {
    // if no split is possible, make the node a leaf
    if (node_obs[node_index].size() < 2 * min_node_size) {
        // compute the chf
        computeChf(node_index);

        // update tree info
        feature_IDs.push_back(0);
        thresholds.push_back({0});   // 0 could actually be a proper threshold, but you can see from feature_IDs that it is a leaf

        // since we are in a terminal node, we save the indices for the observations
        for (int i : node_obs[node_index]) {
            prediction_node_IDs[i] = node_index;
        }
        return true;
    }

    // make sure that at least two spaces are available in node_obs
    //node_obs.push_back(vector<size_t>()); node_obs.push_back(vector<size_t>());
    
    double best_split_val = -1;
    size_t best_feature;
    vector<double> best_threshold;
    vector<size_t> best_left_indices;
    vector<size_t> best_right_indices;

    // sample mtry features
    size_t num_features =  (*data).getNumberOfFeatures();
    vector<size_t> sampled_features = sampleIndices(num_features, mtry, false, random_number_generator);

    /*
    cout << "Line 86: sampled_features = ";
    for (int j : sampled_features) {
        cout << j << ", "; 
    }
    cout << endl;
    */
    // now consider each of the sampled features
    for (int i : sampled_features) {
        //cout << "Line 94: Feature " << i << endl;
        // if the feature is categorical, skip the split for now
        if ((*data).getCategorical()[i]) {
            bestSplitCategorical(node_index, i, best_split_val, best_feature, best_threshold, best_left_indices, best_right_indices);
        }
        else {
            bestSplitContinuous(node_index, i, best_split_val, best_feature, best_threshold, best_left_indices, best_right_indices);
        }
    }

    if (best_split_val < 0) {
        // compute the chf
        computeChf(node_index);

        // update tree info
        feature_IDs.push_back(0);
        thresholds.push_back({0});   // 0 could actually be a proper threshold, but you can see from feature_IDs that it is a leaf

        // since we are in a terminal node, we save the indices for the observations
        for (int i : node_obs[node_index]) {
            prediction_node_IDs[i] = node_index;
        }
        return true;
    }

    node_obs.push_back(best_left_indices);
    node_obs.push_back(best_right_indices);
    feature_IDs.push_back(best_feature);
    thresholds.push_back(best_threshold);
    chf.push_back(vector<double>());      // temporary solution
    return false;
}

// computes the cumulative hazard estimate in node node_index
void SurvivalTree::computeChf(size_t node_index) {
    vector<double>chf = vector<double>(num_unique_event_times, 0);
    computeSurvivalQuantities(node_obs[node_index]);
    /*
    cout << "Line 191: Observations in node " << node_index << ":";
    for (int i : node_obs[node_index]) {
        cout << i << ", ";
    }
    cout << endl;
    */

    if (num_at_risk[0] != 0) {
        chf[0] = double(num_deaths[0])/double(num_at_risk[0]);
    }
    for (int i = 1; i < num_unique_event_times; ++i) {
        if (num_at_risk[i] != 0) {
            chf[i] = chf[i - 1] + double(num_deaths[i]) / double(num_at_risk[i]);
        }
        else {
            chf[i] = chf[i - 1];
        }
    }

    this->chf.push_back(chf);
}

// function to grow a survival tree
void SurvivalTree::grow() {
  // maybe bootstrap weights should be here if we choose to implement general bootstrap schemes

  size_t num_queue = 1;
  size_t depth = 0;
  size_t left_most_node = 0;

  // while not all nodes terminal, continue growing the tree
  size_t i = 0;
  depths.push_back(depth);
  while (num_queue > 0) {
    //cout << "Line 213: i = " << i << endl;
    bool is_leaf = createSplit(i);
    if (is_leaf) {
        num_queue--;
        left_daughters.push_back(0);  // 0 indicates no daughters
        num_terminal_nodes++;
    }
    else {
        num_queue++;
        left_daughters.push_back(num_nodes);
        num_nodes += 2;
        if (i >= left_most_node) {
            depth++;
            left_most_node = num_nodes - 2;
        }
        depths.push_back(depth);
        depths.push_back(depth);
    }
    i++;
  }

  tree_depth = depth;
  cleanUp();
}

// splitting rules for survival trees
//--------------------------------------------------------------------------------------

double SurvivalTree::log_rank(vector<size_t> left_indices, vector<size_t> right_indices) {
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

    /*
    cout << "Line 246: Left number at risk = ";
    for (int n : Y1) {
        cout << n << ", ";
    }
    cout << endl << "Line 250: Left number of deaths = ";
    for (int d : d1) {
        cout << d << ", ";
    }
    cout << endl << "Line 254: Right number at risk = ";
    for (int n : Y2) {
        cout << n << ", ";
    }
    cout << endl << "Line 258: Right number of deaths = ";
    for (int d : d2) {
        cout << d << ", ";
    }
    cout << endl;
    */

    // compute number at risk and number of deaths in the parent node
    vector<double> Y = vector<double>(num_unique_event_times, 0);
    vector<double> d = vector<double>(num_unique_event_times, 0);
    for (int i = 0; i < num_unique_event_times; ++i) {
        Y[i] = Y1[i] + Y2[i];
        d[i] = d1[i] + d2[i];
    }

    // compute the log-rank test
    double sum_num = 0;
    double sum_den = 0;
    for (int i = 0; i < num_unique_event_times; ++i) {
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

/*

double SurvivalTree::log_rank_score() {

}

double SurvivalTree::approx_log_rank() {

}

double SurvivalTree::conserve() {

}

*/

// functions for predicting with survival trees
//--------------------------------------------------------------------------------------

// error estimation for survival trees
//--------------------------------------------------------------------------------------

// miscellaneous functions related to survival
//--------------------------------------------------------------------------------------

vector<double> computeUniqueEventTimes(const vector<double>& times, const vector<size_t>& ind) {
    vector<double> unique_event_times;

    // remove censored times
    vector<double> observed_times;
    for (int i = 0; i < times.size(); ++i) {
      if (ind[i] == 1) {
        observed_times.push_back(times[i]);
      }
    }
    
    // sort and remove duplicated observed times
    sort(observed_times.begin(), observed_times.end());
    observed_times.push_back(-1); // to ensure the last observed time is included
    for (int i = 0; i < observed_times.size() - 1; ++i) {
      if (observed_times[i] != observed_times[i + 1]) {
        unique_event_times.push_back(observed_times[i]);
      }
    }
    return(unique_event_times);
}

// computes the Kaplan-Meyer estimator given a Nelson-Aalen estimator
vector<double> KaplanMeyer(vector<double> na) {
    vector<double> KM = vector<double>(na.size() + 1, 1);
    KM[0] = 1;

    // compute using the recursion given by the product integral
    for (int i = 1; i < na.size() + 1; ++i) {
        KM[i] = KM[i - 1] * (1 - (na[i] - na[i - 1]));
    }
    return(KM);
}

// computes Harrell's C-index for a list of outcomes and survival data
double computeConcordanceIndex(const vector<double>& outcomes, const vector<double>& times, const vector<double>& ind) {
    double concordance = 0;
    double permissible = 0;
    size_t n = times.size();

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // if T_i < T_j and delta_i = 0, the pair is not comparable
            if (times[i] < times[j] && ind[i] == 0) {
                continue;
            }
            // similarly with i and j reversed
            if (times[j] < times[i] && ind[j] == 0) {
                continue;
            }
            // if T_i = T_j and delta_i = delta_j, the pair is also considered incomparable
            if (ind[i] == ind[j] && times[i] == times[j]) {
                continue;
            }

            // if T_i < T_j and outcomes[i] > outcomes[j], the model predicts correctly
            // (similarly with i and j reversed), so add 1 to concordance
            // if the outcomes are identical, the model is indecisive so add 0.5 to concordance
            if (times[i] < times[j] && outcomes[i] > outcomes[j]) {
                concordance += 1;
            }
            else if (times[j] < times[i] && outcomes[j] > outcomes[i]) {
                concordance += 1;
            }
            else if (outcomes[i] == outcomes[j]) {
                concordance += 0.5;
            }
            permissible += 1;
        }
    }

    return(concordance/permissible);
}
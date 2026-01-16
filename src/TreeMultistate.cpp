/*

Functions for multi-state trees

*/

#include "TreeMultistate.h"


// constructor for MultistateTree
//--------------------------------------------------------------------------------------

// NB: we need to know the number of states to initialise the at risk and jumps vectors properly

// the final argument is only used for honest trees
MultistateTree::MultistateTree(shared_ptr<vector<double>> unique_event_times, shared_ptr<vector<size_t>> response_event_time_ids, uint8_t num_states, const vector<size_t>& subset_indices, const vector<size_t>& estimation_indices) : 
    unique_event_times {unique_event_times}, response_time_event_ids {response_time_event_ids}, num_states {num_states} {
        this->node_obs.push_back(subset_indices);
        this->holdout_node_obs.push_back(estimation_indices);
        this->num_unique_event_times = unique_event_times->size();
        this->node_sizes.push_back(subset_indices.size());

        // initialise vector of jumps and individuals at risk
        this->num_jumps.resize(num_unique_event_times * num_states * num_states);
        this->num_at_risk.resize(num_unique_event_times * num_states);
}

// functions for growing multi-state trees
//--------------------------------------------------------------------------------------

void MultistateTree::computeMultistateQuantities(const vector<size_t>& indices, vector<size_t>& at_risk, vector<size_t>& jumps) {
    size_t n = indices.size();
    
    num_jumps.assign(num_unique_event_times * num_states * num_states, 0);
    num_at_risk.assign(num_unique_event_times * num_states, 0);

    
}

void MultistateTree::makeLeaf(size_t node_index) {

}

bool MultistateTree::createSplit(size_t node_index) {

}

void MultistateTree::computeNA(size_t node_index) {

}

// might need to change num_jumps_right to a vector of matrices, otherwise maybe okay with a flattened array of flattened matrices
void MultistateTree::computeMultistateQuantitiesDaughter(size_t node_index, size_t feature, const vector<double>& split_points, vector<size_t>& num_obs_right,
                                         vector<size_t>& delta_num_at_risk_right, vector<size_t>& num_jumps_right, size_t nsplits_final) {

}

void MultistateTree::bestSplitContinuous(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, vector<double>& best_threshold) {

}

void MultistateTree::bestSplitCategorical(size_t node_index, size_t feature, double& best_split_val, size_t& best_feature, 
                           vector<double>& best_threshold, vector<size_t>& best_left_indices, vector<size_t>& best_right_indices) {

}

vector<vector<double>> AalenJohansen(const vector<vector<double>>& na) {

}

// splitting rules for multi-state trees
//--------------------------------------------------------------------------------------

// prediction for multi-state trees
//--------------------------------------------------------------------------------------

vector<double> MultistateTree::computePredictions(const Data& new_data) {
    
}

// error estimation for multi-state trees
//--------------------------------------------------------------------------------------

// miscellaneous functions related to multi-states
//--------------------------------------------------------------------------------------

// for computing the ids in the observed times corresponding to the unique event times (including censored times)
vector<size_t> computeResponseEventTimeIDsMultistate(const vector<double>& unique_event_times, const vector<double>& times) {
    vector<size_t> response_event_time_ids;
    response_event_time_ids.reserve(times.size());
    for (const double& time : times) {
        // only difference to survival: there will be many zeroes since we flatten the times vector with the largest number of total jumps
        if (time == 0) {
            response_event_time_ids.push_back(0);
            continue;
        }

        // use binary search to find lower bound
        auto it = lower_bound(unique_event_times.begin(), unique_event_times.end(), time);
        response_event_time_ids.push_back(static_cast<size_t>(distance(unique_event_times.begin(), it)));
    }
    return(response_event_time_ids);
}
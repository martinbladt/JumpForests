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
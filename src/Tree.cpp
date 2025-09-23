#include "Tree.h"

void Tree::initialise(shared_ptr<Data> data, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, string splitrule, bool honest, unsigned int seed) {
    // initialise with the chosen hyperparameters
    this->data = data;
    this->mtry = mtry;
    this->min_node_size = min_node_size;
    this->nsplits = nsplits;
    this->splitrule = splitrule;
    this->honest = honest;

    // initialise tree info
    num_nodes = 1;
    num_terminal_nodes = 0;
    prediction_node_IDs = vector<size_t>((*data).getNumberOfObs(), 0);  // should this be a choice from the user?

    // set seed
    random_number_generator.seed(seed);
}

void Tree::setRNG(mt19937 rng) {
    random_number_generator = rng;
}

size_t Tree::predictionLeafID(const vector<double>& x) {
    size_t current_node = 0;
    while (left_daughters[current_node] != 0) { // while not yet in a terminal node
        // if the feature is categorical, check whether the coordinate of x belongs to the left or right subset
        if (data->getCategorical()[feature_IDs[current_node]]) {
            const vector<double>& left_subset = thresholds[current_node];
            if (find(left_subset.begin(), left_subset.end(), x[feature_IDs[current_node]]) != left_subset.end()) {
                current_node = left_daughters[current_node];
            }
            else {
                // right daughter is always the left plus one
                current_node = left_daughters[current_node] + 1;
            }
        }
        // if the feature is continuous, check whether the coordinate of x is below the threshold
        else {
            if (x[feature_IDs[current_node]] <= thresholds[current_node][0]) {
                current_node = left_daughters[current_node];
            }
            else {
                // right daughter is always the left plus one
                current_node = left_daughters[current_node] + 1;
            }
        }
    }
    return(current_node);
}

size_t Tree::predictionLeafIDVIMP(const vector<double>& x, size_t feature, mt19937& rng) {
    uniform_int_distribution<size_t> daughter_id(0, 1);
    size_t current_node = 0;
    while (left_daughters[current_node] != 0) { // while not yet in a terminal node
        // if the feature in the current node equals the chosen feature, make daughter assignment random
        if (feature_IDs[current_node] == feature) {
            //size_t left_daughter_size = node_sizes[left_daughters[current_node]];
            //size_t right_daughter_size = node_sizes[left_daughters[current_node] + 1];
            //discrete_distribution<size_t> daughter_id({left_daughter_size, right_daughter_size});
            size_t daughter = daughter_id(rng);
            //cout << "daughter:" << daughter << endl;
            current_node = left_daughters[current_node] + daughter;
        }
        // if not, do prediction as normal
        else {
            // if the feature is categorical, check whether the coordinate of x belongs to the left or right subset
            if (data->getCategorical()[feature_IDs[current_node]]) {
                const vector<double>& left_subset = thresholds[current_node];
                if (find(left_subset.begin(), left_subset.end(), x[feature_IDs[current_node]]) != left_subset.end()) {
                    current_node = left_daughters[current_node];
                }
                else {
                    // right daughter is always the left plus one
                    current_node = left_daughters[current_node] + 1;
                }
            }
            // if the feature is continuous, check whether the coordinate of x is below the threshold
            else {
                if (x[feature_IDs[current_node]] <= thresholds[current_node][0]) {
                    current_node = left_daughters[current_node];
                }
                else {
                    // right daughter is always the left plus one
                    current_node = left_daughters[current_node] + 1;
                }
            }
        }
    }
    return current_node;
}

// samples split points for continuous splits, returns number of final split points (zero indicates no possible splits)
size_t Tree::sampleSplitPoints(vector<double>& split_points, const vector<size_t>& indices, size_t feature) {
  // extract candidate split points
  vector<double> potential_split_points = data->getValues(indices, feature);

  // sort and remove duplicates to obtain the final set of candidate split points
  sort(potential_split_points.begin(), potential_split_points.end());
  potential_split_points.erase(unique(potential_split_points.begin(), potential_split_points.end()), potential_split_points.end());

  // no possible splits
  if (potential_split_points.size() < 2) {
      return 0;
  }

  // now sample nsplits points
  size_t nsplits_final;
  if (potential_split_points.size() <= nsplits) {
      split_points = potential_split_points;
      nsplits_final = potential_split_points.size();
  } else {
      sample(potential_split_points.begin(), potential_split_points.end(), back_inserter(split_points), nsplits, random_number_generator);
      nsplits_final = nsplits;
  }
  sort(split_points.begin(), split_points.end());
  return nsplits_final;
}

// samples partitions for categorical splits, returns true if no possible splits
bool Tree::generateCategoricalPartitions(const vector<double>& feature_values, unordered_set<uint64_t>& partition_masks) {
    size_t num_feature_values = feature_values.size();

    if (num_feature_values < 2) {   // no possible split if only one unique value
        return true;
    }
    if (num_feature_values > 63) {  // prevent 64 bit-integer overflow
        num_feature_values = 63;
    }

    uint64_t total_partitions = (1ULL << (num_feature_values - 1)) - 1;
    size_t num_splits_to_try = nsplits;
    if (nsplits > total_partitions) {
        num_splits_to_try = total_partitions;
    }

    // sample unique partition IDs (bitmasks)
    uniform_int_distribution<uint64_t> dist(1, total_partitions);
    while(partition_masks.size() < num_splits_to_try) {
        partition_masks.insert(dist(random_number_generator));
    }
    return false;
}

// function to grow a tree
void Tree::grow() {
  // maybe bootstrap weights should be here if we choose to implement general bootstrap schemes

  size_t num_queue = 1;
  size_t depth = 0;
  size_t left_most_node = 0;

  // while not all nodes terminal, continue growing the tree
  size_t i = 0;
  depths.push_back(depth);
  while (num_queue > 0) {
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
        // only for info, technically redundant
        depths.push_back(depth);
        depths.push_back(depth);
    }
    i++;
  }

  tree_depth = depth;
  cleanUpTree();
}
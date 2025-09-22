#ifndef FOREST_H
#define FOREST_H

#include "Data.h"
#include "Tree.h"
#include <omp.h>
#include <thread>
//#include <future>
//#include <mutex>

class Forest {
public:
  // function to initialise a general Forest (later add more options such as OOB, honesty etc.)
  void initialise(shared_ptr<Data> data, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, string splitrule,
                  unsigned int ntrees, bool honest, bool swr, double sample_rate, unsigned int seed, unsigned int nworkers);

  virtual ~Forest() = default;

  // maybe this is not even necessary for forests, since prediction is very specific to the type
  //ValueType virtual predict(vector<double> x) = 0;

  // functions to get forest info
  const vector<vector<size_t>> getLeftDaughters() const {
    return left_daughters;
  }
  const vector<vector<size_t>> getFeatureIDs() const {
    return feature_IDs;
  }
  const vector<vector<vector<double>>> getThresholds() const {
    return thresholds;
  }
  const vector<vector<size_t>> getDepths() const {
    return depths;
  }
  const vector<size_t> getNumberOfTerminalNodes() const {
    return num_terminal_nodes;
  }
  const vector<size_t> getNumberOfNodes() const {
    return num_nodes;
  }
  const vector<size_t> getTreeDepths() const {
    return tree_depths;
  }
  const double getAvgNumberOfTerminalNodes() const {
    return avg_num_terminal_nodes;
  }
  const double getAvgNumberOfNodes() const {
    return avg_num_nodes;
  }
  const double getAvgTreeDepth() const {
    return avg_tree_depth;
  }
  const vector<unique_ptr<Tree>>& getTrees() const {
    return trees;
  }
  const vector<vector<bool>> getOOBIndices() const {
    return oob_indices;
  }
  shared_ptr<Data> getData() const {
    return data;
  }

  // frees memory after oob_indices are no longer necessary (maybe not best practice to have it public)
  void cleanUp() {
    vector<vector<bool>>().swap(oob_indices);
  }

protected:
  // pointer to the data
  shared_ptr<Data> data;

  // the trees in the forest
  vector<unique_ptr<Tree>> trees;

  // forest information
  int seed;                                   // global seed for the forest, can be set by user
  vector<vector<size_t>> left_daughters;      // vector of left daughters for each tree
  vector<vector<size_t>> feature_IDs;         // ID of the feature in each node for each tree
  vector<vector<vector<double>>> thresholds;  // thresholds in each node for each tree
  vector<vector<size_t>> depths;              // depths of each node for each tree

  // hyperparameters
  unsigned int mtry;              // number of variables randomly selected for each split
  unsigned int min_node_size;     // minimal number of observations in each node
  unsigned int nsplits;           // number of split values to consider after feature is chosen
  unsigned int ntrees;            // number of trees in the forest
  string splitrule;               // splitting rule
  bool swr;                       // sampling with replacement (true) or not (false)
  double sample_rate;           // the subsampling rate (if double_bootstrap == true, applies to both subsets)
  bool double_bootstrap;          // use bootstrap separately (true) or not (false), only relevant when honest == true
  bool honest;                    // true if the trees in the forest are honest, otherwise false

  // other options (need to think about this further)
  unsigned int nworkers;
  bool save_predictions;              // if true, save predictions for each observation, allowing for much
                                      // faster error computations at the cost of increased memory usage
  vector<vector<bool>> oob_indices;   // the oob indices for each tree

  // misc. information
  vector<size_t> num_terminal_nodes;    // number of terminal nodes in each tree
  vector<size_t> num_nodes;             // number of nodes in each tree
  vector<size_t> tree_depths;           // the depth of each tree
  double avg_num_terminal_nodes;        // average number of terminal nodes
  double avg_num_nodes;                 // average number of nodes
  double avg_tree_depth;                // the average depth of the trees

  // random number generator
  mt19937 random_number_generator;

  // function to compute all quantities of interest after the forest is grown
  void computeForestQuantities();
  vector<vector<double>> shuffledFeatureValues(const vector<vector<size_t>>& oob_indices_non_bool, size_t feature, int feature_seed);
};

void OOBNonBoolIndices(vector<vector<size_t>>& oob_indices_non_bool, const vector<vector<bool>>& oob_indices);

#endif // FOREST_H
// some of these include statements should probably be moved later
#ifndef TREE_H
#define TREE_H

#include "Data.h"
#include <memory>
#include <variant>
#include <optional>
#include <limits>

using namespace std;
// so that we may handle predictions for different types of trees (extend continuously)
using ValueType = variant<double, size_t, vector<double>, vector<vector<double>>>;

class Tree {
public:
  // function to initialise a general Tree (later many more options should be added such as honesty, max_depth etc.)
  void initialise(shared_ptr<Data> data, unsigned int mtry, unsigned int min_node_size, double max_depth,
                  unsigned int nsplits, string splitrule, bool honest, unsigned int seed,
                  double splitrule_par = numeric_limits<double>::quiet_NaN());
  void setRNG(mt19937 rng);

  virtual ~Tree() = default;

  // grows the tree
  void grow();

  // predicted value depending on the type of tree (must be overriden by a derived Tree class)
  virtual ValueType predict(const vector<double>& x) = 0;               // predicting on a single observation

  // returns the ID of the leaf containing x
  size_t predictionLeafID(const vector<double>& x);
  // returns the leaf for a training observation without materialising its feature row
  size_t predictionLeafID(size_t observation);
  // returns the leaf for an observation in another dataset without materialising its feature row
  size_t predictionLeafID(const Data& prediction_data, size_t observation);
  // as above, but substitutes one feature value (used by permutation VIMP)
  size_t predictionLeafIDPermuted(size_t observation, size_t feature, double value);
  // as above, but routes a training observation without materialising its feature row
  size_t predictionLeafIDVIMP(size_t observation, size_t feature, mt19937& rng);

  // returns the size of the node containing x
  size_t nodeSize(const vector<double>& x) {
    return node_sizes[predictionLeafID(x)];
  }

  // functions to get tree info
  const vector<size_t>& getLeftDaughters() const {
    return left_daughters;
  }

  const vector<size_t>& getFeatureIDs() const {
    return feature_IDs;
  }

  const vector<vector<double>> getThresholds() const {
    return thresholds;
  }

  const vector<size_t> getDepths() const {
    return depths;
  }

  size_t getNumberOfTerminalNodes() const {
    return num_terminal_nodes;
  }

  size_t getNumberOfNodes() const {
    return num_nodes;
  }

  size_t getTreeDepth() const {
    return tree_depth;
  }

  const vector<size_t>& getPredictionNodeIDs() const {
    return prediction_node_IDs;
  }

  shared_ptr<Data> getData() const {
    return data;
  }

protected:
  // pointer to the data
  shared_ptr<Data> data;

  // node information
  vector<size_t> left_daughters;      // vector of the ID of the left daughter for each node
  vector<size_t> feature_IDs;         // ID of the feature in each node
  vector<vector<double>> thresholds;  // the threshold for each node
  vector<size_t> depths;              // the depth of each node

  // hyperparameters
  unsigned int mtry;              // number of variables randomly selected for each split
  unsigned int min_node_size;     // minimal number of observations in each node
  unsigned int nsplits;           // number of split values to consider after feature is chosen
  string splitrule;               // splitting rule
  double splitrule_par;           // optional scalar parameter used by parameterised splitting rules
  bool honest;                    // true if the trees in the forest are honest, otherwise false
  double max_depth;               // maximum allowed depth of a tree
  
  // misc. information
  size_t num_terminal_nodes;      // number of terminal nodes in the tree
  size_t num_nodes;               // number of nodes in the tree
  size_t tree_depth;              // depth of the tree
  vector<size_t> node_sizes;      // sizes of the nodes in the tree

  // information for growing trees
  vector<vector<size_t>> node_obs;          // vector of indices of the data currently being split on
  vector<vector<size_t>> holdout_node_obs;  // vector of indices of holdout data in the nodes (only used for honest trees)
  vector<size_t> prediction_node_IDs;       // vector of indices indicating which node, each observation belongs to (for fast prediction)

  // random number generator
  mt19937 random_number_generator;

  // for sampling split points in continuous splits
  size_t sampleSplitPoints(vector<double>& split_points, const vector<size_t>& indices, size_t feature);
  bool generateCategoricalPartitions(const vector<double>& feature_values,
                                     unordered_set<uint64_t>& partition_masks,
                                     bool has_missing_values = false);

  // protected functions to grow trees
  virtual bool createSplit(size_t node_index) = 0;

  // frees memory after fitting is complete
  virtual void cleanUpTree() = 0;
};

// general helper functions used for growing trees
//--------------------------------------------------------------------------------------

// find all unique values of a vector of size_t
vector<double> uniqueValues(vector<double> input);

#endif // TREE_H

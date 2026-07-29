#ifndef TREE_REGRESSION_H
#define TREE_REGRESSION_H

#include "Tree.h"

enum class RegressionSplitRule : uint8_t {
  MSE,
  MAE,
  Binomial,
  NegativeBinomial,
  Poisson,
  Gamma,
  InverseGaussian,
  Tweedie,
  Huber
};

RegressionSplitRule regressionSplitRuleFromString(const string& splitrule);
bool regressionSplitRuleUsesParameter(RegressionSplitRule splitrule);
void validateRegressionResponse(const vector<double>& response, RegressionSplitRule splitrule,
                                double splitrule_par);

class RegressionTree : public Tree {
public:
  RegressionTree(vector<size_t> subset_indices, vector<size_t> estimation_indices = {});

  const vector<double>& getMeans() const {
    return means;
  }

  // prediction for regression trees
  ValueType predict(const vector<double>& x) override {
    return predictValue(x);
  }
  double predictValue(const vector<double>& x) {
    return means[predictionLeafID(x)];
  }
  double predictValue(size_t observation) {
    return means[predictionLeafID(observation)];
  }
  double predictValue(const Data& prediction_data, size_t observation) {
    return means[predictionLeafID(prediction_data, observation)];
  }
  vector<double> computePredictions(const Data& new_data);
  // VIMP prediction for regression trees
  ValueType predictVIMP(const vector<double>& x, size_t feature, mt19937& rng) {
    return means[predictionLeafIDVIMP(x, feature, rng)];
  }

private:
  struct MeanComponents {
    long double sum = 0;
    long double correction = 0;
    size_t count = 0;
  };

  vector<double> means;     // the means in each terminal node

  // temporary quantities used in growing regression trees
  vector<size_t> num_obs_right;    // number of observations at the splitting points in the right node
  vector<long double> sums_right;  // sums of responses at the splitting points in the right node
  vector<long double> sum_node;    // sums of responses in each node
  vector<size_t> feature_indices;  // feature indices sampled at each node
  RegressionSplitRule splitrule_id = RegressionSplitRule::MSE;
  long double tweedie_node_scale = 1;
  long double candidate_parent_mean = 0;
  long double candidate_left_mean = 0;
  long double candidate_right_mean = 0;
  long double candidate_left_difference = 0;
  long double candidate_right_difference = 0;
  long double candidate_failure_parent_mean = 0;
  long double candidate_left_failure_mean = 0;
  long double candidate_right_failure_mean = 0;
  long double candidate_left_failure_difference = 0;
  long double candidate_right_failure_difference = 0;
  MeanComponents candidate_tweedie_left;
  MeanComponents candidate_tweedie_right;
  MeanComponents candidate_tweedie_parent;
  MeanComponents best_tweedie_left;
  MeanComponents best_tweedie_right;
  bool best_tweedie_uses_power_sum = false;

  // growing regression trees
  long double computeSum(const vector<size_t>& indices);
  long double computeMedianSorted(const vector<double>& response_values);
  long double computeHuberCenterSorted(const vector<double>& response_values);
  long double computeMSESplitValue(size_t n_left, long double sum_left,
                                   size_t n_right, long double sum_right);
  long double computeMAESplitValue(const vector<double>& left_responses,
                                   const vector<double>& right_responses,
                                   long double parent_center);
  long double computeBinomialSplitValue(size_t n_left, long double sum_left,
                                        long double failure_sum_left,
                                        size_t n_right, long double sum_right,
                                        long double failure_sum_right);
  long double computeNegativeBinomialSplitValue(
    size_t n_left, long double sum_left, size_t n_right, long double sum_right);
  long double computePoissonSplitValue(size_t n_left, long double sum_left,
                                       size_t n_right, long double sum_right);
  long double computeGammaSplitValue(size_t n_left, long double sum_left,
                                     size_t n_right, long double sum_right);
  long double computeInverseGaussianSplitValue(
    size_t n_left, long double sum_left, size_t n_right, long double sum_right);
  long double computeTweedieSplitValue(size_t n_left, long double sum_left,
                                       size_t n_right, long double sum_right);
  long double computeHuberSplitValue(const vector<double>& left_responses,
                                     const vector<double>& right_responses,
                                     long double parent_center);
  bool useTweediePowerSumComparison(long double split_value) const;
  int compareTweediePowerSums(
    const MeanComponents& first_left, const MeanComponents& first_right,
    const MeanComponents& second_left, const MeanComponents& second_right) const;
  bool isBetterSplit(long double split_value, long double best_split_value,
                     bool found_split);
  long double computeSplitValue(
    size_t n_left, long double sum_left, size_t n_right, long double sum_right,
    long double sum_correction_left, long double sum_correction_right,
    long double failure_sum_left, long double failure_sum_right,
    long double failure_correction_left, long double failure_correction_right,
    const vector<double>& left_responses, const vector<double>& right_responses,
    long double parent_robust_center);
  void reserveTreeMemory(size_t num_obs);
  void makeLeaf(size_t node_index);
  bool createSplit(size_t node_index) override;
  void bestSplitContinuous(size_t node_index, size_t feature,
                           const vector<size_t>& response_order,
                           long double parent_robust_center,
                           long double& best_split_val, bool& found_split,
                           size_t& best_feature,
                           vector<double>& best_threshold,
                           long double& best_sum_left);
  void bestSplitCategorical(size_t node_index, size_t feature,
                            long double& best_split_val, bool& found_split,
                            size_t& best_feature, vector<double>& best_threshold,
                            long double& best_sum_left,
                            const vector<size_t>& response_order,
                            long double parent_robust_center);

  // frees memory from temporary quantities used in growing the tree
  void cleanUpTree() override {
    vector<vector<size_t>>().swap(node_obs);
    vector<vector<size_t>>().swap(holdout_node_obs);
    vector<size_t>().swap(num_obs_right);
    vector<long double>().swap(sums_right);
    vector<long double>().swap(sum_node);
    vector<size_t>().swap(feature_indices);
    left_daughters.shrink_to_fit();
    feature_IDs.shrink_to_fit();
    thresholds.shrink_to_fit();
    depths.shrink_to_fit();
    node_sizes.shrink_to_fit();
    means.shrink_to_fit();
  }
};

double computeMSE(const vector<double>& predictions, const vector<double>& response);
double computeR2(double mse, const vector<double>& response);

#endif // TREE_REGRESSION_H

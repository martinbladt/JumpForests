#include "JumpForests.h"

namespace {

vector<double> eventTimesAtIDs(const vector<double>& event_times, const vector<size_t>& ids) {
  vector<double> selected_event_times;
  selected_event_times.reserve(ids.size());
  for (size_t id : ids) {
    selected_event_times.push_back(event_times[id]);
  }
  return selected_event_times;
}

struct MultistateScoreWeights {
  vector<double> brier;
  vector<double> kl;
};

MultistateScoreWeights multistateScoreWeights(const NumericVector& state_weights, size_t num_states) {
  if (state_weights.size() == 0) {
    /*
      Brier averages its contributions over every state, whereas categorical KL contributes only for the observed state.
      Keeping separate defaults makes a two-state multi-state model agree with the corresponding survival model for both.
    */
    return {
      vector<double>(num_states, 1 / static_cast<double>(num_states)),
      vector<double>(num_states, 1.0)
    };
  }
  if (static_cast<size_t>(state_weights.size()) != num_states) {
    throw runtime_error("The length of state_weights must be equal to the number of states");
  }

  vector<double> state_weights_cpp = as<vector<double>>(state_weights);
  for (double weight : state_weights_cpp) {
    if (!R_finite(weight) || weight < 0) {
      throw runtime_error("state_weights must contain finite non-negative values");
    }
  }
  // explicitly supplied state weights retain their previous meaning for both scores
  return {state_weights_cpp, state_weights_cpp};
}

struct FlemingHarringtonWeights {
  shared_ptr<const vector<double>> a;
  shared_ptr<const vector<double>> b;
};

vector<double> validateFlemingHarringtonExponents(const NumericVector& values, size_t num_states,
                                                  const string& argument_name) {
  if (values.size() == 0) {
    return vector<double>(num_states, 1.0);
  }
  if (static_cast<size_t>(values.size()) != num_states) {
    throw runtime_error(
      argument_name + " must have length " + to_string(num_states) +
      " (one exponent per state); got " + to_string(values.size()));
  }

  vector<double> result = as<vector<double>>(values);
  for (size_t i = 0; i < result.size(); ++i) {
    if (!R_finite(result[i]) || result[i] < 0) {
      throw runtime_error(
        argument_name + "[" + to_string(i + 1) + "] must be finite and non-negative");
    }
  }
  return result;
}

FlemingHarringtonWeights flemingHarringtonWeights(
    const string& splitrule, const NumericVector& weights_a, const NumericVector& weights_b,
    size_t num_states) {
  if (splitrule != "flemingharrington") {
    if (weights_a.size() != 0 || weights_b.size() != 0) {
      throw runtime_error(
        "fh_weights_a and fh_weights_b may only be supplied when "
        "splitrule = \"flemingharrington\"");
    }
    return {};
  }

  return {
    make_shared<const vector<double>>(
      validateFlemingHarringtonExponents(weights_a, num_states, "fh_weights_a")),
    make_shared<const vector<double>>(
      validateFlemingHarringtonExponents(weights_b, num_states, "fh_weights_b"))
  };
}

void saveFlemingHarringtonMetadata(List& result, const FlemingHarringtonWeights& weights) {
  if (weights.a == nullptr || weights.b == nullptr) {
    return;
  }

  result["fh.weights.a"] = wrap(*weights.a);
  result["fh.weights.b"] = wrap(*weights.b);
}

double regressionSplitRuleParameter(RegressionSplitRule splitrule,
                                    const NumericVector& splitrule_par) {
  if (!regressionSplitRuleUsesParameter(splitrule)) {
    if (splitrule_par.size() != 0) {
      throw runtime_error(
        "splitrule_par may only be supplied for negativebinomial, tweedie or huber");
    }
    return numeric_limits<double>::quiet_NaN();
  }
  if (splitrule == RegressionSplitRule::NegativeBinomial &&
      splitrule_par.size() == 0) {
    return 1.0;
  }
  if (splitrule_par.size() != 1) {
    throw runtime_error(
      "splitrule_par must contain exactly one value for a parameterised regression splitrule");
  }

  double value = splitrule_par[0];
  if (!R_finite(value)) {
    throw runtime_error("splitrule_par must be finite");
  }
  if (splitrule == RegressionSplitRule::NegativeBinomial && value <= 0) {
    throw runtime_error(
      "splitrule_par must be strictly positive (negative-binomial size k > 0)");
  }
  if (splitrule == RegressionSplitRule::Huber && value <= 0) {
    throw runtime_error(
      "splitrule_par must be strictly positive (Huber delta > 0)");
  }
  if (splitrule == RegressionSplitRule::Tweedie && value > 0 && value < 1) {
    throw runtime_error(
      "splitrule_par must satisfy xi <= 0 or xi >= 1 for the tweedie splitrule");
  }
  return value;
}

}

/*

The functions below are master functions that wrap the fitted objects the user writes in R to C++. JFCppTree is for fitting
an individual decision tree, while JFCppForest is for fitting a whole forest (the standard application)

tree_type:
1: Regression
2: Classification
3: Survival
4: Multi-state

*/

// growing decision trees
//--------------------------------------------------------------------------------------

// [[Rcpp::export]]
List JFCppTree(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, CharacterVector splitrule, 
               bool honest, NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical, NumericVector unique, 
               unsigned int seed, size_t num_event_times = 0,
               NumericVector splitrule_par = NumericVector()) {
  
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  string splitrule_cpp = as<string>(splitrule);

  if (tree_type != 1 && splitrule_par.size() != 0) {
    throw runtime_error(
      "splitrule_par is only valid for negativebinomial, tweedie or huber regression models");
  }

  // make the data into a C++ format and save it via a shared pointer
  shared_ptr<Data> data = make_shared<Data>(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);

  // use all indices since we grow a single tree
  vector<size_t> subset_indices_cpp;
  for (size_t i = 0; i < data->getNumberOfObs(); ++i) {
    subset_indices_cpp.push_back(i);
  }

  // we save a list with all information about the tree
  List result = List::create(
    Named("num.obs") = data->getNumberOfObs(),
    Named("num.features") = data->getNumberOfFeatures(),
    Named("feature.names") = data->getFeatureNames(), 
    Named("response.names") = data->getResponseNames(),
    Named("splitrule") = splitrule,
    Named("mtry") = mtry,
    Named("min.node.size") = min_node_size,
    Named("nsplits") = nsplits,
    Named("honest") = honest
  );
  
  // the tree is a regression tree
  if (tree_type == 1) {
    if (min_node_size == 0) {
      throw runtime_error("The minimal node size must be at least one");
    }
    const vector<bool>& categorical_features = data->getCategorical();
    vector<size_t> unique_feature_values = data->getUniqueValues();
    for (size_t i = 0; i < categorical_features.size(); ++i) {
      if (categorical_features[i] && unique_feature_values[i] > 63) {
        throw runtime_error("Categorical features with more than 63 values are not supported");
      }
    }

    RegressionSplitRule splitrule_id = regressionSplitRuleFromString(splitrule_cpp);
    double splitrule_par_cpp =
      regressionSplitRuleParameter(splitrule_id, splitrule_par);
    validateRegressionResponse(data->get_y(), splitrule_id, splitrule_par_cpp);
    if (regressionSplitRuleUsesParameter(splitrule_id)) {
      result["splitrule.par"] = splitrule_par_cpp;
    }

    RegressionTree* tree;
    if (!honest) {
      tree = new RegressionTree(subset_indices_cpp);
    } else {
      mt19937 rng(seed + 1);
      pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(subset_indices_cpp, rng);
      tree = new RegressionTree(partition.first, partition.second);
      tree->setRNG(rng);
    }
    tree->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, honest,
                     seed, splitrule_par_cpp);
    tree->grow();

    XPtr<RegressionTree> regression_tree(tree, true);   // cast the regression tree as an R pointer
    result["tree.type"] = "Regression";
    result["Tree"] = regression_tree;                   // add the tree (as a pointer, only to be used for prediction in C++)
    JFCppTreePredict(result);                           // compute and save predictions on the data
    vector<double> response = as<vector<double>>(df[response_indices_cpp[0]]);
    JFCppTreeErrorRegression(result, response);         // compute and save error estimate

    // save information about the tree itself
    result["num.nodes"] = tree->getNumberOfNodes();
    result["num.terminal.nodes"] = tree->getNumberOfTerminalNodes();
    result["tree.depth"] = tree->getTreeDepth();
  }

  // the tree is a classification tree
  if (tree_type == 2) {
    if (min_node_size == 0) {
      throw runtime_error("The minimal node size must be at least one");
    }
    const vector<bool>& categorical_features = data->getCategorical();
    vector<size_t> unique_feature_values = data->getUniqueValues();
    for (size_t i = 0; i < categorical_features.size(); ++i) {
      if (categorical_features[i] && unique_feature_values[i] > 63) {
        throw runtime_error("Categorical features with more than 63 values are not supported");
      }
    }

    // check validity of splitrule argument
    vector<string> valid_splitrules = {"gini", "entropy", "misc", "twoing", "hellinger"};
    if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
      throw runtime_error("Invalid splitrule, please choose between gini, entropy, misc, twoing or hellinger");
    }

    ClassificationTree* tree;
    if (!honest) {
      tree = new ClassificationTree(std::move(subset_indices_cpp));
    } else {
      mt19937 rng(seed + 1);
      pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(subset_indices_cpp, rng);
      tree = new ClassificationTree(std::move(partition.first), std::move(partition.second));
      tree->setRNG(rng);
    }
    tree->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, honest, seed);
    tree->grow();

    XPtr<ClassificationTree> classification_tree(tree, true);   // cast the classification tree as an R pointer
    result["tree.type"] = "Classification";
    result["num.classes"] = data->getNumClasses();
    result["Tree"] = classification_tree;                       // add the tree (as a pointer, only to be used for prediction in C++)
    JFCppTreePredict(result);                                   // compute and save predictions on the data
    vector<double> response = as<vector<double>>(df[response_indices_cpp[0]]);
    JFCppTreeErrorClassification(result, response);             // compute and save error estimate

    // save information about the tree itself
    result["num.nodes"] = tree->getNumberOfNodes();
    result["num.terminal.nodes"] = tree->getNumberOfTerminalNodes();
    result["tree.depth"] = tree->getTreeDepth();
  }

  // the tree is a survival tree
  if (tree_type == 3) {
    if (min_node_size == 0) {
      throw runtime_error("The minimal node size must be at least one");
    }
    const vector<bool>& categorical_features = data->getCategorical();
    vector<size_t> unique_feature_values = data->getUniqueValues();
    for (size_t i = 0; i < categorical_features.size(); ++i) {
      if (categorical_features[i] && unique_feature_values[i] > 63) {
        throw runtime_error("Categorical features with more than 63 values are not supported");
      }
    }

    // determine the unique sorted response times
    vector<double> times = as<vector<double>>(df[response_indices_cpp[0]]);
    vector<double> ind = as<vector<double>>(df[response_indices_cpp[1]]);
    vector<double> unique_event_times = uniqueValues(times);
    vector<double> censoring_times = unique_event_times;

    // if the user has specified a number of event times, thin the vector of unique event times
    if (num_event_times > 0) {
      unique_event_times = thinUniqueEventTimes(unique_event_times, num_event_times);
    }

    vector<size_t> response_event_time_ids = computeResponseEventTimeIDs(unique_event_times, times);
    vector<size_t> true_event_time_ids = computeTrueEventTimeIDs(unique_event_times, response_event_time_ids, ind);
    vector<double> true_unique_event_times = eventTimesAtIDs(unique_event_times, true_event_time_ids);
    vector<size_t> true_response_event_time_ids = computeResponseEventTimeIDs(true_unique_event_times, times);

    // create and grow the survival tree
    shared_ptr<vector<double>> unique_event_times_ptr = make_shared<vector<double>>(unique_event_times);
    shared_ptr<vector<size_t>> response_event_time_ids_ptr = make_shared<vector<size_t>>(response_event_time_ids);
    shared_ptr<vector<size_t>> true_event_time_ids_ptr = make_shared<vector<size_t>>(true_event_time_ids);
    shared_ptr<vector<double>> censoring_times_ptr = make_shared<vector<double>>(censoring_times);

    // check validity of splitrule argument
    vector<string> valid_splitrules = {"logrank", "conserve", "approxlogrank"};
    if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
      throw runtime_error("Invalid splitrule, please choose between logrank, conserve or approxlogrank");
    }

    SurvivalTree* tree;
    if (!honest) {
      tree = new SurvivalTree(unique_event_times_ptr, response_event_time_ids_ptr, true_event_time_ids_ptr,
        std::move(subset_indices_cpp), true, {}, censoring_times_ptr);
    } else {
      mt19937 rng(seed + 1);
      pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(subset_indices_cpp, rng);
      tree = new SurvivalTree(unique_event_times_ptr, response_event_time_ids_ptr, true_event_time_ids_ptr,
        std::move(partition.first), true, std::move(partition.second), censoring_times_ptr);
      tree->setRNG(rng);
    }
    
    tree->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, honest, seed);
    tree->grow();
    
    // specific to survival
    NumericVector unique_event_times_R(true_unique_event_times.begin(), true_unique_event_times.end());
    XPtr<SurvivalTree> survival_tree(tree, true);   // cast the survival tree as an R pointer
    result["num.deaths"] = vector_sum(ind);
    result["tree.type"] = "Survival";
    result["unique.event.times"] = unique_event_times_R;
    result["Tree"] = survival_tree;                 // add the tree (as a pointer, only to be used for prediction in C++)
    JFCppTreePredict(result);                       // compute and save predictions on the data
    JFCppTreeErrorSurvival(result, times, ind, true_unique_event_times, true_response_event_time_ids);     // compute and save error estimate

    // save information about the tree itself
    result["num.nodes"] = tree->getNumberOfNodes();
    result["num.terminal.nodes"] = tree->getNumberOfTerminalNodes();
    result["tree.depth"] = tree->getTreeDepth();
  }
  return result;
}

// [[Rcpp::export]]
List JFCppTreeMultistate(List jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame df_features, unsigned int mtry, unsigned int min_node_size,
                 unsigned int nsplits, CharacterVector splitrule, bool honest, NumericVector feature_indices, LogicalVector categorical, NumericVector unique, 
                 unsigned int seed, NumericVector state_weights, size_t num_event_times, NumericVector fh_weights_a, NumericVector fh_weights_b) {

  // convert the input to C++ vectors
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  string splitrule_cpp = as<string>(splitrule);

  // make the data into a C++ format and save it via a shared pointer
  shared_ptr<Data> data = make_shared<Data>(jump_data, max_response_length, num_states, df_features, feature_indices_cpp, categorical_cpp, unique_cpp);

  // use all indices since we grow a single tree
  vector<size_t> subset_indices_cpp;
  for (int i = 0; i < data->getNumberOfObs(); ++i) {
    subset_indices_cpp.push_back(i);
  }

  // we save a list with all information about the tree
  List result = List::create(
    Named("num.obs") = data->getNumberOfObs(),
    Named("num.features") = data->getNumberOfFeatures(),
    Named("feature.names") = data->getFeatureNames(), 
    Named("splitrule") = splitrule,
    Named("mtry") = mtry,
    Named("min.node.size") = min_node_size,
    Named("nsplits") = nsplits,
    Named("honest") = honest
  );

  // determine the (sorted) unique event times
  vector<double> unique_event_times = uniqueEventTimesMultistate(data->getTimes(), data->getStates());

  // if the user has specified a number of event times, thin the vector of unique event times
  if (num_event_times > 0) {
    unique_event_times = thinUniqueEventTimes(unique_event_times, num_event_times);
  }
  vector<size_t> response_event_time_ids = computeResponseEventTimeIDsMultistate(unique_event_times, data->getTimes(), data->getStates(), data->getMaxResponseLength());
  vector<double> censoring_times = uniqueCensoringTimesMultistate(unique_event_times, data->getTimes(), data->getLastObservedTimes());

  // create and grow the multi-state tree
  shared_ptr<vector<double>> unique_event_times_ptr = make_shared<vector<double>>(unique_event_times);
  shared_ptr<vector<size_t>> response_event_time_ids_ptr = make_shared<vector<size_t>>(response_event_time_ids);
  shared_ptr<vector<double>> censoring_times_ptr = make_shared<vector<double>>(censoring_times);

  // check validity of splitrule argument
  vector<string> valid_splitrules = {"logrank", "gehan", "taroneware", "conserve", "approxlogrank", "petoprentice", "flemingharrington"};
  if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
    throw runtime_error("Invalid splitrule, please choose between logrank, gehan, taroneware, conserve, approxlogrank, petoprentice or flemingharrington");
  }

  FlemingHarringtonWeights fh_weights = flemingHarringtonWeights(
    splitrule_cpp, fh_weights_a, fh_weights_b, num_states);
  saveFlemingHarringtonMetadata(result, fh_weights);

  MultistateTree* tree;
  if (!honest) {
    tree = new MultistateTree(
      unique_event_times_ptr, response_event_time_ids_ptr, subset_indices_cpp, num_states,
      true, {}, censoring_times_ptr, nullptr, nullptr, fh_weights.a, fh_weights.b);
  } else {
    mt19937 rng(seed + 1);
    pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(subset_indices_cpp, rng);
    tree = new MultistateTree(
      unique_event_times_ptr, response_event_time_ids_ptr, partition.first, num_states,
      true, partition.second, censoring_times_ptr, nullptr, nullptr, fh_weights.a, fh_weights.b);
    tree->setRNG(rng);
  }

  tree->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, honest, seed);
  tree->grow();

  // specific to multi-states
  NumericVector unique_event_times_R(unique_event_times.begin(), unique_event_times.end());
  XPtr<MultistateTree> multistate_tree(tree, true);   // cast the multi-state tree as an R pointer
  result["tree.type"] = "Multi-state";
  result["unique.event.times"] = unique_event_times_R;
  result["Tree"] = multistate_tree;               // add the tree (as a pointer, only to be used for prediction in C++)
  JFCppTreePredict(result);                       // compute and save predictions on the data

  MultistateScoreWeights score_weights = multistateScoreWeights(state_weights, num_states);

  // compute and save error estimate (need to come up with something for multi-states)
  vector<double> censoring_indicators(data->getNumberOfObs(), 0);
  const vector<uint8_t> censoring_states = data->getCensoringStates();
  for (size_t i = 0; i < data->getNumberOfObs(); ++i) {
    if (censoring_states[i] == 0) {
      censoring_indicators[i] = 1;    // censoring_state is 0 if and only if censoring has not occured
    }
  }
  JFCppTreeErrorMultistate(result, data->getTimes(), data->getLastObservedTimes(), censoring_indicators,
                           unique_event_times, response_event_time_ids, score_weights.brier, score_weights.kl);

  // save information about the tree itself
  result["num.nodes"] = tree->getNumberOfNodes();
  result["num.terminal.nodes"] = tree->getNumberOfTerminalNodes();
  result["tree.depth"] = tree->getTreeDepth();
  return result;
}

/*

The following functions are for prediction with a single tree. For single trees, predictions for the data used to fit
are always computed and saved in the list (see the JFCppTree function above)

*/

// predicting with decision trees
//--------------------------------------------------------------------------------------

void JFCppTreePredict(List& JFTree) {
  string type = as<string>(JFTree["tree.type"]);
  size_t num_obs = as<size_t>(JFTree["num.obs"]);
  if (type == "Regression") {
    RegressionTree* tree = ((XPtr<RegressionTree>) JFTree["Tree"]).get();
    NumericVector predictions(num_obs);
    const vector<double>& means = tree->getMeans();
    const vector<size_t>& prediction_node_IDs = tree->getPredictionNodeIDs();

    // we saved the corresponding terminal node ID for every observation
    for (int i = 0; i < num_obs; ++i) {
      predictions[i] = means[prediction_node_IDs[i]];
    }
    JFTree["predictions"] = predictions;
  }
  if (type == "Classification") {
    ClassificationTree* tree = ((XPtr<ClassificationTree>) JFTree["Tree"]).get();
    size_t num_classes = as<size_t>(JFTree["num.classes"]);
    NumericVector predictions(num_obs);
    NumericMatrix predictions_prob(num_obs, num_classes);

    // fetch predictions, first vector is the predicted classes, second a flattened vector of class probabilities
    const pair<vector<double>, vector<double>>& predictions_cpp = tree->computePredictions();

    for (size_t i = 0; i < num_obs; ++i) {
      predictions[i] = predictions_cpp.first[i];
      size_t index = i * num_classes;
      for (size_t c = 0; c < num_classes; ++c) {
        predictions_prob(i, c) = predictions_cpp.second[index + c];
      }
    }

    JFTree["predictions"] = predictions;
    JFTree["predictions.prob"] = predictions_prob;
  }
  if (type == "Survival") {
    SurvivalTree* tree = ((XPtr<SurvivalTree>) JFTree["Tree"]).get();
    const vector<size_t>& true_event_time_ids = tree->getTrueEventTimeIDs();
    size_t num_unique_event_times = true_event_time_ids.size();
    size_t num_censoring_times = tree->getCensoringTimes().size();
    NumericMatrix predictions(num_obs, num_unique_event_times);
    NumericMatrix censoring(num_obs, num_unique_event_times);
    NumericMatrix censoring_full(num_obs, num_censoring_times);
    const vector<size_t>& leaf_ids = tree->getPredictionNodeIDs();
    const vector<vector<double>>& chf = tree->getCHF();
    const vector<vector<double>>& km_censoring = tree->getKMCensoring();
    const vector<vector<double>>& km_censoring_full = tree->getKMCensoringFull();

    // we saved the corresponding terminal node ID for every observation
    for (int i = 0; i < num_obs; ++i) {
      size_t leaf_id = leaf_ids[i];
      const vector<double>& pred = chf[leaf_id];
      const vector<double>& cens = km_censoring[leaf_id];
      for (size_t t = 0; t < num_unique_event_times; ++t) {
        size_t event_time_id = true_event_time_ids[t];
        predictions(i, t) = pred[event_time_id];
        censoring(i, t) = cens[event_time_id];
      }
      copy(km_censoring_full[leaf_id].begin(), km_censoring_full[leaf_id].end(), censoring_full.row(i).begin());
    }
    
    JFTree["predictions"] = predictions;
    JFTree["predictions.km"] = KaplanMeier(predictions);
    JFTree["censoring"] = censoring;
    JFTree["censoring.full"] = censoring_full;
    JFTree["censoring.times"] = tree->getCensoringTimes();
  }

  if (type == "Multi-state") {
    MultistateTree* tree = ((XPtr<MultistateTree>) JFTree["Tree"]).get();
    size_t num_states = tree->getData()->getNumberOfStates();
    size_t num_unique_event_times = tree->getNumberOfUniqueEventTimes();
    size_t num_censoring_times = tree->getCensoringTimes().size();
    const vector<size_t>& leaf_ids = tree->getPredictionNodeIDs();
    const vector<vector<double>>& na = tree->getNA();
    const vector<vector<double>>& init = tree->getInitDist();
    const vector<vector<double>>& cens = tree->getKMCensoring();
    const vector<vector<double>>& cens_full = tree->getKMCensoringFull();

    List predictions(num_obs);                                  // each prediction is a list of matrices
    List predictions_init(num_obs);                             // each predicted initial distribution is a vector
    NumericMatrix censoring(num_obs, num_unique_event_times);   // a matrix with the KM estimator for observation i along the i'th row
    NumericMatrix censoring_full(num_obs, num_censoring_times);
    List prediction_cache(na.size());                           // R predictions are materialised once for each visited leaf
    List initial_cache(init.size());

    // we saved the corresponding terminal node ID for every observation
    for (size_t i = 0; i < num_obs; ++i) {
      size_t leaf_id = leaf_ids[i];
      if (Rf_isNull(prediction_cache[leaf_id])) {
        List leaf_prediction(num_unique_event_times);
        const vector<double>& pred = na[leaf_id];
        for (size_t j = 0; j < num_unique_event_times; ++j) {
          auto start_it = pred.begin() + (j * num_states * num_states);
          NumericMatrix pred_time(num_states, num_states, start_it);
          leaf_prediction[j] = transpose(pred_time);
        }
        prediction_cache[leaf_id] = leaf_prediction;

        const vector<double>& leaf_initial = init[leaf_id];
        initial_cache[leaf_id] = NumericVector(leaf_initial.begin(), leaf_initial.end());
      }

      // observations in one leaf can safely share the same immutable R prediction until R's copy-on-write is triggered
      predictions[i] = prediction_cache[leaf_id];
      predictions_init[i] = initial_cache[leaf_id];

      // save censoring KM estimator
      const vector<double>& cens_pred = cens[leaf_id];
      copy(cens_pred.begin(), cens_pred.end(), censoring.row(i).begin());
      const vector<double>& cens_pred_full = cens_full[leaf_id];
      copy(cens_pred_full.begin(), cens_pred_full.end(), censoring_full.row(i).begin());
    }
    JFTree["predictions"] = predictions;
    JFTree["init"] = predictions_init;
    JFTree["censoring"] = censoring;
    JFTree["censoring.full"] = censoring_full;
    JFTree["censoring.times"] = tree->getCensoringTimes();
  }
}

// function to predict on a new dataset (Regression, Classification and Survival)

// [[Rcpp::export]]
NumericMatrix JFCppTreePredict(const List& JFTree, DataFrame df, NumericVector feature_indices, 
                               LogicalVector categorical, NumericVector unique) {
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp;  // need an empty vector for the response indices
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  
  // convert the new data to a suitable Data object
  Data new_data = Data(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);
  size_t num_obs = new_data.getNumberOfObs();

  string type = as<string>(JFTree["tree.type"]);
  if (type == "Regression") {
    RegressionTree* tree = ((XPtr<RegressionTree>) JFTree["Tree"]).get();
    NumericMatrix predictions(num_obs, 1);
    for (size_t i = 0; i < num_obs; ++i) {
      predictions(i, 0) = tree->predictValue(new_data, i);
    }
    return predictions;
  }

  if (type == "Classification") {
    ClassificationTree* tree = ((XPtr<ClassificationTree>) JFTree["Tree"]).get();
    size_t num_classes = as<size_t>(JFTree["num.classes"]);
    NumericMatrix predictions(num_obs, num_classes + 1);
    
    // fetch predictions, first vector is the predicted classes, second a flattened vector of class probabilities
    const pair<vector<double>, vector<double>> predictions_cpp = tree->computePredictions(new_data);

    for (size_t i = 0; i < num_obs; ++i) {
      predictions(i, 0) = predictions_cpp.first[i];
      size_t index = i * num_classes;
      for (size_t c = 0; c < num_classes; ++c) {
        predictions(i, c + 1) = predictions_cpp.second[index + c];
      }
    }
    return predictions;
  }

  if (type == "Survival") {
    SurvivalTree* tree = ((XPtr<SurvivalTree>) JFTree["Tree"]).get();
    size_t num_all_event_times = tree->getEventTimes().size();
    const vector<size_t>& true_event_time_ids = tree->getTrueEventTimeIDs();
    size_t num_unique_event_times = true_event_time_ids.size();
    NumericMatrix predictions(num_obs, num_unique_event_times);

    const vector<double>& predictions_cpp = tree->computePredictions(new_data);
    for (size_t i = 0; i < num_obs; ++i) {
      for (size_t t = 0; t < num_unique_event_times; ++t) {
        predictions(i, t) = predictions_cpp[i * num_all_event_times + true_event_time_ids[t]];
      }
    }
    return predictions;
  }
  throw runtime_error("Type of tree not recognised");
}

// function to compute predictions and the predicted KM estimators for the censoring for survival trees

// [[Rcpp::export]]
List JFCppTreePredictCensoring(const List& JFTree, DataFrame df, NumericVector feature_indices, 
                               LogicalVector categorical, NumericVector unique) {
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp;  // need an empty vector for the response indices
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  
  // convert the new data to a suitable Data object
  Data new_data = Data(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);
  
  SurvivalTree* tree = ((XPtr<SurvivalTree>) JFTree["Tree"]).get();
  size_t num_obs = new_data.getNumberOfObs();
  const vector<size_t>& true_event_time_ids = tree->getTrueEventTimeIDs();
  vector<double> true_event_times = eventTimesAtIDs(tree->getEventTimes(), true_event_time_ids);
  size_t num_all_event_times = tree->getEventTimes().size();
  size_t num_unique_event_times = true_event_time_ids.size();
  size_t num_censoring_times = tree->getCensoringTimes().size();
  NumericMatrix predictions(num_obs, num_unique_event_times);
  NumericMatrix censoring(num_obs, num_unique_event_times);
  NumericMatrix censoring_full(num_obs, num_censoring_times);

  const pair<vector<double>, vector<double>>& predictions_cpp = tree->computePredictionsCensoring(new_data);
  vector<double> censoring_event_cpp = selectCensoringAtTimes(predictions_cpp.second, tree->getCensoringTimes(), true_event_times, num_obs);
  for (size_t i = 0; i < num_obs; ++i) {
    for (size_t t = 0; t < num_unique_event_times; ++t) {
      predictions(i, t) = predictions_cpp.first[i * num_all_event_times + true_event_time_ids[t]];
    }
    copy(censoring_event_cpp.begin() + i * num_unique_event_times, censoring_event_cpp.begin() + (i + 1) * num_unique_event_times, censoring.row(i).begin());
    copy(predictions_cpp.second.begin() + i * num_censoring_times, predictions_cpp.second.begin() + (i + 1) * num_censoring_times, censoring_full.row(i).begin());
  }
  List result = List::create(
    Named("predictions") = predictions,
    Named("censoring") = censoring,
    Named("censoring.full") = censoring_full,
    Named("censoring.times") = tree->getCensoringTimes()
  );
  return result;
}

/*
  function to predict on a new dataset for multi-states
  if compute_initial = false (default), simply return a list of the predictions on the new data df
  if compute_initial = true, add a list containing predicted initial distributions
  if compute_censoring = true, add a list containing the predicted censoring distributions (KM estimator)
*/

// [[Rcpp::export]]
List JFCppTreePredictMultistate(const List& JFTree, DataFrame df, NumericVector feature_indices, LogicalVector categorical,
                        NumericVector unique, bool compute_initial, bool compute_censoring) {
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp;  // need an empty vector for the response indices
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  
  // convert the new data to a suitable Data object (no need to use the multi-state edition for the new data)
  Data new_data = Data(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);

  // fetch multi-state tree and relevant quantities
  MultistateTree* tree = ((XPtr<MultistateTree>) JFTree["Tree"]).get();
  size_t num_states = tree->getData()->getNumberOfStates();
  size_t num_unique_event_times = tree->getNumberOfUniqueEventTimes();
  size_t num_censoring_times = tree->getCensoringTimes().size();
  size_t num_obs = new_data.getNumberOfObs();
  List predictions(num_obs);                                   // each prediction is a list of matrices
  List predictions_init(compute_initial ? num_obs : 0);        // optional outputs are not allocated unless requested
  NumericMatrix censoring(compute_censoring ? num_obs : 0, compute_censoring ? num_unique_event_times : 0);
  NumericMatrix censoring_full(compute_censoring ? num_obs : 0, compute_censoring ? num_censoring_times : 0);

  const vector<vector<double>>& na = tree->getNA();
  const vector<vector<double>>& init = tree->getInitDist();
  const vector<vector<double>>& cens = tree->getKMCensoring();
  const vector<vector<double>>& cens_full = tree->getKMCensoringFull();
  List prediction_cache(na.size());
  List initial_cache(compute_initial ? init.size() : 0);

  for (size_t i = 0; i < num_obs; ++i) {
    size_t leaf_id = tree->predictionLeafID(new_data, i);
    if (Rf_isNull(prediction_cache[leaf_id])) {
      List leaf_prediction(num_unique_event_times);
      const vector<double>& pred = na[leaf_id];
      for (size_t j = 0; j < num_unique_event_times; ++j) {
        auto start_it = pred.begin() + (j * num_states * num_states);
        NumericMatrix pred_time(num_states, num_states, start_it);
        leaf_prediction[j] = transpose(pred_time);
      }
      prediction_cache[leaf_id] = leaf_prediction;

      if (compute_initial) {
        const vector<double>& leaf_initial = init[leaf_id];
        initial_cache[leaf_id] = NumericVector(leaf_initial.begin(), leaf_initial.end());
      }
    }
    predictions[i] = prediction_cache[leaf_id];

    // if compute_initial == true, save initial distribution
    if (compute_initial) {
      predictions_init[i] = initial_cache[leaf_id];
    }
    // if compute_censoring == true, save censoring estimators
    if (compute_censoring) {
      copy(cens[leaf_id].begin(), cens[leaf_id].end(), censoring.row(i).begin());
      copy(cens_full[leaf_id].begin(), cens_full[leaf_id].end(), censoring_full.row(i).begin());
    }
  }

  if (compute_initial || compute_censoring) {
    List result = List::create(
      Named("predictions") = predictions
    );
    if (compute_initial) {
      result["predictions.init"] = predictions_init;
    }
    if (compute_censoring) {
      result["censoring"] = censoring;
      result["censoring.full"] = censoring_full;
      result["censoring.times"] = tree->getCensoringTimes();
    }
    return result;
  } else {
    return predictions;
  }
}

// error computation for decision trees
//--------------------------------------------------------------------------------------

// Regression

void JFCppTreeErrorRegression(List& JFTree, const vector<double>& response) {
  const vector<double>& predictions = as<vector<double>>(JFTree["predictions"]);
  JFTree["mse.error"] = computeMSE(predictions, response);
  JFTree["R2"] = computeR2(JFTree["mse.error"], response);
}

// Classification

void JFCppTreeErrorClassification(List& JFTree, const vector<double>& response) {
  size_t num_classes = as<size_t>(JFTree["num.classes"]);
  NumericVector class_misclassification_errors(num_classes);
  NumericMatrix confusionMatrix(num_classes, num_classes);
  NumericMatrix prob_predictions_matrix = as<NumericMatrix>(JFTree["predictions.prob"]);
  vector<double> prob_predictions(response.size() * num_classes);

  // compute and save overall misclassification error and class-wise misclassification error
  vector<double> misc = computeMisclassificationError(JFTree["predictions"], response, num_classes);
  JFTree["misc.error.total"] = misc[num_classes];
  for (size_t c = 0; c < num_classes; ++c) {
    class_misclassification_errors[c] = misc[c];
  }
  JFTree["misc.error"] = class_misclassification_errors;

  // compute and save the Brier score error
  for (size_t i = 0; i < response.size(); ++i) {
    size_t index = i * num_classes;
    for (size_t c = 0; c < num_classes; ++c) {
      prob_predictions[index + c] = prob_predictions_matrix(i, c);
    }
  }
  double brier_score = computeBrierScoreError(prob_predictions, response, num_classes);
  JFTree["bs"] = brier_score;
  JFTree["bs.normalised"] = brier_score * (double) num_classes * (double) num_classes / (double) (num_classes - 1);

  // compute and save the confusion matrix
  vector<size_t> confusion = computeConfusionMatrix(JFTree["predictions"], response, num_classes);
  for (size_t c1 = 0; c1 < num_classes; ++c1) {
    size_t row = c1 * num_classes;
    for (size_t c2 = 0; c2 < num_classes; ++c2) {
      confusionMatrix(c1, c2) = confusion[row + c2];
    }
  }
  JFTree["confusion"] = confusionMatrix;
}

// Survival

void JFCppTreeErrorSurvival(List& JFTree, const vector<double>& times, const vector<double>& ind, const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids) {
  // first compute Harrell's C-index
  const vector<double>& outcomes = computeOutcomes(JFTree["predictions"]);
  JFTree["outcomes"] = outcomes;
  JFTree["C.error"] = 1 - computeConcordanceIndex(outcomes, times, ind);

  // now compute Brier score
  vector<double> censoring_times = as<vector<double>>(JFTree["censoring.times"]);
  vector<double> IPCW_weights = computeIPCW(ind, unique_event_times, response_event_time_ids, JFTree["censoring.full"], times, {}, censoring_times);
  vector<double> brier = computeBrierScore(times, IPCW_weights, unique_event_times, JFTree["predictions.km"]);
  vector<double> kl = computeKLScore(times, IPCW_weights, unique_event_times, JFTree["predictions.km"]);
  pair<double, double> ibs = computeIntegratedScore(brier, unique_event_times);
  pair<double, double> ikl = computeIntegratedScore(kl, unique_event_times);
  JFTree["ibs"] = ibs.first;
  JFTree["ibs.normalised"] = ibs.second;
  JFTree["ikl"] = ikl.first;
  JFTree["ikl.normalised"] = ikl.second;
}

// Multi-state

void JFCppTreeErrorMultistate(List& JFTree, const vector<double>& times, const vector<size_t>& last_observed_time_ids, const vector<double>& ind, 
                              const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids,
                              const vector<double>& brier_state_weights, const vector<double>& kl_state_weights) {
  MultistateTree* tree = ((XPtr<MultistateTree>) JFTree["Tree"]).get();
  vector<vector<double>> error_predictions = tree->computeErrorPredictions();

  // first compute the IPCW weights directly from the leaf-level censoring predictions
  vector<double> IPCW_weights = computeIPCWCpp(times, ind, unique_event_times, response_event_time_ids, error_predictions[1], {}, last_observed_time_ids, tree->getCensoringTimes());

  // compute the status of whether each observation is in each state at the given event times
  vector<bool> states_ind = computeStateIndicatorsMultistate(*tree->getData(), response_event_time_ids, unique_event_times.size());

  // compute the integrated Brier score (IBS) and the normalised IBS
  vector<double> brier = computeBrierScoreCppMultistate(states_ind, IPCW_weights, unique_event_times,
                                                         error_predictions[0], brier_state_weights);
  pair<double, double> ibs = computeIntegratedScore(brier, unique_event_times, true);

  // compute the integrated Kullback-Leibler loss and the normalised IKL
  vector<double> kl = computeKLScoreCppMultistate(states_ind, IPCW_weights, unique_event_times,
                                                   error_predictions[0], kl_state_weights);
  pair<double, double> ikl = computeIntegratedScore(kl, unique_event_times, true);

  // save all the error metrics in the tree list
  JFTree["ibs"] = ibs.first;
  JFTree["ibs.normalised"] = ibs.second;
  JFTree["ikl"] = ikl.first;
  JFTree["ikl.normalised"] = ikl.second;
}

// computes the error based on a a new dataset, here we don't need to specify the type of tree beforehand
// [[Rcpp::export]]
List JFCppTreeError(const List& JFTree, DataFrame df, NumericVector feature_indices,
                    LogicalVector categorical, NumericVector unique, NumericVector response_indices) {
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // convert the new data to a suitable C++ Data object
  Data new_data = Data(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);
  size_t num_obs = new_data.getNumberOfObs();

  string type = as<string>(JFTree["tree.type"]);
  if (type == "Regression") {
    RegressionTree* tree = ((XPtr<RegressionTree>) JFTree["Tree"]).get();
    vector<double> predictions = tree->computePredictions(new_data);
    const vector<double>& response = new_data.get_y();
    List result;
    double mse = computeMSE(predictions, response);
    result["mse.error"] = mse;
    result["R2"] = computeR2(mse, response);
    return result;
  }
  if (type == "Classification") {
    ClassificationTree* tree = ((XPtr<ClassificationTree>) JFTree["Tree"]).get();
    const pair<vector<double>, vector<double>>& predictions = tree->computePredictions(new_data);
    const vector<double>& response = new_data.get_y();
    size_t num_classes = as<size_t>(JFTree["num.classes"]);

    NumericVector class_misclassification_errors(num_classes);
    NumericMatrix confusionMatrix(num_classes, num_classes);

    double brier_score = computeBrierScoreError(predictions.second, response, num_classes);
    List result = List::create(
      Named("BS.error") = brier_score,
      Named("normalised.BS.error") = brier_score * (double) num_classes * (double) num_classes / (double) (num_classes - 1)
    );

    vector<double> misc = computeMisclassificationError(predictions.first, response, num_classes);
    result["misclassification.error.total"] = misc[num_classes];
    for (size_t c = 0; c < num_classes; ++c) {
      class_misclassification_errors[c] = misc[c];
    }
    result["misclassification.error"] = class_misclassification_errors;

    vector<size_t> confusion = computeConfusionMatrix(predictions.first, response, num_classes);
    for (size_t c1 = 0; c1 < num_classes; ++c1) {
      size_t row = c1 * num_classes;
      for (size_t c2 = 0; c2 < num_classes; ++c2) {
        confusionMatrix(c1, c2) = confusion[row + c2];
      }
    }
    result["confusion.matrix"] = confusionMatrix;
    return result;

  }
  if (type == "Survival") {
    SurvivalTree* tree = ((XPtr<SurvivalTree>) JFTree["Tree"]).get();
    const vector<double>& all_unique_event_times = tree->getEventTimes();
    const vector<size_t>& true_event_time_ids = tree->getTrueEventTimeIDs();
    vector<double> unique_event_times = eventTimesAtIDs(all_unique_event_times, true_event_time_ids);
    size_t num_unique_event_times = all_unique_event_times.size();
    size_t num_true_event_times = unique_event_times.size();
    pair<vector<double>, vector<double>> predictions = tree->computePredictionsCensoring(new_data);

    // truncate the predictions to only include non-censored times
    vector<double> predictions_final = selectColumns(predictions.first, true_event_time_ids, num_unique_event_times);

    // fetch data
    vector<double> times = new_data.get_y_col(0);
    vector<double> ind = new_data.get_y_col(1);

    // compute Harrell's C-index error
    const vector<double>& outcomes = computeOutcomes(predictions_final, num_true_event_times);
    double c_index = computeConcordanceIndex(outcomes, times, ind);
    List result = List::create(Named("C.error") = 1 - c_index);

    // compute the Brier score
    vector<size_t> response_event_time_ids_new_data = computeResponseEventTimeIDs(unique_event_times, times);   // have to compute the ids from scratch
    vector<double> IPCW_weights = computeIPCWCpp(times, ind, unique_event_times, response_event_time_ids_new_data, predictions.second, {}, {}, tree->getCensoringTimes());
    vector<double> km_pred = KaplanMeier(predictions_final, times.size());

    // compute Brier score and integrated Brier score (IBS) and normalised IBS
    vector<double> brier = computeBrierScoreCpp(times, IPCW_weights, unique_event_times, km_pred);
    pair<double, double> ibs = computeIntegratedScore(brier, unique_event_times);

    // compute the Kullback-Leibler loss and integrated Kullback-Leibler loss (IKL) and normalised IKL
    vector<double> kl = computeKLScoreCpp(times, IPCW_weights, unique_event_times, km_pred);
    pair<double, double> ikl = computeIntegratedScore(kl, unique_event_times);
    result["IBS.error"] = ibs.first;
    result["normalised.IBS.error"] = ibs.second;
    result["IKL.error"] = ikl.first;
    result["normalised.IKL.error"] = ikl.second;
    return result;
  }
  else {
    throw runtime_error("Type of tree not recognised");
  }
}

// computes the error based on a a new dataset specifically for multi-state trees
// [[Rcpp::export]]
List JFCppTreeErrorMultistate(const List& JFTree, uint8_t max_response_length, uint8_t num_states, List jump_data, DataFrame df_features,
                      NumericVector feature_indices, LogicalVector categorical, NumericVector unique, NumericVector state_weights) {
  MultistateTree* tree = ((XPtr<MultistateTree>) JFTree["Tree"]).get();
  num_states = tree->getData()->getNumberOfStates();

  // convert the input to C++ vectors
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // convert the new data to a suitable C++ Data object
  Data new_data = Data(jump_data, max_response_length, num_states, df_features, feature_indices_cpp, categorical_cpp, unique_cpp);

  // Score on the fitted tree's event-time grid, including any thinning used at fit time.
  const vector<double>& unique_event_times = tree->getEventTimes();
  vector<size_t> response_event_time_ids = computeResponseEventTimeIDsMultistate(
    unique_event_times, new_data.getTimes(), new_data.getStates(), new_data.getMaxResponseLength());

  MultistateScoreWeights score_weights = multistateScoreWeights(state_weights, num_states);

  // compute only the occupation probabilities and censoring predictions needed by the two scores
  vector<vector<double>> error_predictions = tree->computeErrorPredictions(new_data);
  vector<double> censoring_indicators(new_data.getNumberOfObs(), 0);
  const vector<uint8_t> censoring_states = new_data.getCensoringStates();
  for (size_t i = 0; i < new_data.getNumberOfObs(); ++i) {
    if (censoring_states[i] == 0) {
      censoring_indicators[i] = 1;    // censoring_state is 0 if and only if censoring has not occured
    }
  }
  // compute state indicators
  vector<bool> states_ind = computeStateIndicatorsMultistate(new_data, response_event_time_ids,
    unique_event_times.size());

  // now ready to compute Brier score error, starting with the IPCW weights
  vector<double> IPCW_weights = computeIPCWCpp(new_data.getTimes(), censoring_indicators, unique_event_times, response_event_time_ids,
                                               error_predictions[1], {}, new_data.getLastObservedTimes(), tree->getCensoringTimes());

  // compute IBS and normalised IBS
  vector<double> brier = computeBrierScoreCppMultistate(states_ind, IPCW_weights, unique_event_times, error_predictions[0], score_weights.brier);
  pair<double, double> ibs = computeIntegratedScore(brier, unique_event_times, true);

  // compute IKL and normalised IKL
  vector<double> kl = computeKLScoreCppMultistate(states_ind, IPCW_weights, unique_event_times, error_predictions[0], score_weights.kl);
  pair<double, double> ikl = computeIntegratedScore(kl, unique_event_times, true);

  // save in a list and return
  List result = List::create(
    Named("IBS.error") = ibs.first,
    Named("normalised.IBS.error") = ibs.second,
    Named("IKL.error") = ikl.first,
    Named("normalised.IKL.error") = ikl.second
  );
  return result;
}

// growing random forests
//--------------------------------------------------------------------------------------

// [[Rcpp::export]]
List JFCppForest(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, CharacterVector splitrule,
    unsigned int ntrees, bool honest, bool swr, double sample_rate, bool double_bootstrap, NumericVector response_indices, NumericVector feature_indices, 
    LogicalVector categorical, NumericVector unique, unsigned int seed, unsigned int nworkers,
    bool save_predictions, size_t num_event_times = 0,
    NumericVector splitrule_par = NumericVector()) {

  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  string splitrule_cpp = as<string>(splitrule);

  if (tree_type != 1 && splitrule_par.size() != 0) {
    throw runtime_error(
      "splitrule_par is only valid for negativebinomial, tweedie or huber regression models");
  }

  // make the data into a C++ class and save it via a shared pointer
  shared_ptr<Data> data = make_shared<Data>(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);

  // we save a list with all information about the forest
  size_t subsample_size = floor(data->getNumberOfObs() * sample_rate);

  List result = List::create(
    Named("num.obs") = data->getNumberOfObs(),
    Named("num.features") = data->getNumberOfFeatures(),
    Named("feature.names") = data->getFeatureNames(), 
    Named("response.names") = data->getResponseNames(),
    Named("sampling.type") = swr,
    Named("subsample.size") = subsample_size,
    Named("splitrule") = splitrule,
    Named("mtry") = mtry,
    Named("min.node.size") = min_node_size,
    Named("nsplits") = nsplits,
    Named("honest") = honest,
    Named("double.bootstrap") = double_bootstrap
  );

  // the forest is a regression forest
  if (tree_type == 1) {
    RegressionSplitRule splitrule_id = regressionSplitRuleFromString(splitrule_cpp);
    double splitrule_par_cpp =
      regressionSplitRuleParameter(splitrule_id, splitrule_par);
    validateRegressionResponse(data->get_y(), splitrule_id, splitrule_par_cpp);
    if (regressionSplitRuleUsesParameter(splitrule_id)) {
      result["splitrule.par"] = splitrule_par_cpp;
    }

    // create and grow the regression forest
    RegressionForest* forest = new RegressionForest();
    forest->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, ntrees,
                       honest, swr, sample_rate, double_bootstrap, seed, nworkers,
                       splitrule_par_cpp);
    forest->grow();

    XPtr<RegressionForest> regression_forest(forest, true);
    result["tree.type"] = "Regression";
    result["num.trees"] = ntrees;
    result["Forest"] = regression_forest; // add the forest as a pointer, only to be used for prediction

    vector<double> response = as<vector<double>>(df[response_indices_cpp[0]]);
    JFCppForestPredict(result);                     // compute and save predictions on the data
    JFCppForestErrorRegression(result, response);   // compute and save error result

    result["avg.num.nodes"] = forest->getAvgNumberOfNodes();
    result["avg.num.terminal.nodes"] = forest->getAvgNumberOfTerminalNodes();
    result["avg.tree.depth"] = forest->getAvgTreeDepth();
  }

  // the forest is a classification forest
  if (tree_type == 2) {
    // check validity of splitrule argument
    vector<string> valid_splitrules = {"gini", "entropy", "misc", "twoing", "hellinger"};
    if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
      throw runtime_error("Invalid splitrule, please choose between gini, entropy, misc, twoing og hellinger");
    }

    // create and grow the classification forest
    ClassificationForest* forest = new ClassificationForest();
    forest->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, ntrees, honest, swr, sample_rate, double_bootstrap, seed, nworkers);
    forest->grow();

    XPtr<ClassificationForest> classification_forest(forest, true);
    result["tree.type"] = "Classification";
    result["num.trees"] = ntrees;
    result["num.classes"] = data->getNumClasses();
    result["Forest"] = classification_forest; // add the forest as a pointer, only to be used for prediction

    vector<double> response = as<vector<double>>(df[response_indices_cpp[0]]);
    JFCppForestPredict(result);
    JFCppForestErrorClassification(result, response);

    result["avg.num.nodes"] = forest->getAvgNumberOfNodes();
    result["avg.num.terminal.nodes"] = forest->getAvgNumberOfTerminalNodes();
    result["avg.tree.depth"] = forest->getAvgTreeDepth();
  }

  // the forest is a survival forest
  if (tree_type == 3) {
    // determine the unique sorted response times
    vector<double> times = as<vector<double>>(df[response_indices_cpp[0]]);
    vector<double> ind = as<vector<double>>(df[response_indices_cpp[1]]);
    vector<double> unique_event_times = uniqueValues(times);
    vector<double> censoring_times = unique_event_times;

    // if the user has specified a number of event times, thin the vector of unique event times
    if (num_event_times > 0) {
      unique_event_times = thinUniqueEventTimes(unique_event_times, num_event_times);
    }

    vector<size_t> response_event_time_ids = computeResponseEventTimeIDs(unique_event_times, times);
    vector<size_t> true_event_time_ids = computeTrueEventTimeIDs(unique_event_times, response_event_time_ids, ind);
    vector<double> true_unique_event_times = eventTimesAtIDs(unique_event_times, true_event_time_ids);
    vector<size_t> true_response_event_time_ids = computeResponseEventTimeIDs(true_unique_event_times, times);

    // check validity of splitrule argument
    vector<string> valid_splitrules = {"logrank", "conserve", "approxlogrank"};
    if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
      throw runtime_error("Invalid splitrule, please choose between logrank, conserve, or approxlogrank");
    }

    // create and grow the survival forest
    SurvivalForest* forest = new SurvivalForest(std::move(unique_event_times), std::move(response_event_time_ids),
      std::move(true_event_time_ids), std::move(censoring_times), save_predictions);
    forest->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, ntrees, honest, swr, sample_rate, double_bootstrap, seed, nworkers);
    forest->grow();

    // specific to survival
    NumericVector unique_event_times_R(true_unique_event_times.begin(), true_unique_event_times.end());
    XPtr<SurvivalForest> survival_forest(forest, true);
    result["num.deaths"] = vector_sum(ind);
    result["tree.type"] = "Survival";
    result["num.trees"] = ntrees;
    result["unique.event.times"] = unique_event_times_R;
    result["Forest"] = survival_forest;           // add the forest as a pointer, only to be used for prediction

    if (save_predictions) {
      JFCppForestPredict(result);
      JFCppForestErrorSurvival(result, times, ind, true_unique_event_times, true_response_event_time_ids); // compute and save error result
    } else {
      result["predictions"] = R_NilValue;
      result["oob.predictions"] = R_NilValue;
      result["censoring.oob"] = R_NilValue;
      result["censoring.before.event"] = R_NilValue;
      result["C.error"] = R_NilValue;
      result["ibs"] = R_NilValue;
      result["ibs.normalised"] = R_NilValue;
    }

    result["avg.num.nodes"] = forest->getAvgNumberOfNodes();
    result["avg.num.terminal.nodes"] = forest->getAvgNumberOfTerminalNodes();
    result["avg.tree.depth"] = forest->getAvgTreeDepth();
  }
  return result;
}

// [[Rcpp::export]]
List JFCppForestMultistate(List jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame df_features, unsigned int mtry, unsigned int min_node_size,
  unsigned int nsplits, CharacterVector splitrule, unsigned int ntrees, bool honest, bool swr, double sample_rate, bool double_bootstrap, NumericVector feature_indices, 
  LogicalVector categorical, NumericVector unique, unsigned int seed, unsigned int nworkers, bool save_predictions, NumericVector state_weights,
  size_t num_event_times, NumericVector fh_weights_a, NumericVector fh_weights_b) {
  
  // convert the input to C++ vectors
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  string splitrule_cpp = as<string>(splitrule);

  // make the data into a C++ format and save it via a shared pointer
  shared_ptr<Data> data = make_shared<Data>(jump_data, max_response_length, num_states, df_features, feature_indices_cpp, categorical_cpp, unique_cpp);

  // we save a list with all information about the forest
  size_t subsample_size = floor(data->getNumberOfObs() * sample_rate);

  List result = List::create(
    Named("num.obs") = data->getNumberOfObs(),
    Named("num.features") = data->getNumberOfFeatures(),
    Named("feature.names") = data->getFeatureNames(), 
    Named("sampling.type") = swr,
    Named("subsample.size") = subsample_size,
    Named("splitrule") = splitrule,
    Named("mtry") = mtry,
    Named("min.node.size") = min_node_size,
    Named("nsplits") = nsplits,
    Named("honest") = honest
  );

  // determine the (sorted) unique event times and the corresponding response IDs
  vector<double> unique_event_times = uniqueEventTimesMultistate(data->getTimes(), data->getStates());

  // if the user has specified a number of event times, thin the vector of unique event times
  if (num_event_times > 0) {
    unique_event_times = thinUniqueEventTimes(unique_event_times, num_event_times);
  }
  
  vector<size_t> response_event_time_ids = computeResponseEventTimeIDsMultistate(
    unique_event_times, data->getTimes(), data->getStates(), data->getMaxResponseLength());

  // check validity of splitrule argument
  vector<string> valid_splitrules = {"logrank", "gehan", "taroneware", "conserve", "approxlogrank", "petoprentice", "flemingharrington"};
  if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
    throw runtime_error("Invalid splitrule, please choose between logrank, gehan, taroneware, conserve, approxlogrank, petoprentice or flemingharrington");
  }

  FlemingHarringtonWeights fh_weights = flemingHarringtonWeights(
    splitrule_cpp, fh_weights_a, fh_weights_b, num_states);
  saveFlemingHarringtonMetadata(result, fh_weights);

  // create and grow the multi-state forest
  MultistateForest* forest = new MultistateForest(
    unique_event_times, response_event_time_ids, data->getNumberOfStates(),
    save_predictions, fh_weights.a, fh_weights.b);
  forest->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, ntrees, honest, swr, sample_rate, double_bootstrap, seed, nworkers);
  forest->grow();

  // specific to multi-states
  NumericVector unique_event_times_R(unique_event_times.begin(), unique_event_times.end());
  result["tree.type"] = "Multi-state";
  result["num.states"] = num_states;
  result["num.trees"] = ntrees;
  result["unique.event.times"] = unique_event_times_R;
  XPtr<MultistateForest> multistate_forest(forest, true);
  result["Forest"] = multistate_forest;           // add the forest as a pointer, only to be used for prediction

  if (save_predictions) {
    Rcout << "Computing forest predictions" << endl;
    JFCppForestPredict(result);

    vector<double> censoring_indicators(data->getNumberOfObs(), 0);
    const vector<uint8_t>& censoring_states = data->getCensoringStates();
    for (size_t i = 0; i < data->getNumberOfObs(); ++i) {
      if (censoring_states[i] == 0) {
        censoring_indicators[i] = 1;
      }
    }
    MultistateScoreWeights score_weights = multistateScoreWeights(state_weights, num_states);
    JFCppForestErrorMultistate(result, data->getTimes(), data->getLastObservedTimes(), censoring_indicators,
                               unique_event_times, response_event_time_ids, score_weights.brier, score_weights.kl);
  } else {
    result["predictions"] = R_NilValue;
    result["oob.predictions"] = R_NilValue;
    result["init"] = R_NilValue;
    result["oob.init"] = R_NilValue;
  }

  result["avg.num.nodes"] = forest->getAvgNumberOfNodes();
  result["avg.num.terminal.nodes"] = forest->getAvgNumberOfTerminalNodes();
  result["avg.tree.depth"] = forest->getAvgTreeDepth();
  
  return result;
}

// predicting with random forests
//--------------------------------------------------------------------------------------

// for computing all predictions (in-bag and OOB) of the dataset
void JFCppForestPredict(List& JFForest, bool compute_censoring) {
  string type = as<string>(JFForest["tree.type"]);
  if (type == "Regression") {
    RegressionForest* forest = ((XPtr<RegressionForest>) JFForest["Forest"]).get();
    const pair<vector<double>, vector<double>>& predictions_cpp = forest->computePredictions();
    size_t num_obs = JFForest["num.obs"];

    NumericVector predictions(num_obs);
    NumericVector predictions_oob(num_obs);
    for (size_t i = 0; i < num_obs; ++i) {
      predictions[i] = predictions_cpp.first[i];
      predictions_oob[i] = predictions_cpp.second[i];
    }
    // save predictions
    JFForest["predictions"] = predictions;
    JFForest["oob.predictions"] = predictions_oob;
  }
  if (type == "Classification") {
    ClassificationForest* forest = ((XPtr<ClassificationForest>) JFForest["Forest"]).get();
    const vector<vector<double>>& predictions_cpp = forest->computePredictions();
    size_t num_obs = JFForest["num.obs"];
    size_t num_classes = JFForest["num.classes"];

    NumericVector predictions(num_obs);
    NumericVector predictions_oob(num_obs);
    NumericMatrix predictions_prob(num_obs, num_classes);
    NumericMatrix predictions_prob_oob(num_obs, num_classes);

    for (size_t i = 0; i < num_obs; ++i) {
      predictions[i] = predictions_cpp[0][i];
      predictions_oob[i] = predictions_cpp[1][i];
      size_t index = i * num_classes;
      for (size_t c = 0; c < num_classes; ++c) {
        predictions_prob(i, c) = predictions_cpp[2][index + c];
        predictions_prob_oob(i, c) = predictions_cpp[3][index + c];
      }
    }
    // save predictions
    JFForest["predictions"] = predictions;
    JFForest["oob.predictions"] = predictions_oob;
    JFForest["predictions.prob"] = predictions_prob;
    JFForest["oob.predictions.prob"] = predictions_prob_oob;
  }
  
  if (type == "Survival") {
    // compute predictions via multi-threading
    SurvivalForest* forest = ((XPtr<SurvivalForest>) JFForest["Forest"]).get();

    // if censoring predictions are not saved during fitting, we need to compute these in each node across all trees
    if (compute_censoring && !forest->predictionsSaved()) {
      cout << "Predictions are not saved, populating leaves with censoring KM estimators" << endl;
      forest->computePredictionsCensoring();
    }

    const vector<vector<double>>& predictions_cpp = forest->computePredictions(compute_censoring);
    size_t num_all_event_times = forest->getNumUniqueEventTimes();
    const vector<size_t>& true_event_time_ids = forest->getTrueEventTimeIDs();
    size_t num_unique_event_times = true_event_time_ids.size();
    size_t num_obs = JFForest["num.obs"];
    
    // now fill the matrices of predictions
    NumericMatrix predictions(num_obs, num_unique_event_times);
    NumericMatrix predictions_oob(num_obs, num_unique_event_times);
    NumericMatrix censoring_oob(compute_censoring ? num_obs : 0,
                                compute_censoring ? num_unique_event_times : 0);
    NumericVector censoring_before_event(compute_censoring ? num_obs : 0);
    for (size_t i = 0; i < num_obs; ++i) {
      for (size_t j = 0; j < num_unique_event_times; ++j) {
        size_t index = i * num_all_event_times + true_event_time_ids[j];
        predictions(i, j) = predictions_cpp[0][index];
        predictions_oob(i, j) = predictions_cpp[1][index];
        if (compute_censoring) {
          censoring_oob(i, j) = predictions_cpp[2][index];
        }
      }
      if (compute_censoring) {
        censoring_before_event[i] = predictions_cpp[3][i];
      }
    }

    // save predictions
    JFForest["predictions"] = predictions;
    JFForest["oob.predictions"] = predictions_oob;
    if (compute_censoring) {
      JFForest["censoring.oob"] = censoring_oob;
      JFForest["censoring.before.event"] = censoring_before_event;
    }
  }
  
  if (type == "Multi-state") {
    // compute predictions via multi-threading
    MultistateForest* forest = ((XPtr<MultistateForest>) JFForest["Forest"]).get();

    // Censoring estimators only exist in the leaves when predictions were saved
    // during fitting. Nelson--Aalen and initial-state estimates are always stored.
    bool compute_multistate_censoring = forest->predictionsSaved();
    const vector<vector<double>>& predictions_cpp = forest->computePredictions(true, compute_multistate_censoring);
    size_t num_unique_event_times = forest->getNumUniqueEventTimes();
    size_t num_obs = JFForest["num.obs"];
    size_t num_states = forest->getData()->getNumberOfStates();
    size_t dim = num_states * num_states;
    List predictions(num_obs);        // each prediction is a list of matrices
    List predictions_oob(num_obs);
    List predictions_init(num_obs);   // each predicted initial distribution is a vector
    List predictions_init_oob(num_obs);
    NumericMatrix censoring_oob(compute_multistate_censoring ? num_obs : 0,
                                compute_multistate_censoring ? num_unique_event_times : 0);

    for (size_t i = 0; i < num_obs; ++i) {
      List rpred(num_unique_event_times);                     // list of NumericMatrix for a single prediction
      List rpred_oob(num_unique_event_times);                 // ditto for OOB
      for (size_t t = 0; t < num_unique_event_times; ++t) {
        NumericMatrix pred_time(num_states, num_states);
        NumericMatrix pred_time_oob(num_states, num_states);
        for (size_t j = 0; j < num_states; ++j) {
          for (size_t k = 0; k < num_states; ++k) {
            pred_time(j, k) = predictions_cpp[0][i * num_unique_event_times * dim + t * dim + j * num_states + k];
            pred_time_oob(j, k) = predictions_cpp[1][i * num_unique_event_times * dim + t * dim + j * num_states + k];
          }
        }
        rpred[t] = pred_time;
        rpred_oob[t] = pred_time_oob;
      }
      // save predicted Nelson-Aalen estimators
      predictions[i] = rpred;
      predictions_oob[i] = rpred_oob;

      NumericVector rpred_init(num_states);
      NumericVector rpred_init_oob(num_states);

      // save predicted initial distributions
      for (size_t j = 0; j < num_states; ++j) {
        rpred_init[j] = predictions_cpp[2][i * num_states + j];
        rpred_init_oob[j] = predictions_cpp[3][i * num_states + j];
      }
      predictions_init[i] = rpred_init;
      predictions_init_oob[i] = rpred_init_oob;
      if (compute_multistate_censoring) {
        for (size_t t = 0; t < num_unique_event_times; ++t) {
          censoring_oob(i, t) = predictions_cpp[4][i * num_unique_event_times + t];
        }
      }
    }
    // save predictions
    JFForest["predictions"] = predictions;
    JFForest["oob.predictions"] = predictions_oob;
    JFForest["init"] = predictions_init;
    JFForest["oob.init"] = predictions_init_oob;
    if (compute_multistate_censoring) {
      JFForest["censoring.oob"] = censoring_oob;
    }
  }
}

// [[Rcpp::export]]
void JFCppForestPredictTraining(List& JFForest) {
  JFCppForestPredict(JFForest);
}

// [[Rcpp::export]]
NumericMatrix JFCppForestPredict(const List& JFForest, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique) {
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp;  // need an empty vector for the response indices
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // convert the new data to a suitable C++ Data object
  Data new_data = Data(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);
  size_t num_obs = new_data.getNumberOfObs();

  string type = as<string>(JFForest["tree.type"]);
  if (type == "Regression") {
    RegressionForest* forest = ((XPtr<RegressionForest>) JFForest["Forest"]).get();
    NumericMatrix predictions(num_obs, 1);
    const vector<double>& predictions_cpp = forest->computePredictions(new_data);
    for (size_t i = 0; i < num_obs; ++i) {
      predictions(i, 0) = predictions_cpp[i];
    }
    return predictions;
  }

  if (type == "Classification") {
    ClassificationForest* forest = ((XPtr<ClassificationForest>) JFForest["Forest"]).get();
    size_t num_classes = as<size_t>(JFForest["num.classes"]);
    NumericMatrix predictions(num_obs, num_classes + 1);
    const pair<vector<double>, vector<double>>& predictions_cpp = forest->computePredictions(new_data, true);
    for (size_t i = 0; i < num_obs; ++i) {
      predictions(i, 0) = predictions_cpp.first[i];
      size_t index = i * num_classes;
      for (size_t c = 0; c < num_classes; ++c) {
        predictions(i, c + 1) = predictions_cpp.second[index + c];
      }
    }
    return predictions;
  }

  if (type == "Survival") {
    SurvivalForest* forest = ((XPtr<SurvivalForest>) JFForest["Forest"]).get();
    size_t num_all_event_times = forest->getNumUniqueEventTimes();
    const vector<size_t>& true_event_time_ids = forest->getTrueEventTimeIDs();
    size_t num_unique_event_times = true_event_time_ids.size();
    NumericMatrix predictions(num_obs, num_unique_event_times);
    const vector<vector<double>>& predictions_cpp = forest->computePredictions(new_data, false);
    for (size_t i = 0; i < num_obs; ++i) {
      for (size_t t = 0; t < num_unique_event_times; ++t) {
        predictions(i, t) = predictions_cpp[0][i * num_all_event_times + true_event_time_ids[t]];
      }
    }
    return predictions;
  }
  throw runtime_error("Type of forest not recognised");
}


// function to compute predictions and the predicted KM estimators for the censoring for survival forests

// [[Rcpp::export]]
List JFCppForestPredictCensoring(const List& JFForest, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique) {
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp;  // need an empty vector for the response indices
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  
  // convert the new data to a suitable Data object
  Data new_data = Data(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);

  SurvivalForest* forest = ((XPtr<SurvivalForest>) JFForest["Forest"]).get();
  size_t num_obs = new_data.getNumberOfObs();
  size_t num_all_event_times = forest->getEventTimes().size();
  const vector<size_t>& true_event_time_ids = forest->getTrueEventTimeIDs();
  size_t num_unique_event_times = true_event_time_ids.size();
  NumericMatrix predictions(num_obs, num_unique_event_times);
  NumericMatrix censoring(num_obs, num_unique_event_times);
  
  // if predictions are not saved, it means that the censoring KM estimators need to be computed from scratch in each leaf
  if (!forest->predictionsSaved()) {
    forest->computePredictionsCensoring();
  }

  const vector<vector<double>> predictions_cpp = forest->computePredictions(new_data, true);
  for (size_t i = 0; i < num_obs; ++i) {
    for (size_t t = 0; t < num_unique_event_times; ++t) {
      size_t index = i * num_all_event_times + true_event_time_ids[t];
      predictions(i, t) = predictions_cpp[0][index];
      censoring(i, t) = predictions_cpp[1][index];
    }
  }
  List result = List::create(
    Named("predictions") = predictions,
    Named("censoring") = censoring
  );
  return result;
}

/*
  if compute_initial = false (default), simply return a list of the predictions on the new data df
  if compute_initial = true, return a list of two lists, one containing predictions on the new data
  and the other containing predicted initial distributions
*/

// [[Rcpp::export]]
List JFCppForestPredictMultistate(const List& JFForest, DataFrame df, NumericVector feature_indices,
                          LogicalVector categorical, NumericVector unique, bool compute_initial) {
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp;  // need an empty vector for the response indices
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // convert the new data to a suitable C++ Data object
  Data new_data = Data(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);
  size_t num_obs = new_data.getNumberOfObs();

  MultistateForest* forest = ((XPtr<MultistateForest>) JFForest["Forest"]).get();
  size_t num_unique_event_times = forest->getNumUniqueEventTimes();
  size_t num_states = forest->getData()->getNumberOfStates();
  size_t dim = num_states * num_states;
  List predictions(num_obs);      // each prediction is a list of matrices
  List predictions_init(compute_initial ? num_obs : 0); // only allocate initial distributions when requested
  
  const vector<vector<double>>& predictions_cpp = forest->computePredictions(new_data, compute_initial, false);
  for (size_t i = 0; i < num_obs; ++i) {
    List rpred(num_unique_event_times);
    for (size_t t = 0; t < num_unique_event_times; ++t) {
      NumericMatrix pred_time(num_states, num_states);
      for (size_t j = 0; j < num_states; ++j) {
        for (size_t k = 0; k < num_states; ++k) {
          pred_time(j, k) = predictions_cpp[0][i * num_unique_event_times * dim + dim * t + j * num_states + k];
        }
      }
      rpred[t] = pred_time;
    }
    // save Nelson-Aalen estimator
    predictions[i] = rpred;

    // if compute_initial == true, compute and save initial distribution
    if (compute_initial) {
      NumericVector rpred_init(num_states);
      for (size_t j = 0; j < num_states; ++j) {
        rpred_init[j] = predictions_cpp[1][i * num_states + j];
      }
      predictions_init[i] = rpred_init;
    }
  }

  if (compute_initial) {
    List result = List::create(
      Named("predictions") = predictions,
      Named("initial") = predictions_init
    );
    return result;
  }
  return predictions;
}

// error computation for random forests
//--------------------------------------------------------------------------------------

// Regression

// computes the error based on the OOB predictions of the forest
void JFCppForestErrorRegression(List& JFForest, const vector<double>& response) {
  size_t num_obs = as<size_t>(JFForest["num.obs"]);
  JFForest["mse.error"] = computeMSE(JFForest["oob.predictions"], response);
  JFForest["R2"] = computeR2(JFForest["mse.error"], response);
}

// computes the error for a regression forest based on new predictions
List JFCppForestErrorRegression(const vector<double>& predictions, const vector<double>& response) {
  List result;
  double mse = computeMSE(predictions, response);
  result["error.mse"] = mse;
  result["R2"] = computeR2(mse, response);
  return result;
}

// Classification

void JFCppForestErrorClassification(List& JFForest, const vector<double>& response) {
  size_t num_obs = as<size_t>(JFForest["num.obs"]);
  size_t num_classes = as<size_t>(JFForest["num.classes"]);
  NumericVector class_misclassification_errors(num_classes);
  NumericMatrix confusionMatrix(num_classes, num_classes);
  NumericVector class_predictions_R = as<NumericVector>(JFForest["oob.predictions"]);
  NumericMatrix prob_predictions_matrix = as<NumericMatrix>(JFForest["oob.predictions.prob"]);
  vector<double> class_predictions;
  vector<double> response_oob;
  vector<double> prob_predictions;
  class_predictions.reserve(num_obs);
  response_oob.reserve(num_obs);
  prob_predictions.reserve(num_obs * num_classes);

  for (size_t i = 0; i < num_obs; ++i) {
    if (NumericVector::is_na(class_predictions_R[i])) {
      continue;
    }

    bool complete_probabilities = true;
    for (size_t c = 0; c < num_classes; ++c) {
      if (NumericVector::is_na(prob_predictions_matrix(i, c))) {
        complete_probabilities = false;
        break;
      }
    }
    if (!complete_probabilities) {
      continue;
    }

    class_predictions.push_back(class_predictions_R[i]);
    response_oob.push_back(response[i]);
    for (size_t c = 0; c < num_classes; ++c) {
      prob_predictions.push_back(prob_predictions_matrix(i, c));
    }
  }

  if (class_predictions.empty()) {
    JFForest["misc.error.total"] = NA_REAL;
    for (size_t c = 0; c < num_classes; ++c) {
      class_misclassification_errors[c] = NA_REAL;
      for (size_t c2 = 0; c2 < num_classes; ++c2) {
        confusionMatrix(c, c2) = NA_REAL;
      }
    }
    JFForest["misc.error"] = class_misclassification_errors;
    JFForest["bs"] = NA_REAL;
    JFForest["bs.normalised"] = NA_REAL;
    JFForest["num.oob.predictions"] = 0;
    JFForest["confusion"] = confusionMatrix;
    return;
  }

  // compute and save overall misclassification error and class-wise misclassification error
  vector<double> misc = computeMisclassificationError(class_predictions, response_oob, num_classes);
  JFForest["misc.error.total"] = misc[num_classes];
  for (size_t c = 0; c < num_classes; ++c) {
    class_misclassification_errors[c] = misc[c];
  }
  JFForest["misc.error"] = class_misclassification_errors;

  // compute and save the Brier score error
  double brier_score = computeBrierScoreError(prob_predictions, response_oob, num_classes);
  JFForest["bs"] = brier_score;
  JFForest["bs.normalised"] = brier_score * (double) num_classes * (double) num_classes / (double) (num_classes - 1);
  JFForest["num.oob.predictions"] = class_predictions.size();

  // compute and save the confusion matrix
  vector<size_t> confusion = computeConfusionMatrix(class_predictions, response_oob, num_classes);
  for (size_t c1 = 0; c1 < num_classes; ++c1) {
    size_t row = c1 * num_classes;
    for (size_t c2 = 0; c2 < num_classes; ++c2) {
      confusionMatrix(c1, c2) = confusion[row + c2];
    }
  }
  JFForest["confusion"] = confusionMatrix;
}

// Survival

void JFCppForestErrorSurvival(List& JFForest, const vector<double>& times, const vector<double>& ind, const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids) {
  // first compute Harrell's C-index
  const vector<double>& outcomes = computeOutcomes(JFForest["oob.predictions"]);
  JFForest["outcomes.oob"] = outcomes;
  JFForest["C.error"] = 1 - computeConcordanceIndex(outcomes, times, ind);

  // compute IPCW weights and the censoring Kaplan-Meier estimators
  vector<double> censoring_before_event = as<vector<double>>(JFForest["censoring.before.event"]);
  vector<double> IPCW_weights = computeIPCW(ind, unique_event_times, response_event_time_ids,
    JFForest["censoring.oob"], times, {}, {}, censoring_before_event);
  const NumericMatrix& censoring_km = KaplanMeier(as<NumericMatrix>(JFForest["oob.predictions"]));

  // compute Brier score and integrated Brier score
  vector<double> brier = computeBrierScore(times, IPCW_weights, unique_event_times, censoring_km);
  pair<double, double> ibs = computeIntegratedScore(brier, unique_event_times);

  // compute Kullback-Leibler score and integrated Kullback-Leibler score
  vector<double> kl = computeKLScore(times, IPCW_weights, unique_event_times, censoring_km);
  pair<double, double> ikl = computeIntegratedScore(kl, unique_event_times);

  JFForest["ibs"] = ibs.first;
  JFForest["ibs.normalised"] = ibs.second;
  JFForest["ikl"] = ikl.first;
  JFForest["ikl.normalised"] = ikl.second;
}

// for computing OOB errors for a survival forest when errors are not already saved, but predictions are computed

// [[Rcpp::export]]
List JFCppForestErrorSurvivalExternal(List& JFForest) {
  SurvivalForest* forest = ((XPtr<SurvivalForest>) JFForest["Forest"]).get();
  const vector<double>& times = forest->getData()->get_y_col(0);
  const vector<double>& ind = forest->getData()->get_y_col(1);
  vector<double> unique_event_times = eventTimesAtIDs(forest->getEventTimes(), forest->getTrueEventTimeIDs());
  vector<size_t> response_event_time_ids = computeResponseEventTimeIDs(unique_event_times, times);

  // first compute Harrell's C-index
  const vector<double>& outcomes = computeOutcomes(JFForest["oob.predictions"]);
  JFForest["outcomes.oob"] = outcomes;

  // now compute IPCW weights and Kaplan-Meier estimator for the censoring distribution
  vector<double> censoring_before_event = as<vector<double>>(JFForest["censoring.before.event"]);
  vector<double> IPCW_weights = computeIPCW(ind, unique_event_times, response_event_time_ids,
    JFForest["censoring.oob"], times, {}, {}, censoring_before_event);
  const NumericMatrix& censoring_km = KaplanMeier(as<NumericMatrix>(JFForest["oob.predictions"]));

  // compute Brier score and integrated Brier score
  vector<double> brier = computeBrierScore(times, IPCW_weights, unique_event_times, censoring_km);
  pair<double, double> ibs = computeIntegratedScore(brier, unique_event_times);
  
  // compute Kullback-Leibler score and integrated Kullback-Leibler score
  vector<double> kl = computeKLScore(times, IPCW_weights, unique_event_times, censoring_km);
  pair<double, double> ikl = computeIntegratedScore(kl, unique_event_times);

  // create and return list of errors
  double c_error = 1 - computeConcordanceIndex(outcomes, times, ind);
  List result = List::create(
    Named("C.error") = c_error,
    Named("IBS.error") = ibs.first,
    Named("normalised.IBS.error") = ibs.second,
    Named("KL.error") = ikl.first,
    Named("normalised.KL.error") = ikl.second
  );

  return result;
}

// Multi-state

void JFCppForestErrorMultistate(List& JFForest, const vector<double>& times, const vector<size_t>& last_observed_time_ids, const vector<double>& ind,
                        const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids,
                        const vector<double>& brier_state_weights, const vector<double>& kl_state_weights) {
  MultistateForest* forest = ((XPtr<MultistateForest>) JFForest["Forest"]).get();
  vector<vector<double>> error_predictions = forest->computeErrorPredictions();

  /*
    Use the full censoring grid so G(T-) can include pure censoring times which are absent from the event grid. This is
    also the path used by single multi-state trees and VIMP, keeping all three error workflows coherent.
  */
  vector<double> IPCW_weights = computeIPCWCpp(times, ind, unique_event_times, response_event_time_ids, error_predictions[1], {}, 
                                               last_observed_time_ids, forest->getCensoringTimes());
  vector<bool> states_ind = computeStateIndicatorsMultistate(*forest->getData(), response_event_time_ids, unique_event_times.size());

  vector<double> brier = computeBrierScoreCppMultistate(states_ind, IPCW_weights, unique_event_times,
                                                        error_predictions[0], brier_state_weights);
  pair<double, double> ibs = computeIntegratedScore(brier, unique_event_times, true);
  vector<double> kl = computeKLScoreCppMultistate(states_ind, IPCW_weights, unique_event_times,
                                                  error_predictions[0], kl_state_weights);
  pair<double, double> ikl = computeIntegratedScore(kl, unique_event_times, true);

  JFForest["ibs"] = ibs.first;
  JFForest["ibs.normalised"] = ibs.second;
  JFForest["ikl"] = ikl.first;
  JFForest["ikl.normalised"] = ikl.second;
}

// Computes and saves OOB errors when predictions were not requested during fitting.
// [[Rcpp::export]]
List JFCppForestErrorMultistateExternal(List& JFForest, NumericVector state_weights) {
  MultistateForest* forest = ((XPtr<MultistateForest>) JFForest["Forest"]).get();
  if (!forest->predictionsSaved()) {
    forest->computePredictionsCensoring();
  }

  shared_ptr<Data> data = forest->getData();
  const vector<double>& unique_event_times = forest->getEventTimes();
  const vector<size_t>& response_event_time_ids = forest->getResponseEventTimeIDs();
  vector<double> censoring_indicators(data->getNumberOfObs(), 0);
  const vector<uint8_t>& censoring_states = data->getCensoringStates();
  for (size_t i = 0; i < data->getNumberOfObs(); ++i) {
    if (censoring_states[i] == 0) {
      censoring_indicators[i] = 1;
    }
  }

  size_t num_states = data->getNumberOfStates();
  MultistateScoreWeights score_weights = multistateScoreWeights(state_weights, num_states);
  JFCppForestErrorMultistate(JFForest, data->getTimes(), data->getLastObservedTimes(),
                             censoring_indicators, unique_event_times,
                             response_event_time_ids, score_weights.brier, score_weights.kl);

  return List::create(
    Named("IBS.error") = JFForest["ibs"],
    Named("normalised.IBS.error") = JFForest["ibs.normalised"],
    Named("IKL.error") = JFForest["ikl"],
    Named("normalised.IKL.error") = JFForest["ikl.normalised"]
  );
}

// computes the error based on a a new dataset, here we don't need to specify the type of forest beforehand
// [[Rcpp::export]]
List JFCppForestError(const List& JFForest, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique, NumericVector response_indices) {
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // convert the new data to a suitable C++ Data object
  Data new_data = Data(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);
  size_t num_obs = new_data.getNumberOfObs();

  string type = as<string>(JFForest["tree.type"]);
  if (type == "Regression") {
    RegressionForest* forest = ((XPtr<RegressionForest>) JFForest["Forest"]).get();
    const vector<double>& predictions = forest->computePredictions(new_data);
    const vector<double>& response = new_data.get_y_col(0);
    return JFCppForestErrorRegression(predictions, response);
  }
  if (type == "Classification") {
    ClassificationForest* forest = ((XPtr<ClassificationForest>) JFForest["Forest"]).get();
    const pair<vector<double>, vector<double>>& predictions = forest->computePredictions(new_data, true);
    const vector<double>& response = new_data.get_y();

    size_t num_classes = as<size_t>(JFForest["num.classes"]);
    NumericVector class_misclassification_errors(num_classes);
    NumericMatrix confusionMatrix(num_classes, num_classes);

    // create list, compute and save the Brier Score error
    double brier_score = computeBrierScoreError(predictions.second, response, num_classes);
    List result = List::create(Named("BS.error") = brier_score,
                               Named("normalised.BS.error") = brier_score * (double) num_classes * (double) num_classes / (double) (num_classes - 1));

    // compute and save overall misclassification error and class-wise misclassification error
    vector<double> misc = computeMisclassificationError(predictions.first, response, num_classes);
    result["misclassification.error.total"] = misc[num_classes];
    for (size_t c = 0; c < num_classes; ++c) {
      class_misclassification_errors[c] = misc[c];
    }
    result["misclassification.error"] = class_misclassification_errors;

    // compute and save the confusion matrix
    vector<size_t> confusion = computeConfusionMatrix(predictions.first, response, num_classes);
    for (size_t c1 = 0; c1 < num_classes; ++c1) {
      size_t row = c1 * num_classes;
      for (size_t c2 = 0; c2 < num_classes; ++c2) {
        confusionMatrix(c1, c2) = confusion[row + c2];
      }
    }
    result["confusion.matrix"] = confusionMatrix;
    return result;
  }
  if (type == "Survival") {
    SurvivalForest* forest = ((XPtr<SurvivalForest>) JFForest["Forest"]).get();
    const vector<double>& all_unique_event_times = forest->getEventTimes();
    const vector<size_t>& true_event_time_ids = forest->getTrueEventTimeIDs();
    vector<double> unique_event_times = eventTimesAtIDs(all_unique_event_times, true_event_time_ids);
    size_t num_unique_event_times = all_unique_event_times.size();
    size_t num_true_event_times = unique_event_times.size();

    // check if predictions are saved, if not, the censoring KM estimators have to be computed in every leaf
    if (!forest->predictionsSaved()) {
      forest->computePredictionsCensoring();
    }

    // fetch data
    const vector<double>& times = new_data.get_y_col(0);
    const vector<double>& ind = new_data.get_y_col(1);
    const vector<vector<double>>& predictions = forest->computePredictions(new_data, true, &times);

    // truncate the predictions to only include non-censored times
    vector<double> predictions_final = selectColumns(predictions[0], true_event_time_ids, num_unique_event_times);
    vector<double> censoring_final = selectColumns(predictions[1], true_event_time_ids, num_unique_event_times);

    // compute Harrell's C-index error
    const vector<double>& outcomes = computeOutcomes(predictions_final, num_true_event_times);
    List result = List::create(Named("C.error") = 1 - computeConcordanceIndex(outcomes, times, ind));

    // compute the response event time ids based on the new data, the IPCW weights and the Kaplan-Meier
    // estimators of the predictions
    vector<size_t> response_event_time_ids_new_data = computeResponseEventTimeIDs(unique_event_times, times);
    vector<double> IPCW_weights = computeIPCWCpp(times, ind, unique_event_times,
      response_event_time_ids_new_data, censoring_final, {}, {}, {}, predictions[2]);
    vector<double> km_pred = KaplanMeier(predictions_final, times.size());

    // now compute the Brier score and integrated Brier score
    vector<double> brier = computeBrierScoreCpp(times, IPCW_weights, unique_event_times, km_pred);
    pair<double, double> ibs = computeIntegratedScore(brier, unique_event_times);

    // finally, compute the Kullback-Leibler score and integrated Kullback-Leibler score
    vector<double> kl = computeKLScoreCpp(times, IPCW_weights, unique_event_times, km_pred);
    pair<double, double> ikl = computeIntegratedScore(kl, unique_event_times);

    result["IBS.error"] = ibs.first;
    result["normalised.IBS.error"] = ibs.second;
    result["IKL.error"] = ikl.first;
    result["normalised.IKL.error"] = ikl.second;
    return result;
  }
  else {
    throw runtime_error("Type of forest not recognised");
  }
}

// computes errors on a new dataset specifically for multi-state forests
// [[Rcpp::export]]
List JFCppForestErrorMultistate(const List& JFForest, uint8_t max_response_length, uint8_t num_states,
                                List jump_data, DataFrame df_features, NumericVector feature_indices,
                                LogicalVector categorical, NumericVector unique, NumericVector state_weights) {
  MultistateForest* forest = ((XPtr<MultistateForest>) JFForest["Forest"]).get();
  num_states = forest->getData()->getNumberOfStates();

  if (!forest->predictionsSaved()) {
    forest->computePredictionsCensoring();
  }

  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  Data new_data(jump_data, max_response_length, num_states, df_features, feature_indices_cpp,
                categorical_cpp, unique_cpp);

  const vector<double>& unique_event_times = forest->getEventTimes();
  vector<size_t> response_event_time_ids = computeResponseEventTimeIDsMultistate(
    unique_event_times, new_data.getTimes(), new_data.getStates(), new_data.getMaxResponseLength());
  MultistateScoreWeights score_weights = multistateScoreWeights(state_weights, num_states);

  /*
    Aggregate only the occupation and full-grid censoring curves used by scoring. The public prediction route would
    otherwise materialise N*T*S*S Nelson--Aalen matrices and then immediately discard them after the product integral.
  */
  vector<vector<double>> error_predictions = forest->computeErrorPredictions(new_data);
  vector<double> censoring_indicators(new_data.getNumberOfObs(), 0);
  const vector<uint8_t>& censoring_states = new_data.getCensoringStates();
  for (size_t i = 0; i < new_data.getNumberOfObs(); ++i) {
    if (censoring_states[i] == 0) {
      censoring_indicators[i] = 1;
    }
  }

  vector<bool> states_ind = computeStateIndicatorsMultistate(new_data, response_event_time_ids, unique_event_times.size());
  vector<double> IPCW_weights = computeIPCWCpp(new_data.getTimes(), censoring_indicators, unique_event_times, response_event_time_ids,
                                               error_predictions[1], {}, new_data.getLastObservedTimes(), forest->getCensoringTimes());

  vector<double> brier = computeBrierScoreCppMultistate(states_ind, IPCW_weights, unique_event_times, error_predictions[0], score_weights.brier);
  pair<double, double> ibs = computeIntegratedScore(brier, unique_event_times, true);
  vector<double> kl = computeKLScoreCppMultistate(states_ind, IPCW_weights, unique_event_times, error_predictions[0], score_weights.kl);
  pair<double, double> ikl = computeIntegratedScore(kl, unique_event_times, true);

  return List::create(
    Named("IBS.error") = ibs.first,
    Named("normalised.IBS.error") = ibs.second,
    Named("IKL.error") = ikl.first,
    Named("normalised.IKL.error") = ikl.second
  );
}

// computes VIMP for a specific feature after the forest is grown
// [[Rcpp::export]]
double JFCppForestVIMPFeature(const List& JFForest, CharacterVector feature_name, int feature_seed, CharacterVector method, CharacterVector loss) {
  string type = as<string>(JFForest["tree.type"]);
  string method_cpp = as<string>(method);
  string loss_cpp = as<string>(loss);

  // translate from feature name to feature index
  string feature_name_cpp = as<string>(feature_name);

  if (type == "Regression") {
    RegressionForest* forest = ((XPtr<RegressionForest>) JFForest["Forest"]).get();
    size_t feature = forest->getData()->getFeatureID(feature_name_cpp);
    if (loss_cpp == "default") {
      loss_cpp = "mse";
    }

    // check validity of loss function
    vector<string> valid_loss_functions = {"mse"};
    if (find(valid_loss_functions.begin(), valid_loss_functions.end(), loss_cpp) == valid_loss_functions.end()) {
      throw runtime_error("Invalid loss function, only 'mse' is implemented so far");
    }

    if (method_cpp == "permute") {
      return forest->computeVIMPPermute(feature, feature_seed);
    } else if (method_cpp == "random") {
      return forest->computeVIMPRandom(feature, feature_seed);
    } else {
      throw runtime_error("Type of VIMP computation method not recognised, use 'permute' or 'random'");
    }
  }
  if (type == "Classification") {
    ClassificationForest* forest = ((XPtr<ClassificationForest>) JFForest["Forest"]).get();
    size_t feature = forest->getData()->getFeatureID(feature_name_cpp);
    if (loss_cpp == "default") {
      loss_cpp = "misc";
    }

    // check validity of loss function
    vector<string> valid_loss_functions = {"brier", "misc"};
    if (find(valid_loss_functions.begin(), valid_loss_functions.end(), loss_cpp) == valid_loss_functions.end()) {
      throw runtime_error("Invalid loss function, please choose between 'brier' or 'misc'");
    }

    vector<double> vimp;
    if (method_cpp == "permute") {
      vimp = forest->computeVIMPPermute(feature, feature_seed, loss_cpp);
    } else if (method_cpp == "random") {
      vimp = forest->computeVIMPRandom(feature, feature_seed, loss_cpp);
    } else {
      throw runtime_error("Type of VIMP computation method not recognised, use 'permute' or 'random'");
    }
    return vimp.back();
  }
  if (type == "Survival") {
    SurvivalForest* forest = ((XPtr<SurvivalForest>) JFForest["Forest"]).get();
    size_t feature = forest->getData()->getFeatureID(feature_name_cpp);
    if (loss_cpp == "default") {
      loss_cpp = "concordance";
    }

    // check validity of loss function
    vector<string> valid_loss_functions = {"brier", "concordance", "kl"};
    if (find(valid_loss_functions.begin(), valid_loss_functions.end(), loss_cpp) == valid_loss_functions.end()) {
      throw runtime_error("Invalid loss function, please choose between 'brier', 'concordance', or 'kl'");
    }
    
    if (method_cpp == "permute") {
      return forest->computeVIMPPermute(feature, feature_seed, loss_cpp);
    } else if (method_cpp == "random") {
      return forest->computeVIMPRandom(feature, feature_seed, loss_cpp);
    } else {
      throw runtime_error("Type of VIMP computation method not recognised, use 'permute' or 'random'");
    }
  }
  if (type == "Multi-state") {
    MultistateForest* forest = ((XPtr<MultistateForest>) JFForest["Forest"]).get();
    size_t feature = forest->getData()->getFeatureID(feature_name_cpp);
    if (loss_cpp == "default") {
      loss_cpp = "brier";
    }
    vector<string> valid_loss_functions = {"brier", "kl"};
    if (find(valid_loss_functions.begin(), valid_loss_functions.end(), loss_cpp) == valid_loss_functions.end()) {
      throw runtime_error("Invalid loss function, please choose between 'brier' or 'kl'");
    }
    const size_t num_states = forest->getData()->getNumberOfStates();
    // VIMP uses the same loss-specific defaults as the ordinary multi-state error calculation
    vector<double> state_weights(num_states, loss_cpp == "kl" ? 1.0 : 1.0 / static_cast<double>(num_states));
    if (method_cpp == "permute") {
      return forest->computeVIMPPermute(feature, feature_seed, loss_cpp, state_weights);
    } else if (method_cpp == "random") {
      return forest->computeVIMPRandom(feature, feature_seed, loss_cpp, state_weights);
    } else {
      throw runtime_error("Type of VIMP computation method not recognised, use 'permute' or 'random'");
    }
  }
  throw runtime_error("Type of forest not recognised");
}

// computes VIMP for every variable and saves the list of VIMP-values in JFForest (set seed for reproducibility)
// [[Rcpp::export]]
List JFCppForestVIMP(List& JFForest, int seed, CharacterVector method, CharacterVector loss) {
  mt19937 random_number_generator(seed);
  Forest* forest = ((XPtr<Forest>) JFForest["Forest"]).get();
  size_t num_features = forest->getData()->getNumberOfFeatures();
  vector<string> feature_names = forest->getData()->getFeatureNames();

  // for generating feature specific seeds in VIMP computations
  uniform_int_distribution<int> compute_feature_seed(0, numeric_limits<int>::max());

  List VIMP;
  for (size_t i = 0; i < num_features; ++i) {
    int feature_seed = compute_feature_seed(random_number_generator);
    VIMP[feature_names[i]] = JFCppForestVIMPFeature(JFForest, feature_names[i], feature_seed, method, loss);
  }
  
  JFForest["vimp"] = VIMP;
  return JFForest;
}

// returns a list of the full tree table to be printed if the user wishes

// [[Rcpp::export]]
List getTreeTable(SEXP tree_sexp) {
  Tree* tree = ((XPtr<Tree>) tree_sexp).get();
  List table = List::create(
    Named("left.daughters") = tree->getLeftDaughters(),
    Named("feature.IDs") = tree->getFeatureIDs(),
    Named("thresholds") = tree->getThresholds(),
    Named("categorical") = tree->getData()->getCategorical(),
    Named("depths") = tree->getDepths()
  );
  return(table);
}

// [[Rcpp::export]]
void df_test(DataFrame data, NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical, NumericVector unique) {
  // start by converting the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // create the data as a C++ object
  Data test = Data(data, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);

  // for testing purposes
  Rcout << "Number of features: " << test.getNumberOfFeatures() << endl;
  Rcout << "Number of observations: " << test.getNumberOfObs() << endl;
  if (response_indices_cpp.size() > 0) {
    Rcout << endl << "The response vector is: ";
    for (size_t i = 0; i < test.getNumberOfObs(); ++i) {
      Rcout << test.get_y(i, 0) << ", ";
    }
  }
  Rcout << endl << "The number of unique values of the features are: ";
  for (size_t i  = 0; i < test.getUniqueValues().size(); ++i) {
    Rcout << test.getUniqueValues()[i] << ", ";
  }
  Rcout << endl << "The categorical indicators are: ";
  for (size_t i = 0; i < test.getCategorical().size(); ++i) {
    Rcout << test.getCategorical()[i] << ", ";
  }
  Rcout << endl << "The variable names are: ";
  for (string name : test.getFeatureNames()) {
    Rcout << name << ", ";
  }
  Rcout << endl << "The feature values are: " << endl;
  for (size_t i = 0; i < test.getNumberOfObs(); ++i) {
    printVector(test.get_x_row(i));
  }
}

// a test function to ensure that all Data functionalities work
// [[Rcpp::export]]
void testData(const DataFrame& df, const NumericVector& response_indices, const NumericVector& feature_indices, 
              const LogicalVector& categorical, const NumericVector& unique) {
  
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // create Data object
  Data data = Data(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);

  Rcout << "Number of features: " << data.getNumberOfFeatures() << endl;
  Rcout << "Feature names: ";
  for (string s : data.getFeatureNames()) {
    Rcout << s << ", ";
  }
  Rcout << endl << "Features categorical? ";
  printVector(data.getCategorical());
  Rcout << "Unique values of features: ";
  printVector(data.getUniqueValues());
  // print first 10 feature values
  for (size_t i = 0; i < data.getNumberOfFeatures(); ++i) {
    Rcout << data.getFeatureNames()[i] << ":";
    for (size_t j = 0; j < 30; ++j) {
      Rcout << data.get_x(j, i) << ", ";
    }
    Rcout << endl;
  }

  // compare feature names
  for (size_t i = 0; i < data.getNumberOfFeatures(); ++i) {
    Rcout << "Internal feature ID: " << i << endl;
    Rcout << "Feature name is: " << data.getFeatureNames()[i] << endl;
    Rcout << "getFeatureID: " << data.getFeatureID(data.getFeatureNames()[i]) << endl;
  }
}

// for testing that all data functionalities related to multi-states work
// [[Rcpp::export]]
void testDataMultistate(const List& jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame& feature_df,
                const NumericVector& feature_indices, const LogicalVector& categorical, const NumericVector& unique) {
  // convert the input to C++ vectors
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // create Data object
  Data data = Data(jump_data, max_response_length, num_states, feature_df, feature_indices_cpp, categorical_cpp, unique_cpp);
  
  Rcout << "Number of features: " << data.getNumberOfFeatures() << endl;
  Rcout << "Feature names: ";
  for (string s : data.getFeatureNames()) {
    Rcout << s << ", ";
  }
  Rcout << endl << "Features categorical? ";
  printVector(data.getCategorical());
  Rcout << "Unique values of features: ";
  printVector(data.getUniqueValues());
  // print feature values
  for (size_t i = 0; i < data.getNumberOfFeatures(); ++i) {
    Rcout << data.getFeatureNames()[i] << ":";
    for (size_t j = 0; j < data.getNumberOfObs(); ++j) {
      Rcout << data.get_x(j, i) << ", ";
    }
    Rcout << endl;
  }

  // compare feature names
  for (size_t i = 0; i < data.getNumberOfFeatures(); ++i) {
    Rcout << "Internal feature ID: " << i << endl;
    Rcout << "Feature name is: " << data.getFeatureNames()[i] << endl;
    Rcout << "getFeatureID: " << data.getFeatureID(data.getFeatureNames()[i]) << endl;
  }

  // print jump data
  Rcout << "Times: ";
  printVector(data.getTimes());
  Rcout << "States: ";
  printVector(data.getStates());
  Rcout << "Last observed times: ";
  printVector(data.getLastObservedTimes());
  Rcout << "Censoring states: ";
  printVector(data.getCensoringStates());
  Rcout << "Valid jumps: ";
  for (auto jump : data.getValidJumps()) {
    Rcout << "(" << static_cast<size_t>(jump.first) << ", " << static_cast<size_t>(jump.second) << ")";
  }
  vector<double> unique_event_times = uniqueEventTimesMultistate(data.getTimes(), data.getStates());
  Rcout << endl << "The unique event times are: (total number : " << unique_event_times.size() << "):" << endl;;
  printVector(unique_event_times);
  Rcout << "The response event time ids are: ";
  vector<size_t> response_event_time_ids = computeResponseEventTimeIDsMultistate(
    unique_event_times, data.getTimes(), data.getStates(), data.getMaxResponseLength());
  printVector(response_event_time_ids);
  vector<bool> state_indicators = data.computeStateIndicators(response_event_time_ids, unique_event_times);
  Rcout << "The state indicators are: ";
  printVector(state_indicators);
  Rcout << "The last observed time ids are: ";
  printVector(data.getLastObservedTimes());

  // some metadata
  Rcout << "Number of observations: " << data.getNumberOfObs() << endl;
  Rcout << "Number of states: " << static_cast<size_t>(data.getNumberOfStates()) << endl;
}

// [[Rcpp::export]]
List testUniqueEventTimesThinning(const NumericVector& unique_event_times, size_t num_event_times) {
  vector<double> unique_event_times_cpp = as<vector<double>>(unique_event_times);
  vector<double> thinned_unique_event_times = thinUniqueEventTimes(unique_event_times_cpp, num_event_times);
  List result = List::create(
    Named("thinned_event_times") = thinned_unique_event_times
  );
  return result;
}

// for testing OpenMP
// [[Rcpp::export]]
void test_omp() {
  #pragma omp parallel
  {
    Rcout << "Thread: " << omp_get_thread_num() << "\n";
  }
}

// returns the size of the node in the tree belonging to x (for testing normality)
// [[Rcpp::export]]
int getNodeSize(const List& JFTree, const NumericVector& x) {
  Tree* tree = ((XPtr<Tree>) JFTree["Tree"]).get();
  vector<double> covariate = as<vector<double>>(x);
  return tree->nodeSize(covariate);
}

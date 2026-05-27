#include "JumpForests.h"

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
               unsigned int seed, size_t num_event_times = 0) {
  
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  string splitrule_cpp = as<string>(splitrule);

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
    // check validity of splitrule argument
    vector<string> valid_splitrules = {"mse", "variance", "mae"};
    if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
      throw runtime_error("Invalid splitrule, please choose between mse, variance or mae");
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
    tree->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, honest, seed);
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
    // check validity of splitrule argument
    vector<string> valid_splitrules = {"gini", "entropy", "misc", "twoing", "hellinger"};
    if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
      throw runtime_error("Invalid splitrule, please choose between gini, entropy, misc, twoing og hellinger");
    }

    ClassificationTree* tree;
    if (!honest) {
      tree = new ClassificationTree(subset_indices_cpp);
    } else {
      mt19937 rng(seed + 1);
      pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(subset_indices_cpp, rng);
      tree = new ClassificationTree(partition.first, partition.second);
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
    // determine the unique sorted (true) event times
    vector<double> times = as<vector<double>>(df[response_indices_cpp[0]]);
    vector<double> ind = as<vector<double>>(df[response_indices_cpp[1]]);
    vector<double> unique_event_times = uniqueValues(times);

    // if the user has specified a number of event times, thin the vector of unique event times
    if (num_event_times > 0) {
      unique_event_times = thinUniqueEventTimes(unique_event_times, num_event_times);
    }

    vector<size_t> response_event_time_ids = computeResponseEventTimeIDs(unique_event_times, times);
    vector<size_t> true_event_time_ids = computeTrueEventTimeIDs(unique_event_times, response_event_time_ids, ind);

    // create and grow the survival tree
    shared_ptr<vector<double>> unique_event_times_ptr = make_shared<vector<double>>(unique_event_times);
    shared_ptr<vector<size_t>> response_event_time_ids_ptr = make_shared<vector<size_t>>(response_event_time_ids);
    shared_ptr<vector<size_t>> true_event_time_ids_ptr = make_shared<vector<size_t>>(true_event_time_ids);

    // check validity of splitrule argument
    vector<string> valid_splitrules = {"logrank", "conserve", "approxlogrank"};
    if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
      throw runtime_error("Invalid splitrule, please choose between logrank, conserve or approxlogrank");
    }

    SurvivalTree* tree;
    if (!honest) {
      tree = new SurvivalTree(unique_event_times_ptr, response_event_time_ids_ptr, true_event_time_ids_ptr, subset_indices_cpp, true);
    } else {
      mt19937 rng(seed + 1);
      pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(subset_indices_cpp, rng);
      tree = new SurvivalTree(unique_event_times_ptr, response_event_time_ids_ptr, true_event_time_ids_ptr, partition.first, true, partition.second);
      tree->setRNG(rng);
    }
    
    tree->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, honest, seed);
    tree->grow();
    
    // specific to survival
    NumericVector unique_event_times_R(unique_event_times.begin(), unique_event_times.end());
    XPtr<SurvivalTree> survival_tree(tree, true);   // cast the survival tree as an R pointer
    result["num.deaths"] = vector_sum(ind);
    result["tree.type"] = "Survival";
    result["unique.event.times"] = unique_event_times_R;
    result["Tree"] = survival_tree;                 // add the tree (as a pointer, only to be used for prediction in C++)
    cout << "Computing predictions..." << endl;
    JFCppTreePredict(result);                       // compute and save predictions on the data
    JFCppTreeErrorSurvival(result, times, ind, unique_event_times, response_event_time_ids);     // compute and save error estimate

    // save information about the tree itself
    result["num.nodes"] = tree->getNumberOfNodes();
    result["num.terminal.nodes"] = tree->getNumberOfTerminalNodes();
    result["tree.depth"] = tree->getTreeDepth();
  }
  return result;
}

// [[Rcpp::export]]
List JFCppTreeMM(List jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame df_features, unsigned int mtry, unsigned int min_node_size, 
                 unsigned int nsplits, CharacterVector splitrule, bool honest, NumericVector feature_indices, LogicalVector categorical, NumericVector unique, 
                 unsigned int seed, size_t num_event_times = 0) {

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
  //Rcout << "Computed unique_event_times" << endl;

  // if the user has specified a number of event times, thin the vector of unique event times
  if (num_event_times > 0) {
    unique_event_times = thinUniqueEventTimes(unique_event_times, num_event_times);
  }
  vector<size_t> response_event_time_ids = computeResponseEventTimeIDsMultistate(unique_event_times, data->getTimes(), data->getStates());
  //Rcout << "Computed response_event_time_ids" << endl;

  // create and grow the multi-state tree
  shared_ptr<vector<double>> unique_event_times_ptr = make_shared<vector<double>>(unique_event_times);
  shared_ptr<vector<size_t>> response_event_time_ids_ptr = make_shared<vector<size_t>>(response_event_time_ids);

  // for debugging purposes
  /*
  Rcout << "Number of unique event times:" << unique_event_times.size() << endl;
  Rcout << "Length of response_event_time_ids: " << response_event_time_ids.size() << ", number of obs in flattened vector: " << data->getStates().size() << endl;
  Rcout << "Printing unique_event_times and response_event_time_ids:" << endl;
  printVector(*unique_event_times_ptr);
  printVector(*response_event_time_ids_ptr);
  */

  // check validity of splitrule argument
  vector<string> valid_splitrules = {"logrank", "gehan", "taroneware", "conserve", "approxlogrank"};
  if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
    throw runtime_error("Invalid splitrule, please choose between logrank, gehan, taroneware or approxlogrank");
  }

  MultistateTree* tree;
  if (!honest) {
    tree = new MultistateTree(unique_event_times_ptr, response_event_time_ids_ptr, subset_indices_cpp, num_states, true);
  } else {
    mt19937 rng(seed + 1);
    pair<vector<size_t>, vector<size_t>> partition = partitionHonesty(subset_indices_cpp, rng);
    tree = new MultistateTree(unique_event_times_ptr, response_event_time_ids_ptr, partition.first, num_states, true, partition.second);
    tree->setRNG(rng);
  }

  tree->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, honest, seed);
  //Rcout << "Tree initilised" << endl;
  tree->grow();
  //Rcout << "Tree grown" << endl;

  // specific to multi-states
  NumericVector unique_event_times_R(unique_event_times.begin(), unique_event_times.end());
  XPtr<MultistateTree> multistate_tree(tree, true);   // cast the multi-state tree as an R pointer
  result["tree.type"] = "Multi-state";
  result["unique.event.times"] = unique_event_times_R;
  result["Tree"] = multistate_tree;               // add the tree (as a pointer, only to be used for prediction in C++)
  JFCppTreePredict(result);                       // compute and save predictions on the data
  //Rcout << "Computed predictions" << endl;

  // compute and save error estimate (need to come up with something for multi-states)
  vector<double> state_weights(num_states, 1 / double(num_states));   // just choose default for now, let the user specify later
  vector<double> censoring_indicators(data->getNumberOfObs(), 0);
  const vector<uint8_t> censoring_states = data->getCensoringStates();
  for (size_t i = 0; i < data->getNumberOfObs(); ++i) {
    if (censoring_states[i] != 0) {
      censoring_indicators[i] = 1;    // censoring_state is 0 if and only if censoring has not occured
    }
  }
  //Rcout << "About to run JFCppTreeErrorMultistate" << endl;
  JFCppTreeErrorMultistate(result, data->getTimes(), data->getLastObservedTimes(), censoring_indicators, unique_event_times, response_event_time_ids, state_weights);
  //Rcout << "Done computing error for multi-state tree" << endl;

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

    // we saved the corresponding terminal node ID for every observation
    for (int i = 0; i < num_obs; ++i) {
      double pred = tree->getMeans()[tree->getPredictionNodeIDs()[i]];
      predictions[i] = pred;
    }
    JFTree["predictions"] = predictions;
  }
  if (type == "Classification") {
    ClassificationTree* tree = ((XPtr<ClassificationTree>) JFTree["Tree"]).get();
    size_t num_classes = as<size_t>(JFTree["num.classes"]);
    //size_t num_classes = tree->getData()->getNumClasses();
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
    /*
    SurvivalTree* tree = ((XPtr<SurvivalTree>) JFTree["Tree"]).get();
    NumericMatrix predictions(num_obs, tree->getEventTimes().size());

    // we saved the corresponding terminal node ID for every observation
    for (int i = 0; i < num_obs; ++i) {
      vector<double> pred = tree->getCHF()[tree->getPredictionNodeIDs()[i]];
      NumericVector rpred(pred.begin(), pred.end());
      predictions.row(i) = rpred;
    }

    // now truncate the time axis to only include non-censored times
    predictions = selectColumns(predictions, tree->getTrueEventTimeIDs());
    JFTree["predictions"] = predictions;
    */

    SurvivalTree* tree = ((XPtr<SurvivalTree>) JFTree["Tree"]).get();
    size_t num_unique_event_times = tree->getEventTimes().size();
    NumericMatrix predictions(num_obs, num_unique_event_times);
    NumericMatrix censoring(num_obs, num_unique_event_times);

    // we saved the corresponding terminal node ID for every observation
    for (int i = 0; i < num_obs; ++i) {
      size_t leaf_id = tree->getPredictionNodeIDs()[i];
      vector<double> pred = tree->getCHF()[leaf_id];
      vector<double> cens = tree->getKMCensoring()[leaf_id];
      NumericVector rpred(pred.begin(), pred.end());
      NumericVector rcens(cens.begin(), cens.end());
      predictions.row(i) = rpred;
      censoring.row(i) = rcens;
    }

    // now truncate the time axis to only include non-censored times
    //predictions = selectColumns(predictions, tree->getTrueEventTimeIDs());
    //censoring = selectColumns(censoring, tree->getTrueEventTimeIDs());
    
    JFTree["predictions"] = predictions;
    JFTree["predictions.km"] = KaplanMeier(predictions);
    JFTree["censoring"] = censoring;
  }

  if (type == "Multi-state") {
    MultistateTree* tree = ((XPtr<MultistateTree>) JFTree["Tree"]).get();
    size_t num_states = tree->getData()->getNumberOfStates();
    size_t num_unique_event_times = tree->getNumberOfUniqueEventTimes();
    const vector<size_t>& leaf_ids = tree->getPredictionNodeIDs();
    const vector<vector<double>>& na = tree->getNA();
    /*
    cout << "Checking NA estimators right after fitting (JFCppTreePredict):" << endl;
    for (auto NA : na) {
      if (!NA.empty()) {
        printVector(NA);
      }
    }
    */
    const vector<vector<double>>& init = tree->getInitDist();
    const vector<vector<double>>& cens = tree->getKMCensoring();

    List predictions(num_obs);                                  // each prediction is a list of matrices
    List predictions_init(num_obs);                             // each predicted initial distribution is a vector
    NumericMatrix censoring(num_obs, num_unique_event_times);   // a matrix with the KM estimator for observation i along the i'th row

    // we saved the corresponding terminal node ID for every observation
    for (size_t i = 0; i < num_obs; ++i) {
      size_t leaf_id = leaf_ids[i];
      //cout << "leaf_id = " << leaf_id << endl;
      //cout << "leaf_id (recomputed) = " << tree->predictionLeafID(tree->getData()->get_x_row(i)) << endl;
      List rpred(num_unique_event_times);

      vector<double> pred = na[leaf_id];
      for (size_t j = 0; j < num_unique_event_times; ++j) {
        auto start_it = pred.begin() + (j * num_states * num_states);
        NumericMatrix pred_time(num_states, num_states, start_it);
        rpred[j] = transpose(pred_time);
      }
      // save Nelson-Aalen estimator
      predictions[i] = rpred;

      // save censoring KM estimator
      //cout << "num_unique_event_times = " << num_unique_event_times << endl;
      vector<double> cens_pred = cens[leaf_id];
      //printVector(cens_pred);
      NumericVector rcens(cens_pred.begin(), cens_pred.end());
      censoring.row(i) = rcens;

      // save initial distribution
      NumericVector rpred_init(num_states);
      for (size_t j = 0; j < num_states; ++j) {
        rpred_init[j] = init[leaf_id][j];
      }
      predictions_init[i] = rpred_init;
    }
    JFTree["predictions"] = predictions;
    JFTree["init"] = predictions_init;
    JFTree["censoring"] = censoring;
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
      predictions(i, 0) = get<double>(tree->predict(new_data.get_x_row(i)));
    }
    return predictions;
  }

  if (type == "Classification") {
    ClassificationTree* tree = ((XPtr<ClassificationTree>) JFTree["Tree"]).get();
    size_t num_classes = as<size_t>(JFTree["num.classes"]);
    //size_t num_classes = tree->getData()->getNumClasses();
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

    /*
    NumericMatrix predictions(new_data.getNumberOfObs(), tree->getEventTimes().size());
    for (size_t i = 0; i < new_data.getNumberOfObs(); ++i) {
      vector<double> pred = get<vector<double>>(tree->predict(new_data.get_x_row(i)));
      copy(pred.begin(), pred.end(), predictions.row(i).begin());
    }
    
    // truncate the predictions to only include non-censored times
    predictions = selectColumns(predictions, tree->getTrueEventTimeIDs());
    return predictions;
    */
    
    size_t num_unique_event_times = tree->getEventTimes().size();
    NumericMatrix predictions(num_obs, num_unique_event_times);

    const vector<double>& predictions_cpp = tree->computePredictions(new_data);
    for (size_t i = 0; i < num_obs; ++i) {
      copy(predictions_cpp.begin() + i * num_unique_event_times, predictions_cpp.begin() + (i + 1) * num_unique_event_times, predictions.row(i).begin());
    }

    // truncate the predictions to only include non-censored times
    //predictions = selectColumns(predictions, tree->getTrueEventTimeIDs());
    return predictions;
  }
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
  size_t num_unique_event_times = tree->getEventTimes().size();
  NumericMatrix predictions(num_obs, num_unique_event_times);
  NumericMatrix censoring(num_obs, num_unique_event_times);

  const pair<vector<double>, vector<double>>& predictions_cpp = tree->computePredictionsCensoring(new_data);
  for (size_t i = 0; i < num_obs; ++i) {
    copy(predictions_cpp.first.begin() + i * num_unique_event_times, predictions_cpp.first.begin() + (i + 1) * num_unique_event_times, predictions.row(i).begin());
    copy(predictions_cpp.second.begin() + i * num_unique_event_times, predictions_cpp.second.begin() + (i + 1) * num_unique_event_times, censoring.row(i).begin());
  }

  // truncate the predictions to only include non-censored times
  //predictions = selectColumns(predictions, tree->getTrueEventTimeIDs());
  //censoring = selectColumns(censoring, tree->getTrueEventTimeIDs());
  List result = List::create(
    Named("predictions") = predictions,
    Named("censoring") = censoring
  );
  return result;
}

// function to predict on a new dataset for multi-states
// if compute_initial = false (default), simply return a list of the predictions on the new data df
// if compute_initial = true, add a list containing predicted initial distributions
// if compute_censoring = true, add a list containing the predicted censoring distributions (KM estimator)

// [[Rcpp::export]]
List JFCppTreePredictMM(const List& JFTree, DataFrame df, NumericVector feature_indices, LogicalVector categorical,
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
  size_t num_obs = new_data.getNumberOfObs();
  List predictions(num_obs);                                  // each prediction is a list of matrices
  List predictions_init(num_obs);                             // each predicted initial distribution is a vector
  NumericMatrix censoring(num_obs, num_unique_event_times);   // the censoring KM estimators are saved (if compute_censoring = TRUE) as a matrix like for survival
  
  // for debugging purposes
  /*
  vector<vector<double>> na = tree->getNA();                  // for very odd reasons, the NA are somehow destroyed... (this does not occur for any other tree type)
  cout << "Checking NA estimators before predicting on new data (JFCppTreePredictMM):" << endl;
  for (auto NA : na) {
    if (!NA.empty()) {
      printVector(NA);
    }
  }
  */

  // compute the predictions in C++
  vector<vector<double>> predictions_cpp = tree->computePredictions(compute_initial, compute_censoring, new_data);

  
  /*
  vector<double> predictions_cpp; // = tree->computePredictions(new_data);
  vector<double> predictions_init_cpp;
  vector<double> censoring_cpp;

  // compute predictions depending on the specified options (w/wo initial distributions and or censoring)
  if (compute_initial && compute_censoring) {
    vector<vector<double>> all_predictions_cpp = tree->computeAllPredictions(new_data);   // ordering: NA, initial distributions and censoring predictions
    predictions_cpp = std::move(all_predictions_cpp[0]);
    predictions_init_cpp = std::move(all_predictions_cpp[1]);
    censoring_cpp = std::move(all_predictions_cpp[2]);
  } else if (compute_initial) {
    pair<vector<double>, vector<double>> combined_predictions = tree->computePredictedInitialDistributions(new_data);
    predictions_cpp = std::move(combined_predictions.first);
    predictions_init_cpp = std::move(combined_predictions.second);
  } else if (compute_censoring) {
    pair<vector<double>, vector<double>> combined_predictions = tree->computePredictionsCensoring(new_data);
    predictions_cpp = std::move(combined_predictions.first);
    censoring_cpp = std::move(combined_predictions.second);
  } else {
    //cout << "Computing predictions (no init nor censoring)" << endl;
    predictions_cpp = tree->computePredictions(new_data);
  }
  */
  
  //cout << "predictions_cpp:" << endl;
  //printVector(predictions_cpp);

  //cout << "predictions_init_cpp:" << endl;
  //printVector(predictions_init_cpp);

  /*
  Plan:
  - Change this function to do computations in C++ instead
  - Add the remaining functions for prediction with and without censoring for multi-states
  - Add error functions for multi-states (test with and without predictions saved during fitting like for survival)
  */

  for (size_t i = 0; i < num_obs; ++i) {
    // save Nelson-Aalen estimator
    List rpred(num_unique_event_times);
    //const vector<double>& pred = get<vector<double>>(tree->predict(new_data.get_x_row(i)));
    for (size_t j = 0; j < num_unique_event_times; ++j) {
      auto start_it = predictions_cpp[0].begin() + ((i * num_unique_event_times + j) * num_states * num_states);
      //auto start_it = predictions_cpp.begin() + ((i * num_unique_event_times + j) * num_states * num_states);
      //auto start_it = pred.begin() + (j * num_states * num_states);
      NumericMatrix pred_time(num_states, num_states, start_it);
      rpred[j] = transpose(pred_time);
    }
    predictions[i] = rpred;

    // if compute_initial == true, save initial distribution
    if (compute_initial) {
      NumericVector rpred_init(num_states);
      //vector<double> pred_init = tree->predictInitDist(new_data.get_x_row(i));
      for (size_t j = 0; j < num_states; ++j) {
        //rpred_init[j] = predictions_init_cpp[i * num_states + j];
        rpred_init[j] = predictions_cpp[1][i * num_states + j];
      }
      predictions_init[i] = rpred_init;
    }
    // if compute_censoring == true, save censoring estimators
    if (compute_censoring && compute_initial) {
      copy(predictions_cpp[2].begin() + i * num_unique_event_times, predictions_cpp[2].begin() + (i + 1) * num_unique_event_times, censoring.row(i).begin());
      //copy(censoring_cpp.begin() + i * num_unique_event_times, censoring_cpp.begin() + (i + 1) * num_unique_event_times, censoring.row(i).begin());
    } else if (compute_censoring) {
      copy(predictions_cpp[1].begin() + i * num_unique_event_times, predictions_cpp[1].begin() + (i + 1) * num_unique_event_times, censoring.row(i).begin());
      //copy(censoring_cpp.begin() + i * num_unique_event_times, censoring_cpp.begin() + (i + 1) * num_unique_event_times, censoring.row(i).begin());
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
  JFTree["bs"] = computeBrierScoreError(prob_predictions, response, num_classes);
  JFTree["bs.normalised"] = computeNormalizedBrierScoreError(prob_predictions, response, num_classes);

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
  vector<double> IPCW_weights = computeIPCW(ind, unique_event_times, response_event_time_ids, JFTree["censoring"], times);
  vector<double> brier = computeBrierScore(times, IPCW_weights, unique_event_times, JFTree["predictions.km"]);
  pair<double, double> ibs = computeIBS(brier, unique_event_times);
  JFTree["ibs"] = ibs.first;
  JFTree["ibs.normalised"] = ibs.second;
}

double JFCppErrorSurvival(const NumericMatrix& predictions, const vector<double>& times, const vector<double>& ind) {
  // compute Harrell's C-index
  vector<double> outcomes = computeOutcomes(predictions);
  return 1 - computeConcordanceIndex(outcomes, times, ind);
}

// Multi-state

void JFCppTreeErrorMultistate(List& JFTree, const vector<double>& times, const vector<size_t>& last_observed_time_ids, const vector<double>& ind, 
                              const vector<double>& unique_event_times, const vector<size_t>& response_event_time_ids, const vector<double>& state_weights) {
  // first compute the IPCW weights
  vector<double> IPCW_weights = computeIPCW(ind, unique_event_times, response_event_time_ids, JFTree["censoring"], times, last_observed_time_ids);
  //cout << "Done computing IPCW weights" << endl;
  //printVector(IPCW_weights);
  //cout << "IPCW_weights.size() = " << IPCW_weights.size() << endl;

  // compute the status of whether each observation is in each state at the given event times
  MultistateTree* tree = ((XPtr<MultistateTree>) JFTree["Tree"]).get();
  vector<bool> states_ind = tree->getData()->computeStateIndicators(response_event_time_ids, unique_event_times);
  //cout << "Done computing states_ind" << endl;
  //printVector(states_ind);  // sceptical here, why so few ones?

  // compute occupation probabilities as a flattened vector (IMPORTANT: Let the user specify whether init should be used!)
  vector<double> occupation_probabilities = occupationProbabilities(JFTree["predictions"], JFTree["init"], state_weights.size());
  //cout << "Done computing occupation_probabilities" << endl;
  //printVector(occupation_probabilities);

  // compute the integrated Brier score (IBS) and the normalised IBS
  vector<double> brier = computeBrierScoreCppMM(states_ind, IPCW_weights, unique_event_times, occupation_probabilities, state_weights);
  //cout << "Done computing brier" << endl;
  pair<double, double> ibs = computeIBS(brier, unique_event_times, true);
  //cout << "Done computing ibs" << endl;

  // compute the integrated Kullback-Leibler loss and the normalised IKL
  // add this here when IBS seems to work

  // save all the error metrics in the tree list
  JFTree["ibs"] = ibs.first;
  JFTree["ibs.normalised"] = ibs.second;
}

// General function for a new dataset

// computes the error based on a a new dataset, here we don't need to specify the type of tree beforehand
// [[Rcpp::export]]
List JFCppTreeError(const List& JFTree, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique, NumericVector response_indices) {
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  //Rcout << "Line 280:" << response_indices_cpp[0] << endl;
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // convert the new data to a suitable C++ Data object
  Data new_data = Data(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);
  size_t num_obs = new_data.getNumberOfObs();

  string type = as<string>(JFTree["tree.type"]);
  if (type == "Regression") {
    RegressionTree* tree = ((XPtr<RegressionTree>) JFTree["Tree"]).get();
    vector<double> predictions(num_obs);
    for (size_t i = 0; i < new_data.getNumberOfObs(); ++i) {
      predictions[i] = get<double>(tree->predict(new_data.get_x_row(i)));
    }
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
    const vector<double>& response = new_data.get_y_col(0);
    size_t num_classes = as<size_t>(JFTree["num.classes"]);

    NumericVector class_misclassification_errors(num_classes);
    NumericMatrix confusionMatrix(num_classes, num_classes);

    List result = List::create(
      Named("BS.error") = computeBrierScoreError(predictions.second, response, num_classes),
      Named("normalised.BS.error") = computeNormalizedBrierScoreError(predictions.second, response, num_classes)
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
    vector<double> unique_event_times = tree->getEventTimes();
    size_t num_unique_event_times = unique_event_times.size();
    pair<vector<double>, vector<double>> predictions = tree->computePredictionsCensoring(new_data);

    // truncate the predictions to only include non-censored times
    //const vector<double>& predictions_final = selectColumns(predictions, tree->getTrueEventTimeIDs(), num_unique_event_times);

    // fetch data
    vector<double> times = new_data.get_y_col(0);
    vector<double> ind = new_data.get_y_col(1);

    // compute Harrell's C-index error
    const vector<double>& outcomes = computeOutcomes(predictions.first, num_unique_event_times);
    double c_index = computeConcordanceIndex(outcomes, times, ind);
    List result = List::create(Named("C.error") = 1 - c_index);

    // compute the Brier score
    vector<size_t> response_event_time_ids_new_data = computeResponseEventTimeIDs(unique_event_times, times);   // have to compute the ids from scratch
    vector<double> IPCW_weights = computeIPCWCpp(times, ind, unique_event_times, response_event_time_ids_new_data, predictions.second);
    vector<double> km_pred = KaplanMeier(predictions.first, times.size());
    vector<double> brier = computeBrierScoreCpp(times, IPCW_weights, unique_event_times, km_pred);
    pair<double, double> ibs = computeIBS(brier, unique_event_times);
    result["IBS.error"] = ibs.first;
    result["normalised.IBS.error"] = ibs.second;
    return result;

    /*
    size_t num_unique_event_times = tree->getEventTimes().size();
    NumericMatrix predictions(num_obs, num_unique_event_times);
    for (size_t i = 0; i < num_obs; ++i) {
      copy(predictions_cpp.begin() + i * num_unique_event_times, predictions_cpp.begin() + (i + 1) * num_unique_event_times, predictions.row(i).begin());
    }

    // truncate the predictions to only include non-censored times
    predictions = selectColumns(predictions, tree->getTrueEventTimeIDs());
    */    
  }
  if (type == "Multi-state") {
    
  }
  else {
    throw runtime_error("Type of forest not recognised");
  }
}

// growing random forests
//--------------------------------------------------------------------------------------

// [[Rcpp::export]]
List JFCppForest(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, CharacterVector splitrule,
    unsigned int ntrees, bool honest, bool swr, double sample_rate, bool double_bootstrap, NumericVector response_indices, NumericVector feature_indices, 
    LogicalVector categorical, NumericVector unique, unsigned int seed, unsigned int nworkers, bool save_predictions, size_t num_event_times = 0) {

  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  string splitrule_cpp = as<string>(splitrule);

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
    // check validity of splitrule argument
    vector<string> valid_splitrules = {"mse", "variance", "mae"};
    if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
      throw runtime_error("Invalid splitrule, please choose between mse, variance or mae");
    }

    // create and grow the regression forest
    RegressionForest* forest = new RegressionForest();
    forest->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, ntrees, honest, swr, sample_rate, double_bootstrap, seed, nworkers);
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
    // determine the unique sorted (true) event times
    vector<double> times = as<vector<double>>(df[response_indices_cpp[0]]);
    vector<double> ind = as<vector<double>>(df[response_indices_cpp[1]]);
    vector<double> unique_event_times = uniqueValues(times);

    // if the user has specified a number of event times, thin the vector of unique event times
    if (num_event_times > 0) {
      unique_event_times = thinUniqueEventTimes(unique_event_times, num_event_times);
    }

    vector<size_t> response_event_time_ids = computeResponseEventTimeIDs(unique_event_times, times);
    vector<size_t> true_event_time_ids = computeTrueEventTimeIDs(unique_event_times, response_event_time_ids, ind);

    // check validity of splitrule argument
    vector<string> valid_splitrules = {"logrank", "conserve", "approxlogrank"};
    if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
      throw runtime_error("Invalid splitrule, please choose between logrank, conserve, or approxlogrank");
    }

    // create and grow the survival forest
    SurvivalForest* forest = new SurvivalForest(unique_event_times, response_event_time_ids, true_event_time_ids, save_predictions);
    forest->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, ntrees, honest, swr, sample_rate, double_bootstrap, seed, nworkers);
    forest->grow();

    // specific to survival
    NumericVector unique_event_times_R(unique_event_times.begin(), unique_event_times.end());
    XPtr<SurvivalForest> survival_forest(forest, true);
    result["num.deaths"] = vector_sum(ind);
    result["tree.type"] = "Survival";
    result["num.trees"] = ntrees;
    result["unique.event.times"] = unique_event_times_R;
    result["Forest"] = survival_forest;           // add the forest as a pointer, only to be used for prediction

    if (save_predictions) {
      Rcout << "Computing forest predictions" << endl;
      JFCppForestPredict(result);
      JFCppForestErrorSurvival(result, times, ind, unique_event_times, response_event_time_ids); // compute and save error result
    } else {
      result["predictions"] = R_NilValue;
      result["oob.predictions"] = R_NilValue;
      result["censoring.oob"] = R_NilValue;
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
List JFCppForestMM(List jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame df_features, unsigned int mtry, unsigned int min_node_size, 
  unsigned int nsplits, CharacterVector splitrule, unsigned int ntrees, bool honest, bool swr, double sample_rate, bool double_bootstrap, NumericVector feature_indices, 
  LogicalVector categorical, NumericVector unique, unsigned int seed, unsigned int nworkers, bool save_predictions, size_t num_event_times = 0) {
  
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
  
  vector<size_t> response_event_time_ids = computeResponseEventTimeIDsMultistate(unique_event_times, data->getTimes(), data->getStates());

  // check validity of splitrule argument (just logrank for now)
  vector<string> valid_splitrules = {"logrank", "gehan", "taroneware", "conserve", "approxlogrank"};
  if (find(valid_splitrules.begin(), valid_splitrules.end(), splitrule_cpp) == valid_splitrules.end()) {
    throw runtime_error("Invalid splitrule, please choose between logrank, gehan or taroneware");
  }

  // create and grow the multi-state forest
  MultistateForest* forest = new MultistateForest(unique_event_times, response_event_time_ids, data->getNumberOfStates(), save_predictions);
  forest->initialise(data, mtry, min_node_size, nsplits, splitrule_cpp, ntrees, honest, swr, sample_rate, double_bootstrap, seed, nworkers);
  forest->grow();

  // specific to multi-states
  NumericVector unique_event_times_R(unique_event_times.begin(), unique_event_times.end());
  result["tree.type"] = "Multi-state";
  result["num.trees"] = ntrees;
  result["unique.event.times"] = unique_event_times_R;
  XPtr<MultistateForest> multistate_forest(forest, true);
  // compute total number of jumps? maybe in Data?
  result["Forest"] = multistate_forest;           // add the forest as a pointer, only to be used for prediction

  if (save_predictions) {
    Rcout << "Computing forest predictions" << endl;
    JFCppForestPredict(result);

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
    //const pair<vector<double>, vector<double>>& predictions_cpp = forest->computePredictions(); // computes in-bag and out-of-bag predictions
    //cout << "Done computing predictions (in-bag and out-of-bag)" << endl;

    
    //cout << "Done populating leaves with KM estimators" << endl;
    //const vector<double>& censoring_cpp = forest->computePredictionsCensoringOOB(); // now able to fetch the predicted censoring KM estimators
    //cout << "Done computing predicted censoring KM estimators" << endl;
    size_t num_unique_event_times = forest->getNumUniqueEventTimes();
    size_t num_obs = JFForest["num.obs"];
    
    // now fill the matrices of predictions
    NumericMatrix predictions(num_obs, num_unique_event_times);
    NumericMatrix predictions_oob(num_obs, num_unique_event_times);
    NumericMatrix censoring_oob(num_obs, num_unique_event_times);
    for (size_t i = 0; i < num_obs; ++i) {
      for (size_t j = 0; j < num_unique_event_times; ++j) {
        size_t index = i * num_unique_event_times + j;
        //predictions(i, j) = predictions_cpp.first[index];
        //predictions_oob(i, j) = predictions_cpp.second[index];
        //censoring_oob(i, j) = censoring_cpp[index];
        predictions(i, j) = predictions_cpp[0][index];
        predictions_oob(i, j) = predictions_cpp[1][index];
        if (compute_censoring) {
          censoring_oob(i, j) = predictions_cpp[2][index];
        }
      }
    }

    // now truncate the time axis to only include non-censored times
    //predictions = selectColumns(predictions, forest->getTrueEventTimeIDs());
    //predictions_oob = selectColumns(predictions_oob, forest->getTrueEventTimeIDs());

    // save predictions
    JFForest["predictions"] = predictions;
    JFForest["oob.predictions"] = predictions_oob;
    if (compute_censoring) {
      JFForest["censoring.oob"] = censoring_oob;
    }
  }
  
  if (type == "Multi-state") {
    // compute predictions via multi-threading
    MultistateForest* forest = ((XPtr<MultistateForest>) JFForest["Forest"]).get();

    const vector<vector<double>>& predictions_cpp = forest->computePredictions(true, false);

    // old computation
    //const pair<vector<double>, vector<double>>& predictions_cpp = forest->computePredictions();
    //const pair<vector<double>, vector<double>>& predictions_init_cpp = forest->computePredictedInitialDistributions();

    size_t num_unique_event_times = forest->getNumUniqueEventTimes();
    size_t num_obs = JFForest["num.obs"];
    uint8_t num_states = forest->getData()->getNumberOfStates();
    uint8_t dim = num_states * num_states;
    List predictions(num_obs);        // each prediction is a list of matrices
    List predictions_oob(num_obs);
    List predictions_init(num_obs);   // each predicted initial distribution is a vector
    List predictions_init_oob(num_obs);

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
            //pred_time(j, k) = predictions_cpp.first[i * num_unique_event_times * dim + t * dim + j * num_states + k];
            //pred_time_oob(j, k) = predictions_cpp.second[i * num_unique_event_times * dim + t * dim + j * num_states + k];
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
        //rpred_init[j] = predictions_init_cpp.first[i * num_states + j];
        //rpred_init_oob[j] = predictions_init_cpp.second[i * num_states + j];
      }
      predictions_init[i] = rpred_init;
      predictions_init_oob[i] = rpred_init_oob;
    }
    // save predictions
    JFForest["predictions"] = predictions;
    JFForest["oob.predictions"] = predictions_oob;
    JFForest["init"] = predictions_init;
    JFForest["oob.init"] = predictions_init_oob;
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
    size_t num_unique_event_times = forest->getNumUniqueEventTimes();
    NumericMatrix predictions(num_obs, num_unique_event_times);
    cout << "Computing predictions" << endl;
    const vector<vector<double>>& predictions_cpp = forest->computePredictions(new_data, false);
    //cout << "Finished computing predictions" << endl;
    for (size_t i = 0; i < num_obs; ++i) {
      copy(predictions_cpp[0].begin() + i * num_unique_event_times, predictions_cpp[0].begin() + (i + 1) * num_unique_event_times, predictions.row(i).begin());
    }

    // truncate the predictions to only include non-censored times
    //predictions = selectColumns(predictions, forest->getTrueEventTimeIDs());
    return predictions;
  }
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
  size_t num_unique_event_times = forest->getEventTimes().size();
  NumericMatrix predictions(num_obs, num_unique_event_times);
  NumericMatrix censoring(num_obs, num_unique_event_times);
  
  // if predictions are not saved, it means that the censoring KM estimators need to be computed from scratch in each leaf
  if (!forest->predictionsSaved()) {
    forest->computePredictionsCensoring();
  }

  const vector<vector<double>> predictions_cpp = forest->computePredictions(new_data, true);
  //const pair<vector<double>, vector<double>>& predictions_cpp = forest->computePredictionsCensoring(new_data);
  for (size_t i = 0; i < num_obs; ++i) {
    copy(predictions_cpp[0].begin() + i * num_unique_event_times, predictions_cpp[0].begin() + (i + 1) * num_unique_event_times, predictions.row(i).begin());
    copy(predictions_cpp[1].begin() + i * num_unique_event_times, predictions_cpp[1].begin() + (i + 1) * num_unique_event_times, censoring.row(i).begin());
  }

  // truncate the predictions to only include non-censored times
  //predictions = selectColumns(predictions, tree->getTrueEventTimeIDs());
  //censoring = selectColumns(censoring, tree->getTrueEventTimeIDs());
  List result = List::create(
    Named("predictions") = predictions,
    Named("censoring") = censoring
  );
  return result;
}

// if compute_initial = false (default), simply return a list of the predictions on the new data df
// if compute_initial = true, return a list of two lists, one containing predictions on the new data
// and the other containing predicted initial distributions

// [[Rcpp::export]]
List JFCppForestPredictMM(const List& JFForest, DataFrame df, NumericVector feature_indices,
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
  uint8_t num_states = forest->getData()->getNumberOfStates();
  uint8_t dim = num_states * num_states;
  List predictions(num_obs);      // each prediction is a list of matrices
  List predictions_init(num_obs); // each predicted initial distribution is a vector
  
  const vector<vector<double>>& predictions_cpp = forest->computePredictions(new_data, compute_initial, false);

  // old
  //const vector<double>& predictions_cpp = forest->computePredictions(new_data);
  //vector<double> predictions_init_cpp;

  //if (compute_initial) {
  //  predictions_init_cpp = forest->computePredictedInitialDistributions(new_data);
  //}

  for (size_t i = 0; i < num_obs; ++i) {
    List rpred(num_unique_event_times);
    for (size_t t = 0; t < num_unique_event_times; ++t) {
      NumericMatrix pred_time(num_states, num_states);
      for (size_t j = 0; j < num_states; ++j) {
        for (size_t k = 0; k < num_states; ++k) {
          pred_time(j, k) = predictions_cpp[0][i * num_unique_event_times * dim + dim * t + j * num_states + k];
          //pred_time(j, k) = predictions_cpp[i * num_unique_event_times * dim + dim * t + j * num_states + k];
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
        //rpred_init[j] = predictions_init_cpp[i * num_states + j];
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
  /*
  vector<double> predictions(num_obs);
  const NumericVector& predictions_R = JFForest["oob.predictions"];
  for (size_t i = 0; i < num_obs; ++i) {
    predictions[i] = predictions_R[i];
  }
  */
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
  JFForest["bs"] = computeBrierScoreError(prob_predictions, response_oob, num_classes);
  JFForest["bs.normalised"] = computeNormalizedBrierScoreError(prob_predictions, response_oob, num_classes);
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

  // now compute Brier score
  vector<double> IPCW_weights = computeIPCW(ind, unique_event_times, response_event_time_ids, JFForest["censoring.oob"], times);
  //const NumericMatrix& km_pred = KaplanMeier(as<NumericMatrix>(JFForest["oob.predictions"]));
  //cout << "OOB IPCW weights:" << endl;
  //printVector(IPCW_weights);
  vector<double> brier = computeBrierScore(times, IPCW_weights, unique_event_times, KaplanMeier(as<NumericMatrix>(JFForest["oob.predictions"])));
  //cout << "OOB Brier scores:" << endl;
  //printVector(brier);
  pair<double, double> ibs = computeIBS(brier, unique_event_times);
  JFForest["ibs"] = ibs.first;
  JFForest["ibs.normalised"] = ibs.second;
}

// for computing OOB errors for a survival forest when errors are not already saved, but predictions are computed

// [[Rcpp::export]]
List JFCppForestErrorSurvivalExternal(List& JFForest) {
  SurvivalForest* forest = ((XPtr<SurvivalForest>) JFForest["Forest"]).get();
  const vector<double>& times = forest->getData()->get_y_col(0);
  const vector<double>& ind = forest->getData()->get_y_col(1);
  const vector<double>& unique_event_times = forest->getEventTimes();
  const vector<size_t>& response_event_time_ids = forest->getResponseEventTimeIDs();

  // first compute Harrell's C-index
  const vector<double>& outcomes = computeOutcomes(JFForest["oob.predictions"]);
  JFForest["outcomes.oob"] = outcomes;

  // now compute Brier score
  vector<double> IPCW_weights = computeIPCW(ind, unique_event_times, response_event_time_ids, JFForest["censoring.oob"], times);
  //const NumericMatrix& km_pred = KaplanMeier(as<NumericMatrix>(JFForest["oob.predictions"]));
  //cout << "OOB IPCW weights:" << endl;
  //printVector(IPCW_weights);
  vector<double> brier = computeBrierScore(times, IPCW_weights, unique_event_times, KaplanMeier(as<NumericMatrix>(JFForest["oob.predictions"])));
  //cout << "OOB Brier scores:" << endl;
  //printVector(brier);
  pair<double, double> ibs = computeIBS(brier, unique_event_times);
  
  // create and return list of errors
  double c_error = 1 - computeConcordanceIndex(outcomes, times, ind);
  List result = List::create(
    Named("C.error") = c_error,
    Named("IBS") = ibs.first,
    Named("IBS.normalised") = ibs.second
  );

  return result;
}

// Multi-state

// General function for a new dataset

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
    const vector<double>& response = new_data.get_y_col(0);

    size_t num_classes = as<size_t>(JFForest["num.classes"]);
    NumericVector class_misclassification_errors(num_classes);
    NumericMatrix confusionMatrix(num_classes, num_classes);

    // create list, compute and save the Brier Score error
    List result = List::create(Named("BS.error") = computeBrierScoreError(predictions.second, response, num_classes),
                               Named("normalised.BS.error") = computeNormalizedBrierScoreError(predictions.second, response, num_classes));

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
    vector<double> unique_event_times = forest->getEventTimes();
    size_t num_unique_event_times = unique_event_times.size();

    // check if predictions are saved, if not, the censoring KM estimators have to be computed in every leaf
    if (!forest->predictionsSaved()) {
      forest->computePredictionsCensoring();
    }
    
    const vector<vector<double>>& predictions = forest->computePredictions(new_data, true);
    //pair<vector<double>, vector<double>> predictions = forest->computePredictionsCensoring(new_data);

    // truncate the predictions to only include non-censored times
    //predictions = selectColumns(predictions, forest->getTrueEventTimeIDs());

    // fetch data
    const vector<double>& times = new_data.get_y_col(0);
    const vector<double>& ind = new_data.get_y_col(1);

    // compute Harrell's C-index error
    const vector<double>& outcomes = computeOutcomes(predictions[0], num_unique_event_times);
    List result = List::create(Named("C.error") = 1 - computeConcordanceIndex(outcomes, times, ind));

    // compute the Brier score
    vector<size_t> response_event_time_ids_new_data = computeResponseEventTimeIDs(unique_event_times, times);   // have to compute the ids from scratch
    vector<double> IPCW_weights = computeIPCWCpp(times, ind, unique_event_times, response_event_time_ids_new_data, predictions[1]);
    //cout << "ICPW weights on new data:" << endl;
    //printVector(IPCW_weights);
    vector<double> km_pred = KaplanMeier(predictions[0], times.size());
    vector<double> brier = computeBrierScoreCpp(times, IPCW_weights, unique_event_times, km_pred);
    //cout << "Brier scores on new data:" << endl;
    //printVector(brier);
    pair<double, double> ibs = computeIBS(brier, unique_event_times);
    result["IBS.error"] = ibs.first;
    result["normalised.IBS.error"] = ibs.second;
    return result;
  }
  if (type == "Multi-state") {

  }
  else {
    throw runtime_error("Type of forest not recognised");
  }
}

// computes VIMP for a specific feature after the forest is grown
// [[Rcpp::export]]
double JFCppForestVIMPFeature(const List& JFForest, CharacterVector feature_name, int feature_seed, CharacterVector method) {
  string type = as<string>(JFForest["tree.type"]);
  string method_cpp = as<string>(method);
  //size_t num_obs = as<size_t>(JFForest["num.obs"]);

  // translate from feature name to feature index
  string feature_name_cpp = as<string>(feature_name);

  if (type == "Regression") {
    RegressionForest* forest = ((XPtr<RegressionForest>) JFForest["Forest"]).get();
    size_t feature = forest->getData()->getFeatureID(feature_name_cpp);
    if (method_cpp == "permute") {
      return forest->computeVIMPPermute(feature, feature_seed);
    } else if (method_cpp == "random") {
      return forest->computeVIMPRandom(feature, feature_seed);
    } else {
      throw runtime_error("Type of VIMP computation method not recognised, use 'permute' or 'random'");
    }
  }
  if (type == "Classification") {

  }
  if (type == "Survival") {
    SurvivalForest* forest = ((XPtr<SurvivalForest>) JFForest["Forest"]).get();
    size_t feature = forest->getData()->getFeatureID(feature_name_cpp);
    if (method_cpp == "permute") {
      return forest->computeVIMPPermute(feature, feature_seed);
    } else if (method_cpp == "random") {
      return forest->computeVIMPRandom(feature, feature_seed);
    } else {
      throw runtime_error("Type of VIMP computation method not recognised, use 'permute' or 'random'");
    }
      
    

    /*
    // first compute VIMP predictions
    vector<double> predictions_vimp_cpp;
    if (method_cpp == "permute") {
      predictions_vimp_cpp = forest->computePredictionsVIMPPermute(feature, feature_seed);
    } else if (method_cpp == "random") {
      predictions_vimp_cpp = forest->computePredictionsVIMPRandom(feature, feature_seed);
    } else {
      throw runtime_error("Type of VIMP computation method not recognised, use 'permute' or 'random'");
    }

    //Rcout << "Line 376: Done computing VIMP predictions for feature " << feature << endl;
    size_t num_unique_event_times = forest->getEventTimes().size();
    NumericMatrix predictions_vimp(num_obs, num_unique_event_times);
    for (size_t i = 0; i < num_obs; ++i) {
      for (size_t j = 0; j < num_unique_event_times; ++j) {
        predictions_vimp(i, j) = predictions_vimp_cpp[i * num_unique_event_times + j];
      }
    }

    vector<double> outcomes_vimp = computeOutcomes(predictions_vimp);
    //Rcout << "Line 386: VIMP outcomes computed for feature " << feature << endl;
    //Rcout << "Line 387: times: "; printVector(forest->getData()->get_y_col(0)); 
    //Rcout << "Line 388: ind: "; printVector(forest->getData()->get_y_col(1));
    double vimp_error = (1 - computeConcordanceIndex(outcomes_vimp, forest->getData()->get_y_col(0), forest->getData()->get_y_col(1)));
    double vimp = vimp_error - as<double>(JFForest["C.error"]);
    Rcout << "Line 389: VIMP for feature " << feature << ": " << vimp << endl;
    return vimp;
    */
  }
  if (type == "Multi-state") {
    
  }
}

// computes VIMP for every variable and saves the list of VIMP-values in JFForest (set seed for reproducibility)
// [[Rcpp::export]]
List JFCppForestVIMP(List& JFForest, int seed, CharacterVector method) {
  mt19937 random_number_generator(seed);
  Forest* forest = ((XPtr<Forest>) JFForest["Forest"]).get();
  size_t num_features = forest->getData()->getNumberOfFeatures();
  vector<string> feature_names = forest->getData()->getFeatureNames();

  // for generating feature specific seeds in VIMP computations
  uniform_int_distribution<size_t> compute_feature_seed(0, numeric_limits<size_t>::max());

  List VIMP;
  for (size_t i = 0; i < num_features; ++i) {
    size_t feature_seed = compute_feature_seed(random_number_generator);
    VIMP[feature_names[i]] = JFCppForestVIMPFeature(JFForest, feature_names[i], feature_seed, method);
  }
  
  // why does this not work?
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

// below is a temporary (now outdated) test function to make sure all methods work

/*

// [[Rcpp::export]]
void fitSurvivalTree(DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, 
                     NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical,
                    NumericVector unique, NumericVector subset_indices) {
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);
  vector<size_t> subset_indices_cpp = as<vector<size_t>>(subset_indices);

  // determine the unique sorted (true) event times
  vector<double> times = as<vector<double>>(df[response_indices[0]]);
  vector<double> ind = as<vector<double>>(df[response_indices[1]]);
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
  vector<size_t> response_event_time_ids = computeResponseEventTimeIDs(unique_event_times, times);
  vector<size_t> true_event_time_ids = computeTrueEventTimeIDs(unique_event_times, response_event_time_ids, ind);
  
  // create the SurvivalTree
  SurvivalTree tree = SurvivalTree(unique_event_times, response_event_time_ids, true_event_time_ids, subset_indices_cpp);
  shared_ptr<Data> data = make_shared<Data>(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);
  tree.initialise(data, mtry, min_node_size, nsplits, 2025); // just set seed to something

  // grow the SurvivalTree
  // the bug happens after
  tree.grow();
  Rcout << "Done!" << endl;

  
  // write out predictions (testing)
  Rcout << "The terminal node values are:" << endl;
  vector<vector<double>> predictions = tree.getCHF();
  for (vector<double> vec : predictions) {
    for (int i = 0; i < vec.size(); ++i) {
      Rcout << vec[i] << ", ";
    }
    Rcout << endl;
  }

  // compute predictions (testing)
  Rcout << "The predicted values for the data are: " << endl;
  for (int i = 0; i < (*(tree.getData())).getNumberOfObs(); ++i) {
    vector<double> pred = get<vector<double>>(tree.predict((*(tree.getData())).get_x_row(i)));
    for (double h : pred) {
      Rcout << h << ", ";
    }
    Rcout << endl;
  }
    
}

*/

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

  /*
  // print first 10 observed times
  vector<double> times_trunc = data.get_y_col(0);
  vector<double> ind_trunc  = data.get_y_col(1);
  for (size_t i = 0; i < 10; ++i) {
    Rcout << times_trunc[i] << ", ";
  }
  Rcout << endl;
  for (size_t i = 0; i < 10; ++i) {
    Rcout << ind_trunc[i] << ", ";
  }
  Rcout << endl;
  */
  
}

// for testing that all data functionalities related to multi-states work
// [[Rcpp::export]]
void testDataMM(const List& jump_data, uint8_t max_response_length, uint8_t num_states, DataFrame& feature_df, 
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
  //Rcout << "Censoring times: ";
  //printVector(data.getCensoringTimes());
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
  vector<size_t> response_event_time_ids = computeResponseEventTimeIDsMultistate(unique_event_times, data.getTimes(), data.getStates());
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
  //Rcout << "Original vector of length " << unique_event_times.size() << ":" << endl;
  //printVector(unique_event_times_cpp);
  //Rcout << "Thinned vector of length " << result.size() << ":" << endl;
  //printVector(result);
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
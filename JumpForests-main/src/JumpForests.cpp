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

// [[Rcpp::export]]
List JFCppTree(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, 
    NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical,
    NumericVector unique, unsigned int seed) {
  
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // make the data into a C++ format and save it via a shared pointer
  shared_ptr<Data> data = make_shared<Data>(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);

  // use all indices since we grow a single tree
  vector<size_t> subset_indices_cpp;
  for (int i = 0; i < (*data).getNumberOfObs(); ++i) {
    subset_indices_cpp.push_back(i);
  }

  // we save a list with all information about the tree
  List result = List::create(
    Named("num.obs") = (*data).getNumberOfObs(),
    Named("num.features") = (*data).getNumberOfFeatures(),
    Named("feature.names") = (*data).getFeatureNames(), 
    Named("response.names") = (*data).getResponseNames(),
    Named("mtry") = mtry,
    Named("min.node.size") = min_node_size,
    Named("nsplits") = nsplits
  );
  
  // the tree is a regression tree
  if (tree_type == 1) {

  }

  // the tree is a classification tree
  if (tree_type == 2) {

  }

  // the tree is a survival tree
  if (tree_type == 3) {
    // determine the unique sorted (true) event times
    vector<double> times = as<vector<double>>(df[response_indices_cpp[0]]);
    vector<size_t> ind = as<vector<size_t>>(df[response_indices_cpp[1]]);
    vector<double> unique_event_times = computeUniqueEventTimes(times, ind);

    // create and grow the survival tree
    SurvivalTree* tree = new SurvivalTree(unique_event_times, subset_indices_cpp);
    tree->initialise(data, mtry, min_node_size, nsplits, seed);
    tree->grow();

    //specific to survival
    NumericVector unique_event_times_R(unique_event_times.begin(), unique_event_times.end());
    XPtr<SurvivalTree> survival_tree(tree, true);   // cast the survival tree as an R pointer
    result["tree.type"] = "Survival";
    result["unique.event.times"] = unique_event_times_R;
    result["Tree"] = survival_tree;                 // add the tree (as a pointer, only to be used for prediction in C++)
    JFCppTreePredict(result);                       // compute and save predictions on the data

    // save information about the tree itself
    result["num.nodes"] = (*tree).getNumberOfNodes();
    result["num.terminal.nodes"] = (*tree).getNumberOfTerminalNodes();
    result["tree.depth"] = (*tree).getTreeDepth();
  }

  // the tree is a multi-state tree
  if (tree_type == 4) {
    
  }

  return(result);
}

/*

The following functions are for prediction with a single tree. For single trees, predictions for the data used to fit
are always computed and saved in the list (see the JFCppTree function above)

*/

void JFCppTreePredict(List& JFTree) {
  string type = as<string>(JFTree["tree.type"]);
  if (type == "Regression") {

  }
  if (type == "Classification") {

  }
  if (type == "Survival") {
    SurvivalTree* tree = ((XPtr<SurvivalTree>) JFTree["Tree"]).get();
    NumericMatrix predictions(JFTree["num.obs"], (*tree).getEventTimes().size());

    // we saved the corresponding terminal node ID for every observation
    for (int i = 0; i < as<int>(JFTree["num.obs"]); ++i) {
      vector<double> pred = (*tree).getCHF()[(*tree).getPredictionNodeIDs()[i]];
      NumericVector rpred(pred.begin(), pred.end());
      predictions.row(i) = rpred;
    }

    JFTree["predictions"] = predictions;
  }
  if (type == "Multi-state") {

  }
}

// it is possible that the return type should be different such as a list (especially for multi-states)

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

  string type = as<string>(JFTree["tree.type"]);
  if (type == "Regression") {

  }

  if (type == "Classification") {
    
  }

  if (type == "Survival") {
    SurvivalTree* tree = ((XPtr<SurvivalTree>) JFTree["Tree"]).get();

    NumericMatrix predictions(new_data.getNumberOfObs(), (*tree).getEventTimes().size());
    for (int i = 0; i < new_data.getNumberOfObs(); ++i) {
      printVector(new_data.get_x_row(i));
      vector<double> pred = get<vector<double>>((*tree).predict(new_data.get_x_row(i)));
      copy(pred.begin(), pred.end(), predictions.row(i).begin());
    }
    return(predictions);
  }

  if (type == "Multi-state") {
    
  }
}

// add more options later

// [[Rcpp::export]]
List JFCppForest(uint tree_type, DataFrame df, unsigned int mtry, unsigned int min_node_size, unsigned int nsplits, 
    unsigned int ntrees, NumericVector response_indices, NumericVector feature_indices, LogicalVector categorical,
    NumericVector unique, unsigned int seed) {

  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp = as<vector<size_t>>(response_indices);
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // make the data into a C++ format and save it via a shared pointer
  shared_ptr<Data> data = make_shared<Data>(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);

  // we save a list with all information about the forest
  List result = List::create(
    Named("num.obs") = data->getNumberOfObs(),
    Named("num.features") = data->getNumberOfFeatures(),
    Named("feature.names") = data->getFeatureNames(), 
    Named("response.names") = data->getResponseNames(),
    Named("mtry") = mtry,
    Named("min.node.size") = min_node_size,
    Named("nsplits") = nsplits
  );

  // the forest is a regression forest
  if (tree_type == 1) {

  }

  // the forest is a classification forest
  if (tree_type == 2) {

  }

  // the forest is a survival forest
  if (tree_type == 3) {
    // determine the unique sorted (true) event times
    vector<double> times = as<vector<double>>(df[response_indices_cpp[0]]);
    vector<size_t> ind = as<vector<size_t>>(df[response_indices_cpp[1]]);
    vector<double> unique_event_times = computeUniqueEventTimes(times, ind);

    // create and grow the survival forest
    SurvivalForest* forest = new SurvivalForest(unique_event_times);
    forest->initialise(data, mtry, min_node_size, nsplits, ntrees, seed);
    forest->grow();

    // specific to survival
    NumericVector unique_event_times_R(unique_event_times.begin(), unique_event_times.end());
    XPtr<SurvivalForest> survival_forest(forest, true);
    result["tree.type"] = "Survival";
    result["num.trees"] = ntrees;
    result["unique.event.times"] = unique_event_times_R;
    result["Forest"] = survival_forest;   // add the forest as a pointer, only to be used for prediction

    JFCppForestPredict(result);    // compute and save predictions on the data

    result["avg.num.nodes"] = forest->getAvgNumberOfNodes();
    result["avg.num.terminal.nodes"] = forest->getAvgNumberOfTerminalNodes();
    result["avg.tree.depth"] = forest->getAvgTreeDepth();
  }

  // the forest is a multi-state forest
  if (tree_type == 4) {

  }
  return(result);
}

void JFCppForestPredict(List& JFForest) {
  string type = as<string>(JFForest["tree.type"]);
  if (type == "Regression") {

  }
  if (type == "Classification") {

  }
  
  if (type == "Survival") {
    SurvivalForest* forest = ((XPtr<SurvivalForest>) JFForest["Forest"]).get();
    int num_unique_event_times = forest->getEventTimes().size();
    NumericMatrix predictions(JFForest["num.obs"], num_unique_event_times);
    NumericMatrix predictions_oob(JFForest["num.obs"], num_unique_event_times);
    const auto& trees = forest->getTrees();
    vector<vector<bool>> OOB_indices = forest->getOOBIndices();

    // we saved the corresponding terminal node ID for every observation
    for (int i = 0; i < as<int>(JFForest["num.obs"]); ++i) {
      vector<double> pred(num_unique_event_times, 0);
      vector<double> pred_oob(num_unique_event_times, 0);
      double num_oob_trees = 0;

      for (int j = 0; j < trees.size(); ++j) {
        SurvivalTree* tree = dynamic_cast<SurvivalTree*>(trees[j].get());
        //cout << "OOB indices: ";
        //printVector(OOB_indices[j]);

        // in bag prediction
        vector<double> tree_pred = tree->getCHF()[tree->getPredictionNodeIDs()[i]];
        sum_vectors(pred, tree_pred);
        //cout << "Line 268: Node ID: " << tree->getPredictionNodeIDs()[i] << endl;
        //cout << "Line 269: tree_pred length: ";
        //cout << tree_pred.size() << endl;

        // out of bag prediction
        if (OOB_indices[j][i]) {
          sum_vectors(pred_oob, tree_pred);
          num_oob_trees += 1;
          //cout << "Line 274: Tree prediction ";
          //cout << tree_pred.size() << endl;
        }
      }

      for (int k = 0; k < num_unique_event_times; ++k) {
        pred[k] = pred[k]/as<double>(JFForest["num.trees"]);
        //cout << "Line 282:" << num_oob_trees << endl;
        if (num_oob_trees != 0) {
          pred_oob[k] = pred_oob[k]/num_oob_trees;
        }
      }

      NumericVector rpred(pred.begin(), pred.end());
      predictions.row(i) = rpred;
      NumericVector rpred_oob(pred_oob.begin(), pred_oob.end());
      predictions_oob.row(i) = rpred_oob;
      
    }
    JFForest["predictions"] = predictions;
    JFForest["oob.predictions"] = predictions_oob;

    // remove OOB indices from the forest to save memory
    forest->cleanUp();
  }
  
  if (type == "Multi-state") {

  }
}

// [[Rcpp::export]]
NumericMatrix JFCppForestPredict(const List& JFForest, DataFrame df, NumericVector feature_indices,
                                 LogicalVector categorical, NumericVector unique) {
  // convert the input to C++ vectors
  vector<size_t> response_indices_cpp;  // need an empty vector for the response indices
  vector<size_t> feature_indices_cpp = as<vector<size_t>>(feature_indices);
  vector<bool> categorical_cpp = as<vector<bool>>(categorical);
  vector<size_t> unique_cpp = as<vector<size_t>>(unique);

  // convert the new data to a suitable Data object
  Data new_data = Data(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);

  string type = as<string>(JFForest["tree.type"]);
  if (type == "Regression") {

  }

  if (type == "Classification") {
    
  }

  if (type == "Survival") {
    SurvivalForest* forest = ((XPtr<SurvivalForest>) JFForest["Forest"]).get();
    NumericMatrix predictions(new_data.getNumberOfObs(), forest->getEventTimes().size());
    for (int i = 0; i < new_data.getNumberOfObs(); ++i) {
      vector<double> pred = forest->predict(new_data.get_x_row(i));
      copy(pred.begin(), pred.end(), predictions.row(i).begin());
    }
    return(predictions);
  }

  if (type == "Multi-state") {
    
  }
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
  cout << "Number of features: " << test.getNumberOfFeatures() << endl;
  cout << "Number of observations: " << test.getNumberOfObs() << endl;
  if (response_indices_cpp.size() > 0) {
    cout << endl << "The response vector is: ";
    for (int i = 0; i < test.getNumberOfObs(); ++i) {
      cout << test.get_y(i, 0) << ", ";
    }
  }
  cout << endl << "The number of unique values of the features are: ";
  for (int i  = 0; i < test.getUniqueValues().size(); ++i) {
    cout << test.getUniqueValues()[i] << ", ";
  }
  cout << endl << "The categorical indicators are: ";
  for (int i = 0; i < test.getCategorical().size(); ++i) {
    cout << test.getCategorical()[i] << ", ";
  }
  cout << endl << "The variable names are: ";
  for (string name : test.getFeatureNames()) {
    cout << name << ", ";
  }
  cout << endl << "The feature values are: " << endl;
  for (int i = 0; i < test.getNumberOfObs(); ++i) {
    printVector(test.get_x_row(i));
  }
}

// classification trees and forests
//-------------------------------------------------------------------------------------

// regression trees and forests
//-------------------------------------------------------------------------------------

// survival trees and forests
//-------------------------------------------------------------------------------------

// below is a temporary test function to make sure all methods work

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
  vector<size_t> ind = as<vector<size_t>>(df[response_indices[1]]);
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
  
  // create the SurvivalTree
  SurvivalTree tree = SurvivalTree(unique_event_times, subset_indices_cpp);
  shared_ptr<Data> data = make_shared<Data>(df, response_indices_cpp, feature_indices_cpp, categorical_cpp, unique_cpp);
  tree.initialise(data, mtry, min_node_size, nsplits, 2025); // just set seed to something

  // grow the SurvivalTree
  // the bug happens after
  tree.grow();
  cout << "Done!" << endl;

  
  // write out predictions (testing)
  cout << "The terminal node values are:" << endl;
  vector<vector<double>> predictions = tree.getCHF();
  for (vector<double> vec : predictions) {
    for (int i = 0; i < vec.size(); ++i) {
      cout << vec[i] << ", ";
    }
    cout << endl;
  }

  // compute predictions (testing)
  cout << "The predicted values for the data are: " << endl;
  for (int i = 0; i < (*(tree.getData())).getNumberOfObs(); ++i) {
    vector<double> pred = get<vector<double>>(tree.predict((*(tree.getData())).get_x_row(i)));
    for (double h : pred) {
      cout << h << ", ";
    }
    cout << endl;
  }
    
}
#include "Data.h"
#include "TreeSurvival.h"

using namespace std;
using namespace Rcpp;

// all code below is strictly for testing and do not go into the final library
//-------------------------------------------------------------------------------------

// [[Rcpp::export]]
List testList() {
  List result;
  vector<double> test {1, 2, 3.0};
  result.push_back(2, "number");
  result.push_back(test, "vector<double>");
  return result;
}

// [[Rcpp::export]]
int extract_column(DataFrame data, size_t index) {
  NumericVector col = data[index];
  cout << "The given vector is: " << col << endl;

  // the line below is not an issue
  vector<double> output(col.begin(), col.end());

  return output.size();
}

// [[Rcpp::export]]
List rcpp_hello_world() {

    CharacterVector x = CharacterVector::create( "foo", "bar" )  ;
    NumericVector y   = NumericVector::create( 0.0, 1.0 ) ;
    List z            = List::create( x, y ) ;

    return z ;
}


// [[Rcpp::export]]
double square_cpp(double x) {
    return x * x;
}

// [[Rcpp::export]]
NumericMatrix timesTwoMatrix(NumericMatrix mat) {
  int nrow = mat.nrow();
  int ncol = mat.ncol();

  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      mat(i, j) = 2 * mat(i, j);
    }
  }

  return mat;
}

// [[Rcpp::export]]
void sampler(int n, int k) {
  vector<size_t> vec(n);
  vector<size_t> out;
  for (int i = 0; i < n; ++i) {
    vec[i] = i;
  }
  sample(vec.begin(), vec.end(), back_inserter(out), k,
                std::mt19937 {random_device{}()});
  
  for (int i = 0; i < k; ++i) {
    cout << out[i] << ", ";
  }
  cout << endl;
}

// [[Rcpp::export]]
void silly_sampler(int k) {
  vector<double> vec {1, 0.13, 0.145, -0.16, 0.15};
  vector<double> out;
  sample(vec.begin(), vec.end(), back_inserter(out), k,
                std::mt19937 {random_device{}()});
  for (int i = 0; i < k; ++i) {
    cout << out[i] << ", ";
  }
  cout << endl;
}

// [[Rcpp::export]]
void sampler_wrapper(int n, int k, bool wr) {
  mt19937 generator = mt19937 {random_device{}()};
  vector<size_t> output = sampleIndices(n, k, wr, generator);
  for (size_t o: output) {
    cout << o << ", ";
  }
  cout << endl;
}
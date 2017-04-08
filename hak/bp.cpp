// TO TEST:
// g++ bp.cpp -std=c++11 -g && echo -e "run\nbt" | gdb ./a.out

#include <iostream>
#include <random>
#include <cstdlib>
#include <array>
#include <vector>
#include <tuple>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <initializer_list>

using namespace std;

int m = 2;
int c = 7;

typedef vector<double> Tensor1;
typedef vector<vector<double>> Matrix;
typedef vector<vector<double>> Tensor2;

default_random_engine re;

// Verbosity
int V = 30;




inline void error(const string& s) {
  throw runtime_error(s);
}


double y(double x) {
  double noise = (rand() % 100 - 50)*0.01;
  return m*x+c+noise;
}

void init(vector<double> & t) {
  normal_distribution<double> dist(0,1);

  for (int i = 0; i < t.size(); i ++) {
    t[i] = dist(re);
  }
}

Matrix init_matrix(int rows, int cols) {
  Matrix m(rows, vector<double>(cols,0));
  normal_distribution<double> dist(0,1);

  for (int i = 0; i < rows; i ++) {
    for (int j = 0; j < cols; j ++) {
      m[i][j] = dist(re);
    }
  }
  return m;
}

string matrix_string(Matrix m) {
  stringstream ss;
  for (int i = 0; i < m.size(); i ++) {
    for (int j = 0; j < m[i].size(); j ++) {
      ss << m[i][j]  << " ";
    }
    ss << endl;
  }
  return ss.str();
}

void print_matrix(Matrix m) {
  for (int i = 0; i < m.size(); i ++) {
    for (int j = 0; j < m[i].size(); j ++) {
      cout << m[i][j]  << " ";
    }
    cout << endl;
  }
}

Matrix scale(double t, Matrix a) {
  Matrix result = init_matrix(a.size(), a[0].size());
  for (int i = 0 ; i < result.size(); i ++) {
    for (int j = 0; j < result[i].size(); j ++) {
      result[i][j] = a[i][j] *t;
    }
  }
  return result;
}

Matrix transpose(Matrix a) {
  Matrix result = init_matrix(a[0].size(), a.size());
  for (int i = 0 ; i < result.size(); i ++) {
    for (int j = 0; j < result[i].size(); j ++) {
      result[i][j] = a[j][i];
    }
  }
  return result;
}

Matrix add(Matrix a, Matrix b) {
  if (a.size() != b.size()) error("a and b are not compatible");
  if (a[0].size() != b[0].size()) error("a and b are not compatible");
  Matrix result = init_matrix(a.size(), a[0].size());
  for (int i = 0 ; i < result.size(); i ++) {
    for (int j = 0; j < result[i].size(); j ++) {
      result[i][j] = a[i][j] + b[i][j];
    }
  }
  return result;
}

Matrix termwise_product(Matrix a, Matrix b) {
  if (a.size() != b.size()) error("a and b are not compatible");
  if (a[0].size() != b[0].size()) error("a and b are not compatible");
  Matrix result = init_matrix(a.size(), a[0].size());
  for (int i = 0 ; i < result.size(); i ++) {
    for (int j = 0; j < result[i].size(); j ++) {
      result[i][j] = a[i][j] * b[i][j];
    }
  }
  return result;
}

Matrix multiply(Matrix a, Matrix b) {
  Matrix result = init_matrix(a.size(), b[0].size());
  if (a[0].size() != b.size()) error("a and b are not compatible");

  for (int i = 0 ; i < result.size(); i ++) {
    for (int j = 0; j < result[i].size(); j ++) {
      double val = 0;
      // a row_i \dot b col_j
      for (int k = 0; k < a[i].size(); k ++) {
        val += a[i][k] * b[k][j];
      }
      result[i][j] = val;
    }
  }
  return result;
}

Matrix norm(Matrix v) {
  if (v.size() != 1 && v[0].size() != 1) error("a is not a vector");
  bool row = v.size() == 1;
  int len = row ? v[0].size() : v.size();
  double magnitude = 0;
  for (int i = 0 ; i < len; i ++) {
    double val = row ? v[0][i] : v[i][0];
    magnitude += val*val;
  }
  magnitude = sqrt(magnitude);
  Matrix result = init_matrix(v.size(), v[0].size());
  for (int i = 0 ; i < len; i ++) {
    if (row)
      result[0][i] = v[0][i] / magnitude;
    else
      result[i][0] = v[i][0] / magnitude;
  }
  return result;
}

double logit(double x) {
  return 1 / (1 + exp(-x));
}
double logit_dx(double x) {
  return logit(x)*(1-logit(x));
}

Matrix apply_logit(Matrix a) {
  Matrix result = init_matrix(a.size(), a[0].size());
  for (int i = 0 ; i < result.size(); i ++) {
    for (int j = 0; j < result[i].size(); j ++) {
      result[i][j] = logit(a[i][j]);
    }
  }
  return result;
}

Matrix apply_logit_dx(Matrix a) {
  Matrix result = init_matrix(a.size(), a[0].size());
  for (int i = 0 ; i < result.size(); i ++) {
    for (int j = 0; j < result[i].size(); j ++) {
      result[i][j] = logit_dx(a[i][j]);
    }
  }
  return result;
}

tuple<vector<Matrix>, vector<Matrix>> feed_forward(Matrix inp_x, vector<Matrix> weights, vector<Matrix> biases) {
  Matrix vals = inp_x;

  vector<Matrix> inputs;
  vector<Matrix> activations;

  for (int i = 0; i < weights.size(); i ++) {
    if (V > 10) cout << "feed_forward " << i << endl;
    Matrix W = weights[i];
    Matrix B = biases[i];
    Matrix Z = add(multiply(vals, W), B);
    Matrix A = apply_logit(Z);
    if (V > 10) cout << "W = " << matrix_string(W);
    if (V > 10) cout << "B = " << matrix_string(B);
    if (V > 10) cout << "Z = " << matrix_string(Z);
    if (V > 10) cout << "A = " << matrix_string(A);
    inputs.push_back(Z);
    activations.push_back(A);
    vals = A;
  }
  return tuple<vector<Matrix>, vector<Matrix>>(inputs, activations);
}

void back_propagate(vector<Matrix> & weights, vector<Matrix> & biases,
      Matrix inp_x, vector<Matrix> inputs, vector<Matrix> activations,
      Matrix y, double learning_rate) {
  int n = weights.size();
  Matrix loss = add(y, scale(-1.0, activations[n-1]));
  // loss = (y-a_n])^2
  // error = dLoss/da_n * logit_dx(inputs_n)
  Matrix error = termwise_product(
        scale(2.0, add(activations[n-1], scale(-1.0, y))),
        apply_logit_dx(inputs[n-1]));
  // Error is a row vector.
  if (V > 10) cout << "error0 = " << matrix_string(error);


  for (int i = weights.size()-1; i >= 0; i --) {
    biases[i] = add(biases[i], scale(-learning_rate, error));

    Matrix prev_activations = i > 0 ? activations[i-1] : inp_x;
    if (V > 10) cout << "back_propagate " << i << endl;
    if (V > 10) cout << "error = " << matrix_string(error);
    if (V > 10) cout << "prev_activations = " << matrix_string(prev_activations);
    if (V > 10) cout << "activations_dx = " << matrix_string(multiply(transpose(prev_activations), error));
    weights[i] = add(weights[i], scale(-learning_rate, multiply(transpose(prev_activations), error)));

    if (i != 0) {
      // calc error for i-1
      error = termwise_product(multiply(error, transpose(weights[i])), apply_logit_dx(inputs[i-1]));
    }
  }
}

Matrix row_vec(initializer_list<double> vals) {
  Matrix vec = init_matrix(1,vals.size());
  int i = 0;
  for (double v : vals) {
    vec[0][i++] = v;
  }
  return vec;
}

Matrix calc_y(Matrix x) {
  double x1 = x[0][0];
  double x2 = x[0][1];
  double x3 = x[0][2];
  Matrix y_dir = row_vec({x1+x2, x2+x3});
  // cout << "y_dir = " << matrix_string(y_dir);
  return norm(y_dir);
}

int run_stuff() {
  cout << "foo" << endl;
  // f -> (x1,x2,x3) -> norm(x1+x2, x2+x3)
  // input layer: three number, x


  // hid=10 -> error @100k = 0.0562022 -0.0581912, -0.197466 0.221695, 0.0871238 -0.0180035
  // hid=30 -> error @100k = -0.0322616 0.0764045, 0.0306207 0.0215664, -0.0015342 0.0207133
  // H1=10, H2=5 -> error @100k = -0.0322616 0.0764045, 0.0306207 0.0215664, -0.0015342 0.0207133
  int H1 = 10;
  int H2 = 5;

  // hidden_layer_1, H1 nodes fully connected to input layer.
  Matrix w1 = init_matrix(3,H1);
  Matrix b1 = init_matrix(1,H1);
  cout << "w1 = " << endl;
  print_matrix(w1);


  // hidden_layer_2, H2 nodes fully connected to input layer.
  Matrix w2 = init_matrix(H1,H2);
  Matrix b2 = init_matrix(1,H2);
  cout << "w2 = " << endl;
  print_matrix(w2);


  // output layer, 2 numbers, 5 nodes fully connected to w1.
  Matrix wOut = init_matrix(H2,2);
  Matrix bOut = init_matrix(1,2);
  cout << "wOut = " << endl;
  print_matrix(wOut);

  vector<Matrix> weights = {w1, w2, wOut};
  vector<Matrix> biases = {b1, b2, bOut};

  // Matrix x = row_vec({1, 2, 3});
  // Matrix y = row_vec({(1+2)/sqrt(34), (2+3)/sqrt(34)});

  // auto inputs_activations = feed_forward(x, weights, biases);
  // Matrix pred_y = get<1>(inputs_activations)[1];
  // cout << "x = " << matrix_string(x);
  // cout << "y = " << matrix_string(y);
  // cout << "pred_y = " << matrix_string(pred_y);

  // back_propagate(weights, biases, x, get<0>(inputs_activations), get<1>(inputs_activations), y, 0.1);
  // cout << "bar" << endl;
  // cout << "w1 = " << endl;
  // print_matrix(w1);
  // cout << "w2 = " << endl;
  // print_matrix(w2);
  // cout << "AAAA" << endl << endl;
  // Matrix pred_y2 = get<1>(feed_forward(x, weights, biases))[1];
  // cout << "y = " << matrix_string(y);
  // cout << "pred_y 1 = " << matrix_string(pred_y);
  // cout << "pred_y 2 = " << matrix_string(pred_y2);

  uniform_real_distribution<double> x_dist(0,1);
  for (int i = 0; i <= 100000; i ++) {

    Matrix x = row_vec({x_dist(re), x_dist(re), x_dist(re)});
    Matrix y = calc_y(x);
    // cout << "x = " << matrix_string(x);
    // cout << "y = " << matrix_string(y);

    auto inputs_activations = feed_forward(x, weights, biases);
    Matrix pred_y = get<1>(inputs_activations)[1];
    cout << "pred_y = " << matrix_string(pred_y);
    Matrix diff = add(y, scale(-1, pred_y));
    if (i % 1000 == 0) cout << i << " diff = " << matrix_string(diff);
    back_propagate(weights, biases, x, get<0>(inputs_activations), get<1>(inputs_activations), y, 0.001);
  }
}

int main() {
  // try {
    run_stuff();
  // } catch (runtime_error e) {
      // cout << "Runtime error: " << e.what();
  // }
}
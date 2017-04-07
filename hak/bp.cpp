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
  if (v[0].size() != 1) error("a is not a column vector");
  double magnitude = 0;
  for (int i = 0 ; i < v.size(); i ++) {
    magnitude += v[i][0]*v[i][0];
  }
  magnitude = sqrt(magnitude);
  Matrix result = init_matrix(v.size(), v[0].size());
  for (int i = 0 ; i < v.size(); i ++) {
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
    Matrix W = weights[i];
    Matrix B = biases[i];
    Matrix Z = add(multiply(vals, W), B);
    Matrix A = apply_logit(Z);
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
  for (int i = weights.size()-1; i >= 0; i --) {
    biases[i] = add(biases[i], scale(-learning_rate, error));

    Matrix prev_activations = i > 0 ? activations[i-1] : inp_x;
    cout << "i = " << i << endl;
    cout << "error = " << matrix_string(error);
    cout << "prev_activations = " << matrix_string(prev_activations);
    weights[i] = add(weights[i], scale(-learning_rate, multiply(error, transpose(prev_activations))));

    if (i != 0) {
      // calc error for i-1
      error = termwise_product(multiply(transpose(weights[i]), error), apply_logit_dx(inputs[i-1]));
    }
    weights[i] = add(weights[i], scale(-learning_rate, norm(error)));
  }
}

Matrix col_vec(initializer_list<double> vals) {
  Matrix vec = init_matrix(1,vals.size());
  int i = 0;
  for (double v : vals) {
    vec[0][i++] = v;
  }
  return vec;
}

int run_stuff() {
  cout << "foo" << endl;
  // f -> (x1,x2,x3) -> (x1+x2, x2+x3)
  // input layer: three number, x

  // hidden_layer_1, 10 nodes fully connected to input layer.
  Matrix w1 = init_matrix(3,10);
  Matrix b1 = init_matrix(1,10);
  cout << "w1 = " << endl;
  print_matrix(w1);

  // output layer, 2 numbers, 5 nodes fully connected to w1.
  Matrix w2 = init_matrix(10,2);
  Matrix b2 = init_matrix(1,2);
  cout << "w2 = " << endl;
  print_matrix(w2);

  vector<Matrix> weights = {w1, w2};
  vector<Matrix> biases = {b1, b2};

  Matrix x = col_vec({1, 2, 3});
  Matrix y = col_vec({1+2, 2+3});

  auto inputs_activations = feed_forward(x, weights, biases);
  Matrix pred_y = get<1>(inputs_activations)[1];
  cout << "x = " << matrix_string(x);
  cout << "y = " << matrix_string(y);
  cout << "pred_y = " << matrix_string(pred_y);

  back_propagate(weights, biases, x, get<0>(inputs_activations), get<1>(inputs_activations), y, 0.01);
  cout << "w1 = " << endl;
  print_matrix(w1);
  cout << "w2 = " << endl;
  print_matrix(w2);

  cout << "bar" << endl;
}

int main() {
  // try {
    run_stuff();
  // } catch (runtime_error e) {
      // cout << "Runtime error: " << e.what();
  // }
}
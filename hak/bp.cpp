#include <iostream>
#include <random>
#include <cstdlib>
#include <array>
#include <vector>
#include <cmath>

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

void print_matrix(Matrix m) {
  for (int i = 0; i < m.size(); i ++) {
    for (int j = 0; j < m[i].size(); j ++) {
      cout << m[i][j]  << " ";
    }
    cout << endl;
  }
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

Matrix apply_layers(vector<Matrix> layers) {
  Matrix vals = layers[0];
  auto it = std::begin(layers);
  it ++;
  while (it != std::end(layers)) {
    Matrix inputs = multiply(vals, *it);
    // apply biases
    // apply activation function
    Matrix activations = inputs;
    vals = activations;
    it ++;
  }
  return vals;
}

double run_nn(double x, vector<Matrix> layers) {
  layers[0][0][0] = x;
  return apply_layers(layers)[0][0];
}

int run_stuff() {
  cout << "foo" << endl;

  // input layer: just one number, x
  Matrix inp_x = init_matrix(1,1);

  // hidden_layer_1, 10 nodes fully connected to input layer.
  Matrix h1 = init_matrix(1,10);
  Matrix b1 = init_matrix(1,10); // biases
  cout << "h1 = " << endl;
  print_matrix(h1);

  // hidden_layer_2, 5 nodes fully connected to h1.
  Matrix h2 = init_matrix(10,5);
  // biases, add this onto the vector after multiply by h2, but before applying activation function.
  Matrix b2 = init_matrix(1,5);
  cout << "h2 = " << endl;
  print_matrix(h2);

  // output_layer, 1 node fully connected.
  Matrix out = init_matrix(5,1);
  cout << "out = " << endl;
  print_matrix(out);

  vector<Matrix> layers = {inp_x, h1, h2, out};

  double in_x = 1;
  double out_y = run_nn(in_x, layers);
  cout << "y(1) = " << y(in_x) << endl;
  cout << "run_nn(1) = " << out_y << endl;

  double loss = pow(y(in_x) - out_y, 2);
  cout << "loss = " << loss << endl;

  // optimizer = tf.train.GradientDescentOptimizer(0.01)
  // train = optimizer.minimize(loss)

  // loss

  cout << "y(2) = " << y(2) << endl;
  cout << "Phi(2) = " << run_nn(2, layers) << endl;

  cout << "y(3) = " << y(3) << endl;
  cout << "Phi(3) = " << run_nn(3, layers) << endl;


  cout << "bar" << endl;
}

int main() {
  try {
    run_stuff();
  } catch (runtime_error e) {
      cout << "Runtime error: " << e.what();
  }
}
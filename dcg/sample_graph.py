import dcg
import random

gp = dcg.DifferentiableComputationGraph()

x_inp = gp.add_op([], dcg.Placeholder(5), 'x_in')
m_var = gp.add_op([], dcg.Variable(3), 'm')
c_var = gp.add_op([], dcg.Variable(12), 'c')
y_out = gp.add_op([gp.add_op([x_inp,m_var], dcg.Multiply()),c_var], dcg.Plus(), 'y_out')

real_m = 8
real_c = -4

for i in range(0,1000):
  x = (random.random()-0.5)*10
  x_inp.function.placeholder_value = x
  y = gp.evaluate(y_out)
  real_y = real_m * x + real_c
  loss = real_y - y
  gp.backprop(y_out, loss, 0.01)
  print ("x = %3d, y = %3d, m_var = %f, c_var = %f, loss = %f" %
          (x, y, m_var.function.variable_value, c_var.function.variable_value, loss))

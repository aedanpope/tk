import dcg

lin_graph = dcg.DifferentiableComputationGraph()

x_inp = lin_graph.add_op([], dcg.Placeholder(5), 'x_in')

m_var = lin_graph.add_op([], dcg.Variable(3), 'm')
c_var = lin_graph.add_op([], dcg.Variable(12), 'c')
y_out = lin_graph.add_op([lin_graph.add_op([x_inp,m_var], dcg.Multiply()),c_var], dcg.Plus(), 'y_out')

print "y_out = " + str(lin_graph.evaluate(y_out))
x_inp.function.placeholder_value = -5
print "x = -5"
print "y_out = " + str(lin_graph.evaluate(y_out))

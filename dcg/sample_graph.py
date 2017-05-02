import dcg

lin_graph = dcg.CGraph()

x_inp = lin_graph.add_op([], dcg.Placeholder(5))

m_var = lin_graph.add_op([], dcg.Variable(3))
c_var = lin_graph.add_op([], dcg.Variable(12))
z_var = lin_graph.add_op([lin_graph.add_op([x_inp,m_var], dcg.Multiply()),c_var], dcg.Plus())

print "z_var = " + str(lin_graph.evaluate(z_var))
x_inp.function.placeholder_value = -5
print "x = -5"
print "z_var = " + str(lin_graph.evaluate(z_var))

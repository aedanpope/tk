

# E.g. Matrix multiplication.

# Functons need to define eval(), and some jacobian shit..

# Constant functions.

class Placeholder:
  placeholder_value = None
  def __init__(self, value):
    self.placeholder_value = value

  def eval(self, input_vals):
    print "eval placeholder " + str(self.placeholder_value)
    return self.placeholder_value

class Variable:
  variable_value = None
  def __init__(self, value):
    self.variable_value = value

  def eval(self, input_vals):
    print "eval variable " + str(self.variable_value)
    return self.variable_value


# Aggregate functions
class Plus:
  def eval(self, input_vals):
    print "eval sum " + str(input_vals) + " = " + str(sum(input_vals))
    return sum(input_vals)


class Multiply:
  def eval(self, input_vals):
    if len(input_vals) < 1:
      raise Exception("can't multiply 0 input vals")
    product = input_vals[0];
    for v in input_vals[1:]:
      product *= v
    print "eval multiply " + str(input_vals) + " = " + str(product)
    return product


class Operation:
  inputs = None
  consumers = None
  function = None
  value = None

  def __init__(self):
    self.inputs = []
    self.consumers = []


class CGraph:
  all_ops = None

  def __init__(self):
    self.all_ops = []

  def add_op(self, inputs, function):
    op = Operation()
    op.inputs = inputs
    op.function = function
    for in_op in inputs:
      in_op.consumers.append(op)
    self.all_ops.append(op)
    return op

  def rec_evaluate(self, op):
    input_vals = []
    for inp_op in op.inputs:
      if inp_op.value is None:
        self.rec_evaluate(inp_op)
      input_vals.append(inp_op.value)
    op.value = op.function.eval(input_vals)

  def evaluate(self, placeholder_op):
    # should assert that placeholder_op is of type Placeholder
    # Reset the graph.
    for op in self.all_ops:
      op.value = None
    self.rec_evaluate(placeholder_op)
    return placeholder_op.value



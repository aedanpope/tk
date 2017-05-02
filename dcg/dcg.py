

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
  identifier = None
  inputs = None
  consumers = None
  function = None
  value = None
  gradient = None

  def __init__(self, identifier):
    self.identifier = identifier
    self.inputs = []
    self.consumers = []


class DifferentiableComputationGraph:
  all_ops = None
  next_identifier = 0

  def __init__(self):
    self.all_ops = []

  def add_op(self, inputs, function, identifier=None):
    if identifier is None: identifier = self.next_identifier ++
    op = Operation(identifier)
    op.inputs = inputs
    op.function = function
    for in_op in inputs:
      in_op.consumers.append(op)
    self.all_ops.append(op)
    return op

  def recursive_evaluate(self, op):
    if op.value is not None: return op.value

    input_vals = []
    for inp_op in op.inputs:
      input_vals.append(self.recursive_evaluate(inp_op))
    op.value = op.function.eval(input_vals)

  def evaluate(self, output_op):
    # should assert that output_op is of type Placeholder
    # Reset the graph.
    for op in self.all_ops:
      op.value = None
    self.recursive_evaluate(output_op)
    return output_op.value

  def compute_gradient(self, op):
    if op.gradient is not None:
      return op.gradient

    op.gradient = 0

    for consumer in op.consumers:
      self.compute_gradient(consumer)
      # op.gradient += consumer.



  def backprop(self, output_op, output_grad):
    """ Populates gradients field in all the ops in the graph.
    """
    # Reset
    for op in self.all_ops:
      op.gradient = None

    # Assign gradient on output.
    output_op.grad = output_grad

    # Compute all gradients.
    for op in self.all_ops:
      self.compute_gradient(op)





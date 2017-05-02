

# E.g. Matrix multiplication.

# Functons need to define inner_eval(), and inner_derivative()

class Function:
  def eval(self, input_vals):
    val = self.inner_eval(input_vals)
    return val

  def derivative(self, input_vals, wrt_input_index):
    if not input_vals: raise Exception("Can't compute derivative with no inputs")
    val = self.inner_derivative(input_vals, wrt_input_index)
    return val

  def __str__(self):
    return self.__class__.__name__


# Constant functions.
class Placeholder(Function):
  placeholder_value = None
  def __init__(self, value):
    self.placeholder_value = value

  def inner_eval(self, input_vals):
    return self.placeholder_value


class Variable(Function):
  variable_value = None
  def __init__(self, value):
    self.variable_value = value

  def inner_eval(self, input_vals):
    return self.variable_value


# Aggregate functions
class Plus(Function):
  def inner_eval(self, input_vals):
    return sum(input_vals)

  def inner_derivative(self, input_vals, wrt_input_index):
    return 1


class Multiply(Function):
  def inner_eval(self, input_vals):
    if len(input_vals) < 1:
      raise Exception("can't multiply no input vals")
    product = input_vals[0];
    for v in input_vals[1:]:
      product *= v
    return product

  def inner_derivative(self, input_vals, wrt_input_index):
    input_vals[wrt_input_index] = 1
    return self.inner_eval(input_vals)


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

  def __str__(self):
    return ("Operation {" +
        str(self.identifier) + ", " +
        "function: " + str(self.function) + ", " +
        "value: " + str(self.value) + ", " +
        "gradient: " + str(self.gradient) + "}")

  def derivative(self, input_op):
    """ Given that the gradient of this op is know, returns
        The gradient of the input_op with respect to this op.
        input_op must be in self.inputs
    """
    if not input_op in self.inputs: raise Exception("input_op not in self.inputs")

    return self.gradient * self.function.derivative(
        [inp.value for inp in self.inputs], self.inputs.index(input_op))


class DifferentiableComputationGraph:
  all_ops = None
  next_identifier = 0

  def __init__(self):
    self.all_ops = []

  def add_op(self, inputs, function, identifier=None):
    if identifier is None:
      identifier = "e" + str(self.next_identifier)
      self.next_identifier += 1
    op = Operation(identifier)
    op.inputs = inputs
    op.function = function
    for in_op in inputs:
      in_op.consumers.append(op)
    self.all_ops.append(op)
    return op

  def _evaluate_recursive(self, op):
    if op.value is not None: return op.value

    input_vals = []
    for inp_op in op.inputs:
      input_vals.append(self._evaluate_recursive(inp_op))
    op.value = op.function.eval(input_vals)
    return op.value

  def evaluate(self, output_op):
    # should assert that output_op is of type Placeholder
    # Reset the graph.
    for op in self.all_ops:
      op.value = None
    self._evaluate_recursive(output_op)
    return output_op.value

  def _compute_gradient_recursive(self, op):
    if op.gradient is not None:
      return op.gradient

    op.gradient = 0

    for consumer in op.consumers:
      self._compute_gradient_recursive(consumer)
      op.gradient += consumer.derivative(op)

    return op.gradient

  def compute_gradients(self, output_op, output_grad):
    """ Populates gradients field in all the ops in the graph.
    """
    for op in self.all_ops:
      op.gradient = None

    # Assign gradient on output.
    output_op.gradient = output_grad

    # Compute all gradients.
    for op in self.all_ops:
      # No loops because graph is a DAG
      self._compute_gradient_recursive(op)

  def backprop(self, output_op, output_grad, learning_rate):
    self.compute_gradients(output_op, output_grad)
    for op in self.all_ops:
      if isinstance(op.function, Variable):
        op.function.variable_value += (op.gradient * learning_rate)





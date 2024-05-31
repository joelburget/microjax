import numpy as np
import unittest
import math
from dataclasses import dataclass


def toposort(end_node):
    child_counts = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(node.parents)

    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in node.parents:
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1


trace_id = 0


def trace(start_node, fun, x):
    """Build a computation graph."""
    global trace_id
    trace_id += 1
    try:
        # Wrap 'x' in a box.
        start_box = Box(x, trace_id, start_node)

        # Apply fun() to boxed value. This will carry the value throughout the
        # comutation as well as the box.
        end_box = fun(start_box)

        if isinstance(end_box, Box) and end_box._trace_id == start_box._trace_id:
            # Extract final value (== fun(x)) and its node in the computation
            # graph.
            return end_box._value, end_box._node
        else:
            # Output seems independent of input
            return end_box, None
    finally:
        trace_id -= 1


class Node(object):
    """A node in a computation graph."""

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        """

        Args:
          value: output of fun(*args, **kwargs)
          fun: wrapped numpy that was applied.
          args: all (unboxed) positional arguments.
          kwargs: dict of additional keyword args.
          parent_argnums: integers corresponding to positional indices of boxed
            values.
          parents: Node instances corresponding to parent_argnums.
        """
        self.parents = parents
        self.recipe = (fun, value, args, kwargs, parent_argnums)

    def initialize_root(self):
        self.parents = []
        self.recipe = (lambda x: x, None, (), {}, [])

    @classmethod
    def new_root(cls, *args, **kwargs):
        root = cls.__new__(cls)
        root.initialize_root(*args, **kwargs)
        return root


def primitive(f_raw):
    """Wraps a function so that its gradient (vjp) can be specified and its
    invocation can be recorded."""

    def f_wrapped(*args, **kwargs):
        # Fetch boxed arguments with largest trace_id.  This ensures that the
        # computational graph being constructed only consists of other nodes
        # from the same call to trace().
        boxed_args, trace_id = find_top_boxed_args(args)
        if boxed_args:
            # Replace some elements of args with corresponding unboxed values.
            argvals = list(args)
            for argnum, box in boxed_args:
                argvals[argnum] = box._value
            # Get nodes for each boxed argument.
            parents = tuple(box._node for _, box in boxed_args)

            # Get argument indices for each boxed argument.
            argnums = tuple(argnum for argnum, _ in boxed_args)

            # Calculate result of applying original numpy function.
            #
            # Note that we use a recursive call here in order to also augment
            # outer calls to trace() with lower trace_ids.
            ans = f_wrapped(*argvals, **kwargs)

            # Create a new node
            node = Node(ans, f_wrapped, argvals, kwargs, argnums, parents)
            return Box(ans, trace_id, node)
        else:
            return f_raw(*args, **kwargs)

    return f_wrapped


def find_top_boxed_args(args):
    """Finds boxed arguments with largest trace_id.

    Equivalent to finding the largest trace_id of any argument, keeping args
    with the same, and dropping the remainder.

    Args:
      args: Arguments to function wrapped by primitive().

    Returns:
      top_boxes: List of (index, boxed argument). Arguments have same, largest
        trace_id.
      top_trace_id: trace_id of all elements in top_boxes.
    """
    top_trace_id = -1
    top_boxes = []
    for argnum, arg in enumerate(args):
        if isinstance(arg, Box):
            if arg._trace_id > top_trace_id:
                top_boxes = [(argnum, arg)]
                top_trace_id = arg._trace_id
            elif arg._trace_id == top_trace_id:
                top_boxes.append((argnum, arg))
    return top_boxes, top_trace_id


def make_vjp(fun, x):
    """Make function for vector-Jacobian product.

    Args:
      fun: single-arg function. Jacobian derived from this.
      x: ndarray. Point to differentiate about.

    Returns:
      vjp: single-arg function. vector -> vector-Jacobian[fun, x] product.
      end_value: end_value = fun(start_node)

    """
    start_node = Node.new_root()
    end_value, end_node = trace(start_node, fun, x)

    if end_node is None:

        def vjp(g):
            return np.zeros_like(x)

    else:

        def vjp(g):
            return backward_pass(g, end_node)

    return vjp, end_value


def backward_pass(g, end_node):
    """Backpropagation.

    Traverse computation graph backwards in topological order from the end node.
    For each node, compute local gradient contribution and accumulate.
    """
    outgrads = {end_node: g}
    for node in toposort(end_node):
        outgrad = outgrads.pop(node)
        fun, value, args, kwargs, argnums = node.recipe
        for argnum, parent in zip(argnums, node.parents):
            # Lookup vector-Jacobian product (gradient) function for this
            # function/argument.
            vjp = primitive_vjps[fun][argnum]

            # Compute vector-Jacobian product (gradient) contribution due to
            # parent node's use in this function.
            parent_grad = vjp(outgrad, value, *args, **kwargs)

            # Save vector-Jacobian product (gradient) for upstream nodes.
            # Sum contributions with all others also using parent's output.
            outgrads[parent] = add_outgrads(outgrads.get(parent), parent_grad)
    return outgrad


def add_outgrads(prev_g, g):
    """Add gradient contributions together."""
    if prev_g is None:
        return g
    return prev_g + g


def grad(fun, argnum=0):
    """Constructs gradient function.

    Given a function fun(x), returns a function fun'(x) that returns the
    gradient of fun(x) wrt x.

    Args:
      fun: single-argument function. ndarray -> ndarray.
      argnum: integer. Index of argument to take derivative wrt.

    Returns:
      gradfun: function that takes same args as fun(), but returns the gradient
        wrt to fun()'s argnum-th argument.
    """

    def gradfun(*args, **kwargs):
        # Replace args[argnum] with x. Define a single-argument function to
        # compute derivative wrt.
        def unary_fun(x):
            new_args = list(args)
            new_args[argnum] = x
            return fun(*new_args, **kwargs)

        # Construct vector-Jacobian product
        vjp, ans = make_vjp(unary_fun, args[argnum])
        return vjp(np.ones_like(ans))

    return gradfun


class Anp:
    def __init__(self):
        self.negative = primitive(np.negative)
        self.exp = primitive(np.exp)
        self.log = primitive(np.log)
        self.tanh = primitive(np.tanh)
        self.sinh = primitive(np.sinh)
        self.cosh = primitive(np.cosh)
        self.multiply = primitive(np.multiply)
        self.add = primitive(np.add)
        self.subtract = primitive(np.subtract)
        self.abs = primitive(np.abs)
        self.true_divide = primitive(np.true_divide)
        self.divide = self.true_divide
        self.sign = primitive(np.sign)
        self.power = primitive(np.power)
        self.where = primitive(np.where)
        self.zeros = primitive(np.zeros)
        self.array = primitive(np.array)
        self.matmul = primitive(np.matmul)


anp = Anp()


primitive_vjps = {
    anp.negative: {0: lambda g, ans, x: -g},
    anp.multiply: {
        0: lambda g, ans, x, y: y * g,
        1: lambda g, ans, x, y: x * g,
    },
    anp.exp: {0: lambda g, ans, x: ans * g},
    anp.log: {0: lambda g, ans, x: g / x},
    anp.tanh: {0: lambda g, ans, x: g / anp.cosh(x) ** 2},
    anp.sinh: {0: lambda g, ans, x: g * anp.cosh(x)},
    anp.cosh: {0: lambda g, ans, x: g * anp.sinh(x)},
    anp.divide: {
        0: lambda g, ans, x, y: g / y,
        1: lambda g, ans, x, y: -g * x / y**2,
    },
    anp.add: {
        0: lambda g, ans, x, y: g,
        1: lambda g, ans, x, y: g,
    },
    anp.subtract: {
        0: lambda g, ans, x, y: g,
        1: lambda g, ans, x, y: -g,
    },
    anp.abs: {0: lambda g, ans, x: g * anp.sign(x)},
    anp.sign: {0: lambda g, ans, x: 0},
    anp.power: {
        0: lambda g, ans, x, y: g * y * x ** anp.where(y, y - 1, 1.0),
        1: lambda g, ans, x, y: g * anp.log(anp.where(x, x, 1.0)) * x**y,
    },
    anp.where: {
        0: lambda g, ans, c, x=None, y=None: anp.where(c, g, anp.zeros(g.shape)),
        1: lambda g, ans, c, x=None, y=None: anp.where(c, anp.zeros(g.shape), g),
    },
    anp.matmul: {
        0: lambda g, ans, x, y: g * y.T,
        1: lambda g, ans, x, y: x.T * g,
    },
}


def relu(x):
    return (x + abs(x)) / 2


class Box:
    """Box for np.ndarray.

    Anything you can do with an np.ndarray, you can do with a Box.
    """

    def __init__(self, value, trace_id, node):
        self._value = value
        self._node = node
        self._trace_id = trace_id

    # Tell Numpy to use this type instead of np.array. When is this useful?
    # When adding `np.array([1.0]) + Box(np.array([2.0]))`, we want the result
    # to be a Box, not an np.array of Box. This is bad:
    # >>> np.array([1.0]) + microjax.Box(np.array([1.0]), -1, None)
    # array([<microjax.Box object at 0x10c79c680>], dtype=object)
    # >>> microjax.Box(np.array([1.0]), -1, None) + np.array([1.0])
    # <microjax.Box object at 0x10c79cef0>
    __array_priority__ = 100.0

    @primitive
    def __getitem__(A, idx):
        return A[idx]

    # Constants w.r.t float data just pass though
    shape = property(lambda self: self._value.shape)
    ndim = property(lambda self: self._value.ndim)
    size = property(lambda self: self._value.size)
    dtype = property(lambda self: self._value.dtype)
    T = property(lambda self: anp.transpose(self))

    def __len__(self):
        return len(self._value)

    def __neg__(self):
        return anp.negative(self)

    def __add__(self, other):
        return anp.add(self, other)

    def __sub__(self, other):
        return anp.subtract(self, other)

    def __mul__(self, other):
        return anp.multiply(self, other)

    def __pow__(self, other):
        return anp.power(self, other)

    def __div__(self, other):
        return anp.divide(self, other)

    def __mod__(self, other):
        return anp.mod(self, other)

    def __truediv__(self, other):
        return anp.true_divide(self, other)

    def __matmul__(self, other):
        return anp.matmul(self, other)

    def __radd__(self, other):
        return anp.add(other, self)

    def __rsub__(self, other):
        return anp.subtract(other, self)

    def __rmul__(self, other):
        return anp.multiply(other, self)

    def __rpow__(self, other):
        return anp.power(other, self)

    def __rdiv__(self, other):
        return anp.divide(other, self)

    def __rmod__(self, other):
        return anp.mod(other, self)

    def __rtruediv__(self, other):
        return anp.true_divide(other, self)

    def __rmatmul__(self, other):
        return anp.matmul(other, self)

    def __abs__(self):
        return anp.abs(self)


def neuron_call(w, b, x, nonlin):
    act_mul = w @ x
    act = act_mul + b
    return relu(act) if nonlin else act


@dataclass
class Neuron:
    w: np.array
    b: np.array

    def __init__(self, nin, nonlin=True):
        self.w = np.random.uniform(-1, 1, (nin,))
        self.b = np.array([0.0])
        self.nonlin = nonlin

    def value_and_grad(self, x):
        return (
            self(x),
            (
                grad(neuron_call, 0)(self.w, self.b, x, self.nonlin),
                grad(neuron_call, 1)(self.w, self.b, x, self.nonlin),
            ),
        )

    def __call__(self, x):
        return neuron_call(self.w, self.b, x, self.nonlin)


@dataclass
class Layer:
    neurons: list[Neuron]

    def __init__(self, nin, nout, nonlin):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out


@dataclass
class MLP:
    layers: list[Layer]

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestNeuron(unittest.TestCase):
    def test_call(self):
        neuron = Neuron(1, nonlin=True)
        neuron.w = np.array([[1.0, 2.0], [3.0, 4.0]])
        neuron.b = np.array([1.0, 2.0])
        x = np.array([1.0, 2.0])
        np.testing.assert_allclose(neuron(x), np.array((6, 13)), rtol=1e-5, atol=0)

    def test_value_and_grad(self):
        neuron = Neuron(2, nonlin=True)
        neuron.w = np.ones((2,))
        neuron.b = np.array([1.0])
        value, (w_grad, b_grad) = neuron.value_and_grad(np.ones((2,)))
        np.testing.assert_allclose(value, 3.0, rtol=1e-5, atol=0)
        np.testing.assert_allclose(w_grad, np.array([1.0, 1.0]), rtol=1e-5, atol=0)
        np.testing.assert_allclose(b_grad, np.array([1.0]), rtol=1e-5, atol=0)


class TestLayer(unittest.TestCase):
    def test_doesnt_throw(self):
        layer = Layer(4, 6, nonlin=True)
        layer(np.array([1.0, 2.0, 3.0, 4.0]))


class TestMLP(unittest.TestCase):
    def test_doesnt_throw(self):
        mlp = MLP(2, [16, 16, 1])
        mlp(np.array([1.0, 2.0]))

    # class TestTraining(unittest.TestCase):
    #     def test_simple(self):
    #         mlp = MLP(2, [16, 16, 1])
    #
    #         def loss(x, y):
    #             (mlp(x) - y) ** 2
    #
    #         g = grad(loss)(np.array([1.0, 2.0]), 1.0)
    #         print("g", g)


class TestMicroJax(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(anp.negative(1.0), -1.0)
        self.assertAlmostEqual(anp.exp(1.0), 2.718281828459045)
        self.assertAlmostEqual(anp.exp(0.0), 1.0)
        self.assertAlmostEqual(anp.log(1.0), 0.0)
        self.assertAlmostEqual(anp.tanh(0.0), 0.0)
        self.assertAlmostEqual(anp.sinh(0.0), 0.0)
        self.assertAlmostEqual(anp.cosh(0.0), 1.0)

    def test_grad_negative(self):
        self.assertAlmostEqual(grad(anp.negative)(1.0), -1.0)
        self.assertAlmostEqual(grad(grad(anp.negative))(1.0), 0.0)
        self.assertAlmostEqual(grad(grad(grad(anp.negative)))(1.0), 0.0)
        self.assertAlmostEqual(grad(grad(grad(grad(anp.negative))))(1.0), 0.0)

    def test_grad_tanh(self):
        self.assertAlmostEqual(grad(anp.tanh)(0.0), 1.0)
        self.assertAlmostEqual(grad(grad(anp.tanh))(0.0), 0.0)

    def test_grad_exp(self):
        self.assertAlmostEqual(grad(anp.exp)(1.0), math.e)
        self.assertAlmostEqual(grad(grad(anp.exp))(1.0), math.e)
        self.assertAlmostEqual(grad(grad(grad(anp.exp)))(1.0), math.e)
        self.assertAlmostEqual(grad(grad(grad(grad(anp.exp))))(1.0), math.e)

    def test_relu(self):
        self.assertAlmostEqual(relu(1.0), 1.0)
        self.assertAlmostEqual(relu(-1.0), 0.0)
        self.assertAlmostEqual(grad(relu)(1.0), 1.0)
        self.assertAlmostEqual(grad(relu)(-1.0), 0.0)
        self.assertAlmostEqual(grad(grad(relu))(1.0), 0.0)
        self.assertAlmostEqual(grad(grad(relu))(-1.0), 0.0)


if __name__ == "__main__":
    unittest.main()

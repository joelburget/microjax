from dataclasses import dataclass
from typing import Any, Callable

import numpy
import math
import unittest


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


@dataclass(frozen=True)
class Node[T]:
    """A node in a computation graph.

    parents and argnums should be the same length. There is one for each boxed argument.
    args could be longer if there are non-Box arguments.
    """

    parents: tuple["Node[T]"]  # For each boxed argument, the node that produced it.
    value: T  # The output of the primitive function.
    fun: Callable[..., T]  # Wrapped primitive function
    args: tuple[Any]  # All (unboxed) positional arguments
    kwargs: dict[str, Any]  # Additional keyword arguments
    argnums: tuple[int]  # Positional indices of boxed values

    @staticmethod
    def new_root():
        global trace_id
        return Node((), None, lambda x: x, (), {}, ())

    # We have to override hash because dicts aren't hashable.
    def __hash__(self):
        return id(self)


def primitive(f_raw):
    """Wraps a function so that its gradient (vjp) can be specified and its
    invocation can be recorded."""

    def f_wrapped(*args, **kwargs):
        # Fetch boxed arguments with largest trace_id. This ensures that the
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
            raw_result = f_wrapped(*argvals, **kwargs)

            node = Node(parents, raw_result, f_wrapped, argvals, kwargs, argnums)
            return Box(raw_result, trace_id, node)
        else:
            return f_raw(*args, **kwargs)

    f_wrapped.__name__ = f_raw.__name__

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
            return numpy.zeros_like(x)
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
        for argnum, parent in zip(node.argnums, node.parents):
            # Lookup vector-Jacobian product (gradient) function for this
            # function/argument.
            vjp = primitive_vjps[node.fun][argnum]

            # Compute vector-Jacobian product (gradient) contribution due to
            # parent node's use in this function.
            parent_grad = vjp(outgrad, node.value, *node.args, **node.kwargs)

            # Save vector-Jacobian product (gradient) for upstream nodes.
            # Sum contributions with all others also using parent's output.
            outgrads[parent] = add_outgrads(outgrads.get(parent), parent_grad)
    return outgrad


def add_outgrads(prev_g, g):
    """Add gradient contributions together."""
    if prev_g is None:
        return g
    return prev_g + g


def jacobian(fun, argnum=0):
    """Compute Jacobian of fun at x."""

    def jacfun(*args, **kwargs):
        def unary_fun(x):
            new_args = list(args)
            new_args[argnum] = x
            return fun(*new_args, **kwargs)

        vjp, ans = make_vjp(unary_fun, args[argnum])
        return vjp(numpy.ones_like(ans))

    return jacfun


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
        if isinstance(ans, numpy.ndarray):
            if ans.shape != ():
                raise ValueError(
                    f"Gradient only defined for scalar-output functions. Output had shape: {ans.shape}"
                )

        return vjp(numpy.ones_like(ans))

    return gradfun


class Np:
    def __init__(self):
        self.negative = primitive(numpy.negative)
        self.exp = primitive(numpy.exp)
        self.log = primitive(numpy.log)
        self.tanh = primitive(numpy.tanh)
        self.sinh = primitive(numpy.sinh)
        self.cosh = primitive(numpy.cosh)
        self.multiply = primitive(numpy.multiply)
        self.add = primitive(numpy.add)
        self.subtract = primitive(numpy.subtract)
        self.abs = primitive(numpy.abs)
        self.true_divide = primitive(numpy.true_divide)
        self.divide = self.true_divide
        self.sign = primitive(numpy.sign)
        self.power = primitive(numpy.power)
        self.where = primitive(numpy.where)
        self.zeros = primitive(numpy.zeros)
        self.array = primitive(numpy.array)
        self.matmul = primitive(numpy.matmul)
        self.ndim = primitive(numpy.ndim)


np = Np()


def matmul_vjp0(g, ans, x, y):
    if max(np.ndim(x), np.ndim(y)) > 2:
        raise NotImplementedError("Current matmul vjps only support ndim <= 2.")

    if np.ndim(x) == 0:
        return np.sum(y * g)
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        return g * y
    if np.ndim(x) == 2 and np.ndim(y) == 1:
        return g[:, None] * y
    if np.ndim(x) == 1 and np.ndim(y) == 2:
        return np.matmul(y, g)
    return np.matmul(g, y.T)


def matmul_vjp1(g, ans, x, y):
    if max(np.ndim(x), np.ndim(y)) > 2:
        raise NotImplementedError("Current matmul vjps only support ndim <= 2.")

    if np.ndim(y) == 0:
        return np.sum(x * g)
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        return g * x
    if np.ndim(x) == 2 and np.ndim(y) == 1:
        return np.matmul(g, x)
    if np.ndim(x) == 1 and np.ndim(y) == 2:
        return x[:, None] * g
    return np.matmul(x.T, g)


primitive_vjps = {
    np.negative: {0: lambda g, ans, x: -g},
    np.multiply: {
        0: lambda g, ans, x, y: y * g,
        1: lambda g, ans, x, y: x * g,
    },
    np.exp: {0: lambda g, ans, x: ans * g},
    np.log: {0: lambda g, ans, x: g / x},
    np.tanh: {0: lambda g, ans, x: g / np.cosh(x) ** 2},
    np.sinh: {0: lambda g, ans, x: g * np.cosh(x)},
    np.cosh: {0: lambda g, ans, x: g * np.sinh(x)},
    np.divide: {
        0: lambda g, ans, x, y: g / y,
        1: lambda g, ans, x, y: -g * x / y**2,
    },
    np.add: {
        0: lambda g, ans, x, y: g,
        1: lambda g, ans, x, y: g,
    },
    np.subtract: {
        0: lambda g, ans, x, y: g,
        1: lambda g, ans, x, y: -g,
    },
    np.abs: {0: lambda g, ans, x: g * np.sign(x)},
    np.sign: {0: lambda g, ans, x: 0},
    np.power: {
        0: lambda g, ans, x, y: g * y * x ** np.where(y, y - 1, 1.0),
        1: lambda g, ans, x, y: g * np.log(np.where(x, x, 1.0)) * x**y,
    },
    np.where: {
        0: lambda g, ans, c, x=None, y=None: np.where(c, g, np.zeros(g.shape)),
        1: lambda g, ans, c, x=None, y=None: np.where(c, np.zeros(g.shape), g),
    },
    np.matmul: {
        0: matmul_vjp0,  # lambda g, ans, x, y: g * y.T,
        1: matmul_vjp1,  # lambda g, ans, x, y: x.T * g,
    },
}


class Box:
    """Box for numpy.ndarray.

    Anything you can do with an numpy.ndarray, you can do with a Box.
    """

    def __init__(self, value, trace_id, node):
        self._value = value
        self._node = node
        self._trace_id = trace_id

    # Tell Numpy to use this type instead of np.array. (It tells Numpy which
    # type to use when there are two possibilities.) When is this useful?
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
    T = property(lambda self: np.transpose(self))

    def __len__(self):
        return len(self._value)

    def __neg__(self):
        return np.negative(self)

    def __add__(self, other):
        return np.add(self, other)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __pow__(self, other):
        return np.power(self, other)

    def __div__(self, other):
        return np.divide(self, other)

    def __mod__(self, other):
        return np.mod(self, other)

    def __truediv__(self, other):
        return np.true_divide(self, other)

    def __matmul__(self, other):
        return np.matmul(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __rsub__(self, other):
        return np.subtract(other, self)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __rpow__(self, other):
        return np.power(other, self)

    def __rdiv__(self, other):
        return np.divide(other, self)

    def __rmod__(self, other):
        return np.mod(other, self)

    def __rtruediv__(self, other):
        return np.true_divide(other, self)

    def __rmatmul__(self, other):
        return np.matmul(other, self)

    def __abs__(self):
        return np.abs(self)


class TestMicroJax(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(np.negative(1.0), -1.0)
        self.assertAlmostEqual(np.exp(1.0), 2.718281828459045)
        self.assertAlmostEqual(np.exp(0.0), 1.0)
        self.assertAlmostEqual(np.log(1.0), 0.0)
        self.assertAlmostEqual(np.tanh(0.0), 0.0)
        self.assertAlmostEqual(np.sinh(0.0), 0.0)
        self.assertAlmostEqual(np.cosh(0.0), 1.0)

    def test_grad_negative(self):
        f = grad(np.negative)
        self.assertAlmostEqual(f(1.0), -1.0)
        for _ in range(10):
            f = grad(f)
            self.assertAlmostEqual(f(1.0), 0.0)

    def test_grad_tanh(self):
        self.assertAlmostEqual(grad(np.tanh)(0.0), 1.0)
        self.assertAlmostEqual(grad(grad(np.tanh))(0.0), 0.0)
        self.assertAlmostEqual(grad(grad(grad(np.tanh)))(0.0), -2.0)

    def test_grad_exp(self):
        f = np.exp
        for _ in range(10):
            self.assertAlmostEqual(grad(f)(1.0), math.e)
            f = grad(f)


if __name__ == "__main__":
    unittest.main()

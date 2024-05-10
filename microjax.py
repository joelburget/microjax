import numpy as np
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
        if isinstance(args[0], Anp):  # XXX HACK
            args = args[1:]
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


anp = Anp()

primitive_vjps = {
    anp.negative: {0: lambda g, ans, x: -g},
    # anp.multiply: {0 lambda g, ans, x: ...}
    anp.exp: {0: lambda g, ans, x: ans * g},
    anp.log: {0: lambda g, ans, x: g / x},
    anp.tanh: {0: lambda g, ans, x: g / np.cosh(x) ** 2},
    anp.sinh: {0: lambda g, ans, x: g * np.cosh(x)},
    anp.cosh: {0: lambda g, ans, x: g * np.sinh(x)},
}


class Box:
    """Box for np.ndarray.

    Anything you can do with an np.ndarray, you can do with an Box.
    """

    def __init__(self, value, trace_id, node):
        self._value = value
        self._node = node
        self._trace_id = trace_id

    def __bool__(self):
        return bool(self._value)

    __nonzero__ = __bool__

    def __str__(self):
        return "Autograd {0} with value {1}".format(
            type(self).__name__, str(self._value)
        )

    # Used by NumPy to determine which type gets returned when there are
    # multiple possibilities. Larger numbers == higher priority.
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

    def astype(self, *args, **kwargs):
        return anp._astype(self, *args, **kwargs)

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

    def __eq__(self, other):
        return anp.equal(self, other)

    def __ne__(self, other):
        return anp.not_equal(self, other)

    def __gt__(self, other):
        return anp.greater(self, other)

    def __ge__(self, other):
        return anp.greater_equal(self, other)

    def __lt__(self, other):
        return anp.less(self, other)

    def __le__(self, other):
        return anp.less_equal(self, other)

    def __abs__(self):
        return anp.abs(self)

    def __hash__(self):
        return id(self)


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
        # needs power

        # needs multiply
        # def test_grad_exp(self):
        #     self.assertAlmostEqual(grad(anp.exp)(1.0), math.e)
        #     self.assertAlmostEqual(grad(grad(anp.exp))(1.0), math.e)
        #     self.assertAlmostEqual(grad(grad(grad(anp.exp)))(1.0), math.e)
        #     self.assertAlmostEqual(grad(grad(grad(grad(anp.exp))))(1.0), math.e)


if __name__ == "__main__":
    unittest.main()

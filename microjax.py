from contextlib import contextmanager
from collections import defaultdict
from itertools import count
import numpy as np


def subvals(x, ivs):
    """Replace the i-th value of x with v.

    Args:
      x: iterable of items.
      ivs: list of (int, value) pairs.

    Returns:
      x modified appropriately.
    """
    x_ = list(x)
    for i, v in ivs:
        x_[i] = v
    return tuple(x_)


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


def trace(start_node, fun, x):
    """Build a computation graph."""
    with trace_stack.new_trace() as trace_id:
        # Wrap 'x' in a box.
        start_box = new_box(x, trace_id, start_node)

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
        args = args[1:]
        # Fetch boxed arguments with largest trace_id.  This ensures that the
        # computational graph being constructed only consists of other nodes
        # from the same call to trace().
        boxed_args, trace_id = find_top_boxed_args(args)
        if boxed_args:
            # Replace some elements of args with corresponding unboxed values.
            argvals = subvals(
                args, [(argnum, box._value) for argnum, box in boxed_args]
            )
            # Get nodes for each boxed argument.
            parents = tuple(box._node for _, box in boxed_args)

            # Get argument indices for each boxed argument.
            argnums = tuple(argnum for argnum, _ in boxed_args)

            # Calculate result of applying original numpy function.
            #
            # Note that we use a recursive call here in order to also augment
            # outer calls to trace() with lower trace_ids. See TraceStack's
            # docstring for details.
            ans = f_wrapped(*argvals, **kwargs)

            # Create a new node
            node = Node(ans, f_wrapped, argvals, kwargs, argnums, parents)
            return new_box(ans, trace_id, node)
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


class TraceStack(object):
    """Tracks number of times trace() has been called.

    This is critical to ensure calling grad() on a function that also calls
    grad() resolves correctly. For example,

    ```
    def f(x):
      def g(y):
        return x * y
      return grad(g)(x)

    y = grad(f)(5.)
    ```

    First, grad(f)(5.) wraps 5. in a Box and calls f(Box(5)). Then, grad(g)(x)
    wraps Box(5) again and calls g(Box(Box(5)). When computing grad(g), we want
    to treat x=Box(5) as fixed -- it's not a direct argument to g(). How does
    Autograd know that x is fixed, when all it can see is
    np.multipy(Box(5.), Box(Box(5.))? Because the second argument has a larger
    trace_id than the former!
    """

    def __init__(self):
        self.top = -1

    @contextmanager
    def new_trace(self):
        """Increment trace depth."""
        self.top += 1
        yield self.top
        self.top -= 1


trace_stack = TraceStack()


class Box(object):
    """Boxes a value within a computation graph."""

    # Type -> subclasses of Box. Types may be instances of Box. Subclasses must
    # take same arguments for __init__().
    type_mappings = {}

    # Non-Box types that can be boxed.
    types = set()

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

    @classmethod
    def register(cls, value_type):
        """Register a class as a Box for type 'value_type'.

        Should be called immediately after declaration.

        Args:
          cls: Inherits from Box. Type to box values of type 'value_type'.
          value_type: Type to be boxed.
        """
        Box.types.add(cls)
        Box.type_mappings[value_type] = cls

        # The Box implementation for a Box type is itself. Why? Imagine a nested
        # call to grad(). One doesn't want the inner Box's computation graph to
        # interact with the outer Box's.
        Box.type_mappings[cls] = cls


box_type_mappings = Box.type_mappings


def new_box(value, trace_id, node):
    """Box an unboxed value.

    Args:
      value: unboxed value
      trace_id: int. Trace stack depth.
      node: Node corresponding to this boxed value.

    Returns:
      Boxed value.
    """
    try:
        return box_type_mappings[type(value)](value, trace_id, node)
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))


box_types = Box.types


# Get value from a Box.
def getval(x):
    getval(x._value) if isinstance(x, Box) else x


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


primitive_vjps = defaultdict(dict)


def defvjp(fun, *vjps, **kwargs):
    """Register vector-Jacobian product functions.

    Let fun(x, y, ...) = ans be a function. We wish to register a
    vector-Jacobian product for each of fun's arguments. That is, functions

      vjp_x(g, ans, x, y, ...) = g df/dx
      vjp_y(g, ans, x, y, ...) = g df/dy
      ...

    This function registers said callbacks.

    Args:
      fun: function for which one wants to define vjps for.
      *vjps: functions. vector-Jacobian products. One per argument to fun().
      **kwargs: additional keyword arugments. Only 'argnums' is used.
    """
    argnums = kwargs.get("argnums", count())
    for argnum, vjp in zip(argnums, vjps):
        primitive_vjps[fun][argnum] = vjp


class Anp:
    negative = primitive(np.negative)
    exp = primitive(np.exp)
    log = primitive(np.log)
    tanh = primitive(np.tanh)
    sinh = primitive(np.sinh)
    cosh = primitive(np.cosh)


anp = Anp()

defvjp(anp.negative, lambda g, ans, x: -g)
defvjp(anp.exp, lambda g, ans, x: ans * g)
defvjp(anp.log, lambda g, ans, x: g / x)
defvjp(anp.tanh, lambda g, ans, x: g / anp.cosh(x) ** 2)
defvjp(anp.sinh, lambda g, ans, x: g * anp.cosh(x))
defvjp(anp.cosh, lambda g, ans, x: g * anp.sinh(x))

if __name__ == "__main__":
    print(anp.negative(1))
    print(anp.cosh(0))

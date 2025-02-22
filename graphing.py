from graphviz import Digraph
from microjax import Node, trace


def draw_dot(end_node, format="svg", rankdir="LR"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for parent in v.parents:
                edges.add((parent, v))
                build(parent)

    build(end_node)

    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    for n in nodes:
        dot.node(name=str(id(n)), label=n.fun.__name__, shape="circle")

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))

    return dot


def trace_and_draw(f, x):
    start_node = Node.new_root()
    _end_box, end_node = trace(start_node, f, x)
    return draw_dot(end_node)

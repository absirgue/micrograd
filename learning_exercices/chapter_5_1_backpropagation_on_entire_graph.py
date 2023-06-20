import math
from graphviz import Digraph


class Value:
    '''
    Definition of fields:
    data -> the numerical value held by this Value node
    _prev -> the numerical values that were used to produce this Value, its ancestors, if any
    _op -> the operation, if any, that was applied to the ancestors to produce this Value object
    label -> allows us to give a recognizable name to the Value (e.g. 'a')
    grad -> the derivative of the end result of our graph (NN), our loss function, with respect to this Value
    backward -> a function that "does the chain rule at each node that took input to produce an output
    '''

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        # Initialized to 0 because we assume that, by default, this value does not have any impact on our loss function.
        self.grad = 0
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data+other.data, (self, other), '+')

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data*other.data, (self, other), '*')

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad = (1 - t**2) * out.grad

        out._backward = _backward
        return out


def trace(root):
    # Builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    # LR = Left to Right
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # For any value in the graph, create a rectangular ('record) node for it
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f}" %
                 (n.label, n.data, n.grad), shape='record')
        if n._op:
            # If this value is the result of some operation, create an op node for it ...
            dot.node(name=uid+n._op, label=n._op)
            # ... and connect this node to it.
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # Connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2))+n2._op)

    return dot


x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
x1w1 = x1*w1
x1w1.label = 'x1 * w1'
x2w2 = x2*w2
x2w2.label = 'x2 * w2'
sum = x1w1 + x2w2
sum.label = 'Sum dendrite values'
bias = Value(6.8813735870195432, label='bias')
body = sum + bias
body.label = 'Cell body value'
o = body.tanh()
o.label = 'Result of activation function'

topo = []
visited = set()


def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)


build_topo(o)
print("Topological_sorting:", topo)

'''
Automating backpropagation through the graph. 
This code is moved to the backward() method of the Value object to have a cleaner code and 
to be able to start backpropagation from any node.
'''
o.grad = 1.0

build_topo(o)
for node in reversed(topo):
    node._backward()

draw_dot(o).render(directory='expression-graph-visualizations', view=True)

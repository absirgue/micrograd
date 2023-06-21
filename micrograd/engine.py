import math


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
        # This is so that other can be a normal int/float and not necessarily a Value object.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        # This is so that other can be a normal int/float and not necessarily a Value object.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # rmul is a fallback. If Python can't do 2*Value(3.0), it will check if an r_mull is defined and will do that instead.
    def __rmul__(self, other):
        return self*other

    def __truediv__(self, other):
        return self*other**-1

    def __neg__(self, other):
        return self*-1

    def __sub__(self, other):
        return self+(-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "** only supports int or float"
        # This is so that other can be a normal int/float and not necessarily a Value object.
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other-1)*out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        # This is so that other can be a normal int/float and not necessarily a Value object.
        out = Value(math.exp(self.data), (self, ), 'exp')

        def _backward():
            # We use out.data because derivative of e^x is e^x
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    # Automatic backpropgation from this Value object
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        self.grad = 1.0
        build_topo(self)
        for node in reversed(topo):
            node._backward()

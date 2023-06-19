class Value:
    '''
    Definition of fields:
    data -> the numerical value held by this Value node
    _prev -> the numerical values that were used to produce this Value, its ancestors, if any
    _op -> the operation, if any, that was applied to the ancestors to produce this Value object
    label -> allows us to give a recognizable name to the Value (e.g. 'a')
    grad -> the derivative of the end result of our graph (NN), our loss function, with respect to this Value
    '''

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        # Initialized to 0 because we assume that, by default, this value does not have any impact on our loss function.
        self.grad = 0
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data+other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data*other.data, (self, other), '*')
        self.grad = other.data
        other.grad = self.data
        return out

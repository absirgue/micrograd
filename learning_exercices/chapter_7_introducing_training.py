import random

import math



class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        print("IN ADD")
        print(self.data)
        print(other)
        other = other if isinstance(other, Value) else Value(other)
        print(other)
        out = Value(self.data+other.data, (self, other), '+')
        print(out)
        def _backward():
            self.grad += 1.0 * out.grad
            print(out)
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self*other

    def __radd__(self, other):
        return self+other

    def __truediv__(self, other):
        return self*other**-1

    def __neg__(self):
        return self*-1

    def __sub__(self, other):
        return self+(-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "** only supports int or float"
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
        out = Value(math.exp(self.data), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

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

class Neuron:
    '''
    Definition of fields:
    w -> the list of the neuron's weights
    b -> the neuron's bias
    '''

    # nin = number of inputs coming into this neuron
    def __init__(self, nin):
        self.w = [Value(random.uniform(1, -1)) for _ in range(nin)]
        self.b = Value(random.uniform(1, -1))

    '''
    Returns the value of the activation function for this Neuron.
    It is called in the following context:
    x = [2,4]
    n = Neuron(2) <- two dimmensional because 2 neurons
    n(x) <- calls __call__ on x
    '''

    def __call__(self, x):
        # Zip allows us to iterate over tuples made of a member of self.w and a member of x (a weight and an input)
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        # Return all the parameters invloved in this neuron's activation function.
        return self.w + [self.b]


class Layer:
    # nout defines the number of neurons in this layer, nin defines the number of neurons in the previous layer
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    '''
    We can now do:
    x = [2,4]
    l = Layer(2,3) <- created a layer with 3 two-dimensional neurons
    l(x) <- calls __call__ on x
    '''

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neurons in self.neurons for p in neurons.parameters()]


class MLP:
    # MLP= Multi-Layer Perceptron

    # nouts -> the number of neurons we want in each layer
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    '''
    We can now do:
    x = [2,3,-1]
    mlp = MLP(3,[4,4,1]) <- created a MLP with 3 layers respectively containing 4, 4 and 1 neurons 3-dimensional neurons
    mlp(x) <- calls __call__ on x
    '''

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # Returns all the weights and biases inside the entire Neural Network
        return [p for layer in self.layers for p in layer.parameters()]


'''
Our network first gives random outputs because all weights and biases are assigned randomly.
'''
training_set = [[2, 3, -1], [3, -1, 0.5], [0.5, 1, 1], [1, 1, -1]]
expected_outputs = [1, -1, -1, 1]

mlp = MLP(3, [4, 4, 1])
# print(predictions)

'''
Example output: [Value(data=0.6180565868270727), Value(data=-0.11489913259458244), Value(data=0.4395301709187512), Value(data=0.5383537214379654)]
In this case, we want to nudge prediction 1 up, prediction 2 down, prediction 3 down, and prediction 4 up.
'''


# We can our netork's loss
# predictions = [mlp(x) for x in training_set]
# loss = sum((prediction-true_value)**2 for true_value,prediction in zip(expected_outputs, predictions))
# print("LOSS",loss)
'''
Adjusting our model's parameters (all the weights and biases) based on their gradients.
'''

# Backpropagating on loss
# loss.backward()

# GRADIENT_DESCENT_STEP_SIZE = 0.01

# for p in mlp.parameters():
#     p.data += -GRADIENT_DESCENT_STEP_SIZE * p.grad


'''
Performing gradient descent.
'''
NUMBER_OF_ITERATIONS = 1000
LEARNING_RATE = 0.05
for k in range(NUMBER_OF_ITERATIONS):
    # forward pass
    predictions = [mlp(x) for x in training_set]
    print(predictions)
    loss = sum((prediction-true_value)**2 for true_value,prediction in zip(expected_outputs, predictions))
    
    #zero grad
    for p in mlp.parameters():
        p.grad = 0

    #backward pass
    loss.backward()

    #update
    for p in mlp.parameters():
        p.data += -LEARNING_RATE * p.grad
    
    print(k,loss.data)

'''
Example loss achieved with 20 iterations and learning rate is 0.05: 0.48027261204030147
Without zero grad: 4.580485041872402e-09 <- the reason why this work is because this is a very very simple answer
and it is very easy for our neural network to converge.

With 1000 iterations: 0.000539486977737428
'''


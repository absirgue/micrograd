from value_object import Value
import random


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

# Chapter 4: Introducing Neurons

We are now going to backpropagate through a Neuron. In the simplest case, these neuron will be Multilateral Perceptrons.

![Alt text](./illustrations/neuron_cs231n.png?raw=true "Representation of a Neuron")
Representation of a Neuron (taken on Google Images from Stanford's CS231N course)

## Walkthrough of a neuron:

- Input axis carry the input values, denoted x, to synapses
- Synapses have weights, denoted w, and alter the input value so that the value that is effectively fed in the cell body is the result of x\*w
- The cell body adds all the results of w\*x and an additional bias, denoted b, and passes it to an activation function
- The activation is usually some "squashing function" such as sigmoid or tanh.

We can therefore represent our simple neuron as follows:
![Alt text](./illustrations/neuron_operation_graph.png?raw=true "Graph of a Neuron's operations")

## Adding TanH to our Value class

It is interesting to note that we do not necessarily have to add atomic functions to our Value class. We can write functions at different levels of abstractions, the only thing that matters is that we know how to differentiate through any one function.
As such, we can write a functiont that directly calculates tanH and do not have to go through implementing exponentiation and division first.

## Backpropagating through our Neuron

Important to keep in mind:

- what we care the most about is the gradient of our neuron's weigths.
- here we have just one neuron but in a Neural Network there will be, as the name indicates, a vast network of neurons

Useful to know: d/dx tanh(x) = 1- tanh(x)^2

## Working on the Weights

Our Neural Network takes in x*w, x being an input and w beign a weight.
We note that, with our knowledge from previous chapters, if x*w=y
dL/dw = dy/dw \* dL/dy
<=> dL/dw = x \* dL/dy

We therefore see that the value of the gradient for our weight heavily depends on the value of the input. For example, if x is 0, then dL/dw will be 0, no matter the value of w and the structure of the graph after this x\*w operation.

Applying backward propagation automatically
To apply backward propagation automatically, we define a backward lambda field for each Value object that will hold
the function necessary to calculate the grad of its ancestors based on its own grad. This function's behavior
is specific to the operation that was applied that was applied to its ancestor to generate its value.

The result of this backpropagation on our network is:
![Alt text](./illustrations/backpropagation_result.png?raw=true "Result of Backpropagation")

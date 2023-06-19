What is micrograd ?
Is an Autograd engine, Autograd is short of autoamted gradient. It implements backpropagation. Backpropagation is a the mathematical core of any modern Deep Neural Network library like PyTorch.

Micrograd is in fact a scalar valued autograd engine. It is working on the level of individual scalars. This is excssive and would neevr be done in production, this is simply for educational purposes. If you have to train bigger networks, you use tensors. With tensors, none of the math changes, it is just a question of efficiency by combining numbers into arrays which can then be used to take advantage of a computer's ability to parallelize processes.

Micrograd is meant to help you build out mathematical expressions.
It supports addition, multiplication, raising to a power, aplying ReLU, divide,...

It allows us to build an expressiong raph. It enables to not only do a forward path (getting the final result) but we can also go at any node in our operation graph and call .backward() on node g which will basically intialize backpropagation at the node g.

This is going to recursively apply the chain rule from Calculus. This allows us to evaluate the derivative of g with respect to all the internal nodes and to the inputes. This derivative is very important information because ti is telling us how each of the input is affecting node g.

Neural Networks are just mathematical expressions, like the one taken by micrograd. Neural Networks take the input data as an input and the weights of the network as another input and the output is the prediction of the Neural Network or the loss of the network.

It is only 150 lines of code! A lot of pwoer comes from those 150 lines, it's all of Deep Neural Network, the rest is efficiency.

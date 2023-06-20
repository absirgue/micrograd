# Chapter 3: Backpropagation

## Backpropagation - the Importance of the Chain Rule

With backpropagation, our goal is to know the impact of each variable, for Neural Networks of each weight, on the final output, for Neural Networks often the value of the loss function.

![Alt text](./micrograd/expression-graph-visualizations/Forward_pass_visualization.png?raw=true "Operation Tree")
Backpropagation consists in walking backward in this tree to know how each Value node impacted the end result, L.

To do so for the first layer is quite trivial:
dL/dd = f

It becomes more interesting when exploring further layers. For example, when attemtpign to compute dL/de. To compute this derivative, we need the Chain Rule. The Chain Rule states that:
dL/de = dL/dd \* dd/de

Hence, we can compute that dL/de = f\*1 (because dL/dd = f and dd/de=1)

Observation: a '+' node will just "share" the derivative to all its child nodes.
This is because, for a+b=c, dc/db=1 and dc/da=1.

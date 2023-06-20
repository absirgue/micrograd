import torch

'''
Tensors are usually used as matrices (X*Y array) of scalars, here we are creating a Tensor that has
only a single element.

The use of Tensors is the only difference. These constructs allow us to do a lot of operations in parallel.
'''
x1 = torch.Tensor([2.0]).double()
# Because they are leaf nodes (inputs to the network), by default PyTorhc assumes that they do not
# require gradient so we need to manually set gradent to True.
x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()
x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()
w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()
w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double()
b.requires_grad = True
n = x1*w1+x2*w2+b
o = torch.tanh(n)

print('o', o.data.item())
o.backward()

print('--------')
print('x1', x1.grad.item())
print('x2', x2.grad.item())
print('w1', w1.grad.item())
print('w2', w2.grad.item())

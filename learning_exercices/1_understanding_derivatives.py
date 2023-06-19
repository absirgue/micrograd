import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 3*x**2 - 4*x+5


print(f(3))

# Applyin the function to a range of numbers from -5 to 5 in 0.25 increments
xs = np.arange(-5, 5, 0.25)
ys = f(xs)

print(ys)

plt.plot(xs, ys)

# Uncomment next line to display the graph.
# plt.show()

"""
What is the derivative of this function ?
We do not want to calculate this "by hand" because the expressions for Neural Networks are too large to do that.
Rather, we want to understand what this derivative is telling us and find an alternative way to calculate it.

Going back to high school math, a derivative is the limit when h goes to 0 of (f(a+h)-f(a))/h

We can therefore evaluate it by taking a very small h, this is not the strict mathematical truth but,
with a small enough value for h, it can give us a sufficient approximation.

Note: the smaller the h the better the approximate UNTIl a certain point, because we are using
floating point arithmetic and the representation of all these numbers in computer memory is finite.
"""


h = 0.000000001
x = 3.0
print((f(x+h)-f(x))/h)

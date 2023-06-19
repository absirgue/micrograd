# Let's get more complex by introducing multiple variables in our function


a = 2
b = -3
c = 10

d = a*b+c
print(d)

'''
What is the derivative of d in regards to a, b, and c ?

To evaluate it, we are going to, once again, use a very small value of h. 
'''

#  We will demonstrate the idea by calculating our approximation of the derivative of d in respect to a.
h = 0.00000001
a += h
d1 = a*b+c
print("D=", d)
print("D1=", d1)
print("Slope:", (d1-d)/h)

# Now, we can repeat the same logic with b and c to evaluate the derivative of in respect to each of b and c.

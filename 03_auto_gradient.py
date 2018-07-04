import numpy as np

import torch
from torch.autograd import Variable
import pdb

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]),  requires_grad=True)  # Any random value

# our model forward pass


def forward(x):
    return x * w

# Loss function

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# Before training
print("predict (before training)",  4, forward(4).data[0])

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print("progress:", epoch, l.data[0])

# After training
print("predict (after training)",  4, forward(4).data[0])


def forward(x, w, b):
    """
    x = [x1, x2] = [x, x^2]
    w = [w1, w2]
    """
    x = torch.Tensor(x)
    w = torch.Tensor(w)
    return torch.dot(x, w) + b

#%%
def estimate_grad(func, x, step=1e-3):
    """estimate gradient [vector] of func evaluated at x 
    using the centered finite difference method.
    - func: takes x as the argument
    - x: a list of arguments"""
    x = np.array(x)
    grad = []
#     pdb.set_trace()
    for i,xi in enumerate(x):
        dx = np.zeros_like(x)
        dx[i] = step
        print(dx)
        pdb.set_trace()
        grad.append( (func(*(x+dx)) - func(*(x-dx)))/(2*step) )
    return grad
def test_estimate_grad():
    f = lambda x: x**2
    print(estimate_grad(f, [1.0]))
    
test_estimate_grad()
    
    
    
    
    
    
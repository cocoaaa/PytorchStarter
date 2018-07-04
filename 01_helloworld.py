import numpy as np
import matplotlib.pyplot as plt

def forward(x):
    return x*w

def loss(x,y):
    #Note loss is a function of input and target output, not y_hat and y
    y_hat = forward(x)
    return .5*(y-y_hat)**2

w_list = []
mse_list = []
for w in np.arange(0.0, 

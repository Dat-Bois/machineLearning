# Activation functions apply a non-linear transformation and decide whether a neuron should be activated or not
# Without activation functions it's essentially just a linear regression model
# After each layer we usually apply an activation function

# Most popular activation functions
# 1. Step Function
#    - If x>=0 then output is 1, otherwise 0
# 2. Sigmoid
#    - 1 / 1+e^-x (0 to 1)
# 3. TanH (hyperbolic tangent)
#    - Scaled sigmoid function, slightly shifted (-1 to 1)
#    - Good for in-between hidden layers
# 4. ReLU
#    - Most popular choice in most networks
#    - f(x) = max(0,x) (essentially 0 for negative values, otherwise passthrough)
#    - If you don't know which to use, just use a RelLU
# 5. Leaky ReLU
#    - Tries to solve the vanishing gradient problem
#    - if x >= 0 then f(x) = x, otherwise f(x) = a*x (where a is a very small number)
#    - With a normal ReLU, the negative numbers are zero so the gradient is zero and during back propagation the weights won't update (dead neurons)
# 6. Softmax
#    - Good for the last layer in multi-class classification
#    - Essentially turns output into probability dist

import torch
import torch.nn as nn
import torch.nn.functional as F # some activation functions only in here
import numpy as np

# option 1 (create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# option 2 (use activation directly in forward pass)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out
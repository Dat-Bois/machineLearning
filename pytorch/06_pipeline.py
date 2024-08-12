# First step is to design the model (input, output size, forward pass)
# Second step is to construct the loss and optimizer
# Third step is to train the model:
# - forward pass: compute prediction
# - backward pass: gradients
# - update weights
#

import torch
import torch.nn as nn

# X is the input tensor, Y is the output tensor
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# w is the weight tensor, requires_grad=True means that we want to compute the gradient
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

# loss
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 5 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
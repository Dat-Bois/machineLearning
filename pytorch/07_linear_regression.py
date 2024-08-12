import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# This generates a random regression problem. x_numpy is the generated inputs, and y_numpy is the generated outputs
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# This turns them into tensors so we can use them
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# This reshapes the y-tensor so it can be used in the model
y = y.view(y.shape[0], 1)

# This is the number of samples and features
n_samples, n_features = x.shape

# 1. model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

# 2. Loss and optimizer

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. Training loop
num_epochs = 100

for epoch in range(num_epochs):

    #forward pass + loss
    y_pred = model(x)
    loss = criterion(y_pred, y)

    #backward pass
    loss.backward()

    # update
    optimizer.step()

    # clear the gradient
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}, loss = {loss.item():.4f}')

#plot
predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()

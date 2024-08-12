# Softmax essentially squishes the output of the model into a probability distribution.
# For instance if your input was [2.0, 1.0, 0.1] the softmax output would be like [0.7, 0.2, 0.1].

import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy: ', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

# Cross entropy loss measures the performance of our classification model whose output is a prob between 0 and 1
# It can handle multi-class. The better the prediction the lower the loss. Often combined with softmax.

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])   # This would effectively normalize it by dividing it by the num of samples

# This is one hot encoding so like this would be class 0 (class 1 would be [0,1,0] etc)
Y = np.array([1,0,0])

y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)
print(f'Loss1: {l1:.4f}') # Has a low loss so this is good
print(f'Loss2: {l2:.4f}')

# The pytorch version automatically applies the softmax
# Y has class labels, not one-hot
# Again Y_pred must be raw scores, no softmax
loss = nn.CrossEntropyLoss()

# Lets say we have 3 samples
Y = torch.tensor([2, 0, 1]) # The first sample is supposed to be class 2, the second class 0, the third class 1
# nsamples * nclasses = 3x3
y_pred_good = torch.tensor([[0.1, 1.0, 5.1], [2.0, 1.0, 0.1], [0.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 3.0, 0.3], [0.5, 3.0, 0.3], [2.5, 0.3, 0.3]])

l1 = loss(y_pred_good, Y) # this has a lower loss as expected
l2 = loss(y_pred_bad, Y)
print(l1.item())
print(l2.item())

# Essentially grabs the largest prob from each sample
_, pred1 = torch.max(y_pred_good, 1) # Predicted 2, 0, 1
_, pred2 = torch.max(y_pred_bad, 1) # Predicted 1, 1, 0
print(pred1)
print(pred2)



# Multiclass NN
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # goes from input_size to hidden_size
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes) # goes from hidden_size to num_classes

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at end
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() # applies the softmax

# For a binary problem we use a sigmoid function instead of softmax (have to manually apply it), also use nn.BCELoss()

# Binary Classification
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # goes from input_size to hidden_size
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1) # goes from hidden_size to 1

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at end
        y_pred = torch.sigmoid(out)
        return y_pred
    
model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss() # Binary cross entropy loss

# Look at tutorial number 08 to see how we can use the above models and criterion to train a neural net
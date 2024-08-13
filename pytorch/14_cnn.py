# CNNs are used primarily on image data
# Typical structure involves multiple convolutional layers with activation functions after each, and occasionally a 
# pooling layer after an activation function
# Works by using sliding kernel method for convolutional filters
# Pooling layers (max pooling) downsizes an image by for instance taking the largest value of every 2x2 square
# At the end we need to flatten and needs to be a fully connected layer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
epochs = 20
batch_size = 4
learning_rate = 0.001

# the dataset has PILImage images of with channels in range [0,255]
# we turn them into tensors of a normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(), # this line first divides each channel by 255 to normalize between 0 to 1. Also changes order to (C, H, W)
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # sets the mean and stdev to 0.5 per channel. This scales them to [-1,1]

#CIFAR dataset, images start at 32x32
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def find_size(input_width, filter_width, padding, stride):
    return (input_width - filter_width + 2 * padding)/stride + 1

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # Input channels (RGB), output channels (arbitrary), kernel size
        self.pool = nn.MaxPool2d(2, 2) # size and stride (means 2x2 kernel shifted to the right by 2 every time)
        self.conv2 = nn.Conv2d(6, 16, 5) # Input channels need to be same as prev output
        # fully connected layers
        # The 16*5*5 was determined by assuming the following structure:
        # conv1 -> pool -> conv2 -> pool
        # find_size shows the formula of finding the size given the input_width and filter width
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84) # 120 and 84 are purely arbitrary
        self.fc3 = nn.Linear(84, 10) # output must be 10 for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # no softmax since we do it in the cross entropy anyways
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss() # Since multiclass
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stochastic Gradient descent

# training loop
n_total_steps = len(train_loader)
pretime = time.time()
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        # reshape images first since its currently 100, 1, 28, 28
        # input size = 28*28=784 so should be 100,784
        imgs = imgs.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad() # clear gradient
        loss.backward()
        optimizer.step()

        if (i+1) % 200 == 0 or i+1 == n_total_steps:
            elapsed_time = time.time()-pretime
            print(f'epoch: {epoch+1} / {epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}, elapsed time (200 steps): {elapsed_time:.4f}', end='\r')
            pretime = time.time()
print()
# test
model = model.eval()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images) # 2D tensor with shape (batch_size, num_classes)
        # Each row corresponds to a single image in the batch, and each column corresponds to the model's
        # raw output (logits) for one of the classes (digits 0-9).

        # returns the max value of each row (the logit) and index of max (corresponds to the class index)
        _, pred = torch.max(outputs, 1)
        n_samples += labels.shape[0] # number of samples in current batch
        n_correct += (pred == labels).sum().item()

        for i in range(batch_size):
            if(labels[i]==pred[i]):
                n_class_correct[labels[i]] += 1
            n_class_samples[labels[i]] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy = {acc}%')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc}%')
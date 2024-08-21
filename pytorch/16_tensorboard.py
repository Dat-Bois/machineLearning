

# pip3 install tensorboard
# pip3 install -U torch-tb-profiler
# tensorboard --logdir=runs

#------TAKEN FROM 13_feed_forward.py

# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


import torch.nn.functional as F
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
logdir = "runs/mnist/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(logdir)

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 # 28*28 dimension of imgs which gets flattened
hidden_size = 100 # arbitrary
num_classes = 10
epochs = 5
batch_size = 32
learning_rate = 0.001

# MNIST data set
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
    transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
    shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, 
    shuffle=True)

#-----UNCOMMENT TO VISUALIZE
examples = iter(train_loader)
samples, labels = examples._next_data()
# torch.Size([100, 1, 28, 28]) torch.Size([100]) (batch size, color channels, img width, img height) (100 labels (1 for each img in batch))
# print(samples.shape, labels.shape)

# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
writer.flush()
# ----------------------

class NeuralNet(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_sz, hidden_sz)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_sz, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)
model = model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.reshape(-1, 28*28))
writer.flush()

# training loop
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        # reshape images first since its currently 100, 1, 28, 28
        # input size = 28*28=784 so should be 100,784
        imgs = imgs.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad() # clear gradient
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        _, pred = torch.max(outputs, 1)
        running_correct+= (pred == labels).sum().item()

        if (i+1) % 10 == 0 or i+1 == n_total_steps:
            print(f'epoch: {epoch+1} / {epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}', end='\r')
            accuracy = running_correct / (100 * imgs.size(0))
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', accuracy, epoch *  n_total_steps + i)
            running_loss = 0.0
            running_correct = 0

print()
# test
model = model.eval()

labels_list = []
preds_list = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images) # 2D tensor with shape (batch_size, num_classes)
        # Each row corresponds to a single image in the batch, and each column corresponds to the model's
        # raw output (logits) for one of the classes (digits 0-9).

        # returns the max value of each row (the logit) and index of max (corresponds to the class index)
        _, pred = torch.max(outputs, 1)
        n_samples += labels.shape[0] # number of samples in current batch
        n_correct += (pred == labels).sum().item()

        class_predictions = [F.softmax(output, dim=0) for output in outputs]

        preds_list.append(class_predictions)
        labels_list.append(pred)

    preds_list = torch.cat([torch.stack(batch) for batch in preds_list])
    labels_list = torch.cat(labels_list)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy = {acc}')

    classes = range(10)
    for i in classes:
        labels_i = labels_list==i
        preds_i = preds_list[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
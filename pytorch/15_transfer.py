import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torch.utils
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# Centers the distribution around 0
# This is because the input is between [0-1], and using the formula
# (pixel - mean) / std_dev
# You can see how a max of 1 - 0.5 = 0.5 and 0 - 0.5 = -0.5
# A smaller std_dev scales the output to be larger
# -0.5 / 0.25 = -2 and 0.5 / 0.25 = 2
# Typically a stdev of 0.5 or more is better because it ensures the range is between [-1 to 1]
# which is ideal for sigmoids and SGD. Helps the network converge faster.
# However a larger range allows for a more expressive network that might be able to 
# gather more detail at the cost of volatility.
mean = np.array([0.5, 0.5, 0.5]) 
std = np.array([0.25, 0.25, 0.25])

# These are data augmentations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # Crops image randomly to a size of 224 x 224
        transforms.RandomHorizontalFlip(), # Randomly flips
        transforms.ToTensor(), # Converts to tensor [0-255] is scaled to [0-1]
        transforms.Normalize(mean, std) # This scales the channels to [-2-2]
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))

# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])

def train_model(model : nn.Module, criterion, optimizer : optim.SGD, scheduler : lr_scheduler.StepLR, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# This model trained in 4m 13s with a best validation acc of 0.947712
# This method takes the weights from resnet and then trains / tunes all of them 
#  (meaning gradients are calculated for all layers)
# METHOD 1----------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2) # output is 2 classes only
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler
# This will update the learning rate

# Every 7 epochs our learning rate is multiplied by 0.1
step_lr = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr, num_epochs=20)
#--------------------------------

# This method trained in 2m 20s with a best validation acc or 0.960784
# This method not only performed better, but also trained faster
# METHOD 2-----------This will freeze the beginning layers and only train the last ones
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False # This will freeze all the layers at the beginning

num_ftrs = model.fc.in_features

# By default a new layer is with grad
model.fc = nn.Linear(num_ftrs, 2) # output is 2 classes only
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler
# This will update the learning rate

# Every 7 epochs our learning rate is multiplied by 0.1
step_lr = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr, num_epochs=20)
#--------------------------------
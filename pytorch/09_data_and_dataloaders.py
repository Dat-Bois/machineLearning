import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from typing import Any, Iterator

class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) #n_sample, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index) -> Any:
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples


if __name__ == "__main__": #This needs to be in a main clause bc otherwise homebrew throws errors (?)
    dataset = WineDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)


    # training loop
    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / 4)
    print(total_samples, n_iterations)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # forward and then backward pass
            if(i+1)%5 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Step: {i+1}\{n_iterations}, Inputs: {inputs.shape}")

    # torchvision.datasets.MNIST()
    # fashion-mnist, cifar, coco
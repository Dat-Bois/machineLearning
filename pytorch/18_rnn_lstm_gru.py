import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
# input_size = 784 # 28*28 dimension of imgs which gets flattened
# hidden_size = 100 # arbitrary
num_classes = 10
epochs = 10
batch_size = 100
learning_rate = 0.001

input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 2

# Global mean and std deviation of the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


# MNIST data set
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
    transform=transform, download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
    transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
    shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, 
    shuffle=True)

#-----UNCOMMENT TO VISUALIZE
# examples = iter(train_loader)
# samples, labels = examples._next_data()
# # torch.Size([100, 1, 28, 28]) torch.Size([100]) (batch size, color channels, img width, img height) (100 labels (1 for each img in batch))
# print(samples.shape, labels.shape)

# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()
# ----------------------

class RNN(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_sz = hidden_sz
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_sz, hidden_sz, num_layers, batch_first=True) # Means that batch is first dimension
        # self.lstm = nn.RNN(input_sz, hidden_sz, num_layers, batch_first=True) 
        # self.gru = nn.GRU(input_sz, hidden_sz, num_layers, batch_first=True) 
        # x-> (batch_size, seq, input_size)
        self.fc = nn.Linear(hidden_sz, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_sz).to(device)
        out, _ = self.rnn(x, h0)
        # --- FOR GRU
        # out, _ = self.gru(x, h0)
        # --- FOR LSTM
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_sz).to(device)
        # out, _ = self.lstm(x, (c0,h0))
        # -------------
        # out: batch_sz, seq_len, hidden_size
        # decode hidden state of only last time-step
        out = out[:,-1,:]
        # out (N, 128)
        out = self.fc(out)
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes)
model = model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad() # clear gradient
        loss.backward()
        optimizer.step()

        if (i+1) % 200 == 0 or i+1 == n_total_steps:
            print(f'epoch: {epoch+1} / {epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}', end='\r')
print()
# test
model = model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images) # 2D tensor with shape (batch_size, num_classes)
        # Each row corresponds to a single image in the batch, and each column corresponds to the model's
        # raw output (logits) for one of the classes (digits 0-9).

        # returns the max value of each row (the logit) and index of max (corresponds to the class index)
        _, pred = torch.max(outputs, 1)
        n_samples += labels.shape[0] # number of samples in current batch
        n_correct += (pred == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy = {acc}')


save_model = input("Save model? (y/n)")

if save_model == 'y':
    # Save the model
    FILE = "./data/mnist_rnn.pth"
    torch.save(model.state_dict(), FILE)
    print(f"Model saved as {FILE}")

vis = input("Visualize? (y/n)")

if vis != 'y':
    exit()

# Visualize results
with torch.no_grad():
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = dataiter._next_data()
    images = images.reshape(-1, sequence_length, input_size).to(device)
    labels = labels.to(device)

    # Get predictions
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    # Move back to CPU for visualization
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    # Plot the images with their true and predicted labels
    fig = plt.figure(figsize=(10, 8))
    for i in range(6):  # Display first 6 images
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Expected: {labels[i].item()} | Pred: {preds[i].item()}")
        plt.axis('off')

    plt.show()    
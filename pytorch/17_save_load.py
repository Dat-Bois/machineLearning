import torch
import torch.nn as nn

'''
Types of save and load:

# Complete model (lazy option)
torch.save(model, PATH) #(serialize using Python's pickle module)

# Model class must be defined somewhere)
model  = torch.load(PATH)
model.eval()

# STATE_DICT (preferred option)
torch.save(model.state_dict(), PATH)
# saves the parameters of the model

# Model class must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

'''

# Example

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = Model(n_input_features=6)
# Train the model...

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())   # Print the optimizer's state_dict

# To create a checkpoint during training
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

# Save the checkpoint
FILE = "./data/checkpoint.pth"
torch.save(checkpoint, FILE)

# To load the checkpoint
loaded_checkpoint = torch.load(FILE)
epoch = loaded_checkpoint["epoch"]

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)
model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

# Save the entire model
FILE = "./data/test_model.pth"
torch.save(model, FILE)

# Load the model
model : nn.Module = torch.load(FILE)
model.eval()

for param in model.parameters():
    print(param)

# Save the model parameters
FILE = "./data/test_model_state_dict.pth"
torch.save(model.state_dict(), FILE)

# Load the model parameters
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)


# To save on gpu and then load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), FILE)
# load all tensors onto the CPU
device = torch.device("cpu")
model = Model(n_input_features=6)
model.load_state_dict(torch.load(FILE, map_location=device))

# Save on GPU and load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), FILE)

model = Model(n_input_features=6)
model.load_state_dict(torch.load(FILE))
model.to(device)

# Save on CPU and load on GPU
torch.save(model.state_dict(), FILE)

device = torch.device("cuda")
model = Model(n_input_features=6)
model.load_state_dict(torch.load(FILE, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)

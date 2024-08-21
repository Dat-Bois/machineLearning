import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io


# load the model

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

input_size = 784 # 28*28 dimension of imgs which gets flattened
hidden_size = 100 # arbitrary
num_classes = 10
model = NeuralNet(input_size, hidden_size, num_classes)

model.load_state_dict(torch.load('mnist_ffnn.pth'))
model.eval()

# image -> tensor

def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((28, 28)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


# predict

def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)
    _, pred = torch.max(outputs, 1)
    return pred



import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
logdir = "runs/rnn/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(logdir)

from utils import (VALID_LETTERS,
                   N_LETTERS,
                   load_data,
                   letter_to_tensor,
                   line_to_tensor,
                   random_training_example)

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # input to hidden
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # input to output
        self.softmax = nn.LogSoftmax(dim=1) #Shape is 1x57 so we want second dimension

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    

category_lines, all_catagories = load_data()
n_catagories = len(all_catagories)

def category_from_output(output):
    category_index = torch.argmax(output).item()
    return all_catagories[category_index]

#--HYPER PARAMS
input_size = N_LETTERS
hidden_size = 128
output_size = n_catagories

epochs = 100
learning_rate = 0.001
FILE_PATH = "./data/rnn.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#------------
# Our dataset is kinda sucky since its very unbalanced. Some are very large
# Arabic : 2000
# English : 4000
# Russian : 10,000
weights = torch.zeros(n_catagories)
weights[0] = 0.6
weights[4] = 0.6
weights[14] = 0.5
criterion = nn.NLLLoss() # good for classification problems
rnn = RNN(input_size, hidden_size, output_size)
rnn.to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
step_lr = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def train(line_tensor : torch.Tensor, category_tensor : torch.Tensor):
    hidden = rnn.init_hidden()
    line_tensor = line_tensor.to(device)
    category_tensor = category_tensor.to(device)
    hidden = hidden.to(device)

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        hidden = hidden.to(device)

    loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

def train_model():
    current_loss = 0
    all_losses = []
    plot_steps, print_steps = 1000, 5000
    n_iters = epochs * print_steps
    for i in range(n_iters):
        category, line, category_tensor, line_tensor = random_training_example(category_lines, all_catagories)

        output, loss = train(line_tensor, category_tensor)
        current_loss += loss

        if(i+1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            writer.add_scalar('training loss', current_loss / plot_steps, int((i+1)/plot_steps))
            writer.flush()
            current_loss = 0
        
        if(i+1) % print_steps == 0:
            step_lr.step()
            guess = category_from_output(output)
            correct = "CORRECT" if guess == category else f"WRONG ({category})"
            print(f"{i+1} {(i+1)/n_iters*100:.2f}% {loss:.4f} {line} / {guess} {correct}")

    writer.close()

    torch.save(rnn.state_dict(), FILE_PATH)

# plt.figure()
# plt.plot(all_losses)
# plt.show()

def predict(input_line, file_path):

    model = RNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(file_path))
    model.to(device)
    model.eval()
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        hidden = model.init_hidden()
        line_tensor = line_tensor.to(device)
        hidden = hidden.to(device)
        
        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)
            hidden = hidden.to(device)

        guess = category_from_output(output)
        print(guess)


if __name__ == "__main__":
    train_model()
    while True:
        try:
            name = input("Enter name here: ")
            if name == "quit":
                break
            predict(name, FILE_PATH)
        except Exception as e:
            print(e)
            break


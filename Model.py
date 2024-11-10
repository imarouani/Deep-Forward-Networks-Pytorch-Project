# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFFN(nn.Module):
    def __init__(self):
        super(SimpleFFN, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 32 * 32 * 3)
        # Apply ReLU activation and pass through each layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = SimpleFFN()

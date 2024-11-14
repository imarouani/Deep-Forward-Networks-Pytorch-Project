import torch
import torch.nn as nn

# Defining the FFN
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()

        # First fully connected layer: input size 32*32*3, output size 1024
        # This represents the first linear transformation described in Chapter 6
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)

        # ReLU activation function introduces non-linearity, as discussed in Section 6.3
        self.relu = nn.ReLU()

        # Second fully connected layer: input size 1024, output size 512
        # Another linear transformation, similar to what is described in the book
        self.fc2 = nn.Linear(1024, 512)

        # Final fully connected layer: input size 512, output size 10 (CIFAR-10 classes)
        # This layer outputs logits for each class
        self.fc3 = nn.Linear(512, 10)
    
    # Define the forward pass
    def forward(self, x):
        # Flatten the input image to a vector
        # In Chapter 6, the process of flattening inputs is covered in relation to data representation
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))  # Apply first layer and activation
        x = self.relu(self.fc2(x))  # Apply second layer and activation
        x = self.fc3(x)  # Output layer without activation (CrossEntropyLoss handles this)
        return x
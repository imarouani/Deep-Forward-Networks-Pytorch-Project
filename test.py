# test.py
import torch
from model import SimpleFFN
from data_preparation import testloader, classes

# Instantiate the model
model = SimpleFFN()

# Load model weights if saved (optional)
# model.load_state_dict(torch.load("model_weights.pth"))

correct = 0
total = 0
# No need to track gradients for testing
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10,000 test images: {100 * correct / total}%")

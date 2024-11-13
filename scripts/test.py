import torch
from models.feedforward import FeedForwardNetwork
from scripts.data_preparation import testloader, classes  # Import the testloader and classes

# Load the trained model
model = FeedForwardNetwork()
model.load_state_dict(torch.load('weights_epoch_20.pth'))  # Load saved weights
model.eval()

correct = 0
total = 0

# Disable gradient computation for testing
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f'Accuracy on the 10,000 test images: {accuracy:.2f}%')

# Class-wise accuracy
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            label = labels[i]
            if predicted[i] == label:
                class_correct[label] += 1
            class_total[label] += 1

for i in range(10):
    print(f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

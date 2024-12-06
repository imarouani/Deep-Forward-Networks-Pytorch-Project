import torch
from model.feedforward import FeedForwardNetwork
from scripts.data_preparation import prepare_data
from torch.utils.data import DataLoader

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_model(model, testloader, device):
    """
    Evaluates the trained model on the test dataset.

    Args:
        model (nn.Module): The trained neural network model to evaluate.
        testloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to use for computation (CPU or GPU).

    Returns:
        float: Overall accuracy of the model on the test dataset.
        list: Class-wise accuracies for each class in the dataset.
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    class_correct = [0] * len(classes)  # Dynamically handle number of classes
    class_total = [0] * len(classes)

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Class-wise accuracy calculation
            for i in range(len(labels)):
                label = labels[i]
                if predicted[i] == label:
                    class_correct[label] += 1
                class_total[label] += 1

    accuracy = (correct / total) * 100
    class_accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(len(classes))]

    return accuracy, class_accuracies


# Load dataset
_, testset, classes = prepare_data()  # Dynamically retrieve class labels
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Print classes for verification (optional)
print("Classes in the dataset:", classes)

# Load the trained model
model = FeedForwardNetwork()
model.load_state_dict(torch.load('weights/weights_epoch_20.pth'))  # Load saved weights

# Evaluate the model
accuracy, class_accuracies = evaluate_model(model, testloader, device)

# Print overall accuracy
print(f'Accuracy on the 10,000 test images: {accuracy:.2f}%')

# Print class-wise accuracies
for i, class_name in enumerate(classes):
    print(f'Accuracy of {class_name}: {class_accuracies[i]:.2f}%')

import torch
import torch.optim as optim
import torch.nn as nn
from model.feedforward import FeedForwardNetwork
from scripts.data_preparation import prepare_data
from torch.utils.data import DataLoader

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
trainset, _, classes = prepare_data()  # Extract dynamic class labels from prepare_data
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Print classes for verification (optional)
print("Classes in the dataset:", classes)

# Set hyperparameters
learning_rate = 0.001
num_epochs = 20

# Initialize the model and move it to the device
model = FeedForwardNetwork().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Store loss values for visualization
loss_values = []


def train_model(model, trainloader, criterion, optimizer, num_epochs, device):
    """
    Trains the FeedForwardNetwork model on the CIFAR-10 dataset.

    Args:
        model (nn.Module): The neural network model to be trained.
        trainloader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to use for computation (CPU or GPU).

    Returns:
        list: A list containing the average loss for each epoch.
    """
    loss_values = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0

        for images, labels in trainloader:
            # Move images and labels to the device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear gradients from the previous step
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update the model's parameters

            epoch_loss += loss.item()  # Accumulate the loss

        avg_loss = epoch_loss / len(trainloader)  # Calculate average loss for the epoch
        loss_values.append(avg_loss)  # Store the loss value
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Save weights every 5 epochs in the 'weights/' directory
        if (epoch + 1) % 5 == 0:
            weights_path = f'weights/weights_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), weights_path)
            print(f"Model weights saved for epoch {epoch + 1} at {weights_path}")

    return loss_values


# Train the model
loss_values = train_model(model, trainloader, criterion, optimizer, num_epochs, device)
print("Finished Training")

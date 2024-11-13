import torch
import torch.optim as optim
import torch.nn as nn
from model.feedforward import FeedForwardNetwork  # Adjusted import based on your directory
from scripts.data_preparation import trainloader  # Import the trainloader

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Training loop
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
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Save weights every 5 epochs in the 'weights/' directory
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'weights/weights_epoch_{epoch+1}.pth')
        print(f"Model weights saved for epoch {epoch + 1}")


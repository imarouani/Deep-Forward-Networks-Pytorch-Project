import torch
import torch.optim as optim
import torch.nn as nn
from models.feedforward import FeedForwardNetwork
from utils.data_preparation import trainloader  # Import the trainloader

# Set hyperparameters
learning_rate = 0.001
num_epochs = 20

# Initialize the model
model = FeedForwardNetwork()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Store loss values for visualization
loss_values = []

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(trainloader)
    loss_values.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Save weights every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'weights_epoch_{epoch+1}.pth')
        print(f"Model weights saved for epoch {epoch + 1}")

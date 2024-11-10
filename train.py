# train.py
import torch
import torch.optim as optim
from model import SimpleFFN
from data_preparation import trainloader

# Instantiate the model, define the loss function and optimizer
model = SimpleFFN()
criterion = torch.nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Stochastic Gradient Descent

# Training loop
for epoch in range(10):  # Loop over the dataset multiple times
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize the parameters

        running_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

print("Finished Training")

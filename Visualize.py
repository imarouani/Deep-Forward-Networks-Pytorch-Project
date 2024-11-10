# visualize.py
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleFFN
from data_preparation import testloader, classes

# Function to show images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random test images
dataiter = iter(testloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))
# Print ground truth
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]}' for j in range(4)))

# Instantiate and test the model
model = SimpleFFN()
# Load model weights if saved (optional)
# model.load_state_dict(torch.load("model_weights.pth"))

# Predict
outputs = model(images)
_, predicted = torch.max(outputs, 1)
# Print predictions
print('Predicted: ', ' '.join(f'{classes[predicted[j]]}' for j in range(4)))

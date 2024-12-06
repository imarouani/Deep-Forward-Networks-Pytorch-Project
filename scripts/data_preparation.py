import os
import torch
import torchvision
import torchvision.transforms as transforms

# Define the prepare_data function
def prepare_data(save_path="data/dataset.pt"):
    """
    Prepares and saves the CIFAR-10 dataset for training and testing.

    Args:
        save_path (str): Path to save the dataset.

    Returns:
        tuple: Training and testing datasets.
    """
    # Check if the dataset is already saved
    if os.path.exists(save_path):
        print(f"Loading dataset from {save_path}")
        dataset = torch.load(save_path)
    else:
        print("Preparing and saving dataset...")
        # Define transformations for the training and testing sets
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
        ])

        # Download and prepare datasets
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Save the dataset as a dictionary
        dataset = {"train": trainset, "test": testset, "classes": trainset.classes}
        torch.save(dataset, save_path)

    return dataset["train"], dataset["test"], dataset["classes"]

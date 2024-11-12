### Overview: Feedforward Network on CIFAR-10
 This repository contains resources for the seminar Deep Learning Concepts at the University of Osnabrück (2024/2025), led by Lukas Niehaus and Robin Rawiel. The repository includes a presentation and code to introduce foundational theory and practical examples of Feed Forward Neural Networks (FFNNs), based on the 6th chapter of the book Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.  This work is prepared by Iheb Marouani and Zuzanna Bojarska to provide an accessible yet comprehensive introduction to FFNNs, covering both theoretical insights and hands-on examples.
 This chaptter can be directly accessed using this link: https://www.deeplearningbook.org/contents/mlp.html

## Installation Guide

This project demonstrates how a simple feedforward network learns to classify images from the CIFAR-10 dataset using PyTorch. It includes scripts for training, testing, and visualizing model performance.

## Project Structure
- `models/`: Contains the FeedForward network definition.
- `utils/`: Contains data loading and preprocessing code.
- `scripts/`: Contains scripts for training and testing.
- `notebooks/`: Contains Jupyter notebooks for visualizing training and weights.

## Getting Started
1. Clone the repository: `git clone <repository_url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python scripts/train.py`
4. Test the model: `python scripts/test.py`
5. Visualize results using the notebooks in `notebooks/`.

## Requirements
- torch
- torchvision
- matplotlib
- jupyter

## Usage
- **Training**: Run `scripts/train.py` to train the model. The script saves weights every 5 epochs.
- **Testing**: Run `scripts/test.py` to evaluate the model on the test set and get class-wise accuracy.
- **Visualization**: Use the provided Jupyter notebooks to visualize training metrics and weight changes.

## License
No licence required:
This project is provided "as is" with no explicit license. Use, modify, and distribute freely at your own discretion, with no warranties or guarantees.

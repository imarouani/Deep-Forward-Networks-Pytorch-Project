### Overview: Feedforward Network on CIFAR-10
 This repository contains resources for the seminar Deep Learning Concepts at the University of Osnabrück (2024/2025), led by Lukas Niehaus and Robin Rawiel. The repository includes a presentation and code to introduce foundational theory and practical examples of Feed Forward Neural Networks (FFNNs), based on the 6th chapter of the book Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.  This work is prepared by Iheb Marouani and Zuzanna Bojarska to provide an accessible yet comprehensive introduction to FFNNs, covering both theoretical insights and hands-on examples.
 This chaptter can be directly accessed using this link: https://www.deeplearningbook.org/contents/mlp.html

## Installation Guide

## Project Structure
- `models/`: Contains the FeedForward network definition.
- `utils/`: Contains data loading and preprocessing code.
- `scripts/`: 
  - `train.py`: Script for training the model.
  - `test.py`: Script for testing the model.
- `notebooks/`: 
  - `hands_on_theory.ipynb`: Explains the theory from Chapter 6 with code examples.
  - `visualize_training.ipynb`: Plots the loss and accuracy curves.
  - `visualize_weights.ipynb`: Visualizes how the weights change over epochs.

## Prerequisites
- **Python Version**: This project requires Python 3.8 to ensure compatibility with all dependencies.
- **Package Management**: Using Conda is recommended for managing the environment.

## Setting Up the Environment

### Using Conda (Recommended)
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven’t already.
2. Create the environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml

## License
No licence required:
This project is provided "as is" with no explicit license. Use, modify, and distribute freely at your own discretion, with no warranties or guarantees.



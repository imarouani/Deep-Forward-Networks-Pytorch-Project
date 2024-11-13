### Overview: Feedforward Network on CIFAR-10
 This repository contains resources for the seminar Deep Learning Concepts at the University of Osnabrück (2024/2025), led by Lukas Niehaus and Robin Rawiel. The repository includes a presentation and code to introduce foundational theory and practical examples of Feed Forward Neural Networks (FFNNs), based on the 6th chapter of the book Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.  This work is prepared by Iheb Marouani and Zuzanna Bojarska to provide an accessible yet comprehensive introduction to FFNNs, covering both theoretical insights and hands-on examples.
This chapter can be directly accessed using this [link](https://www.deeplearningbook.org/contents/mlp.html).

## Setting Up the Environment

The code automatically uses your NVIDIA GPU for faster computations if CUDA is available (v 11.8). If not, it defaults to using the CPU.
If you do have an NVIDIA GPU and want to use it for training, you can [download the CUDA Toolkit here](https://developer.nvidia.com/cuda-downloads).


1. Install [Miniconda here](https://developer.nvidia.com/cuda-11-8-0-download-archive) if you haven’t already.
2. Create the environment using the provided `environment.yml` file:
   ```bash
   conda env create --file environment.yml

## Project Structure

model/ffn.py: Defines the FeedForwardNetwork class.
utils/data_preparation.py: Loads and prepares the CIFAR-10 dataset.
visuals/: Contains scripts for visualizing training loss and weights.
weights/: Directory to save model weights.
main.ipynb: The main notebook to run the project.
- **Python Version**: This project requires Python 3.8 to ensure compatibility with all dependencies.
- **Package Management**: Using Conda is recommended for managing the environment.

## License
No licence required:
This project is provided "as is" with no explicit license. Use, modify, and distribute freely at your own discretion, with no warranties or guarantees.



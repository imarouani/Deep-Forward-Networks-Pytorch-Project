### **Overview: Feedforward Network on CIFAR-10 - Hands-on Example**
This repository contains resources for the seminar *Deep Learning Concepts* at the University of Osnabrück (2024/2025), led by Lukas Niehaus and Robin Rawiel. It includes a presentation and code to introduce foundational theory and practical examples of Feedforward Neural Networks (FFNNs), based on the [6th chapter](https://www.deeplearningbook.org/contents/mlp.html) of the book *Deep Learning* by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.

This work, prepared by Iheb Marouani and Zuzanna Bojarska, is the second part of a presentation, which aims to provide an accessible yet comprehensive introduction to FFNNs, combining theoretical insights with hands-on examples.

If you're not familiar with the theory, we recommend referring to the first part of the presentation or the mentioned book. Otherwise, the [main notebook](https://github.com/imarouani/Deep-Forward-Networks-Pytorch-Project/blob/main/Main.ipynb) is a comprehensive guide to train and experiment with a FFNN using the CIFAR-10 dataset.

---

### **Setting Up the Environment**

This project supports NVIDIA GPUs for faster computations if CUDA is available. For MacOS users with Apple Silicon GPUs, the code uses Metal Performance Shaders (MPS). If no GPU is available, the code defaults to the CPU.

#### **1. Install the Environment**
You can set up the environment using either **Conda** or **Pip**:

- **Using Conda (Recommended):**
   ```bash
   conda env create --file environment.yml
   conda activate <environment_name>
   ```

- **Using Pip:**
   ```bash
   pip install -r requirements.txt
   ```

#### **2. Verify PyTorch Installation**
Ensure that your PyTorch installation matches your system's hardware:
- For NVIDIA GPUs: Install with `cuda` support.
- For MacOS GPUs: Install with `mps` support.

> For help with compatibility issues, contact: `imarouani@uos.de`.

---

### **How to Run the Project**

#### **1. Train the Model**
Run the training script to train the FFNN on CIFAR-10:
```bash
python scripts/train.py
```
The trained weights will be saved in the `weights/` directory. By default, the script saves weights every 5 epochs.

#### **2. Test the Model**
Run the testing script to evaluate the trained model:
```bash
python scripts/test.py
```
The script prints overall accuracy and class-wise accuracy to the console.

#### **3. Main Notebook**
For an interactive experience, open and run the main notebook:
```bash
jupyter notebook main.ipynb
```

---

### **Project Structure**

- **`model/ffn.py`**: Defines the `FeedForwardNetwork` class.
- **`scripts/`**:
  - **`train.py`**: Script for training the model and saving weights.
  - **`test.py`**: Script for evaluating the model on the test set.
  - **`data_preparation.py`**: Prepares and saves the CIFAR-10 dataset for reuse.
- **`weights/`**: Directory for storing trained model weights.
- **`main.ipynb`**: Interactive notebook to explore and train the FFNN.

---

### **Dataset Preparation**

The dataset is prepared and saved during the first run to avoid redundancy. It will be stored in the `data/` directory as `dataset.pt`. Subsequent runs will load this file automatically.

To regenerate the dataset, delete `data/dataset.pt`.

---

### **Error Handling**

- **Common Issues and Solutions:**
  - **Dataset File Not Found:** If `dataset.pt` is missing or corrupted, rerun the project to regenerate it.
  - **CUDA/MPS Errors:** Verify your PyTorch installation matches your system's hardware. Refer to [PyTorch installation guide](https://pytorch.org/get-started/locally/).
  - **FileNotFoundError for Weights:** Ensure the weights file exists in the `weights/` directory. If missing, train the model using `train.py`.

---

### **Python and Dependencies**

- **Python Version:** The project requires Python 3.8 for compatibility.
- **Dependencies:** Install via `environment.yml` (Conda) or `requirements.txt` (Pip).

---

### **License**
No license required.

--- 

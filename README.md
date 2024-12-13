# CSCI-567 Project

This repository contains code and scripts for conducting experiments on the CIFAR-10 dataset. The experiments explore the effects of different initialization methods, data augmentation techniques, and skip connections on model performance.

---

## Setup

### Clone the repository

```sh
git clone https://github.com/NicklasHolmberg/csci-567-project.git
cd csci-567-project
```

### Create Conda Environment

To set up the environment required for running the experiments, use the provided `environment.yml` file to create and activate a Conda environment:

```sh
conda env create -f environment.yml
conda activate cs567
```
install conda if needed: https://docs.anaconda.com/miniconda/install/

---

## Run Experiments

To execute the experiments, navigate to the `src` directory and run the `run.sh` script:

```sh
bash src/run.sh
```

This script runs multiple configurations, logs results, and saves the best-performing models.

---

## Project Structure

The project is organized as follows:

- **`arxiv/`**: Jupyter notebook for baseline experiments using AlexNet (You don’t need to run it; it is provided for demonstration purposes only).
- **`best_model.pth`**: Pre-trained model weights for the best model.
- **`Cifar10/`**: Directory containing CIFAR-10 dataset batches.
- **`environment.yml`**: Conda environment configuration file.
- **`logs/`**: Directory containing logs of training and testing results.
  - **`test/`**: Logs of test results.
  - **`trainval/`**: Logs of training and validation results.
- **`models/`**: Directory containing saved model weights from various experiments.
- **`README.md`**: This file, providing an overview and instructions.
- **`src/`**: Source code directory.
  - **`run_model.py`**: Main script to train and evaluate models.
  - **`run.sh`**: Shell script to execute experiments with multiple configurations.

---

## File Descriptions

### **`src/run_model.py`**

The `run_model.py` script includes functions for training and evaluating models on the CIFAR-10 dataset:

- **`train_model`**: Trains the model with specified parameters and saves the best model.
- **`evaluate_best_model`**: Evaluates the saved model on test data and logs the results.
- **`get_label_distribution`**: Calculates the distribution of labels in the dataset.
- **`display_label_distribution`**: Displays label counts alongside class names.
- **`im_convert`**: Converts tensors to NumPy arrays for visualization purposes.
- **`overview_data`**: Visualizes sample data from the dataset.

### **`src/run.sh`**

The `run.sh` script executes experiments across multiple configurations, including combinations of random seeds, initialization methods, data augmentations, and skip connections.

---

## Running Experiments

To run the experiments, execute the `run.sh` script as described in the setup section. Results will be stored in the following directories:

- **Training and Validation Logs**: Saved in `logs/trainval/`.
- **Test Results**: Saved in `logs/test/`.
- **Best Model Weights**: Saved in the `models/` directory.

---

## Logs and Models

- **Logs**: All logs related to training, validation, and testing processes are stored in the `logs/` directory.
- **Model Weights**: Best-performing model weights from each experiment are stored in the `models/` directory.

---

## Notes

Feel free to explore and modify the scripts for your specific needs.

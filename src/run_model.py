import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.mps as mps
from torchvision.datasets import CIFAR10
from torch.utils.data import  DataLoader, Subset
import torch.optim as optim
# VISUALIZATION
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
# 3-fold EXPERIMENTS
import argparse
from init_method import initialize_model  # INIT METHOD (PARAMETER-WISE)
from data_augmentation import CustomCIFAR10, build_transforms, transform_configs # DATA AUGMENTATION (Data-wise)
from model import AlexNet, AlexNetWithSkipConnections  # SKIP CONNECTIONS (Model-wise)

import warnings
warnings.filterwarnings("ignore")

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run the model with specified parameters.")

    # Add arguments
    parser.add_argument(
        '--random_seed', 
        type=int, 
        required=True, 
        help="[RANDOM SEED]: 42, 1, 99"
    )
    parser.add_argument(
        '--init_method', 
        type=str, 
        choices=['random', 'xavier', 'he'], 
        required=True, 
        help="[INITIALIZATION METHOD] random, xavier, he"
    )
    parser.add_argument(
        '--augment', 
        type=str, 
        choices=['1', '2', '3', '4', '5', '6', '7'], # 1: resize, 2: random_rotation, 3: random_flip, 4: color_jitter, 5: random_sharpness, 6: random_erasing, 7: everything 
        required=True, 
        help="[DATA AUGMENTATION] 1: resize, 2: random_rotation, 3: random_flip, 4: color_jitter, 5: random_sharpness, 6: random_erasing, 7: everything "
    )
    parser.add_argument(
        '--skip_connections', 
        type=str, 
        choices=['yes', 'no'], 
        required=True, 
        help="[SKIP CONNECTIONS] yes or no"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Total # of experiments: 33=3*(3+8+2)-6=3*13-6= # random_seed * (# init method + # augment method + # skip connections) - redundant
    # Print the arguments
    print("*"*30)
    print(f"Random Seed: {args.random_seed}")
    print(f"Initialization Method: {args.init_method}")
    print(f"Data Augmentation: {args.augment}")
    print(f"Skip Connections: {args.skip_connections}")
    print("*"*30)
    
    # GPU Usage Guide:
    # - For Apple Silicon Mac users:
    #   1. Install `torch`, `torchaudio`, and `torchvision` first.
    #   2. Use the following code with 'mps' to leverage GPU (Metal Performance Shaders) for training.
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # - For users with NVIDIA GPUs:
    #   - If CUDA is available, you can use 'cuda' for GPU acceleration by replacing the code with:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')

    # classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)
    
    # hyper-parameters (that should be tuned)
    batch_size = 128
    num_epochs = 15
    learning_rate = 0.005
    weight_decay = 0.005
    momentum = 0.9
    
    # Set random seed for reproducibility, 'args.random_seed' will be used for this purpose.
    random_seed = args.random_seed
    reset_random(random_seed)

    # 'args.augment' argument will be used to determine whether to use data augmentation or not.
    train_dataset, dev_dataset, test_dataset = load_data(data_dir='../Cifar10', args=args)
    
    # For debugging purpose only.  
    # original_train_size = len(train_dataset)
    # reduced_train_size = original_train_size // 200
    # train_indices = np.random.permutation(original_train_size)[:reduced_train_size]  # Randomly select indices
    # train_dataset = Subset(train_dataset, train_indices)
    # original_dev_size = len(dev_dataset)
    # reduced_dev_size = original_dev_size // 10
    # dev_indices = np.random.permutation(original_dev_size)[:reduced_dev_size]  # Randomly select indices
    # dev_dataset = Subset(dev_dataset, dev_indices)
    # original_test_size = len(test_dataset)
    # reduced_test_size = original_test_size // 20
    # test_indices = np.random.permutation(original_test_size)[:reduced_test_size]  # Randomly select indices
    # test_dataset = Subset(test_dataset, test_indices)
    
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = get_dataloader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    # # Overview of the CIFAR-10 Dataset (statistics of train, dev, test respectively)
    # # Calculate label distributions for train, dev, and test datasets
    # train_label_counts = get_label_distribution(train_dataset)
    # dev_label_counts = get_label_distribution(dev_dataset)
    # test_label_counts = get_label_distribution(test_dataset)
    # # Display label distributions with class names
    # display_label_distribution(train_label_counts, "Train Dataset", classes) # The default training set size is 45,000. The training dataset size depends on args.augment: for augment 1, it is 45,000, while for augment 2–8, it is 90,000 (45,000 × 2). Please ensure the number of augmented versions is correct.
    # display_label_distribution(dev_label_counts, "Dev Dataset", classes)
    # display_label_distribution(test_label_counts, "Test Dataset", classes)
    
    # Preview some data. Activate this if needed.
    # overview_data(train_loader, classes, num_images=num_classes)
    
    # Select model based on '--skip_connections'
    if args.skip_connections == 'yes':
        model = AlexNetWithSkipConnections(num_classes=num_classes)
        print("Using AlexNet with Skip Connections")
    else:
        model = AlexNet(num_classes=num_classes)
        print("Using AlexNet without Skip Connections")
        
    # Initialize the model with the specified '--init_method'
    model = initialize_model(model, args.init_method)
    
    # Train and validate
    train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device = device,
        classes = classes,
        args=args,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        save_path=f"../models/best_model_{args.random_seed}_{args.init_method}_{args.augment}_{args.skip_connections}.pth"
    )
    
    evaluate_best_model(
        model=model, 
        device = device,
        classes=classes,
        test_loader=test_loader, 
        args=args,
        load_path=f"../models/best_model_{args.random_seed}_{args.init_method}_{args.augment}_{args.skip_connections}.pth",
    )
    

# reset random for reproducibility
def reset_random(random_seed):
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(random_seed)
  random.seed(random_seed)

def load_data(data_dir, args):
    """
    Load CIFAR-10 datasets with appropriate augmentations.
    """
    # If augment is 'yes', include both the unaugmented and augmented configs
    selected_method = int(args.augment)
    default_transform_config = transform_configs[1] # None
    transform_config = transform_configs[selected_method]
    
    if args.augment == '1':
        train_transform_configs = [transform_config]
    else:
        train_transform_configs = [default_transform_config, transform_config]  # Only unaugmented config
    
    # Always use the default transform for dev and test datasets
    dev_test_transform = build_transforms(default_transform_config)

    # Load CIFAR-10 base dataset
    base_dataset = CIFAR10(root=data_dir, train=True, download=True)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=dev_test_transform)

    # Split train dataset into train and dev (90-10 split)
    train_indices, dev_indices = train_test_split(range(len(base_dataset)), test_size=0.1, random_state=args.random_seed)
    train_subset = Subset(base_dataset, train_indices)
    dev_subset = Subset(base_dataset, dev_indices)

    # Build the augmented datasets for training
    train_transforms = [build_transforms(config) for config in train_transform_configs]
    train_dataset = CustomCIFAR10(train_subset, transform=train_transforms)

    # Dev dataset (unaugmented)
    dev_dataset = CustomCIFAR10(dev_subset, transform=dev_test_transform)

    return train_dataset, dev_dataset, test_dataset
    
def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Function to get label distribution
def get_label_distribution(dataset):
    labels = [label for _, label in dataset]  # Extract all labels
    return Counter(labels)

# Function to display label counts with class names
def display_label_distribution(label_counts, dataset_name, classes):
    print(f"\n{dataset_name} - Label Distribution:")
    total = 0
    for label, count in sorted(label_counts.items()):
        print(f"(Label {label}) {classes[label]} : {count}")
        total += count
    print(f"*** Total: {total} ***")
    
# Overview of the CIFAR-10 Dataset (image)
# We need to convert the images to numpy arrays as tensors are not compatible with matplotlib.
def im_convert(tensor):
    mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261] # best values for CIFAR-10

    img = tensor.cpu().clone().detach().numpy() #
    img = img.transpose(1, 2, 0)
    img = img * np.array(tuple(mean)) + np.array(tuple(std))
    img = img.clip(0, 1) # Clipping the size to print the images later
    return img

# Display an overview of data
def overview_data(loader, classes, num_images=5):
    data_iterable = iter(loader)  # Convert the loader to an iterable
    images, labels = next(data_iterable)  # Get the first batch

    num_images = min(num_images, len(images))  # Limit to the batch size
    fig = plt.figure(figsize=(20, 10))  # Create figure

    for idx in range(num_images):
        ax = fig.add_subplot(1, num_images, idx + 1, xticks=[], yticks=[])  # Add subplot
        plt.imshow(im_convert(images[idx]))  # Convert and display image
        ax.set_title(classes[labels[idx].item()])  # Set title

    # plt.show()  # Display the figure

def train_model(
    model, 
    train_loader, 
    dev_loader, 
    device,
    classes,
    args,
    num_epochs,
    learning_rate,
    weight_decay, 
    momentum, 
    save_path,
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    num_classes = len(classes)
    
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_acc = 0.0  # Best validation accuracy for saving model

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Metrics calculation
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update tqdm description
            train_bar.set_postfix(loss=loss.item())

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print("-"*50)
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

        # # Training confusion matrix and report
        # cm_train = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
        # print("\nTraining Confusion Matrix:")
        # print(cm_train)

        # if classes:
        #     print("\nTraining Classification Report:")
        #     print(classification_report(all_labels, all_preds, target_names=classes))

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        val_bar = tqdm(dev_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Metrics calculation
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update tqdm description
                val_bar.set_postfix(loss=loss.item())

        # Calculate epoch metrics
        epoch_loss = running_loss / len(dev_loader)
        epoch_acc = 100 * correct / total
        val_loss.append(epoch_loss)
        val_acc.append(epoch_acc)
        
        print("-"*50)
        print(f"Val Loss: {epoch_loss:.4f}, Val Accuracy: {epoch_acc:.2f}%")

        # Validation confusion matrix and report
        cm_val = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
        print("\nValidation Confusion Matrix:")
        print(cm_val)

        if classes:
            print("\nValidation Classification Report:")
            print(classification_report(all_labels, all_preds, target_names=classes))

        # Save the best model
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} with Val Accuracy: {epoch_acc:.2f}%")

        # Save the log of training process for each epoch, but in one file
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "val_loss": epoch_loss,
            "val_accuracy": epoch_acc,
            "val_confusion_matrix": cm_val.tolist(),
            "val_classification_report": classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
        }

        log_file_path = f"../logs/trainval/trainvallog_{args.random_seed}_{args.init_method}_{args.augment}_{args.skip_connections}.json"
        with open(log_file_path, 'a') as log_file:
            json.dump(epoch_log, log_file, indent=4)
            log_file.write('\n')
        
    return train_loss, val_loss, train_acc, val_acc
 
def evaluate_best_model(
    model, 
    device,
    classes,
    test_loader,
    args, 
    load_path,
):
    """
    Evaluate the best saved model on test data and display results.
    """
    # Load the best model weights
    model.load_state_dict(torch.load(load_path))
    model = model.to(device)
    model.eval()

    num_classes = len(classes)
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    print("\nEvaluating the best model on test data...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=True):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Metrics calculation
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions and labels for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # GPU memory management
            del images, labels, outputs

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=np.arange(num_classes))
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    if classes:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=classes))

    # Display Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes if classes else np.arange(num_classes))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: Test Data")
    # plt.show()
    
    # Save accuracy, confusion matrix, and classification report to log file
    test_log = {
        "test_accuracy": accuracy,
        "test_confusion_matrix": cm.tolist(),  # Convert numpy array to list for JSON serialization
        "test_classification_report": classification_report(all_labels, all_predictions, target_names=classes, output_dict=True)  # Convert classification report to dict for JSON serialization
    }

    test_log_file_path = f"../logs/test/testlog_{args.random_seed}_{args.init_method}_{args.augment}_{args.skip_connections}.json"
    with open(test_log_file_path, 'w') as test_log_file:
        json.dump(test_log, test_log_file, indent=4)
    print(f"Test log saved to {test_log_file_path}")
    print(f"{args.random_seed}_{args.init_method}_{args.augment}_{args.skip_connections} completed. Exiting program.")

    return accuracy, cm

if __name__ == "__main__":
    main()

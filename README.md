# Image Classification Project using PyTorch

This project implements an image classification pipeline using PyTorch to train a deep learning model on a custom dataset. The models supported are:

- ResNet-18 (built from scratch using PyTorch layers)
- VGG-16 (pre-trained from torchvision with a custom classifier head)

The project supports training, validation, testing, data augmentation, and visualization of predictions and learning curves.

---

## Dataset Structure

The dataset should be organized in the following structure:

```
/content/
├── train/                  # Training data (subfolders = class names)
│   ├── class1/
│   ├── class2/
│   └── ...
│
├── test/                    # Test data (subfolders = class names)
│   ├── class1/
│   ├── class2/
│   └── ...
```

Each class folder should contain images belonging to that class.

---

## Requirements

Install the required libraries:

```
pip install torch torchvision matplotlib numpy pillow
```

---

## Data Augmentation

### Training Data Augmentations

The following augmentations are applied only to the training data:

- Resize to 256x256
- Random horizontal flip (50% chance)
- Random rotation (up to ±20 degrees)
- Random color jitter (brightness, contrast, saturation - small changes)
- Random resized crop (random zoom-in effect)
- Normalize using ImageNet mean and standard deviation:
    - Mean: [0.485, 0.456, 0.406]
    - Standard Deviation: [0.229, 0.224, 0.225]

### Validation and Test Data Processing

For validation and test data, the following transformations are applied (no augmentation):

- Resize to 256x256
- Normalize using ImageNet mean and standard deviation

---

## Models

### Simple CNN 
- Implemented a simple CNN architecture to compare

### ResNet-18 (from scratch)

- Implemented entirely using `torch.nn` (no torchvision models).
- Contains 4 stages, each with 2 residual blocks.
- Each residual block has 2 convolutional layers and a skip connection.
- Global Average Pooling before the final dense layer.


---

## Training Process

The training script performs:

1. Loads images from `/content/train` and `/content/test`.
2. Splits training data into train and validation (80%/20% split).
3. Applies augmentations to training images.
4. Initializes either ResNet-18 (from scratch) or VGG-16 (pre-trained).
5. Trains the model using CrossEntropyLoss and Adam optimizer.
6. Tracks training/validation accuracy and loss.
7. Plots accuracy and loss curves after training.
8. Evaluates the model on the test set.
9. Displays sample predictions with true and predicted labels (colored for correctness).

---

## Example Output (Accuracy and Loss Plot)

The training script also generates:

- A plot showing training and validation accuracy over epochs.
- A plot showing training and validation loss over epochs.
- Sample test images with predicted labels shown (green for correct, red for incorrect).

---

## How to Run

1. Place your dataset into `/content/train` and `/content/test` folders.
2. Run the training script.
3. After training, results will be displayed, including:
    - Final test accuracy.
    - Accuracy/loss plots.
    - Sample predictions.

---

## Example Folder Structure

```
/content/
├── train/                  # Training data (subfolders = class names)
├── test/                    # Test data (subfolders = class names)
├── project_script.py        # Full training code (data loading, model, training, etc.)
├── README.md                 # This file
```




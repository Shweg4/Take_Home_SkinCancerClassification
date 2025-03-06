# Skin Cancer- Benign and Malignant Classification 

This project implements an image classification pipeline using PyTorch to train a deep learning model on a custom dataset. The models supported are:

- ResNet-18 (built from scratch using PyTorch layers)
- VGG-16 (pre-trained from torchvision with a custom classifier head)
- Simple CNN (Convolution layers with fc layers as final layers)
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
2. Run the Jupyter notebook - The repository contains three files for three models(CNN, ResNet & VGG16).
3. Each notebook contains the entire pipeline from importing the dataset, to pre-processing, training, evaluation & model inference.
4. Ensure the dataset is imported correctly and run all the remaining blocks. 
5. After training, run the next blocks for results to be displayed, including:
    - Final test accuracy.
    - Accuracy/loss plots.
    - Sample predictions.
##Results

The table below summarizes the results of different architectures

<table>
  <tr>
    <th>Simple CNN loss and accuracy</th>
    <th>Resnet loss and accuracy</th>
    <th>VGG-16 pretrained loss and accuracy</th>
  </tr>
  <tr>
    <td>
      <img src="images/Simple CNN - Training Plots.png" alt="Simple CNN loss and accurac" width="400">
      <p><strong>Summary:</strong> This graph depicts the loss and accuracy of a simple CNN architecture. The accuracy reaches around 85% after 100 epochs. The test accuracy for this was 84%.</p>
  </td>
    <td>
      <img src="Resnet - Training Plots.png" alt="Resnet loss and accuracy" width="400">
      <p><strong>Summary:</strong> This graph depicts the loss and accuracy of the Resnet architecture. The validation accuracy is around 89% and the model starts to overfit aroung 40th epoch. The test accuracy is around 93.5%</p>
    </td>
    <td>
      <img src="VGG - Training Plots.png" alt="VGG-16 pretrained loss and accuracy" width="400">
      <p><strong>Summary:</strong> This graph shows the loss and accuracy of a VGG 16 model which is loaded with pre-trained weights. The validation accuracy is around 92% and the test accuracy is 93.55%.</p>
    </td>
  </tr>
</table>

The best performing model was VGG-16 with pretrained weights


### Inference 

<table>
  <tr>
    <th>Simple CNN </th>
    <th>Resnet</th>
    <th>VGG-16 pretrained</th>
  </tr>
  <tr>
    <td>
      <img src="Simple CNN - Sample Predictions(Inference).png" alt="Simple CNN loss and accurac" width="400">
  </td>
    <td>
      <img src="ResNet- Sample Predictions(Inference).png" alt="Resnet loss and accuracy" width="400">
    </td>
    <td>
      <img src="VGG Sample Predictions (Inference).png" alt="VGG-16 pretrained loss and accuracy" width="400">
    </td>
  </tr>
</table>
## Example Folder Structure

```
/content/
├── train/                  # Training data (subfolders = class names)
├── test/                    # Test data (subfolders = class names)
├── project_script.py        # Full training code (data loading, model, training, etc.)
├── README.md                 # This file
```




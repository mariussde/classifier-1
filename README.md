# CIFAR-10 Image Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images across 10 different classes. The model is trained on 50,000 images and tested on 10,000 images.

## Project Structure

- **Data Loading**: The CIFAR-10 dataset is automatically downloaded and loaded into training and testing datasets. The images are normalized and converted into tensors.

- **Model Definition**: The CNN architecture includes two convolutional layers followed by three fully connected layers, designed to classify the images into one of the 10 classes.

- **Training**: The model is trained using Stochastic Gradient Descent (SGD) with a learning rate of 0.001 and momentum of 0.9. The training process runs for 30 epochs.

- **Model Saving**: After training, the model is saved to a file, allowing it to be loaded and used later without retraining.

- **Evaluation**: The accuracy of the trained model is evaluated on the test dataset, and the resulting accuracy is displayed.

- **Inference**: The trained model can make predictions on new images by processing them through the network.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- PIL (Python Imaging Library)

You can install the required packages using pip:

```bash
pip install torch torchvision numpy pillow
```
# How to Run

## Download and Prepare Data
- The CIFAR-10 dataset will be automatically downloaded and prepared when the script is run.

## Train the Model
- The model is trained over multiple epochs, during which the network's weights are adjusted to minimize the loss.

## Save the Model
- After training, the model is saved for future use, allowing you to load and use it without retraining.

## Evaluate the Model
- The performance of the model is evaluated on a separate test dataset, with accuracy being the primary evaluation metric.

## Inference on New Images
- The trained model can be used to process new images and generate predictions.

# Notes
- The current implementation achieves an accuracy of around 10% on the CIFAR-10 dataset. Further improvements to the model architecture or training process may be necessary to enhance performance.
- Be cautious when using `torch.load` due to potential security issues, especially when loading models from untrusted sources.

# License
This project is licensed under the MIT License.

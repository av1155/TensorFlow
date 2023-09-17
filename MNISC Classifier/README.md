# MNIST Digit Classifier using TensorFlow

This repository contains code for a simple MNIST digit classification using TensorFlow and Keras.

## Overview

The provided code demonstrates a neural network-based digit classifier for the MNIST dataset. MNIST is a collection of 28x28 grayscale images of handwritten digits (0-9). The goal is to build a model that accurately predicts the digit represented by each image.

The classification model follows these steps:

1. Data Loading and Preprocessing:
The MNIST dataset is loaded using TensorFlow's mnist.load_data() method. The images are normalized to have pixel values between 0 and 1. Labels are one-hot encoded for proper training.

2. Model Architecture:
The neural network consists of:

- **Flatten Layer**: Flattens the 28x28 input images.
- **Dense Hidden Layers**: Two dense hidden layers with ReLU activation, having 128 and 64 neurons respectively.
- **Dropout Layer**: Helps prevent overfitting by randomly setting a fraction of the inputs to zero during training.
- **Output Layer**: Dense layer with softmax activation, producing probabilities for each digit (0-9).

3. Model Compilation:
The model is compiled using the Adam optimizer and categorical cross-entropy loss function, suitable for multi-class classification.

4. Model Training:
The model is trained using the training dataset, with a validation split to monitor performance. Training is performed for a specified number of epochs with a defined batch size.

5. Model Evaluation:
After training, the model is evaluated using the test dataset, calculating accuracy as a measure of performance.

6. Sample Predictions:
Random samples from the test set are chosen, and the model's predictions are displayed alongside the true labels.

7. Additional Metrics:
Further evaluation metrics such as a classification report and a confusion matrix are generated to provide a detailed understanding of the model's performance.

## Tools Used

* **TensorFlow**: An open-source machine learning framework for training and deploying deep learning models.
* **Keras**: A high-level neural networks API, written in Python and capable of running on top of TensorFlow.
* **NumPy**: A powerful library for numerical computing in Python, used for array operations and data manipulation.
* **Matplotlib and Seaborn**: Python libraries for creating visualizations to display training metrics, sample images, and evaluation results.

## Results and Evaluation

The model achieves an accuracy of approximately 97.73% on the test set. The classification report and confusion matrix provide additional insights into the model's performance for each digit class.

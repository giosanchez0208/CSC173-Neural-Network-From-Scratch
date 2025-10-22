# CSC173 Activity 01 - Neural Network from Scratch

**Date:** October 09, 2025  
**Team:** Gio Kiefer Sanchez, Mark Angelo Gallardo, Angelyn Jimeno

## Project Overview

This project implements a simple neural network for binary classification using breast cancer diagnostic data. The network is built completely from scratch using only Python and NumPy, with no machine learning libraries. The goal is to deepen understanding of neural network fundamentals including forward propagation, loss computation, backpropagation, gradient descent training, and model evaluation.

## Data Preparation

We used the Breast Cancer Wisconsin Diagnostic dataset obtained from these sources:
- [Scikit-learn breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [UCI Machine Learning Repository (Breast Cancer Wisconsin Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  

Two features were selected from the dataset for the input layer of the network:
- concave_points3
- perimeter3

These two inputs were chosen due to their strong correlation with the diagnosis label.

## Network Architecture

- Input layer: 2 neurons (representing the two selected features from the dataset)
- Hidden layer: 4 neurons with the Sigmoid activation function
- Output layer: 1 neuron to produce binary classification output

This structure allows the model to learn nonlinear patterns from the two input variables and output a probability between 0 and 1.

## Implementation Details

- Weight and bias parameters initialized randomly.
- Forward propagation implements layer-wise computations with chosen activation functions.
- Loss computed using Binary Cross-Entropy (BCE) loss instead of Mean Squared Error (MSE) since BCE is better suited for binary classification.
- Backpropagation calculates gradients of weights and biases.
- Parameters updated using gradient descent.
- Training performed for 500 to 1000 iterations, gradually minimizing the BCE loss.

## Results & Visualization

After training, the program displays:
- The decision boundary separating the two classes in the dataset.
- The training loss curve, showing how the network minimizes error over time.

These visualizations help confirm that the neural network successfully learned to classify the data.

## Team Collaboration

Each member contributed to different components of the network:
- Weight and bias initialization
- Forward propagation coding
- Loss function implementation
- Backpropagation and gradient computation
- Training loop and visualization

## How to Run

1. Clone the GitHub repository:
   ```
   git clone https://github.com/giosanchez0208/CSC173-Neural-Network-From-Scratch.git
   ```
2. Open the Jupyter notebook or Colab file.
3. Run all cells sequentially.
4. Explore training loss plot and decision boundary visualizations.

## Summary

This activity provided hands-on experience in building a neural network without relying on high-level ML frameworks. The group collaboratively developed the model, analyzed its training behavior visually, and demonstrated understanding of fundamental AI concepts through both code and documentation.

Video: link
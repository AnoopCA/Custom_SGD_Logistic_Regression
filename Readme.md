Custom SGD Classifier with Logloss and L2 Regularization:
This repository contains a custom implementation of the Stochastic Gradient Descent (SGD) classifier for binary classification. The classifier is implemented with Logarithmic Loss (Logloss) and L2 regularization, all without relying on the scikit-learn library. The purpose of this project is to demonstrate the inner workings of the SGD classifier and provide an alternative implementation.

Implementation Highlights:
The custom dataset is generated using the make_classification function from sklearn.datasets. The dataset consists of 50,000 samples with 15 features and binary labels.
The dataset is split into training and testing sets using the train_test_split function from sklearn.model_selection.
The SGD classifier from sklearn.linear_model is used as a reference to compare results with the custom implementation.
The custom implementation consists of functions to initialize weights, compute the sigmoid function, compute the log loss, and compute gradients for the weights and bias.
Logistic Regression is implemented using the custom SGD approach. The model is trained using a specified number of epochs, learning rate, and L2 regularization constant.
The train and test loss are calculated and plotted against the number of epochs to visualize the convergence of the model.

Usage:
Ensure you have the required packages installed: numpy, pandas, mpmath, and matplotlib.
Clone or download the repository to your local machine.
Open the notebook and run the provided code step by step.
The primary purpose is to demonstrate the underlying concepts of the SGD classifier with Logloss and L2 regularization.

Acknowledgments:
The project utilizes the following resources:
NumPy for numerical operations.
Matplotlib for data visualization.
mpmath for high-precision mathematics.

Note:
The custom implementation aims to provide insights into the SGD classifier's inner workings. It may not match the efficiency or features of the scikit-learn implementation. For practical use cases, consider using established libraries like scikit-learn for machine learning tasks.
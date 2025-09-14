import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Prepare features and target
def prepare_data(data):
    """Prepares the data for training the model.

    This function takes a DataFrame of stock data, creates a 'Target' column
    by shifting the 'Close' price by one day, and then separates the features
    (X) and the target (y).

    Args:
        data (pandas.DataFrame): A DataFrame containing historical stock data
                                 with at least a 'Close' column.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The feature set (X).
            - numpy.ndarray: The target values (y).
    """
    data = data[['Close']].copy()
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()
    X = data[['Close']].values
    y = data['Target'].values
    return X, y

# 3. Train/test split
def split_data(X, y):
    """Splits the data into training and testing sets.

    Args:
        X (numpy.ndarray): The feature set.
        y (numpy.ndarray): The target values.

    Returns:
        tuple: A tuple containing the split data:
               (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train model
def train_model(X_train, y_train):
    """Trains a linear regression model.

    Args:
        X_train (numpy.ndarray): The training feature set.
        y_train (numpy.ndarray): The training target values.

    Returns:
        sklearn.linear_model.LinearRegression: The trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# 5. Predict and visualize
def plot_predictions(data, y_test, y_pred):
    """Plots the actual vs. predicted stock prices.

    Args:
        data (pandas.DataFrame): The original DataFrame (used for plotting context).
        y_test (numpy.ndarray): The actual target values.
        y_pred (numpy.ndarray): The predicted target values.
    """
    plt.figure(figsize=(12,6))
    plt.plot(range(len(y_test)), y_test, label='Actual')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted')
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

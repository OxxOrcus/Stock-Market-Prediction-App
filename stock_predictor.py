import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Download historical stock data
def fetch_data(ticker, period='5y'):
    data = yf.download(ticker, period=period)
    return data

# 2. Prepare features and target
def prepare_data(data):
    data = data[['Close']].copy()
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()
    X = data[['Close']].values
    y = data['Target'].values
    return X, y

# 3. Train/test split
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# 5. Predict and visualize
def plot_predictions(data, y_test, y_pred):
    plt.figure(figsize=(12,6))
    plt.plot(range(len(y_test)), y_test, label='Actual')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted')
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL): ")
    data = fetch_data(ticker)
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_predictions(data, y_test, y_pred)

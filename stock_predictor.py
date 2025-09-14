import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Download historical stock data
def fetch_data(ticker, period='5y'):
    """Fetches historical stock data from Yahoo Finance."""
    data = yf.download(ticker, period=period)
    return data

# 2. Prepare features and target
def prepare_data(data, time_step=60):
    """Prepares the data for training the LSTM model."""
    data_close = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data_close)

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# 3. Train/test split
def split_data(X, y):
    """Splits the data into training and testing sets."""
    return train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Build and Train model
def build_and_train_model(X_train, y_train):
    """Builds and trains the LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1)
    return model

# 5. Predict and visualize
def plot_predictions(y_test, y_pred, scaler):
    """Plots the actual vs. predicted stock prices."""
    y_test = y_test.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    y_test = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(y_pred)

    plt.figure(figsize=(12,6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL): ")
    data = fetch_data(ticker)
    X, y, scaler = prepare_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_and_train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred, scaler)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from stock_predictor import fetch_data, prepare_data, split_data, build_and_train_model, plot_predictions

def test_fetch_data(mocker):
    # Mock the yfinance.download function
    mock_download = mocker.patch('yfinance.download')

    # Create a sample DataFrame to be returned by the mock
    sample_data = pd.DataFrame({
        'Open': [100, 102, 101],
        'High': [103, 104, 102],
        'Low': [99, 101, 100],
        'Close': [102, 103, 101],
        'Volume': [1000, 1200, 1100]
    })
    mock_download.return_value = sample_data

    # Call the function to be tested
    ticker = 'AAPL'
    data = fetch_data(ticker)

    # Assert that the mock was called with the correct ticker
    mock_download.assert_called_once_with(ticker, period='5y')

    # Assert that the function returns the mocked data
    pd.testing.assert_frame_equal(data, sample_data)

def test_prepare_data():
    # Create a sample DataFrame
    sample_data = pd.DataFrame({
        'Close': np.arange(100, 200)
    })

    # Call the function to be tested
    X, y, scaler = prepare_data(sample_data, time_step=60)

    # Assert the shapes of the output
    assert X.shape == (40, 60, 1)
    assert y.shape == (40,)
    assert isinstance(scaler, MinMaxScaler)

def test_split_data():
    # Create sample data
    X = np.random.rand(100, 60, 1)
    y = np.random.rand(100)

    # Call the function
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Assert the shapes of the output
    assert X_train.shape == (80, 60, 1)
    assert X_test.shape == (20, 60, 1)
    assert y_train.shape == (80,)
    assert y_test.shape == (20,)

def test_build_and_train_model():
    # Create sample data
    X_train = np.random.rand(80, 60, 1)
    y_train = np.random.rand(80)

    # Call the function
    model = build_and_train_model(X_train, y_train)

    # Assert that the model is a Sequential instance
    assert isinstance(model, Sequential)

    # Assert that the model is compiled
    assert model.optimizer is not None
    assert model.loss is not None

def test_plot_predictions(mocker):
    # Mock matplotlib.pyplot
    mock_plt = mocker.patch('stock_predictor.plt')

    # Create sample data
    y_test = np.random.rand(20)
    y_pred = np.random.rand(20)
    scaler = MinMaxScaler()
    scaler.fit(np.arange(100,200).reshape(-1, 1))

    # Call the function
    plot_predictions(y_test, y_pred, scaler)

    # Assert that the plot functions were called
    mock_plt.figure.assert_called_once_with(figsize=(12,6))
    assert mock_plt.plot.call_count == 2
    mock_plt.legend.assert_called_once()
    mock_plt.title.assert_called_once_with('Stock Price Prediction')
    mock_plt.xlabel.assert_called_once_with('Time')
    mock_plt.ylabel.assert_called_once_with('Price')
    mock_plt.show.assert_called_once()

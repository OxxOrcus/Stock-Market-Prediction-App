import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from stock_predictor import prepare_data, split_data, train_model, plot_predictions

def test_prepare_data():
    # Create a sample DataFrame
    sample_data = pd.DataFrame({
        'Close': [100, 102, 105, 103]
    })

    # Call the function to be tested
    X, y = prepare_data(sample_data)

    # Define the expected output
    expected_X = np.array([[100], [102], [105]])
    expected_y = np.array([102, 105, 103])

    # Assert that the output is correct
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)

def test_split_data():
    # Create sample data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])

    # Call the function
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Assert the shapes of the output
    assert X_train.shape == (4, 1)
    assert X_test.shape == (1, 1)
    assert y_train.shape == (4,)
    assert y_test.shape == (1,)

    # Assert the content of the output (no shuffling)
    np.testing.assert_array_equal(X_train, np.array([[1], [2], [3], [4]]))
    np.testing.assert_array_equal(X_test, np.array([[5]]))
    np.testing.assert_array_equal(y_train, np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(y_test, np.array([5]))

def test_train_model():
    # Create sample data
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([1, 2, 3, 4])

    # Call the function
    model = train_model(X_train, y_train)

    # Assert that the model is a LinearRegression instance
    assert isinstance(model, LinearRegression)

    # Assert that the model is fitted
    assert hasattr(model, 'coef_')

def test_plot_predictions(mocker):
    # Mock matplotlib.pyplot
    mock_plt = mocker.patch('stock_predictor.plt')

    # Create sample data
    sample_data = pd.DataFrame({'Close': [1, 2, 3, 4, 5]})
    y_test = np.array([4, 5])
    y_pred = np.array([4.1, 5.1])

    # Call the function
    plot_predictions(sample_data, y_test, y_pred)

    # Assert that the plot functions were called
    mock_plt.figure.assert_called_once_with(figsize=(12,6))
    assert mock_plt.plot.call_count == 2
    mock_plt.legend.assert_called_once()
    mock_plt.title.assert_called_once_with('Stock Price Prediction')
    mock_plt.xlabel.assert_called_once_with('Time')
    mock_plt.ylabel.assert_called_once_with('Price')
    mock_plt.show.assert_called_once()

# Stock Market Prediction App

This application predicts stock prices using historical data and machine learning models. It is designed for educational and prototyping purposes, providing a simple interface to fetch stock data, train a model, and visualize predictions.

## Features

- Fetches historical stock data using Yahoo Finance (`yfinance`)
- Preprocesses and prepares data for modeling
- Trains a simple Linear Regression model (using `scikit-learn`)
- Predicts future stock prices
- Visualizes actual vs. predicted prices with `matplotlib`
- Easily extensible for more advanced models (e.g., LSTM, portfolio simulation)

## Requirements

- Python 3.8+
- Packages: scikit-learn, pandas, numpy, matplotlib, yfinance

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/OxxOrcus/Stock-Market-Prediction-App.git
   cd Stock-Market-Prediction-App
   ```

2. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the script:

   ```bash
   python stock_predictor.py
   ```

2. Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA) when prompted.
3. The script will fetch data, train the model, and display a plot of actual vs. predicted prices.

## Example

```
Enter stock ticker (e.g., AAPL): AAPL
[*********************100%***********************]  1 of 1 completed
```

A plot window will appear showing the comparison.

## Project Structure

- `stock_predictor.py` — Main script for data fetching, modeling, and visualization
- `requirements.txt` — List of required Python packages
- `README.md` — Project documentation

## Extending the App

- Replace Linear Regression with more advanced models (e.g., LSTM using TensorFlow or PyTorch)
- Add real-time data visualization (e.g., with Plotly Dash or Streamlit)
- Implement portfolio simulation and backtesting
- Integrate more features for technical analysis

## License

This project is for educational purposes. See LICENSE for more details.

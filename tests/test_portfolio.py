import pandas as pd
import numpy as np
from portfolio import Portfolio, simulate_trading
from sklearn.linear_model import LinearRegression

def test_portfolio_buy():
    portfolio = Portfolio(initial_capital=1000)
    assert portfolio.buy('AAPL', 150, 5)
    assert portfolio.cash == 250
    assert portfolio.holdings['AAPL'] == 5

def test_portfolio_sell():
    portfolio = Portfolio(initial_capital=1000)
    portfolio.buy('AAPL', 150, 5)
    assert portfolio.sell('AAPL', 160, 3)
    assert portfolio.cash == 250 + 160 * 3
    assert portfolio.holdings['AAPL'] == 2

def test_portfolio_get_value():
    portfolio = Portfolio(initial_capital=1000)
    portfolio.buy('AAPL', 150, 5)
    prices = {'AAPL': 160}
    assert portfolio.get_value(prices) == 250 + 160 * 5

def test_simulate_trading(mocker):
    # Create a mock model
    mock_model = mocker.MagicMock(spec=LinearRegression)
    mock_model.predict.side_effect = [[102], [103], [101]]

    # Create sample data
    sample_data = pd.DataFrame({
        'Close': [100, 102, 105, 103]
    })

    # Call the function
    portfolio_values = simulate_trading(mock_model, sample_data)

    # Assert the output
    assert isinstance(portfolio_values, pd.DataFrame)
    assert 'value' in portfolio_values.columns
    assert len(portfolio_values) == len(sample_data) - 1

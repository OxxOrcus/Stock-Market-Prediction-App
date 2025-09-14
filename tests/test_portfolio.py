import pandas as pd
import numpy as np
from portfolio import Portfolio
from datetime import datetime

def test_portfolio_buy():
    portfolio = Portfolio(initial_capital=1000)
    assert portfolio.buy('AAPL', 150, 5)
    assert portfolio.cash == 250
    assert portfolio.positions['AAPL'] == 5

def test_portfolio_sell():
    portfolio = Portfolio(initial_capital=1000)
    portfolio.buy('AAPL', 150, 5)
    assert portfolio.sell('AAPL', 160, 3)
    assert portfolio.cash == 250 + 160 * 3
    assert portfolio.positions['AAPL'] == 2

def test_portfolio_get_total_value():
    portfolio = Portfolio(initial_capital=1000)
    portfolio.buy('AAPL', 150, 5)
    prices = pd.Series({'AAPL': 160})
    assert portfolio.get_total_value(prices) == 250 + 160 * 5

def test_update_holdings_history():
    portfolio = Portfolio(initial_capital=1000)
    portfolio.buy('AAPL', 150, 5)
    date = datetime(2023, 1, 1)
    prices = pd.Series({'AAPL': 160})
    portfolio.update_holdings_history(date, prices)
    assert len(portfolio.holdings_history) == 1
    assert portfolio.holdings_history[0][0] == date
    assert portfolio.holdings_history[0][1]['AAPL'] == 5 * 160
    assert portfolio.holdings_history[0][1]['cash'] == 250
    assert portfolio.holdings_history[0][1]['total'] == 250 + 5 * 160

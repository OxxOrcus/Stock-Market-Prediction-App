import pytest
from portfolio import Portfolio

def test_portfolio_initialization():
    portfolio = Portfolio(initial_cash=50000)
    assert portfolio.initial_cash == 50000
    assert portfolio.cash == 50000
    assert portfolio.holdings == {}
    assert portfolio.transactions == []

def test_buy_stock():
    portfolio = Portfolio(initial_cash=10000)
    assert portfolio.buy('AAPL', 10, 150)
    assert portfolio.cash == 10000 - (10 * 150)
    assert portfolio.holdings == {'AAPL': 10}
    assert len(portfolio.transactions) == 1
    assert portfolio.transactions[0]['side'] == 'BUY'

def test_buy_stock_insufficient_funds():
    portfolio = Portfolio(initial_cash=1000)
    assert not portfolio.buy('AAPL', 10, 150)
    assert portfolio.cash == 1000
    assert portfolio.holdings == {}
    assert len(portfolio.transactions) == 0

def test_sell_stock():
    portfolio = Portfolio(initial_cash=10000)
    portfolio.buy('AAPL', 10, 150)
    assert portfolio.sell('AAPL', 5, 160)
    assert portfolio.cash == 10000 - (10 * 150) + (5 * 160)
    assert portfolio.holdings == {'AAPL': 5}
    assert len(portfolio.transactions) == 2
    assert portfolio.transactions[1]['side'] == 'SELL'

def test_sell_stock_insufficient_holdings():
    portfolio = Portfolio(initial_cash=10000)
    portfolio.buy('AAPL', 10, 150)
    assert not portfolio.sell('GOOG', 5, 200)
    assert not portfolio.sell('AAPL', 15, 160)
    assert portfolio.holdings == {'AAPL': 10}
    assert len(portfolio.transactions) == 1

def test_get_value():
    portfolio = Portfolio(initial_cash=10000)
    portfolio.buy('AAPL', 10, 150)
    portfolio.buy('GOOG', 5, 200)
    current_prices = {'AAPL': 160, 'GOOG': 210}
    expected_value = (10000 - 10*150 - 5*200) + (10 * 160) + (5 * 210)
    assert portfolio.get_value(current_prices) == expected_value

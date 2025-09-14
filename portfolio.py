import pandas as pd
import numpy as np

class Portfolio:
    """
    Represents a trading portfolio.
    """
    def __init__(self, initial_capital=100000):
        """
        Initializes the portfolio.
        Args:
            initial_capital (float): The starting capital.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.history = []
        self.holdings_history = []

    def buy(self, ticker, price, quantity):
        """
        Buys a stock.
        """
        cost = price * quantity
        if self.cash >= cost:
            self.cash -= cost
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
            self.history.append({'action': 'buy', 'ticker': ticker, 'price': price, 'quantity': quantity})
            return True
        return False

    def sell(self, ticker, price, quantity):
        """
        Sells a stock.
        """
        if self.positions.get(ticker, 0) >= quantity:
            self.cash += price * quantity
            self.positions[ticker] -= quantity
            if self.positions[ticker] == 0:
                del self.positions[ticker]
            self.history.append({'action': 'sell', 'ticker': ticker, 'price': price, 'quantity': quantity})
            return True
        return False

    def get_total_value(self, prices):
        """
        Calculates the total value of the portfolio for a given set of prices.
        Args:
            prices (pd.Series): A pandas Series with tickers as index and prices as values.
        """
        value = self.cash
        for ticker, quantity in self.positions.items():
            value += prices[ticker] * quantity
        return value

    def update_holdings_history(self, date, prices):
        """
        Records the daily holdings and cash.
        Args:
            date (datetime): The current date.
            prices (pd.Series): A pandas Series with tickers as index and prices as values.
        """
        holdings = {}
        for ticker, quantity in self.positions.items():
            holdings[ticker] = quantity * prices[ticker]
        holdings['cash'] = self.cash
        holdings['total'] = self.get_total_value(prices)
        self.holdings_history.append((date, holdings))

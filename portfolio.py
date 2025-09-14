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
        self.holdings = {}
        self.history = []

    def buy(self, ticker, price, quantity):
        """
        Buys a stock.
        """
        cost = price * quantity
        if self.cash >= cost:
            self.cash -= cost
            self.holdings[ticker] = self.holdings.get(ticker, 0) + quantity
            self.history.append({'action': 'buy', 'ticker': ticker, 'price': price, 'quantity': quantity})
            return True
        return False

    def sell(self, ticker, price, quantity):
        """
        Sells a stock.
        """
        if self.holdings.get(ticker, 0) >= quantity:
            self.cash += price * quantity
            self.holdings[ticker] -= quantity
            if self.holdings[ticker] == 0:
                del self.holdings[ticker]
            self.history.append({'action': 'sell', 'ticker': ticker, 'price': price, 'quantity': quantity})
            return True
        return False

    def get_value(self, prices):
        """
        Calculates the total value of the portfolio.
        """
        value = self.cash
        for ticker, quantity in self.holdings.items():
            value += prices[ticker] * quantity
        return value

def simulate_trading(model, data, initial_capital=100000):
    """
    Simulates a trading strategy based on the model's predictions.

    Args:
        model: The trained machine learning model.
        data (pandas.DataFrame): The historical stock data.
        initial_capital (float): The starting capital for the portfolio.

    Returns:
        pandas.DataFrame: A DataFrame containing the portfolio's value over time.
    """
    portfolio = Portfolio(initial_capital)
    portfolio_values = []

    for i in range(len(data) - 1):
        current_price = data['Close'].iloc[i]
        prediction = model.predict([[current_price]])[0]

        if prediction > current_price:
            # Buy signal
            portfolio.buy('stock', current_price, 1)
        elif prediction < current_price and portfolio.holdings.get('stock', 0) > 0:
            # Sell signal
            portfolio.sell('stock', current_price, 1)

        portfolio_values.append(portfolio.get_value({'stock': current_price}))

    return pd.DataFrame({'value': portfolio_values}, index=data.index[:-1])

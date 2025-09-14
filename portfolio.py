import pandas as pd

class Portfolio:
    """
    Manages a trading portfolio, including cash, stock holdings, and transactions.
    """
    def __init__(self, initial_cash=100000):
        """
        Initializes the portfolio with a starting cash balance.
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {}  # {ticker: quantity}
        self.transactions = []  # List of transaction dicts

    def buy(self, ticker, quantity, price):
        """
        Executes a stock purchase and updates portfolio holdings.
        """
        cost = quantity * price
        if self.cash >= cost:
            self.cash -= cost
            self.holdings[ticker] = self.holdings.get(ticker, 0) + quantity
            self._record_transaction('BUY', ticker, quantity, price)
            return True
        return False

    def sell(self, ticker, quantity, price):
        """
        Executes a stock sale and updates portfolio holdings.
        """
        if self.holdings.get(ticker, 0) >= quantity:
            self.cash += quantity * price
            self.holdings[ticker] -= quantity
            if self.holdings[ticker] == 0:
                del self.holdings[ticker]
            self._record_transaction('SELL', ticker, quantity, price)
            return True
        return False

    def _record_transaction(self, side, ticker, quantity, price):
        """
        Records a transaction to the transaction log.
        """
        self.transactions.append({
            'side': side,
            'ticker': ticker,
            'quantity': quantity,
            'price': price,
            'timestamp': pd.Timestamp.now()
        })

    def get_value(self, current_prices):
        """
        Calculates the total current value of the portfolio.
        """
        stocks_value = sum(self.holdings.get(ticker, 0) * current_prices.get(ticker, 0) for ticker in self.holdings)
        return self.cash + stocks_value

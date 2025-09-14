import pandas as pd

class Backtester:
    """
    Orchestrates a backtest of a trading strategy.
    """
    def __init__(self, portfolio, strategy, data):
        """
        Initializes the backtester.
        """
        self.portfolio = portfolio
        self.strategy = strategy
        self.data = data
        self.signals = self.strategy.generate_signals(self.data)
        self.results = pd.DataFrame(index=self.data.index)

    def run(self):
        """
        Runs the backtest.
        """
        for i in range(len(self.data)):
            if self.signals['positions'].iloc[i] == 1.0:
                # 'Buy' signal
                self.portfolio.buy(
                    ticker='AAPL',  # Hardcoded for now
                    quantity=100,  # Hardcoded for now
                    price=self.data['Close'].iloc[i]
                )
            elif self.signals['positions'].iloc[i] == -1.0:
                # 'Sell' signal
                self.portfolio.sell(
                    ticker='AAPL',  # Hardcoded for now
                    quantity=100,  # Hardcoded for now
                    price=self.data['Close'].iloc[i]
                )
            self._update_results(i)
        return self.results

    def _update_results(self, i):
        """
        Updates the results DataFrame with the portfolio's current state.
        """
        current_prices = {'AAPL': self.data['Close'].iloc[i]}  # Hardcoded for now
        self.results.at[self.data.index[i], 'portfolio_value'] = self.portfolio.get_value(current_prices)
        self.results.at[self.data.index[i], 'cash'] = self.portfolio.cash

    def get_summary(self):
        """
        Returns a summary of the backtest performance.
        """
        total_return = (self.results['portfolio_value'].iloc[-1] / self.portfolio.initial_cash) - 1
        return {
            'total_return': total_return,
            'final_portfolio_value': self.results['portfolio_value'].iloc[-1]
        }

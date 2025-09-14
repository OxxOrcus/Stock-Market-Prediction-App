import pandas as pd
import yfinance as yf
from portfolio import Portfolio
from strategy import Strategy

class Backtest:
    """
    A class to run a backtest of a trading strategy.
    """
    def __init__(self, portfolio: Portfolio, strategy: Strategy, tickers: list, start_date: str, end_date: str):
        """
        Initializes the Backtest.

        Args:
            portfolio (Portfolio): The portfolio to use for the backtest.
            strategy (Strategy): The trading strategy to use.
            tickers (list): A list of stock tickers to trade.
            start_date (str): The start date of the backtest (YYYY-MM-DD).
            end_date (str): The end date of the backtest (YYYY-MM-DD).
        """
        self.portfolio = portfolio
        self.strategy = strategy
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._fetch_data()

    def _fetch_data(self):
        """
        Fetches historical data for the given tickers.

        Returns:
            pd.DataFrame: A DataFrame with historical data for the tickers.
        """
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        return data

    def run(self):
        """
        Runs the backtest.
        """
        signals = {}
        for ticker in self.tickers:
            ticker_data = self.data['Close'][ticker].to_frame(name='Close')
            signals[ticker] = self.strategy.generate_signals(ticker_data)

        for date in self.data.index:
            prices_for_day = self.data.loc[date]
            for ticker in self.tickers:
                if ticker in signals and date in signals[ticker].index:
                    signal = signals[ticker].loc[date]
                    if signal['positions'] == 1.0:
                        self.portfolio.buy(ticker, prices_for_day['Close'][ticker], 1)
                    elif signal['positions'] == -1.0:
                        if self.portfolio.positions.get(ticker, 0) > 0:
                            self.portfolio.sell(ticker, prices_for_day['Close'][ticker], 1)

            # Update portfolio holdings history for the day
            # We need to handle the case where a ticker might not have a price for a given day
            daily_prices = {}
            for ticker in self.portfolio.positions.keys():
                if not pd.isna(prices_for_day['Close'][ticker]):
                    daily_prices[ticker] = prices_for_day['Close'][ticker]

            if daily_prices:
                self.portfolio.update_holdings_history(date, pd.Series(daily_prices))

    def get_performance_summary(self):
        """
        Returns a summary of the portfolio's performance.
        """
        holdings_df = pd.DataFrame([h for d, h in self.portfolio.holdings_history], index=[d for d, h in self.portfolio.holdings_history])

        total_return = (holdings_df['total'].iloc[-1] / self.portfolio.initial_capital) - 1

        # Annualized return
        days = (holdings_df.index[-1] - holdings_df.index[0]).days
        annualized_return = (1 + total_return) ** (365.0 / days) - 1

        # Sharpe ratio
        returns = holdings_df['total'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) # Assuming 252 trading days in a year

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio
        }

    def plot_performance(self):
        """
        Plots the portfolio's performance over time.
        """
        import matplotlib.pyplot as plt

        holdings_df = pd.DataFrame([h for d, h in self.portfolio.holdings_history], index=[d for d, h in self.portfolio.holdings_history])

        plt.figure(figsize=(12, 6))
        plt.plot(holdings_df.index, holdings_df['total'])
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.show()

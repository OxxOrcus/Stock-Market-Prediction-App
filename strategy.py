import pandas as pd
import numpy as np

class Strategy:
    """
    Base class for a trading strategy.
    """
    def generate_signals(self, data):
        """
        Generates trading signals for a given dataset.

        Args:
            data (pd.DataFrame): A DataFrame with historical price data for a single ticker.

        Returns:
            pd.DataFrame: A DataFrame with signals.
        """
        raise NotImplementedError("Should implement generate_signals()")


class MovingAverageCrossoverStrategy(Strategy):
    """
    A strategy that uses two moving averages to generate trading signals.
    """
    def __init__(self, short_window=40, long_window=100):
        """
        Initializes the MovingAverageCrossoverStrategy.

        Args:
            short_window (int): The short moving average window.
            long_window (int): The long moving average window.
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        """
        Generates trading signals based on the moving average crossover.

        Args:
            data (pd.DataFrame): A DataFrame with historical price data for a single ticker.
                                 It must contain a 'Close' column.

        Returns:
            pd.DataFrame: A DataFrame with signals for the ticker.
                          The 'positions' column contains the trading signals (1 for buy, -1 for sell).
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        # Create short simple moving average on the 'Close' price
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window, center=False).mean()

        # Create long simple moving average
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window, center=False).mean()

        # Create signals
        signals.loc[signals.index[self.short_window:], 'signal'] = np.where(signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1.0, 0.0)

        # Generate trading orders
        signals['positions'] = signals['signal'].diff()

        return signals

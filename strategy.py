import pandas as pd
import numpy as np

class Strategy:
    """
    Base class for a trading strategy.
    """
    def generate_signals(self, data):
        """
        Generates trading signals for the given data.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class MovingAverageCrossoverStrategy(Strategy):
    """
    A strategy based on the crossover of two moving averages.
    """
    def __init__(self, short_window=40, long_window=100):
        """
        Initializes the strategy with short and long window sizes.
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        """
        Generates trading signals based on moving average crossover.
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        # Create short and long simple moving averages
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window, min_periods=1, center=False).mean()

        # Generate signal when short moving average crosses above long moving average
        signals.loc[signals.index[self.short_window:], 'signal'] = \
            np.where(signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1.0, 0.0)

        # Generate trading orders
        signals['positions'] = signals['signal'].diff()
        return signals

import pandas as pd
import numpy as np
from strategy import MovingAverageCrossoverStrategy

def test_moving_average_crossover_strategy():
    # Create more realistic sample data with a crossover
    t = np.linspace(0, 10, 100)
    price = 150 + 10 * np.sin(t) + 5 * np.cos(2 * t)
    data = pd.DataFrame({
        'Close': price
    }, index=pd.to_datetime(pd.date_range('2022-01-01', periods=100)))

    # Initialize the strategy
    strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)

    # Generate signals
    signals = strategy.generate_signals(data)

    # Assert the output shape and columns
    assert signals.shape == (100, 4)
    assert 'signal' in signals.columns
    assert 'short_mavg' in signals.columns
    assert 'long_mavg' in signals.columns
    assert 'positions' in signals.columns

    # Assert that there are buy and sell signals
    assert 1.0 in signals['positions'].values
    assert -1.0 in signals['positions'].values

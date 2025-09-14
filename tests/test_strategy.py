import pandas as pd
import numpy as np
import pytest
from strategy import Strategy, MovingAverageCrossoverStrategy

def test_strategy_base_class():
    strategy = Strategy()
    with pytest.raises(NotImplementedError):
        strategy.generate_signals(None)

def test_moving_average_crossover_strategy():
    # Create sample data with a clear crossover event
    data = pd.DataFrame({
        'Close': [110, 105, 100, 95, 90, 85, 90, 95, 100, 105, 110, 115]
    }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=12)))

    # Create a strategy with short and long windows
    strategy = MovingAverageCrossoverStrategy(short_window=3, long_window=6)
    signals = strategy.generate_signals(data)

    # Assert the columns are correct
    assert 'signal' in signals.columns
    assert 'short_mavg' in signals.columns
    assert 'long_mavg' in signals.columns
    assert 'positions' in signals.columns

    # The crossover should happen on 2023-01-09
    # short mavg on 2023-01-08 is (85+90+95)/3 = 90
    # long mavg on 2023-01-08 is (100+95+90+85+90+95)/6 = 92.5
    # signal is 0
    # short mavg on 2023-01-09 is (90+95+100)/3 = 95
    # long mavg on 2023-01-09 is (95+90+85+90+95+100)/6 = 92.5
    # signal is 1
    # So, positions should be 1 on 2023-01-09
    assert signals['positions'].loc['2023-01-09'] == 1.0
    assert signals['positions'].loc['2023-01-10'] == 0.0

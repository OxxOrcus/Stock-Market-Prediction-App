import pytest
import pandas as pd
import numpy as np
from portfolio import Portfolio
from strategy import MovingAverageCrossoverStrategy
from backtester import Backtester

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Close': np.arange(100, 200)
    }, index=pd.to_datetime(pd.date_range('2022-01-01', periods=100)))

def test_backtester_run(sample_data):
    portfolio = Portfolio()
    strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)
    backtester = Backtester(portfolio, strategy, sample_data)
    results = backtester.run()

    assert isinstance(results, pd.DataFrame)
    assert 'portfolio_value' in results.columns
    assert 'cash' in results.columns
    assert not results.empty

def test_backtester_summary(sample_data):
    portfolio = Portfolio()
    strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)
    backtester = Backtester(portfolio, strategy, sample_data)
    backtester.run()
    summary = backtester.get_summary()

    assert 'total_return' in summary
    assert 'final_portfolio_value' in summary
    assert isinstance(summary['total_return'], float)
    assert isinstance(summary['final_portfolio_value'], float)

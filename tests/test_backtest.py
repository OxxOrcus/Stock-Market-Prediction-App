import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock
from portfolio import Portfolio
from strategy import Strategy
from backtest import Backtest

@pytest.fixture
def mock_portfolio():
    portfolio = MagicMock(spec=Portfolio)
    portfolio.positions = MagicMock()
    return portfolio

@pytest.fixture
def mock_strategy():
    strategy = MagicMock(spec=Strategy)
    signals = pd.DataFrame({
        'positions': [0.0, 1.0, 0.0, -1.0, 0.0]
    }, index=pd.to_datetime(pd.date_range('2023-01-01', periods=5)))
    strategy.generate_signals.return_value = signals
    return strategy

@pytest.fixture
def sample_data():
    dates = pd.to_datetime(pd.date_range('2023-01-01', periods=5))
    aapl_close = [100, 102, 105, 103, 106]
    goog_close = [200, 202, 205, 203, 206]

    df = pd.DataFrame(index=dates)
    df[('Close', 'AAPL')] = aapl_close
    df[('Close', 'GOOG')] = goog_close
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_backtest_initialization(mocker, mock_portfolio, mock_strategy):
    tickers = ['AAPL', 'GOOG']
    start_date = '2023-01-01'
    end_date = '2023-01-05'
    mocker.patch('backtest.yf.download', return_value=pd.DataFrame())

    backtest = Backtest(mock_portfolio, mock_strategy, tickers, start_date, end_date)

    assert backtest.portfolio == mock_portfolio
    assert backtest.strategy == mock_strategy
    assert backtest.tickers == tickers
    assert backtest.start_date == start_date
    assert backtest.end_date == end_date

def test_run_executes_trades(mocker, mock_portfolio, mock_strategy, sample_data):
    tickers = ['AAPL']
    start_date = '2023-01-01'
    end_date = '2023-01-05'

    mocker.patch('backtest.yf.download', return_value=sample_data)

    backtest = Backtest(mock_portfolio, mock_strategy, tickers, start_date, end_date)

    mock_portfolio.positions.get.return_value = 1

    backtest.run()

    buy_call = mock_portfolio.buy.call_args
    assert buy_call.args[0] == 'AAPL'
    assert buy_call.args[1] == 102
    assert buy_call.args[2] == 1

    sell_call = mock_portfolio.sell.call_args
    assert sell_call.args[0] == 'AAPL'
    assert sell_call.args[1] == 103
    assert sell_call.args[2] == 1

def test_get_performance_summary(mocker):
    portfolio = Portfolio(initial_capital=100000)
    portfolio.holdings_history = [
        (pd.to_datetime('2023-01-01'), {'total': 100000}),
        (pd.to_datetime('2023-01-02'), {'total': 101000}),
        (pd.to_datetime('2023-01-03'), {'total': 102000}),
    ]
    mocker.patch('backtest.yf.download', return_value=pd.DataFrame())
    backtest = Backtest(portfolio, MagicMock(), [], '2023-01-01', '2023-01-03')

    backtest.data = pd.DataFrame(index=pd.to_datetime(pd.date_range('2023-01-01', periods=3)))

    summary = backtest.get_performance_summary()

    assert 'total_return' in summary
    assert 'annualized_return' in summary
    assert 'sharpe_ratio' in summary
    assert np.isclose(summary['total_return'], 0.02)

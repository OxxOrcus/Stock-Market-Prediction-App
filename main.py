from portfolio import Portfolio
from strategy import MovingAverageCrossoverStrategy
from backtest import Backtest

if __name__ == "__main__":
    # -- Backtest Configuration --
    initial_capital = 100000.0
    tickers = ['AAPL', 'GOOG', 'MSFT']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    # --------------------------

    # Create a portfolio
    portfolio = Portfolio(initial_capital)

    # Create a strategy
    strategy = MovingAverageCrossoverStrategy(short_window=40, long_window=100)

    # Create a backtest
    backtest = Backtest(portfolio, strategy, tickers, start_date, end_date)

    # Run the backtest
    backtest.run()

    # Get performance summary
    performance_summary = backtest.get_performance_summary()
    print("--- Performance Summary ---")
    print(f"Total Return: {performance_summary['total_return']:.2%}")
    print(f"Annualized Return: {performance_summary['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {performance_summary['sharpe_ratio']:.2f}")
    print("---------------------------")

    # Plot performance
    backtest.plot_performance()

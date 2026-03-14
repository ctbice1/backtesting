import sys

import numpy as np
import pandas as pd

from backtesting.core.portfolio import Portfolio
from backtesting.core.simulate import parallel
from backtesting.core.config import get_shared_test_config, load_yaml_config
from backtesting.core.dates import (
    bound_to_available_dates,
    coerce_date,
    slice_history,
)
from backtesting.data import get_historical_data
from backtesting.performance import (
    portfolio_performance_summary,
    portfolio_values_from_share_history,
    risk_free_rate_from_irx,
)


def test_strategy(
    strategy_config: tuple,
    start_date: np.datetime64,
    historical_data: tuple,
    dates: tuple,
    test_parameters: tuple,
    trace: bool = False,
    track: bool = False,
):
    """Tests a strategy with the provided parameters and data."""
    strategy_cls, strategy_args, securities = strategy_config
    allocation_schedule, initial_allocation, yearly_allocation = test_parameters

    tickers, weights = zip(*securities)
    portfolio = Portfolio(tickers, weights)

    strategy = strategy_cls(
        initial_allocation=initial_allocation,
        allocation_schedule=allocation_schedule,
        yearly_allocation=yearly_allocation,
        start_date=start_date,
        track=track,
        **strategy_args,
    )

    final_portfolio, parameters = strategy.execute(historical_data, dates, portfolio, trace)
    return final_portfolio, parameters, strategy


def single_test(config: dict):
    """Runs a single configured test."""
    try:
        historical_data = get_historical_data((security[0] for security in config["securities"]))
    except RuntimeError as error:
        print(error)
        sys.exit(-1)

    (price_history_dates, price_history), (distribution_history_dates, distribution_history) = historical_data
    requested_start, requested_stop = config["dates"]

    start_date = coerce_date(requested_start, price_history_dates[0])
    stop_date = coerce_date(requested_stop, price_history_dates[-1])
    start_date = bound_to_available_dates(start_date, price_history_dates, upper_bound=False)
    stop_date = bound_to_available_dates(stop_date, price_history_dates, upper_bound=True)

    price_history_dates, price_history = slice_history(price_history_dates, price_history, start_date, stop_date)
    if distribution_history is not None:
        distribution_history_dates, distribution_history = slice_history(
            distribution_history_dates, distribution_history, start_date, stop_date
        )

    def _series_for_ticker(
        ticker: str,
        primary_tickers: tuple[str, ...],
        primary_dates: np.ndarray,
        primary_prices: np.ndarray,
        secondary_tickers: tuple[str, ...] = (),
        secondary_dates: np.ndarray | None = None,
        secondary_prices: np.ndarray | None = None,
    ) -> pd.Series:
        index = pd.to_datetime(primary_dates)
        if ticker in primary_tickers:
            ticker_index = primary_tickers.index(ticker)
            return pd.Series(primary_prices[:, ticker_index], index=index, dtype=float)

        if secondary_dates is None or secondary_prices is None or ticker not in secondary_tickers:
            return pd.Series(dtype=float)

        ticker_index = secondary_tickers.index(ticker)
        secondary_index = pd.to_datetime(secondary_dates)
        secondary_series = pd.Series(secondary_prices[:, ticker_index], index=secondary_index, dtype=float)
        return secondary_series.reindex(index).dropna()

    print(
        f"Time period: {price_history_dates[0].item().strftime('%A %B %d, %Y')} "
        f"to {price_history_dates[-1].item().strftime('%A %B %d, %Y')}"
    )

    strategy_config = (config["strategy"], config["strategy_args"], config["securities"])
    portfolio_tickers = tuple(ticker for ticker, _ in config["securities"])
    benchmark_ticker = config["benchmark_ticker"]
    risk_free_ticker = config["risk_free_ticker"]

    secondary_tickers = []
    for ticker in (benchmark_ticker, risk_free_ticker):
        if ticker and ticker not in portfolio_tickers and ticker not in secondary_tickers:
            secondary_tickers.append(ticker)

    secondary_price_dates = None
    secondary_price_history = None
    if secondary_tickers:
        try:
            (secondary_price_dates, secondary_price_history), _ = get_historical_data(tuple(secondary_tickers))
            secondary_price_dates, secondary_price_history = slice_history(
                secondary_price_dates,
                secondary_price_history,
                start_date,
                stop_date,
            )
        except RuntimeError as error:
            print(error)
            sys.exit(-1)

    allocation_schedule = config["allocation_schedule"]
    test_parameters = (allocation_schedule, config["initial_allocation"], config["yearly_allocation"])

    portfolio, strategy_parameters, strategy = test_strategy(
        strategy_config,
        start_date,
        (price_history, distribution_history),
        (price_history_dates, distribution_history_dates),
        test_parameters,
        trace=config["trace"],
        track=config["track_performance"],
    )

    final_prices = price_history[(price_history_dates == stop_date)][0]
    effective_allocation_schedule = strategy_parameters.get("allocation_schedule", allocation_schedule)
    portfolio_values = portfolio_values_from_share_history(
        strategy.share_history,
        price_history,
        price_history_dates,
    )
    benchmark_values = _series_for_ticker(
        benchmark_ticker,
        portfolio_tickers,
        price_history_dates,
        price_history,
        tuple(secondary_tickers),
        secondary_price_dates,
        secondary_price_history,
    )
    risk_free_values = _series_for_ticker(
        risk_free_ticker,
        portfolio_tickers,
        price_history_dates,
        price_history,
        tuple(secondary_tickers),
        secondary_price_dates,
        secondary_price_history,
    )
    risk_free_rate = risk_free_rate_from_irx(risk_free_values)
    if np.isnan(risk_free_rate):
        risk_free_rate = 0.0

    summary = portfolio_performance_summary(
        portfolio=portfolio,
        final_prices=final_prices,
        portfolio_values=portfolio_values,
        benchmark_values=benchmark_values,
        risk_free_rate=risk_free_rate,
    )

    def _fmt_metric(value: float, percent: bool = False) -> str:
        if np.isnan(value):
            return "N/A"
        if percent:
            return f"{value:.2%}"
        return f"{value:.4f}"

    print(f"{strategy_config[0].__name__} strategy")
    if effective_allocation_schedule is None:
        print("No allocation schedule configured.")
    else:
        print(
            f"{effective_allocation_schedule.fmt} {effective_allocation_schedule.value} allocation schedule."
        )
    print(f"Final balance: ${summary['final_balance']:,.2f}")
    print(f"Net new capital: ${summary['net_new_capital']:,.2f}")
    print(f"Distributions: ${summary['distributions']:,.2f}")
    print(f"Net profit: ${summary['net_profit']:,.2f}")
    print(f"Total return: {_fmt_metric(summary['total_return'], percent=True)}")
    print(f"Benchmark: {benchmark_ticker}")
    print(f"Risk-free source: {risk_free_ticker} ({_fmt_metric(risk_free_rate, percent=True)})")
    print(f"Sortino ratio: {_fmt_metric(summary['sortino_ratio'])}")
    print(f"Treynor ratio: {_fmt_metric(summary['treynor_ratio'])}")
    print(f"Alpha: {_fmt_metric(summary['alpha'], percent=True)}")
    print(f"Beta: {_fmt_metric(summary['beta'])}")
    if len(portfolio_values) == 0:
        print("Risk metrics unavailable: no tracked portfolio return series was produced.")
    print()

def main():
    if len(sys.argv) < 2:
        print("Please provide the path to a configuration file.")
        sys.exit(-1)

    config = load_yaml_config(sys.argv[1])
    tests = config.get("test", {})
    if len(tests) == 0:
        print("Please provide at least one test configuration.")
        sys.exit(-1)

    for test_type, test_configs in tests.items():
        if test_type == "single":
            for single_test_config in test_configs:
                single_test(get_shared_test_config(single_test_config, test_type="single"))
        elif test_type == "simulate":
            for simulation_config in test_configs:
                parallel(get_shared_test_config(simulation_config, test_type="simulate"))
        else:
            print(f"Invalid test type: {test_type}")

    sys.exit()


if __name__ == "__main__":
    main()
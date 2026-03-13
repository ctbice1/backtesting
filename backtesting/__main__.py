import sys

import numpy as np

from backtesting.core.portfolio import Portfolio
from backtesting.core.simulate import parallel
from backtesting.core.util import (
    bound_to_available_dates,
    coerce_date,
    get_historical_data,
    get_shared_test_config,
    load_yaml_config,
    slice_history,
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

    return strategy.execute(historical_data, dates, portfolio, trace)


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

    print(
        f"Time period: {price_history_dates[0].item().strftime('%A %B %d, %Y')} "
        f"to {price_history_dates[-1].item().strftime('%A %B %d, %Y')}"
    )

    strategy_config = (config["strategy"], config["strategy_args"], config["securities"])
    allocation_schedule = config["allocation_schedule"]
    test_parameters = (allocation_schedule, config["initial_allocation"], config["yearly_allocation"])

    portfolio, strategy_parameters = test_strategy(
        strategy_config,
        start_date,
        (price_history, distribution_history),
        (price_history_dates, distribution_history_dates),
        test_parameters,
        trace=config["trace"],
        track=config["track_performance"],
    )

    final_prices = price_history[(price_history_dates == stop_date)][0]
    final_value = portfolio.current_value(final_prices)
    effective_allocation_schedule = strategy_parameters.get("allocation_schedule", allocation_schedule)

    print(f"{strategy_config[0].__name__} strategy")
    if effective_allocation_schedule is None:
        print("No allocation schedule configured.")
    else:
        print(
            f"{effective_allocation_schedule.fmt} {effective_allocation_schedule.value} allocation schedule."
        )
    print(f"Final balance: ${final_value:,.2f}")
    print(f"Net new capital: ${portfolio.total_new_capital:,.2f}")
    print(f"Distributions: ${portfolio.total_distribution:,.2f}\n")

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
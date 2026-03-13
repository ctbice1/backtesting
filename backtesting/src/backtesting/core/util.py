"""Core utility helpers used by backtesting workflows."""

import heapq
import importlib
import os
import pickle
import pkgutil
import sys
import tempfile
import time
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
import yfinance as yf
import yaml

from backtesting.core.schedule import Schedule, ScheduleFormat

def coerce_date(value: str | None, fallback: np.datetime64) -> np.datetime64:
    """Converts a date-like config value to day precision datetime64."""
    if value is None:
        return np.datetime64(fallback, "D")
    return np.datetime64(value, "D")

def bound_to_available_dates(
    date: np.datetime64, available_dates: np.ndarray, upper_bound: bool
) -> np.datetime64:
    """Bounds a date to the available range and nearest existing index date."""
    if upper_bound:
        bounded = min(date, available_dates[-1])
        index = np.searchsorted(available_dates, bounded, side="right") - 1
    else:
        bounded = max(date, available_dates[0])
        index = np.searchsorted(available_dates, bounded, side="left")
    return available_dates[index]

def slice_history(
    dates: np.ndarray, history: np.ndarray, start_date: np.datetime64, stop_date: np.datetime64
) -> tuple[np.ndarray, np.ndarray]:
    """Slices history and dates to the inclusive date range."""
    mask = (dates >= start_date) & (dates <= stop_date)
    return dates[mask], history[mask]

def load_yaml_config(config_file_path: str) -> dict:
    """Loads YAML config from disk."""
    if not os.path.exists(config_file_path):
        print(f"Configuration file {config_file_path} not found.")
        sys.exit(-1)

    with open(config_file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        print("Configuration file must deserialize to a mapping/object.")
        sys.exit(-1)

    return config

def _normalize_schedule_config(schedule_config: object) -> dict[str, object] | None:
    """Normalizes shorthand and mapping schedule config declarations."""
    if schedule_config is None:
        return None
    if isinstance(schedule_config, str):
        return {"format": schedule_config}
    if not isinstance(schedule_config, dict):
        print(
            f"Invalid schedule configuration: {schedule_config}. "
            "Expected a mapping/object or string."
        )
        sys.exit(-1)
    return schedule_config

def _build_schedule(schedule_config: dict, context: str) -> Schedule:
    """Builds a Schedule from {format, value} config fields."""
    schedule_format = schedule_config.get("format", schedule_config.get("schedule_format", None))
    if schedule_format is None:
        print(f"Missing {context}.schedule.format for static mode.")
        sys.exit(-1)
    try:
        schedule_value = (
            schedule_config["value"]
            if "value" in schedule_config
            else schedule_config.get("schedule_value", None)
        )
        return Schedule(ScheduleFormat[schedule_format], schedule_value)
    except KeyError:
        print(f"Invalid ScheduleFormat: {schedule_format}")
        sys.exit(-1)

def get_shared_test_config(config: dict[str, object], test_type: str | None = None) -> dict[str, object]:
    """Loads shared test configuration parameters."""

    # Check if securities to test were provided
    securities = config.get("securities", [])
    if len(securities) == 0:
        print("Please provide securities to test.")
        sys.exit(-1)

    # Check security format
    for security in securities:
        if "ticker" not in security:
            print(f"Security: {security} must have a ticker symbol specified as: ticker = \"symbol\".")
            sys.exit(-1)
        if "weight" not in security:
            print(f"Security: {security} must have a ticker weight specified as: weight = X.X")
            sys.exit(-1)
        if isinstance(security["weight"], bool):
            print(f"Invalid weight for {security['ticker']}: {security['weight']}. Must be numeric.")
            sys.exit(-1)

    # Check ticker weights
    try:
        total_weight = float(sum(float(security["weight"]) for security in config["securities"]))
    except (TypeError, ValueError):
        print("Invalid securities weights. Each weight must be numeric.")
        sys.exit(-1)
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        print(f"Invalid securities weights. Must sum to 1.0 (100%).")
        sys.exit(-1)

    # Order tickers, convert to tuples to preserve order
    config["securities"] = tuple((security["ticker"], float(security["weight"])) for security in securities)

    # Get weights configuration (optional)
    weights = config.get("weights", {})
    if weights is None:
        weights = {}

    # Allow shorthand like: weights: static
    if isinstance(weights, str):
        weights = {"mode": weights}
    elif not isinstance(weights, dict):
        print(f"Invalid weights configuration: {weights}. Expected a mapping or string.")
        sys.exit(-1)

    weights_mode = weights.get("mode", weights.get("type", None))
    if weights_mode is None:
        # Preserve existing simulation behavior: multi-ticker simulations vary weights by default.
        weights_mode = "dynamic" if test_type == "simulate" and len(config["securities"]) > 1 else "static"
    weights_mode = str(weights_mode).strip().lower()
    if weights_mode not in ("static", "dynamic"):
        print(f"Invalid weights mode: {weights_mode}. Expected one of: static, dynamic")
        sys.exit(-1)
    if weights_mode == "dynamic" and test_type != "simulate":
        print("Dynamic weights are only supported for simulate tests.")
        sys.exit(-1)

    config["weights_mode"] = weights_mode
    config["weights_increment"] = None
    if weights_mode == "dynamic":
        increment = weights.get("increment", 0.05)
        try:
            increment = float(increment)
        except (TypeError, ValueError):
            print(f"Invalid weights increment: {increment}. Must be numeric.")
            sys.exit(-1)
        # Allow either ratio increments (0.05) or percent increments (5 for 5%).
        if increment > 1.0:
            increment = increment / 100.0
        if increment <= 0.0 or increment > 1.0:
            print(f"Invalid weights increment: {increment}. Must be > 0 and <= 1.0.")
            sys.exit(-1)
        config["weights_increment"] = increment

    # Get strategy configuration
    test_strategy_config: dict = config.get("strategy", None)
    if test_strategy_config is None:
        print(f"Please provide a strategy Class name and configuration.")
        sys.exit(-1)

    # Parse optional rebalance schedule strategy configuration up front and
    # remove it from the strategy args payload passed to constructors.
    strategy_schedule_config = _normalize_schedule_config(test_strategy_config.pop("schedule", None))
    if strategy_schedule_config is None and "schedule_format" in test_strategy_config:
        strategy_schedule_config = {
            "format": test_strategy_config.pop("schedule_format"),
        }
        if "schedule_value" in test_strategy_config:
            strategy_schedule_config["value"] = test_strategy_config.pop("schedule_value")

    # Dynamically import the specified Strategy, provided it can be found and is correctly implemented
    test_strategy_name = test_strategy_config.pop("name", None)
    if test_strategy_name is None:
        print(f"Please provide a strategy Class name.")
        sys.exit(-1)
    else:

        # Get Strategy module for subclass checks
        strategy_base_module = importlib.import_module("backtesting.core.strategy")
        strategy_base_class = getattr(strategy_base_module, "Strategy")

        # Set strategy module for checking references to strategies
        strategy_pkg = importlib.import_module("backtesting.strategies")

        # Default the strategy to None to make failure case simple
        config["strategy"] = None

        # Iterate over all modules in the package, load strategy if found
        for _, modname, _ in pkgutil.walk_packages(strategy_pkg.__path__, strategy_pkg.__name__ + '.'):
            try:
                strategy_module = importlib.import_module(modname)
                if hasattr(strategy_module, test_strategy_name):
                    strategy = getattr(strategy_module, test_strategy_name)
                    if not issubclass(strategy, strategy_base_class):
                        print(f"Strategy \"{test_strategy_name}\" found but is not a subclass of {strategy_base_class}")
                        sys.exit(-1)
                    config["strategy"] = getattr(strategy_module, test_strategy_name)
                    config["strategy_args"] = test_strategy_config
            except ImportError:
                print(f"Could not import module {modname}")
                continue

        # Exit if strategy could not be loaded
        if config["strategy"] is None:
            print(f"Strategy \"{test_strategy_name}\" not found.")
            sys.exit(-1)

    # Optional strategy rebalance schedule configuration.
    rebalance_mode = "static"
    rebalance_schedule = None
    rebalance_increment = None
    if strategy_schedule_config is not None:
        rebalance_mode = str(
            strategy_schedule_config.get("mode", strategy_schedule_config.get("type", "static"))
        ).strip().lower()
        if rebalance_mode not in ("static", "dynamic"):
            print(f"Invalid strategy.schedule mode: {rebalance_mode}. Expected one of: static, dynamic")
            sys.exit(-1)
        if rebalance_mode == "dynamic":
            if test_type != "simulate":
                print("Dynamic strategy schedules are only supported for simulate tests.")
                sys.exit(-1)
            increment = strategy_schedule_config.get("increment", 4)
            try:
                increment = int(increment)
            except (TypeError, ValueError):
                print(f"Invalid strategy.schedule increment: {increment}. Must be a positive integer.")
                sys.exit(-1)
            if increment < 1:
                print(f"Invalid strategy.schedule increment: {increment}. Must be >= 1.")
                sys.exit(-1)
            rebalance_increment = increment
        else:
            rebalance_schedule = _build_schedule(strategy_schedule_config, "strategy")
            config["strategy_args"]["schedule"] = rebalance_schedule

    # Get allocation configuration (required)
    allocation = config.get("allocation", None)
    if allocation is None:
        print("Missing required allocation block in configuration.")
        sys.exit(-1)
    if not isinstance(allocation, dict):
        print("Invalid allocation configuration. Expected a mapping/object.")
        sys.exit(-1)

    if "initial" not in allocation:
        print("Missing required allocation.initial value in configuration.")
        sys.exit(-1)
    initial_allocation = allocation.get("initial")
    if isinstance(initial_allocation, bool):
        print(f"Invalid allocation.initial: {initial_allocation}. Must be numeric and > 0.")
        sys.exit(-1)
    try:
        initial_allocation = float(initial_allocation)
    except (TypeError, ValueError):
        print(f"Invalid allocation.initial: {initial_allocation}. Must be numeric and > 0.")
        sys.exit(-1)
    if initial_allocation <= 0:
        print(f"Invalid allocation.initial: {initial_allocation}. Must be > 0.")
        sys.exit(-1)
    config["initial_allocation"] = initial_allocation

    if "yearly" not in allocation:
        print("Missing required allocation.yearly value in configuration.")
        sys.exit(-1)
    yearly_allocation = allocation.get("yearly")
    if isinstance(yearly_allocation, bool):
        print(f"Invalid allocation.yearly: {yearly_allocation}. Must be an integer >= 0.")
        sys.exit(-1)
    if isinstance(yearly_allocation, str):
        yearly_str = yearly_allocation.strip()
        if not yearly_str:
            print("Invalid allocation.yearly: empty string. Must be an integer >= 0.")
            sys.exit(-1)
        if yearly_str[0] in "+-":
            if len(yearly_str) == 1 or not yearly_str[1:].isdigit():
                print(f"Invalid allocation.yearly: {yearly_allocation}. Must be an integer >= 0.")
                sys.exit(-1)
        elif not yearly_str.isdigit():
            print(f"Invalid allocation.yearly: {yearly_allocation}. Must be an integer >= 0.")
            sys.exit(-1)
        yearly_allocation = int(yearly_str)
    else:
        if isinstance(yearly_allocation, float) and not yearly_allocation.is_integer():
            print(f"Invalid allocation.yearly: {yearly_allocation}. Must be an integer >= 0.")
            sys.exit(-1)
        try:
            yearly_allocation = int(yearly_allocation)
        except (TypeError, ValueError):
            print(f"Invalid allocation.yearly: {yearly_allocation}. Must be an integer >= 0.")
            sys.exit(-1)
    if yearly_allocation < 0:
        print(f"Invalid allocation.yearly: {yearly_allocation}. Must be >= 0.")
        sys.exit(-1)
    config["yearly_allocation"] = yearly_allocation

    allocation_mode = allocation.get("mode", allocation.get("type", None))
    if allocation_mode is None:
        # Preserve existing simulation behavior when no static schedule is provided.
        allocation_schedule_config = _normalize_schedule_config(allocation.get("schedule", None))
        allocation_mode = (
            "dynamic"
            if test_type == "simulate"
            and allocation_schedule_config is None
            and allocation.get("schedule_format", None) is None
            else "static"
        )
    allocation_mode = str(allocation_mode).strip().lower()
    if allocation_mode not in ("static", "dynamic"):
        print(f"Invalid allocation mode: {allocation_mode}. Expected one of: static, dynamic")
        sys.exit(-1)
    if allocation_mode == "dynamic" and test_type != "simulate":
        print("Dynamic allocation schedules are only supported for simulate tests.")
        sys.exit(-1)

    config["allocation_mode"] = allocation_mode
    config["allocation_increment"] = None

    if allocation_mode == "static":
        allocation_schedule_config = _normalize_schedule_config(allocation.get("schedule", None))
        if allocation_schedule_config is None and "schedule_format" in allocation:
            allocation_schedule_config = {"format": allocation.get("schedule_format")}
            if "schedule_value" in allocation:
                allocation_schedule_config["value"] = allocation.get("schedule_value")

        if allocation_schedule_config is None:
            config["allocation_schedule"] = None
        else:
            config["allocation_schedule"] = _build_schedule(allocation_schedule_config, "allocation")
    else:
        # Dynamic schedules iterate day-based allocation intervals.
        increment = allocation.get("increment", 4)
        try:
            increment = int(increment)
        except (TypeError, ValueError):
            print(f"Invalid allocation increment: {increment}. Must be a positive integer.")
            sys.exit(-1)
        if increment < 1:
            print(f"Invalid allocation increment: {increment}. Must be >= 1.")
            sys.exit(-1)
        config["allocation_schedule"] = None
        config["allocation_increment"] = increment

    # Get start and stop dates from nested dates mapping.
    dates_config = config.get("dates", {})
    if dates_config is None:
        dates_config = {}
    if not isinstance(dates_config, dict):
        print("Invalid dates configuration. Expected: dates: {start: ..., end: ...}")
        sys.exit(-1)
    config["dates"] = (
        dates_config.get("start", dates_config.get("start_date", config.get("start_date", None))),
        dates_config.get("end", dates_config.get("stop_date", config.get("stop_date", None))),
    )

    # Performance tracking
    config["track_performance"] = config.get("track_performance", False)

    # Return a normalized config payload to avoid carrying duplicate/raw blocks
    # (e.g. `weights` + `weights_mode`, `allocation` + `allocation_mode`).
    return {
        "securities": config["securities"],
        "strategy": config["strategy"],
        "strategy_args": config["strategy_args"],
        "allocation_schedule": config["allocation_schedule"],
        "initial_allocation": config["initial_allocation"],
        "yearly_allocation": config["yearly_allocation"],
        "weights_mode": config["weights_mode"],
        "weights_increment": config["weights_increment"],
        "allocation_mode": config["allocation_mode"],
        "allocation_increment": config["allocation_increment"],
        "rebalance_mode": rebalance_mode,
        "rebalance_schedule": rebalance_schedule,
        "rebalance_increment": rebalance_increment,
        "dates": config["dates"],
        "trace": config.get("trace", False),
        "track_performance": config["track_performance"],
    }

def get_historical_data(securities: tuple[str]) -> tuple:
    """Loads, cleans, and returns historical data."""

    price_history = pd.DataFrame()
    distribution_history = pd.DataFrame()

    # Create a dir for ticker price histories
    tickers_dir = os.path.join(os.getcwd(), "tickers")
    if not os.path.exists(tickers_dir):
        os.mkdir(tickers_dir)

    # Load price history for each ticker
    for ticker in securities:

        ticker_file = os.path.join(tickers_dir, f'{ticker}.pkl')
        if not os.path.exists(ticker_file):
            download_historical_data(ticker, tickers_dir)

        try:
            raw = pd.read_pickle(ticker_file)
        except (EOFError, pickle.UnpicklingError, ValueError):
            # Recover from a partial/corrupt cache artifact by redownloading once.
            download_historical_data(ticker, tickers_dir)
            raw = pd.read_pickle(ticker_file)

        # Flatten the multi-indexed columns
        raw = raw.droplevel(1, axis=1)

        # Drop all unused columns
        raw = raw.drop(columns=["Open", "High", "Low", "Volume", "Stock Splits", "Adj Close"])
        distribution_data = raw.drop(columns=["Close"])
        price_data = raw.drop(columns=["Dividends"])

        # Rename dividends to distributions
        distribution_data = distribution_data.rename(columns={"Dividends": "Distributions"})

        # Merge and drop capital gains if it exists
        if "Capital Gains" in raw.columns:
            price_data = price_data.drop(columns=["Capital Gains"])
            distribution_data["Distributions"] = distribution_data["Distributions"] + distribution_data["Capital Gains"]
            distribution_data = distribution_data.drop(columns=["Capital Gains"])

        # Remove rows with no distribution data
        distribution_data = distribution_data.loc[distribution_data["Distributions"] > 0.0]

        # Rename for indexing historical data by ticker
        distribution_data = distribution_data.rename(columns={"Distributions": ticker})
        price_data = price_data.rename(columns={"Close": ticker})

        # Join histories
        distribution_history = distribution_history.join(distribution_data, how="outer")
        price_history = price_history.join(price_data, how="outer")

    # Drop or fill NaNs
    distribution_history = distribution_history.fillna(0.0)
    price_history = price_history.dropna()

    # Adjust distribution data if it exists
    if not distribution_history.empty:

        # Price history likely extends past price history, truncate it before the first price date
        distribution_history = distribution_history[price_history.index[0]:]

        # Normalize dates as datetime64, bundle
        distribution_history_dates = distribution_history.index.normalize().to_numpy(dtype="datetime64[D]")
        distribution_history = distribution_history.to_numpy()
        distribution_data = (distribution_history_dates, distribution_history)
    else:
        distribution_data = (None, None)

    # Convert price and distribution histories to numpy arrays, then create indices for fast lookups
    price_history_dates = price_history.index.normalize().to_numpy(dtype="datetime64[D]")
    price_history = price_history.to_numpy()
    price_data = (price_history_dates, price_history)

    return price_data, distribution_data

def download_historical_data(ticker: str, tickers_dir: str) -> None:
    """Downloads ticker historical data."""

    # Fail if no data was returned
    data = yf.download(tickers=[ticker], period="max", interval="1d", actions=True, auto_adjust=False)
    if len(data) == 0:
        raise RuntimeError(f"Unable to download data for {ticker}")

    target_file = os.path.join(tickers_dir, f"{ticker}.pkl")
    with tempfile.NamedTemporaryFile(dir=tickers_dir, suffix=".tmp", delete=False) as tmp_file:
        temp_path = tmp_file.name
    try:
        pd.to_pickle(data, temp_path)
        # Concurrent workers may race to publish the same ticker cache file.
        # Retry publish and allow already-published readable files to win.
        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                os.replace(temp_path, target_file)
                break
            except PermissionError:
                if os.path.exists(target_file):
                    try:
                        pd.read_pickle(target_file)
                        break
                    except (EOFError, pickle.UnpicklingError, ValueError, PermissionError):
                        pass
                if attempt == max_attempts - 1:
                    raise
                time.sleep(0.15 * (attempt + 1))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def top_n_grouped_incremental(
    n: int, m: Iterable, size: int,
    results: dict, checkpoint: int,
    checkpoint_file: str, results_file: str
) -> int:
    """
    Returns the top N items grouped by np.datetime64.
    
    date_fn: Should return an np.datetime64 object.
    top_fn: Should return the numerical score for ranking.
    """
    checkpoint_target = 250000  # Persist after every 250,000 results
    last_date = None
    current_heap = None
    for item in tqdm(m, total=size):

        date, score = item[0], item[1]

        # Cache heap if possible
        if date == last_date:
            heap = current_heap
        else:
            if date not in results:
                results[date] = []
            heap = results[date]
            current_heap = heap
            last_date = date

        # Push items with a higher score than the smallest item onto the heap
        # Keep payload unchanged; use a separate tiebreaker to avoid comparing
        # non-orderable payload values (e.g., numpy arrays) in heap operations.
        heap_item = (score, checkpoint, item[1:])
        if len(heap) < n:
            heapq.heappush(heap, heap_item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, heap_item)

        # Persist results periodically
        checkpoint += 1
        if checkpoint % checkpoint_target == 0:
            with open(checkpoint_file, "w") as f:
                f.write(str(checkpoint))

            with open(results_file, "wb") as f:
                pickle.dump(results, f)

    return checkpoint
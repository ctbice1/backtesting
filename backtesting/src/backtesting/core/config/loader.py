"""Configuration loading and normalization for test definitions."""

import importlib
import os
import pkgutil
import sys

import numpy as np
import yaml

from backtesting.core.types import Schedule, ScheduleFormat


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
    securities = config.get("securities", [])
    if len(securities) == 0:
        print("Please provide securities to test.")
        sys.exit(-1)

    for security in securities:
        if "ticker" not in security:
            print(f'Security: {security} must have a ticker symbol specified as: ticker = "symbol".')
            sys.exit(-1)
        if "weight" not in security:
            print(f"Security: {security} must have a ticker weight specified as: weight = X.X")
            sys.exit(-1)
        if isinstance(security["weight"], bool):
            print(f"Invalid weight for {security['ticker']}: {security['weight']}. Must be numeric.")
            sys.exit(-1)

    try:
        total_weight = float(sum(float(security["weight"]) for security in config["securities"]))
    except (TypeError, ValueError):
        print("Invalid securities weights. Each weight must be numeric.")
        sys.exit(-1)
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        print("Invalid securities weights. Must sum to 1.0 (100%).")
        sys.exit(-1)

    config["securities"] = tuple((security["ticker"], float(security["weight"])) for security in securities)

    weights = config.get("weights", {})
    if weights is None:
        weights = {}
    if isinstance(weights, str):
        weights = {"mode": weights}
    elif not isinstance(weights, dict):
        print(f"Invalid weights configuration: {weights}. Expected a mapping or string.")
        sys.exit(-1)

    weights_mode = weights.get("mode", weights.get("type", None))
    if weights_mode is None:
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
        if increment > 1.0:
            increment = increment / 100.0
        if increment <= 0.0 or increment > 1.0:
            print(f"Invalid weights increment: {increment}. Must be > 0 and <= 1.0.")
            sys.exit(-1)
        config["weights_increment"] = increment

    test_strategy_config: dict = config.get("strategy", None)
    if test_strategy_config is None:
        print("Please provide a strategy Class name and configuration.")
        sys.exit(-1)

    strategy_schedule_config = _normalize_schedule_config(test_strategy_config.pop("schedule", None))
    if strategy_schedule_config is None and "schedule_format" in test_strategy_config:
        strategy_schedule_config = {"format": test_strategy_config.pop("schedule_format")}
        if "schedule_value" in test_strategy_config:
            strategy_schedule_config["value"] = test_strategy_config.pop("schedule_value")

    test_strategy_name = test_strategy_config.pop("name", None)
    if test_strategy_name is None:
        print("Please provide a strategy Class name.")
        sys.exit(-1)

    strategy_base_module = importlib.import_module("backtesting.core.strategy")
    strategy_base_class = getattr(strategy_base_module, "Strategy")
    strategy_pkg = importlib.import_module("backtesting.strategies")
    config["strategy"] = None

    for _, modname, _ in pkgutil.walk_packages(strategy_pkg.__path__, strategy_pkg.__name__ + "."):
        try:
            strategy_module = importlib.import_module(modname)
            if hasattr(strategy_module, test_strategy_name):
                strategy = getattr(strategy_module, test_strategy_name)
                if not issubclass(strategy, strategy_base_class):
                    print(
                        f'Strategy "{test_strategy_name}" found but is not a subclass of {strategy_base_class}'
                    )
                    sys.exit(-1)
                config["strategy"] = getattr(strategy_module, test_strategy_name)
                config["strategy_args"] = test_strategy_config
        except ImportError:
            print(f"Could not import module {modname}")
            continue

    if config["strategy"] is None:
        print(f'Strategy "{test_strategy_name}" not found.')
        sys.exit(-1)

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

    performance_config = config.get("performance", {})
    if performance_config is None:
        performance_config = {}
    if not isinstance(performance_config, dict):
        print("Invalid performance configuration. Expected a mapping/object.")
        sys.exit(-1)

    benchmark_ticker = performance_config.get("benchmark", config.get("benchmark", None))
    if isinstance(benchmark_ticker, bool):
        print("Invalid benchmark ticker. Expected a string ticker or null.")
        sys.exit(-1)
    if benchmark_ticker is not None:
        benchmark_ticker = str(benchmark_ticker).strip()
        if not benchmark_ticker:
            benchmark_ticker = None

    risk_free_ticker = performance_config.get(
        "risk_free_ticker",
        config.get("risk_free_ticker", "^IRX"),
    )
    if isinstance(risk_free_ticker, bool):
        print("Invalid risk_free_ticker. Expected a string ticker or null.")
        sys.exit(-1)
    if risk_free_ticker is not None:
        risk_free_ticker = str(risk_free_ticker).strip()
        if not risk_free_ticker:
            risk_free_ticker = None

    if test_type == "single":
        if benchmark_ticker is None:
            print(
                "Single tests require performance.benchmark (or top-level benchmark) "
                "to calculate benchmark-relative metrics."
            )
            sys.exit(-1)
        if risk_free_ticker is None:
            print(
                "Single tests require performance.risk_free_ticker (or top-level risk_free_ticker) "
                "to calculate risk-adjusted metrics."
            )
            sys.exit(-1)

    config["benchmark_ticker"] = benchmark_ticker
    config["risk_free_ticker"] = risk_free_ticker
    if test_type == "single":
        config["track_performance"] = True
    else:
        config["track_performance"] = config.get("track_performance", False)

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
        "benchmark_ticker": config["benchmark_ticker"],
        "risk_free_ticker": config["risk_free_ticker"],
    }

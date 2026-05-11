"""Configuration loading and normalization for test definitions."""

import importlib
import os
import pkgutil
import sys

import numpy as np
import yaml

from backtesting.core.types import Schedule, ScheduleFormat
from backtesting.data.market_data import is_fred_cash_series


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


def _parse_contribution_weights(
    weights_obj: object,
    securities: tuple[tuple[str, float], ...],
    context: str,
) -> tuple[float, ...]:
    """Parses a ticker -> weight mapping into a tuple aligned with ``securities`` order."""
    if weights_obj is None:
        print(f"Missing weights mapping for {context}.")
        sys.exit(-1)
    if not isinstance(weights_obj, dict):
        print(f"Invalid {context}: expected a mapping of ticker -> weight, got {type(weights_obj).__name__}.")
        sys.exit(-1)

    tickers = [s[0] for s in securities]
    ticker_set = set(tickers)
    unknown = set(weights_obj.keys()) - ticker_set
    if unknown:
        print(f"Invalid {context}: unknown ticker(s): {sorted(unknown)}.")
        sys.exit(-1)

    values: list[float] = []
    for ticker in tickers:
        if ticker not in weights_obj:
            print(f"Invalid {context}: missing weight for ticker {ticker}.")
            sys.exit(-1)
        w = weights_obj[ticker]
        if isinstance(w, bool):
            print(f"Invalid {context} weight for {ticker}: {w}. Must be numeric.")
            sys.exit(-1)
        try:
            wf = float(w)
        except (TypeError, ValueError):
            print(f"Invalid {context} weight for {ticker}: {w}. Must be numeric.")
            sys.exit(-1)
        values.append(wf)

    try:
        total = float(sum(values))
    except (TypeError, ValueError):
        print(f"Invalid {context}: weights must be numeric.")
        sys.exit(-1)
    if not np.isclose(total, 1.0, atol=1e-6):
        print(f"Invalid {context}: weights must sum to 1.0 (100%), got {total}.")
        sys.exit(-1)

    return tuple(round(v, 4) for v in values)


def _raw_nested_capital_gains_rate(tax_config: dict[str, object], term: str) -> object | None:
    """Reads ``tax.capital_gains.<term>.rate`` when present; otherwise None."""
    cg = tax_config.get("capital_gains")
    if not isinstance(cg, dict):
        return None
    block = cg.get(term)
    if isinstance(block, dict) and "rate" in block:
        return block["rate"]
    return None


def _parse_tax_rate(value: object, context: str) -> float:
    """Parses a tax rate as a decimal, accepting percentage-style values above 1."""
    if isinstance(value, bool):
        print(f"Invalid {context}: {value}. Must be numeric and >= 0.")
        sys.exit(-1)
    try:
        rate = float(value)
    except (TypeError, ValueError):
        print(f"Invalid {context}: {value}. Must be numeric and >= 0.")
        sys.exit(-1)
    if rate > 1.0:
        rate = rate / 100.0
    if rate < 0.0 or rate > 1.0:
        print(f"Invalid {context}: {value}. Must resolve to a rate between 0 and 1.")
        sys.exit(-1)
    return rate


def _parse_bool(value: object, context: str) -> bool:
    """Parses a boolean config value with common string spellings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "1", "on"):
            return True
        if normalized in ("false", "no", "0", "off"):
            return False
    print(f"Invalid {context}: {value}. Must be true or false.")
    sys.exit(-1)


def get_shared_test_config(config: dict[str, object], test_type: str | None = None) -> dict[str, object]:
    """Loads shared test configuration parameters."""
    securities = config.get("securities", [])
    if len(securities) == 0:
        print("Please provide securities to test.")
        sys.exit(-1)

    synthetic_securities: dict[str, dict[str, object]] = {}
    distribution_taxable_as: list[tuple[float, float, float]] = []
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

        ticker = str(security["ticker"]).strip()
        if not ticker:
            print(f"Invalid security ticker: {security['ticker']}. Must be a non-empty string.")
            sys.exit(-1)

        distribution_taxable_as_config = security.get("distribution_taxable_as", {})
        if distribution_taxable_as_config is None:
            distribution_taxable_as_config = {}
        if not isinstance(distribution_taxable_as_config, dict):
            print(
                f"Invalid distribution_taxable_as for {ticker}: expected a mapping with "
                "short_term, long_term, and return_of_capital percentages."
            )
            sys.exit(-1)
        unknown_distribution_tax_keys = set(distribution_taxable_as_config.keys()) - {
            "short_term",
            "long_term",
            "return_of_capital",
        }
        if unknown_distribution_tax_keys:
            print(
                f"Invalid distribution_taxable_as for {ticker}: unknown key(s): "
                f"{sorted(unknown_distribution_tax_keys)}."
            )
            sys.exit(-1)
        distribution_short_term = _parse_tax_rate(
            distribution_taxable_as_config.get("short_term", 0.0),
            f"securities.{ticker}.distribution_taxable_as.short_term",
        )
        distribution_long_term = _parse_tax_rate(
            distribution_taxable_as_config.get("long_term", 0.0),
            f"securities.{ticker}.distribution_taxable_as.long_term",
        )
        distribution_return_of_capital = _parse_tax_rate(
            distribution_taxable_as_config.get("return_of_capital", 0.0),
            f"securities.{ticker}.distribution_taxable_as.return_of_capital",
        )
        if distribution_short_term + distribution_long_term + distribution_return_of_capital > 1.0 + 1e-6:
            print(
                f"Invalid distribution_taxable_as for {ticker}: short_term, long_term, and return_of_capital "
                "percentages must sum to 1.0 or less."
            )
            sys.exit(-1)
        distribution_taxable_as.append(
            (distribution_short_term, distribution_long_term, distribution_return_of_capital)
        )

        if "daily_return_multiplier" not in security:
            continue
        daily_return_multiplier = security["daily_return_multiplier"]
        if isinstance(daily_return_multiplier, bool):
            print(
                f"Invalid daily_return_multiplier for {ticker}: {daily_return_multiplier}. Must be numeric."
            )
            sys.exit(-1)
        try:
            daily_return_multiplier = float(daily_return_multiplier)
        except (TypeError, ValueError):
            print(
                f"Invalid daily_return_multiplier for {ticker}: {daily_return_multiplier}. Must be numeric."
            )
            sys.exit(-1)
        if daily_return_multiplier <= 0:
            print(
                f"Invalid daily_return_multiplier for {ticker}: {daily_return_multiplier}. Must be > 0."
            )
            sys.exit(-1)

        underlying_ticker = security.get("underlying_ticker", ticker)
        if isinstance(underlying_ticker, bool):
            print(
                f"Invalid underlying_ticker for {ticker}: {underlying_ticker}. Must be a non-empty string."
            )
            sys.exit(-1)
        underlying_ticker = str(underlying_ticker).strip()
        if not underlying_ticker:
            print(f"Invalid underlying_ticker for {ticker}. Must be a non-empty string.")
            sys.exit(-1)

        expense_ratio = security.get("expense_ratio", 0.0)
        if isinstance(expense_ratio, bool):
            print(f"Invalid expense_ratio for {ticker}: {expense_ratio}. Must be numeric.")
            sys.exit(-1)
        try:
            expense_ratio = float(expense_ratio)
        except (TypeError, ValueError):
            print(f"Invalid expense_ratio for {ticker}: {expense_ratio}. Must be numeric.")
            sys.exit(-1)
        if expense_ratio < 0.0:
            print(f"Invalid expense_ratio for {ticker}: {expense_ratio}. Must be >= 0.")
            sys.exit(-1)

        synthetic_securities[ticker] = {
            "underlying_ticker": underlying_ticker,
            "daily_return_multiplier": daily_return_multiplier,
            "expense_ratio": expense_ratio,
        }

    try:
        total_weight = float(sum(float(security["weight"]) for security in config["securities"]))
    except (TypeError, ValueError):
        print("Invalid securities weights. Each weight must be numeric.")
        sys.exit(-1)
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        print("Invalid securities weights. Must sum to 1.0 (100%).")
        sys.exit(-1)

    config["securities"] = tuple((str(security["ticker"]).strip(), float(security["weight"])) for security in securities)
    config["synthetic_securities"] = synthetic_securities
    config["distribution_taxable_as"] = tuple(distribution_taxable_as)

    financing_config_raw = config.get("financing", {})
    if financing_config_raw is None:
        financing_config_raw = {}
    if not isinstance(financing_config_raw, dict):
        print("Invalid financing configuration. Expected a mapping/object.")
        sys.exit(-1)

    financing_enabled = financing_config_raw.get("enabled", True)
    if not isinstance(financing_enabled, bool):
        print(f"Invalid financing.enabled: {financing_enabled}. Must be true or false.")
        sys.exit(-1)

    financing_rate_source = financing_config_raw.get("rate_source", "fred")
    if isinstance(financing_rate_source, bool):
        print(f"Invalid financing.rate_source: {financing_rate_source}. Must be a string.")
        sys.exit(-1)
    financing_rate_source = str(financing_rate_source).strip().lower()
    if financing_rate_source not in ("fred",):
        print(f"Invalid financing.rate_source: {financing_rate_source}. Supported sources: fred.")
        sys.exit(-1)

    financing_series_id = financing_config_raw.get("series_id", "DFF")
    if isinstance(financing_series_id, bool):
        print(f"Invalid financing.series_id: {financing_series_id}. Must be a non-empty string.")
        sys.exit(-1)
    financing_series_id = str(financing_series_id).strip()
    if not financing_series_id:
        print("Invalid financing.series_id. Must be a non-empty string.")
        sys.exit(-1)

    config["synthetic_financing"] = {
        "enabled": financing_enabled,
        "rate_source": financing_rate_source,
        "series_id": financing_series_id,
    }

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

    allocation_weights: tuple[float, ...] | None = None
    alloc_weights_raw = allocation.get("weights")
    if alloc_weights_raw is not None:
        allocation_weights = _parse_contribution_weights(
            alloc_weights_raw, config["securities"], "allocation.weights"
        )

    distribution_block = config.get("distribution")
    if distribution_block is not None and not isinstance(distribution_block, dict):
        print("Invalid distribution configuration. Expected a mapping/object or omit the key.")
        sys.exit(-1)

    distribution_weights: tuple[float, ...] | None = allocation_weights
    if isinstance(distribution_block, dict):
        dist_weights_raw = distribution_block.get("weights")
        if dist_weights_raw is not None:
            distribution_weights = _parse_contribution_weights(
                dist_weights_raw, config["securities"], "distribution.weights"
            )

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

    risk_free_ticker = performance_config.get("risk_free_ticker", config.get("risk_free_ticker", None))
    if isinstance(risk_free_ticker, bool):
        print(f"Invalid risk_free_ticker: {risk_free_ticker}. Expected a string ticker or null.")
        sys.exit(-1)
    if risk_free_ticker is not None:
        risk_free_ticker = str(risk_free_ticker).strip()
        if not risk_free_ticker:
            risk_free_ticker = None

    cash_source_raw = config.get("cash_source", "TB3MS")
    if isinstance(cash_source_raw, bool):
        print(f"Invalid cash_source: {cash_source_raw}. Expected a string ticker.")
        sys.exit(-1)
    cash_source = str(cash_source_raw).strip()
    if not cash_source:
        cash_source = "TB3MS"
    if is_fred_cash_series(cash_source):
        cash_source = cash_source.upper()
    held = {str(t).strip().upper() for t, _ in config["securities"]}
    if cash_source.upper() in held:
        print(
            f'cash_source "{cash_source}" duplicates a portfolio security ticker; '
            "use a distinct cash instrument or FRED series id."
        )
        sys.exit(-1)

    config["cash_source"] = cash_source

    cash_management_raw = config.get("cash_management", "sweep")
    if isinstance(cash_management_raw, bool):
        print(f"Invalid cash_management: {cash_management_raw}. Expected 'distribute' or 'sweep'.")
        sys.exit(-1)
    cash_management = str(cash_management_raw).strip().lower()
    if not cash_management:
        cash_management = "sweep"
    if cash_management not in ("distribute", "sweep"):
        print(
            f'Invalid cash_management: {cash_management_raw!r}. '
            "Expected 'distribute' (reinvest cash yield into risky sleeve) or 'sweep' (keep in cash sleeve)."
        )
        sys.exit(-1)
    config["cash_management"] = cash_management

    tax_config = config.get("tax", config.get("tax_drag", {}))
    if tax_config is None:
        tax_config = {}
    if not isinstance(tax_config, dict):
        print("Invalid tax configuration. Expected a mapping/object.")
        sys.exit(-1)
    st_raw = _raw_nested_capital_gains_rate(tax_config, "short_term")
    st_ctx = "tax.capital_gains.short_term.rate"
    if st_raw is None:
        st_raw = tax_config.get(
            "short_term_capital_gains_rate",
            tax_config.get("short_term_rate", tax_config.get("short_term", 0.0)),
        )
        st_ctx = "tax.short_term_capital_gains_rate"
    short_term_capital_gains_rate = _parse_tax_rate(st_raw, st_ctx)

    lt_raw = _raw_nested_capital_gains_rate(tax_config, "long_term")
    lt_ctx = "tax.capital_gains.long_term.rate"
    if lt_raw is None:
        lt_raw = tax_config.get(
            "long_term_capital_gains_rate",
            tax_config.get("long_term_rate", tax_config.get("long_term", 0.0)),
        )
        lt_ctx = "tax.long_term_capital_gains_rate"
    long_term_capital_gains_rate = _parse_tax_rate(lt_raw, lt_ctx)
    net_investment_income = _parse_bool(
        tax_config.get("net_investment_income", True),
        "tax.net_investment_income",
    )
    config["short_term_capital_gains_rate"] = short_term_capital_gains_rate
    config["long_term_capital_gains_rate"] = long_term_capital_gains_rate
    config["net_investment_income"] = net_investment_income

    if risk_free_ticker is None:
        risk_free_ticker = "TB3MS"
    elif is_fred_cash_series(risk_free_ticker):
        risk_free_ticker = str(risk_free_ticker).strip().upper()

    if test_type == "single":
        if benchmark_ticker is None:
            print(
                "Single tests require performance.benchmark (or top-level benchmark) "
                "to calculate benchmark-relative metrics."
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
        "synthetic_securities": config["synthetic_securities"],
        "synthetic_financing": config["synthetic_financing"],
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
        "allocation_weights": allocation_weights,
        "distribution_weights": distribution_weights,
        "cash_source": config["cash_source"],
        "cash_management": config["cash_management"],
        "short_term_capital_gains_rate": config["short_term_capital_gains_rate"],
        "long_term_capital_gains_rate": config["long_term_capital_gains_rate"],
        "net_investment_income": config["net_investment_income"],
        "distribution_taxable_as": config["distribution_taxable_as"],
    }

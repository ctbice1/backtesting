"""Portfolio-level performance metric helpers."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from backtesting.core.portfolio import Portfolio
from backtesting.performance.metrics import portfolio_value_history


def portfolio_values_from_share_history(
    share_history: Mapping[Any, np.ndarray],
    price_history: np.ndarray,
    price_history_dates: np.ndarray,
) -> pd.Series:
    """Returns aggregate portfolio value history as a single time series."""
    ticker_value_history = portfolio_value_history(share_history, price_history, price_history_dates)
    if ticker_value_history.empty:
        return pd.Series(dtype=float)
    return ticker_value_history.sum(axis=1).astype(float)


def _as_series(values: Sequence[float] | np.ndarray | pd.Series | None) -> pd.Series:
    """Converts an array-like value sequence into a clean float series."""
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.Series):
        return values.astype(float).dropna()
    return pd.Series(values, dtype=float).dropna()


def _returns_from_values(values: Sequence[float] | np.ndarray | pd.Series | None) -> pd.Series:
    """Converts a value history into periodic percentage returns."""
    series = _as_series(values)
    if len(series) < 2:
        return pd.Series(dtype=float)
    returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return returns.astype(float)


def _aligned_series(
    portfolio_series: Sequence[float] | np.ndarray | pd.Series | None,
    benchmark_series: Sequence[float] | np.ndarray | pd.Series | None,
) -> tuple[pd.Series, pd.Series]:
    """Aligns portfolio and benchmark series on a shared index."""
    portfolio_returns = _as_series(portfolio_series)
    benchmark_returns = _as_series(benchmark_series)
    if portfolio_returns.empty or benchmark_returns.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    combined = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if combined.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    return combined["portfolio"], combined["benchmark"]


def beta(
    portfolio_returns: Sequence[float] | np.ndarray | pd.Series,
    benchmark_returns: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Calculates beta using covariance(portfolio, benchmark) / variance(benchmark)."""
    p, b = _aligned_series(portfolio_returns, benchmark_returns)
    if len(p) < 2 or len(b) < 2:
        return float("nan")

    benchmark_variance = b.var(ddof=1)
    if benchmark_variance == 0 or np.isnan(benchmark_variance):
        return float("nan")

    covariance = p.cov(b)
    return float(covariance / benchmark_variance)


def sortino_ratio(
    portfolio_returns: Sequence[float] | np.ndarray | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """Calculates annualized Sortino ratio from periodic returns."""
    returns = _as_series(portfolio_returns)
    if len(returns) < 2:
        return float("nan")

    period_risk_free = risk_free_rate / periods_per_year
    period_target = target_return / periods_per_year

    excess_returns = returns - period_risk_free
    downside = (returns - period_target).clip(upper=0.0)
    downside_deviation = np.sqrt((downside.pow(2)).mean())
    if downside_deviation == 0 or np.isnan(downside_deviation):
        return float("nan")

    annualized_excess_return = excess_returns.mean() * periods_per_year
    annualized_downside_deviation = downside_deviation * np.sqrt(periods_per_year)
    return float(annualized_excess_return / annualized_downside_deviation)


def alpha(
    portfolio_returns: Sequence[float] | np.ndarray | pd.Series,
    benchmark_returns: Sequence[float] | np.ndarray | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculates annualized CAPM alpha."""
    p, b = _aligned_series(portfolio_returns, benchmark_returns)
    if len(p) < 2:
        return float("nan")

    beta_value = beta(p, b)
    if np.isnan(beta_value):
        return float("nan")

    annualized_portfolio_return = p.mean() * periods_per_year
    annualized_benchmark_return = b.mean() * periods_per_year
    expected_return = risk_free_rate + beta_value * (annualized_benchmark_return - risk_free_rate)
    return float(annualized_portfolio_return - expected_return)


def treynor_ratio(
    portfolio_returns: Sequence[float] | np.ndarray | pd.Series,
    benchmark_returns: Sequence[float] | np.ndarray | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculates annualized Treynor ratio using benchmark-derived beta."""
    p, b = _aligned_series(portfolio_returns, benchmark_returns)
    if len(p) < 2:
        return float("nan")

    beta_value = beta(p, b)
    if beta_value == 0 or np.isnan(beta_value):
        return float("nan")

    annualized_portfolio_return = p.mean() * periods_per_year
    annualized_excess_return = annualized_portfolio_return - risk_free_rate
    return float(annualized_excess_return / beta_value)


def risk_free_rate_from_irx(irx_values: Sequence[float] | np.ndarray | pd.Series | None) -> float:
    """
    Estimates annualized risk-free rate from `^IRX` close values.

    `^IRX` is quoted in annualized yield percentage terms (e.g. 4.85 -> 4.85%).
    """
    irx_series = _as_series(irx_values)
    if irx_series.empty:
        return float("nan")
    return float((irx_series / 100.0).mean())


def portfolio_performance_summary(
    portfolio: Portfolio,
    final_prices: np.ndarray,
    portfolio_values: Sequence[float] | np.ndarray | pd.Series | None = None,
    benchmark_values: Sequence[float] | np.ndarray | pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Builds a summary using Portfolio top-line metrics plus risk-adjusted metrics.

    Top-line values mirror the `single_test` print output:
    - final_balance
    - net_new_capital
    - distributions
    """
    final_balance = float(portfolio.current_value(final_prices))
    net_new_capital = float(portfolio.total_new_capital)
    distributions = float(portfolio.total_distribution)
    net_profit = final_balance - net_new_capital
    total_return = (net_profit / net_new_capital) if net_new_capital > 0 else float("nan")

    portfolio_returns = _returns_from_values(portfolio_values)
    benchmark_returns = _returns_from_values(benchmark_values)

    metric_beta = beta(portfolio_returns, benchmark_returns)
    metric_sortino = sortino_ratio(
        portfolio_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    metric_alpha = alpha(
        portfolio_returns,
        benchmark_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    metric_treynor = treynor_ratio(
        portfolio_returns,
        benchmark_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    return {
        "final_balance": final_balance,
        "net_new_capital": net_new_capital,
        "distributions": distributions,
        "net_profit": net_profit,
        "total_return": float(total_return),
        "sortino_ratio": metric_sortino,
        "treynor_ratio": metric_treynor,
        "alpha": metric_alpha,
        "beta": metric_beta,
    }

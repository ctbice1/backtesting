"""Portfolio-level performance metric helpers."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from backtesting.core.portfolio import Portfolio
from backtesting.performance.metrics import portfolio_value_history


def trailing_twelve_month_yield(
    distribution_history: Mapping[Any, float] | None,
    final_balance: float,
    end_date: Any | None = None,
) -> float:
    """Calculates TTM yield as last-12-month distributions divided by portfolio value."""
    if final_balance <= 0 or np.isnan(final_balance):
        return float("nan")
    if not distribution_history:
        return 0.0

    distributions = pd.Series(distribution_history, dtype=float).dropna()
    if distributions.empty:
        return 0.0

    distributions.index = pd.to_datetime(distributions.index)
    if end_date is None:
        end_timestamp = distributions.index.max()
    else:
        end_timestamp = pd.to_datetime(end_date)

    window_start = end_timestamp - pd.DateOffset(months=12)
    ttm_distributions = distributions[
        (distributions.index >= window_start) & (distributions.index <= end_timestamp)
    ].sum()
    return float(ttm_distributions / final_balance)


def money_weighted_irr(
    contribution_flows: Sequence[tuple[Any, float]],
    terminal_value: float,
    terminal_date: Any,
) -> float:
    """
    Money-weighted IRR (annual effective rate) from dated investor contributions and terminal wealth.

    Contributions are treated as negative investor cash flows; terminal portfolio value is a positive
    inflow on ``terminal_date``. Dates are converted with ``pandas.to_datetime`` and spaced using
    calendar days / 365.25 for discount exponents.
    """
    if terminal_value <= 0 or np.isnan(terminal_value):
        return float("nan")
    if terminal_date is None:
        return float("nan")
    if not contribution_flows:
        return float("nan")

    net_cf: dict[pd.Timestamp, float] = defaultdict(float)
    for raw_date, amount in contribution_flows:
        ts = pd.to_datetime(raw_date).normalize()
        net_cf[ts] -= float(amount)

    term_ts = pd.to_datetime(terminal_date).normalize()
    net_cf[term_ts] += float(terminal_value)

    ordered = sorted(net_cf.keys())
    if len(ordered) < 2:
        return float("nan")

    times = np.array([(d - ordered[0]).days / 365.25 for d in ordered], dtype=float)
    cf = np.array([net_cf[d] for d in ordered], dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(cf / np.power(1.0 + rate, times)))

    def npv_prime(rate: float) -> float:
        return float(np.sum(-times * cf / np.power(1.0 + rate, times + 1.0)))

    rate = 0.05
    for _ in range(80):
        value = npv(rate)
        if abs(value) < 1e-10:
            return float(rate)
        deriv = npv_prime(rate)
        if abs(deriv) < 1e-18:
            break
        step = value / deriv
        rate_next = rate - step
        rate = max(rate_next, -0.9999)

    lo, hi = -0.9999, 100.0
    v_lo, v_hi = npv(lo), npv(hi)
    if np.isnan(v_lo) or np.isnan(v_hi):
        return float("nan")
    if v_lo * v_hi > 0:
        for scale in (1_000.0, 1_000_000.0):
            hi = scale
            v_hi = npv(hi)
            if np.isnan(v_hi):
                continue
            if v_lo * v_hi <= 0:
                break
        else:
            return float("nan")

    for _ in range(256):
        mid = 0.5 * (lo + hi)
        v_mid = npv(mid)
        if abs(v_mid) < 1e-12:
            return float(mid)
        if v_lo * v_mid <= 0:
            hi, v_hi = mid, v_mid
        else:
            lo, v_lo = mid, v_mid
    return float(0.5 * (lo + hi))


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


def _elapsed_years(values: pd.Series, periods_per_year: int) -> float:
    """Returns elapsed years from a value series index or observation count."""
    if isinstance(values.index, pd.DatetimeIndex):
        elapsed_days = (values.index[-1] - values.index[0]).days
        if elapsed_days > 0:
            return elapsed_days / 365.25
        return float("nan")

    if periods_per_year <= 0:
        return float("nan")
    return (len(values) - 1) / periods_per_year


def compound_annual_growth_rate(
    values: Sequence[float] | np.ndarray | pd.Series | None,
    periods_per_year: int = 252,
) -> float:
    """Calculates CAGR from the first and last values in a portfolio value history."""
    series = _as_series(values)
    if len(series) < 2:
        return float("nan")

    start_value = float(series.iloc[0])
    end_value = float(series.iloc[-1])
    if start_value <= 0 or end_value <= 0:
        return float("nan")

    years = _elapsed_years(series, periods_per_year)
    if years <= 0 or np.isnan(years):
        return float("nan")

    return float((end_value / start_value) ** (1 / years) - 1)


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


def sharpe_ratio(
    portfolio_returns: Sequence[float] | np.ndarray | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculates annualized Sharpe ratio from periodic returns."""
    returns = _as_series(portfolio_returns)
    if len(returns) < 2:
        return float("nan")

    period_risk_free = risk_free_rate / periods_per_year
    excess_returns = returns - period_risk_free
    excess_volatility = excess_returns.std(ddof=1)
    if excess_volatility == 0 or np.isnan(excess_volatility):
        return float("nan")

    annualized_excess_return = excess_returns.mean() * periods_per_year
    annualized_excess_volatility = excess_volatility * np.sqrt(periods_per_year)
    return float(annualized_excess_return / annualized_excess_volatility)


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


def jensens_alpha(
    portfolio_returns: Sequence[float] | np.ndarray | pd.Series,
    benchmark_returns: Sequence[float] | np.ndarray | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculates annualized Jensen's Alpha using CAPM expected return."""
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


def alpha(
    portfolio_returns: Sequence[float] | np.ndarray | pd.Series,
    benchmark_returns: Sequence[float] | np.ndarray | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculates annualized CAPM alpha."""
    return jensens_alpha(
        portfolio_returns,
        benchmark_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )


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
    Mean annualized risk-free rate from yields quoted as percentage points per annum.

    Applies to `^IRX` closes and aligned FRED yields such as TB3MS (e.g. 4.85 → 4.85%/year).
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
    end_date: Any | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Builds a summary using Portfolio top-line metrics plus risk-adjusted metrics.

    Top-line values mirror the `single_test` print output:
    - final_balance
    - net_new_capital
    - distributions
    - irr (money-weighted, from dated flows in ``portfolio.contribution_flows``)
    """
    final_balance = float(portfolio.current_value(final_prices))
    net_new_capital = float(portfolio.total_new_capital)
    distributions = float(portfolio.total_distribution)
    total_taxes_paid = float(getattr(portfolio, "total_tax_paid", 0.0))
    metric_ttm_yield = trailing_twelve_month_yield(
        portfolio.distribution_history,
        final_balance,
        end_date=end_date,
    )
    net_profit = final_balance - net_new_capital
    pre_tax_net_profit = net_profit + total_taxes_paid
    if total_taxes_paid == 0.0:
        portfolio_tax_drag = 0.0
    elif pre_tax_net_profit > 0.0:
        portfolio_tax_drag = total_taxes_paid / pre_tax_net_profit
    else:
        portfolio_tax_drag = float("nan")
    total_return = (net_profit / net_new_capital) if net_new_capital > 0 else float("nan")
    metric_irr = money_weighted_irr(portfolio.contribution_flows, final_balance, end_date)

    portfolio_returns = _returns_from_values(portfolio_values)
    benchmark_returns = _returns_from_values(benchmark_values)

    metric_cagr = compound_annual_growth_rate(portfolio_values, periods_per_year=periods_per_year)
    metric_beta = beta(portfolio_returns, benchmark_returns)
    metric_sharpe = sharpe_ratio(
        portfolio_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    metric_sortino = sortino_ratio(
        portfolio_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    metric_jensens_alpha = jensens_alpha(
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
        "total_taxes_paid": total_taxes_paid,
        "portfolio_tax_drag": float(portfolio_tax_drag),
        "net_profit": net_profit,
        "total_return": float(total_return),
        "irr": metric_irr,
        "ttm_yield": metric_ttm_yield,
        "cagr": metric_cagr,
        "sharpe_ratio": metric_sharpe,
        "sortino_ratio": metric_sortino,
        "treynor_ratio": metric_treynor,
        "jensens_alpha": metric_jensens_alpha,
        "alpha": metric_jensens_alpha,
        "beta": metric_beta,
    }

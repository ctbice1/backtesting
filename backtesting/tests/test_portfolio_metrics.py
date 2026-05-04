"""Tests for portfolio performance metric helpers."""

import unittest

import numpy as np
import pandas as pd

from backtesting.core.portfolio import Portfolio
from backtesting.performance.portfolio_metrics import (
    compound_annual_growth_rate,
    portfolio_performance_summary,
    trailing_twelve_month_yield,
)


class PortfolioMetricTests(unittest.TestCase):

    def test_compound_annual_growth_rate_uses_observation_count_without_dates(self) -> None:
        """A two-period value series at two periods per year covers one year."""
        values = pd.Series([100.0, 110.0, 121.0])

        cagr = compound_annual_growth_rate(values, periods_per_year=2)

        self.assertAlmostEqual(cagr, 0.21)

    def test_compound_annual_growth_rate_uses_datetime_index_when_available(self) -> None:
        """Dated values annualize by elapsed calendar time."""
        values = pd.Series(
            [100.0, 121.0],
            index=pd.DatetimeIndex(["2020-01-01", "2022-01-01"]),
        )

        cagr = compound_annual_growth_rate(values)

        self.assertAlmostEqual(cagr, 0.1, places=3)

    def test_compound_annual_growth_rate_requires_positive_values(self) -> None:
        """CAGR is undefined for non-positive start or ending values."""
        cagr = compound_annual_growth_rate(pd.Series([0.0, 121.0]), periods_per_year=1)

        self.assertTrue(np.isnan(cagr))

    def test_portfolio_performance_summary_includes_cagr(self) -> None:
        """The top-level portfolio summary exposes CAGR alongside total return."""
        portfolio = Portfolio(("SPY",), (1.0,))
        portfolio.current_shares = np.array([1.21])
        portfolio.total_new_capital = 100.0
        portfolio_values = pd.Series([100.0, 110.0, 121.0])

        summary = portfolio_performance_summary(
            portfolio=portfolio,
            final_prices=np.array([100.0]),
            portfolio_values=portfolio_values,
            periods_per_year=2,
        )

        self.assertIn("cagr", summary)
        self.assertAlmostEqual(summary["cagr"], 0.21)

    def test_trailing_twelve_month_yield_uses_distribution_history_window(self) -> None:
        """TTM yield includes distributions in the final 12 calendar months only."""
        distribution_history = {
            pd.Timestamp("2022-12-31"): 5.0,
            pd.Timestamp("2023-06-30"): 20.0,
            pd.Timestamp("2023-12-31"): 30.0,
        }

        ttm_yield = trailing_twelve_month_yield(
            distribution_history,
            final_balance=1000.0,
            end_date=pd.Timestamp("2023-12-31"),
        )

        self.assertAlmostEqual(ttm_yield, 0.055)

    def test_portfolio_performance_summary_includes_ttm_yield(self) -> None:
        """The top-level summary exposes current yield from portfolio distributions."""
        portfolio = Portfolio(("SPY",), (1.0,))
        portfolio.current_shares = np.array([10.0])
        portfolio.record_distribution(pd.Timestamp("2023-03-31"), 12.5)
        portfolio.record_distribution(pd.Timestamp("2023-09-30"), 12.5)

        summary = portfolio_performance_summary(
            portfolio=portfolio,
            final_prices=np.array([10.0]),
            end_date=pd.Timestamp("2023-12-31"),
        )

        self.assertIn("ttm_yield", summary)
        self.assertAlmostEqual(summary["ttm_yield"], 0.25)


if __name__ == "__main__":
    unittest.main()

"""Tests for cash_management (cash yield reinvestment vs. staying in cash sleeve)."""

import heapq
import unittest

import numpy as np
import pandas as pd

from backtesting.core.portfolio import Portfolio
from backtesting.core.strategy import Strategy
from backtesting.core.types import Distribute, Schedule, ScheduleFormat


class _DistributionsOnlyStrategy(Strategy):
    """Schedules distributions only so integration tests avoid allocation ordering."""

    def _set_passive_activity_schedule(
        self, price_history_dates: np.ndarray, distribution_history_dates: np.ndarray | None
    ) -> None:
        self.activity_schedule = []
        heapq.heapify(self.activity_schedule)
        if distribution_history_dates is None:
            return
        if self._start_date is not None:
            pending = [
                date for date in distribution_history_dates if date >= self._start_date
            ]
        else:
            pending = list(distribution_history_dates)
        for date in pd.to_datetime(pending):
            heapq.heappush(
                self.activity_schedule,
                Distribute(pd.Timestamp(date), self._distribution_contribution_weights),
            )

    def procedure(
        self, historical_data: tuple, indices: tuple, portfolio: Portfolio, trace: bool
    ) -> None:
        """No discretionary activities."""
        pass


class CashManagementTests(unittest.TestCase):
    """distribution vs sweep behavior on the cash sleeve column."""

    def _run_once(self, *, cash_management: str) -> np.ndarray:
        d0 = np.datetime64("2020-01-03", "D")
        portfolio = Portfolio(("EQ",), (1.0,), cash_source_ticker="CASH")
        portfolio.current_shares[:] = np.array([10.0, 1.0], dtype=float)

        price_history_dates = np.array([d0], dtype="datetime64[D]")
        price_history = np.array([[100.0, 100.0]], dtype=float)
        distribution_history_dates = price_history_dates.copy()
        distribution_history = np.array([[0.0, 20.0]], dtype=float)

        strategy = _DistributionsOnlyStrategy(
            initial_allocation=0,
            yearly_allocation=0,
            allocation_schedule=Schedule(ScheduleFormat.YEARLY, 1),
            start_date=d0,
            track=False,
            distribution_weights=(1.0,),
            cash_management=cash_management,
        )
        portfolio_out, _ = strategy.execute(
            (price_history, distribution_history),
            (price_history_dates, distribution_history_dates),
            portfolio,
            trace=False,
        )
        return portfolio_out.current_shares.copy()

    def test_sweep_keeps_cash_yield_in_cash_sleeve(self) -> None:
        shares = self._run_once(cash_management="sweep")
        self.assertAlmostEqual(float(shares[0]), 10.0, places=3)
        self.assertAlmostEqual(float(shares[1]), 1.2, places=3)

    def test_distribute_sends_full_payout_through_allocate(self) -> None:
        shares = self._run_once(cash_management="distribute")
        self.assertAlmostEqual(float(shares[0]), 10.2, places=3)
        self.assertAlmostEqual(float(shares[1]), 1.0, places=3)

    def test_niit_reduces_distribution_before_reinvestment(self) -> None:
        d0 = np.datetime64("2020-01-03", "D")
        portfolio = Portfolio(("EQ",), (1.0,), cash_source_ticker="CASH", net_investment_income=True)
        portfolio.current_shares[:] = np.array([10.0, 1.0], dtype=float)

        price_history_dates = np.array([d0], dtype="datetime64[D]")
        price_history = np.array([[100.0, 100.0]], dtype=float)
        distribution_history_dates = price_history_dates.copy()
        distribution_history = np.array([[0.0, 20.0]], dtype=float)

        strategy = _DistributionsOnlyStrategy(
            initial_allocation=0,
            yearly_allocation=0,
            allocation_schedule=Schedule(ScheduleFormat.YEARLY, 1),
            start_date=d0,
            track=False,
            distribution_weights=(1.0,),
            cash_management="distribute",
        )
        portfolio_out, _ = strategy.execute(
            (price_history, distribution_history),
            (price_history_dates, distribution_history_dates),
            portfolio,
            trace=False,
        )

        self.assertAlmostEqual(float(portfolio_out.current_shares[0]), 10.1924, places=4)
        self.assertAlmostEqual(portfolio_out.total_tax_paid, 0.76)


if __name__ == "__main__":
    unittest.main()

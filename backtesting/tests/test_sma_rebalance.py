"""Tests for simple moving average rebalance crossover behavior."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from backtesting.core.portfolio import Portfolio
from backtesting.core.types import Schedule, ScheduleFormat
from backtesting.strategies.rebalance import (
    SMACrossRebalance,
    ScheduledRebalance,
    SimpleMovingAverageRebalance,
    StdDevRebalance,
    VolatilityAdjustedSMARebalance,
)


class SimpleMovingAverageRebalanceTests(unittest.TestCase):
    def test_rebalances_only_on_price_crosses_of_configured_sma(self) -> None:
        """Primary/alternate switches happen when target price crosses its SMA."""
        dates = np.array(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-13",
            ],
            dtype="datetime64[D]",
        )
        prices = np.array([10.0, 12.0, 14.0, 12.0, 10.0, 8.0, 10.0, 12.0, 14.0])

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertEqual(tickers, ("SPY",))
            return (dates, prices.reshape(-1, 1)), None

        portfolio = Portfolio(("UPRO", "TLT", "CASH"), (0.6, 0.3, 0.1))
        strategy = SimpleMovingAverageRebalance(
            target="SPY",
            primary="UPRO",
            alternate="TLT",
            length=3,
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((None, None), (dates, None), portfolio, trace=False)

        self.assertEqual([activity.date for activity in strategy.activity_schedule], [dates[3], dates[6]])
        np.testing.assert_array_equal(strategy.activity_schedule[0].weights, np.array([0.0, 1.0, 0.0]))
        np.testing.assert_array_equal(strategy.activity_schedule[1].weights, np.array([1.0, 0.0, 0.0]))

    def test_trace_false_suppresses_rebalance_output(self) -> None:
        """The strategy respects the procedure trace flag."""
        dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        prices = np.array([10.0, 8.0])

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertEqual(tickers, ("SPY",))
            return (dates, prices.reshape(-1, 1)), None

        portfolio = Portfolio(("UPRO", "TLT"), (1.0, 0.0))
        strategy = SimpleMovingAverageRebalance(
            target="SPY",
            primary="UPRO",
            alternate="TLT",
            length=2,
        )

        output = io.StringIO()
        with (
            patch("backtesting.strategies.rebalance.get_historical_data", historical_data),
            redirect_stdout(output),
        ):
            strategy.procedure((None, None), (dates, None), portfolio, trace=False)

        self.assertEqual(output.getvalue(), "")

    def test_first_in_window_date_ignores_prior_history_for_crossover(self) -> None:
        """Pre-start target history does not create in-window crossover signals."""
        dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        prices = np.array([10.0, 8.0])

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertEqual(tickers, ("SPY",))
            return (dates, prices.reshape(-1, 1)), None

        portfolio = Portfolio(("UPRO", "TLT"), (1.0, 0.0))
        strategy = SimpleMovingAverageRebalance(
            target="SPY",
            primary="UPRO",
            alternate="TLT",
            length=2,
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((None, None), (dates[1:], None), portfolio, trace=False)

        self.assertEqual(strategy.activity_schedule, [])

    def test_minimum_primary_rebalance_days_only_delays_primary_return(self) -> None:
        """Configured cooldown delays return to primary but not risk-off alternate switches."""
        dates = np.array(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
            ],
            dtype="datetime64[D]",
        )
        prices = np.array([10.0, 12.0, 10.0, 12.0, 12.0, 14.0, 10.0])

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertEqual(tickers, ("SPY",))
            return (dates, prices.reshape(-1, 1)), None

        portfolio = Portfolio(("UPRO", "TLT"), (1.0, 0.0))
        strategy = SimpleMovingAverageRebalance(
            target="SPY",
            primary="UPRO",
            alternate="TLT",
            length=2,
            minimum_primary_rebalance_days=3,
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((None, None), (dates, None), portfolio, trace=False)

        self.assertEqual([activity.date for activity in strategy.activity_schedule], [dates[2], dates[5], dates[6]])
        np.testing.assert_array_equal(strategy.activity_schedule[0].weights, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(strategy.activity_schedule[1].weights, np.array([1.0, 0.0]))
        np.testing.assert_array_equal(strategy.activity_schedule[2].weights, np.array([0.0, 1.0]))

    def test_all_rebalance_strategies_run_without_dxy_data(self) -> None:
        """Strategy procedures do not fetch or require DXY-adjusted target prices."""
        dates = np.array(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
            ],
            dtype="datetime64[D]",
        )
        prices = np.array([10.0, 12.0, 10.0, 12.0, 14.0, 10.0])
        volatility = np.array([15.0, 18.0, 25.0, 22.0, 17.0, 16.0])

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertNotEqual(tickers, ("DX-Y.NYB",))
            values = volatility if tickers == ("VIX",) else prices
            return (dates, values.reshape(-1, 1)), None

        strategies = [
            ScheduledRebalance(schedule=Schedule(ScheduleFormat.DAYS, 2)),
            SimpleMovingAverageRebalance(target="SPY", primary="UPRO", alternate="TLT", length=2),
            StdDevRebalance(
                target="SPY",
                primary="UPRO",
                alternate="TLT",
                length=2,
                long_length=2,
                short_length=2,
            ),
            SMACrossRebalance(
                target="SPY",
                primary="UPRO",
                alternate="TLT",
                length=2,
                long_length=3,
                short_length=2,
            ),
            VolatilityAdjustedSMARebalance(
                target="SPY",
                primary="UPRO",
                alternate="TLT",
                length=2,
                volatility="VIX",
                volatility_length=2,
            ),
        ]

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            for strategy in strategies:
                portfolio = Portfolio(("UPRO", "TLT"), (1.0, 0.0))
                strategy.procedure((None, None), (dates, None), portfolio, trace=False)


if __name__ == "__main__":
    unittest.main()

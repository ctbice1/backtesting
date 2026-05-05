"""Tests for simple moving average rebalance crossover behavior."""

import unittest
from unittest.mock import patch

import numpy as np

from backtesting.core.portfolio import Portfolio
from backtesting.strategies.rebalance import SimpleMovingAverageRebalance


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


if __name__ == "__main__":
    unittest.main()

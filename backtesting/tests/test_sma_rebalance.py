"""Tests for simple moving average rebalance crossover behavior."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from backtesting.core.portfolio import Portfolio
from backtesting.core.types import NewATH, Schedule, ScheduleFormat, TakeProfit, ZScore
from backtesting.indicators.basic import SimpleZScore
from backtesting.strategies.rebalance import (
    SMACrossRebalance,
    ScheduledRebalance,
    SimpleMovingAverageRebalance,
    StdDevRebalance,
    VolatilityAdjustedSMARebalance,
)


class SimpleMovingAverageRebalanceTests(unittest.TestCase):
    def test_simple_z_score_uses_price_distance_from_sma_over_rolling_std_dev(self) -> None:
        """SimpleZScore normalizes price/SMA distance by rolling variability."""
        z_score = SimpleZScore(2, np.array([100.0, 110.0, 125.0])).history

        self.assertTrue(np.isnan(z_score[0]))
        np.testing.assert_allclose(z_score[1:], np.array([0.70710678, 0.70710678]))

    def test_z_score_trigger_uses_precomputed_z_score(self) -> None:
        """Z-score triggers compare against a precomputed Z-score series."""
        trigger = ZScore(1.0, 2)
        prices = np.array([100.0, 110.0, 120.0])
        z_score = np.array([np.nan, 0.5, 1.5])

        self.assertFalse(trigger.hit(1, prices, z_score))
        self.assertTrue(trigger.hit(2, prices, z_score))

    def test_rebalances_only_on_price_crosses_of_configured_sma(self) -> None:
        """Primary/alternate switches preserve non-sleeve baseline weights."""
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
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
            length=3,
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((None, None), (dates, None), portfolio, trace=False)

        self.assertEqual([activity.date for activity in strategy.activity_schedule], [dates[3], dates[6]])
        np.testing.assert_array_equal(strategy.activity_schedule[0].weights, np.array([0.0, 0.9, 0.1]))
        np.testing.assert_array_equal(strategy.activity_schedule[1].weights, np.array([0.9, 0.0, 0.1]))
        self.assertIsNone(strategy.activity_schedule[0].rebalance_indices)
        self.assertIsNone(strategy.activity_schedule[1].rebalance_indices)

    def test_rebalance_on_rotation_false_scopes_trades_to_rotation_tickers(self) -> None:
        """Partial rotations leave non-primary/alternate positions untouched."""
        dates = np.array(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
            ],
            dtype="datetime64[D]",
        )
        prices = np.array([10.0, 12.0, 14.0, 12.0])

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertEqual(tickers, ("SPY",))
            return (dates, prices.reshape(-1, 1)), None

        portfolio = Portfolio(("UPRO", "TLT", "CASH"), (0.6, 0.3, 0.1))
        strategy = SimpleMovingAverageRebalance(
            target="SPY",
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
            length=3,
            rebalance_on_rotation=False,
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((None, None), (dates, None), portfolio, trace=False)

        self.assertEqual([activity.date for activity in strategy.activity_schedule], [dates[3]])
        np.testing.assert_array_equal(strategy.activity_schedule[0].weights, np.array([0.0, 0.9, 0.1]))
        self.assertEqual(strategy.activity_schedule[0].rebalance_indices, (0, 1))

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
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
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
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
            length=2,
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((None, None), (dates[1:], None), portfolio, trace=False)

        self.assertEqual(strategy.activity_schedule, [])

    def test_short_window_sma_uses_full_history_before_truncation(self) -> None:
        """Short configured date ranges still use pre-start prices for the SMA."""
        dates = np.array(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
            dtype="datetime64[D]",
        )
        prices = np.array([100.0, 100.0, 1.0, 1.0])

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertEqual(tickers, ("SPY",))
            return (dates, prices.reshape(-1, 1)), None

        portfolio = Portfolio(("UPRO", "TLT"), (1.0, 0.0))
        strategy = SimpleMovingAverageRebalance(
            target="SPY",
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
            length=3,
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((None, None), (dates[2:], None), portfolio, trace=False)

        self.assertEqual([activity.date for activity in strategy.activity_schedule], [dates[3]])
        np.testing.assert_array_equal(strategy.activity_schedule[0].weights, np.array([0.0, 1.0]))

    def test_primary_min_reentry_days_only_delays_primary_return(self) -> None:
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
            primary={"ticker": "UPRO", "min_reentry_days": 3},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
            length=2,
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((None, None), (dates, None), portfolio, trace=False)

        self.assertEqual([activity.date for activity in strategy.activity_schedule], [dates[2], dates[5], dates[6]])
        np.testing.assert_array_equal(strategy.activity_schedule[0].weights, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(strategy.activity_schedule[1].weights, np.array([1.0, 0.0]))
        np.testing.assert_array_equal(strategy.activity_schedule[2].weights, np.array([0.0, 1.0]))

    def test_alternate_min_reentry_days_delays_alternate_return(self) -> None:
        """Alternate can also require a cooldown before being reentered."""
        dates = np.array(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
            ],
            dtype="datetime64[D]",
        )
        prices = np.array([10.0, 12.0, 10.0, 12.0, 12.0, 14.0, 10.0, 8.0])

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertEqual(tickers, ("SPY",))
            return (dates, prices.reshape(-1, 1)), None

        portfolio = Portfolio(("UPRO", "TLT"), (1.0, 0.0))
        strategy = SimpleMovingAverageRebalance(
            target="SPY",
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 4},
            length=2,
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((None, None), (dates, None), portfolio, trace=False)

        self.assertEqual([activity.date for activity in strategy.activity_schedule], [dates[2], dates[3], dates[7]])
        np.testing.assert_array_equal(strategy.activity_schedule[0].weights, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(strategy.activity_schedule[1].weights, np.array([1.0, 0.0]))
        np.testing.assert_array_equal(strategy.activity_schedule[2].weights, np.array([0.0, 1.0]))

    def test_min_consecutive_reentry_days_work_with_reentry_cooldown(self) -> None:
        """Each leg requires enough elapsed days and consecutive SMA closes."""
        dates = np.array(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
            ],
            dtype="datetime64[D]",
        )
        prices = np.array([10.0, 12.0, 10.0, 8.0, 12.0, 14.0, 10.0, 8.0])

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertEqual(tickers, ("SPY",))
            return (dates, prices.reshape(-1, 1)), None

        portfolio = Portfolio(("UPRO", "TLT"), (1.0, 0.0))
        strategy = SimpleMovingAverageRebalance(
            target="SPY",
            primary={
                "ticker": "UPRO",
                "min_reentry_days": 2,
                "min_consecutive_reentry_days": 2,
            },
            alternate={
                "ticker": "TLT",
                "min_reentry_days": 2,
                "min_consecutive_reentry_days": 2,
            },
            length=2,
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((None, None), (dates, None), portfolio, trace=False)

        self.assertEqual([activity.date for activity in strategy.activity_schedule], [dates[3], dates[5], dates[7]])
        np.testing.assert_array_equal(strategy.activity_schedule[0].weights, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(strategy.activity_schedule[1].weights, np.array([1.0, 0.0]))
        np.testing.assert_array_equal(strategy.activity_schedule[2].weights, np.array([0.0, 1.0]))

    def test_take_profit_new_ath_respects_cooldown_and_keeps_cash(self) -> None:
        """New active-ticker highs can harvest excess value without rebalancing."""
        dates = np.array(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            dtype="datetime64[D]",
        )
        prices = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        portfolio_prices = np.column_stack((prices, np.full(len(prices), 10.0), np.full(len(prices), 10.0)))

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertEqual(tickers, ("SPY",))
            return (dates, prices.reshape(-1, 1)), None

        portfolio = Portfolio(("UPRO", "TLT", "CASH"), (0.8, 0.1, 0.1))
        strategy = SimpleMovingAverageRebalance(
            target="SPY",
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
            length=2,
            take_profit={"cooldown": 2, "rebalance": False, "trigger": "newath"},
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((portfolio_prices, None), (dates, None), portfolio, trace=False)

        self.assertEqual([activity.date for activity in strategy.activity_schedule], [dates[1], dates[3]])
        self.assertTrue(all(isinstance(activity, TakeProfit) for activity in strategy.activity_schedule))
        self.assertEqual([activity.ticker_index for activity in strategy.activity_schedule], [0, 0])
        self.assertEqual([activity.target_weight for activity in strategy.activity_schedule], [0.9, 0.9])

    def test_take_profit_z_score_can_schedule_rebalance(self) -> None:
        """Z-score triggers keep rebalance behavior inside TakeProfit activities."""
        dates = np.array(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            dtype="datetime64[D]",
        )
        prices = np.array([10.0, 10.0, 13.0, 13.0])
        portfolio_prices = np.column_stack((prices, np.full(len(prices), 10.0), np.full(len(prices), 10.0)))

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertEqual(tickers, ("SPY",))
            return (dates, prices.reshape(-1, 1)), None

        portfolio = Portfolio(("UPRO", "TLT", "CASH"), (0.8, 0.1, 0.1))
        strategy = SimpleMovingAverageRebalance(
            target="SPY",
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
            length=2,
            take_profit={
                "cooldown": 0,
                "rebalance": True,
                "trigger": {"zscore": {"distance": 0.1}},
                "max_cash_ratio": 0.1,
            },
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((portfolio_prices, None), (dates, None), portfolio, trace=False)

        self.assertEqual([activity.date for activity in strategy.activity_schedule], [dates[2]])
        self.assertIsInstance(strategy.activity_schedule[0], TakeProfit)
        self.assertTrue(strategy.activity_schedule[0].rebalance)
        self.assertEqual(strategy.activity_schedule[0].max_cash_ratio, 0.1)
        np.testing.assert_array_equal(strategy.activity_schedule[0].weights, np.array([0.9, 0.0, 0.1]))

    def test_take_profit_z_score_uses_target_not_active_leg(self) -> None:
        """Z-score take profit uses the rotation target series, not the leveraged ticker."""
        dates = np.array(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            dtype="datetime64[D]",
        )
        target_prices = np.array([10.0, 10.0, 10.0, 10.0])
        primary_prices = np.array([10.0, 10.0, 100.0, 100.0])
        portfolio_prices = np.column_stack(
            (primary_prices, np.full(len(dates), 10.0), np.full(len(dates), 10.0))
        )

        def historical_data(tickers: tuple[str, ...]) -> tuple[tuple[np.ndarray, np.ndarray], None]:
            self.assertEqual(tickers, ("SPY",))
            return (dates, target_prices.reshape(-1, 1)), None

        portfolio = Portfolio(("UPRO", "TLT", "CASH"), (0.8, 0.1, 0.1))
        strategy = SimpleMovingAverageRebalance(
            target="SPY",
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
            length=2,
            take_profit={
                "cooldown": 0,
                "rebalance": True,
                "trigger": {"zscore": {"distance": 0.1}},
            },
        )

        with patch("backtesting.strategies.rebalance.get_historical_data", historical_data):
            strategy.procedure((portfolio_prices, None), (dates, None), portfolio, trace=False)

        self.assertFalse(
            strategy.activity_schedule,
            msg="Target is flat on its SMA while only the primary leg spikes; no take profit.",
        )

    def test_zscore_length_can_differ_from_sma_length(self) -> None:
        """Z-score rolling window on ``zscore`` may differ from the rotation SMA length."""
        s = SimpleMovingAverageRebalance(
            target="SPY",
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
            length=200,
            take_profit={"trigger": {"zscore": {"distance": 0.1, "length": 50}}},
        )
        self.assertEqual(s.length, 200)
        self.assertIsInstance(s.take_profit.trigger, ZScore)
        self.assertEqual(s.take_profit.trigger.length, 50)

    def test_zscore_length_defaults_to_sma_length(self) -> None:
        """Omitting ``zscore.length`` uses the strategy SMA length."""
        s = SimpleMovingAverageRebalance(
            target="SPY",
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
            length=42,
            take_profit={"trigger": {"zscore": {"distance": 0.0}}},
        )
        self.assertIsInstance(s.take_profit.trigger, ZScore)
        self.assertEqual(s.take_profit.trigger.length, 42)

    def test_trigger_accepts_class_name_string_for_no_arg_triggers(self) -> None:
        """A bare class name string configures triggers that take no properties."""
        s = SimpleMovingAverageRebalance(
            target="SPY",
            primary={"ticker": "UPRO", "min_reentry_days": 0},
            alternate={"ticker": "TLT", "min_reentry_days": 0},
            length=2,
            take_profit={"trigger": "newath"},
        )
        self.assertIsInstance(s.take_profit.trigger, NewATH)

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
            SimpleMovingAverageRebalance(
                target="SPY",
                primary={"ticker": "UPRO", "min_reentry_days": 0},
                alternate={"ticker": "TLT", "min_reentry_days": 0},
                length=2,
            ),
            StdDevRebalance(
                target="SPY",
                primary={"ticker": "UPRO", "min_reentry_days": 0},
                alternate={"ticker": "TLT", "min_reentry_days": 0},
                length=2,
                long_length=2,
                short_length=2,
            ),
            SMACrossRebalance(
                target="SPY",
                primary={"ticker": "UPRO", "min_reentry_days": 0},
                alternate={"ticker": "TLT", "min_reentry_days": 0},
                length=2,
                long_length=3,
                short_length=2,
            ),
            VolatilityAdjustedSMARebalance(
                target="SPY",
                primary={"ticker": "UPRO", "min_reentry_days": 0},
                alternate={"ticker": "TLT", "min_reentry_days": 0},
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

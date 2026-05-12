"""Tests for portfolio allocation and rebalance behavior."""

import unittest

import numpy as np
import pandas as pd

from backtesting.core.portfolio import Portfolio
from backtesting.core.types import Allocate, Rebalance, TakeProfit


class PortfolioTests(unittest.TestCase):
    def test_default_allocation_follows_latest_rebalance_weights(self) -> None:
        """Omitted contribution weights use the current active portfolio target."""
        portfolio = Portfolio(("TQQQ", "QQQ"), (1.0, 0.0))
        prices = np.array([10.0, 10.0])

        portfolio.allocate("initial", prices, 100.0, weights=None)
        portfolio.rebalance("risk off", prices, weights=np.array([0.0, 1.0]))
        portfolio.allocate("contribution", prices, 50.0, weights=None)

        np.testing.assert_array_equal(portfolio.weights, np.array([0.0, 1.0]))
        self.assertAlmostEqual(portfolio.current_shares[0], 0.0)
        self.assertAlmostEqual(portfolio.current_shares[1], 15.0)

    def test_scoped_rebalance_leaves_other_positions_untouched(self) -> None:
        """Primary/alternate rotations can avoid rebalancing unrelated tickers."""
        portfolio = Portfolio(("TQQQ", "QQQ", "FDL"), (0.9, 0.0, 0.1))

        portfolio.allocate("initial", np.array([10.0, 10.0, 10.0]), 100.0, weights=None)
        portfolio.rebalance(
            "risk off",
            np.array([10.0, 10.0, 20.0]),
            weights=np.array([0.0, 0.9, 0.1]),
            rebalance_indices=(0, 1),
        )

        np.testing.assert_array_equal(portfolio.weights, np.array([0.0, 0.9, 0.1]))
        self.assertAlmostEqual(portfolio.current_shares[0], 0.0)
        self.assertAlmostEqual(portfolio.current_shares[1], 9.0)
        self.assertAlmostEqual(portfolio.current_shares[2], 1.0)

    def test_rebalance_invests_cash_sweep(self) -> None:
        """Full portfolio rebalances include cash-sweep value in target value."""
        portfolio = Portfolio(("TQQQ", "FDL"), (0.9, 0.1), cash_source_ticker="CASH")
        portfolio.current_shares = np.array([9.0, 1.0, 10.0])

        portfolio.rebalance("rebalance day", np.array([10.0, 10.0, 1.0]), weights=np.array([0.9, 0.1]))

        self.assertAlmostEqual(portfolio.current_shares[0], 9.9)
        self.assertAlmostEqual(portfolio.current_shares[1], 1.1)
        self.assertAlmostEqual(portfolio.current_shares[2], 0.0)

    def test_rebalance_does_not_sell_restricted_tickers(self) -> None:
        """Ticker config can prevent rebalances from selling selected holdings."""
        portfolio = Portfolio(
            ("LOCK", "BUY"),
            (1.0, 0.0),
            cash_source_ticker="CASH",
            sell_on_rebalance=(False, True),
        )
        portfolio.current_shares = np.array([10.0, 0.0, 20.0])

        portfolio.rebalance("rebalance day", np.array([10.0, 10.0, 1.0]), weights=np.array([0.0, 1.0]))

        self.assertAlmostEqual(portfolio.current_shares[0], 10.0)
        self.assertAlmostEqual(portfolio.current_shares[1], 2.0)
        self.assertAlmostEqual(portfolio.current_shares[2], 0.0)
        np.testing.assert_array_equal(portfolio.weights, np.array([0.0, 1.0]))

    def test_take_profit_sells_excess_to_cash_sweep(self) -> None:
        """Profit taking harvests only the excess value above the target weight."""
        portfolio = Portfolio(("TQQQ", "FDL"), (0.9, 0.1), cash_source_ticker="CASH")
        portfolio.current_shares = np.array([10.0, 1.0, 0.0])

        portfolio.take_profit("profit day", np.array([12.0, 10.0, 1.0]), ticker_index=0, target_weight=0.9)

        self.assertAlmostEqual(portfolio.current_shares[0], 9.75)
        self.assertAlmostEqual(portfolio.current_shares[1], 1.0)
        self.assertAlmostEqual(portfolio.current_shares[2], 3.0)
        self.assertAlmostEqual(portfolio.current_value(np.array([12.0, 10.0, 1.0])), 130.0)

    def test_take_profit_can_target_max_cash_ratio_single_risk_bucket(self) -> None:
        """Max cash ratio sizing applies when no other risky sleeve absorbs proceeds."""
        portfolio = Portfolio(("TQQQ",), (1.0,), cash_source_ticker="CASH")
        portfolio.current_shares = np.array([10.0, 0.0])

        portfolio.take_profit(
            "profit day",
            np.array([12.0, 1.0]),
            ticker_index=0,
            target_weight=1.0,
            max_cash_ratio=0.1,
        )

        self.assertAlmostEqual(portfolio.current_shares[0], 9.0)
        self.assertAlmostEqual(portfolio.current_shares[1], 12.0)
        self.assertAlmostEqual(portfolio.current_value(np.array([12.0, 1.0])), 120.0, places=2)

    def test_take_profit_with_additional_risk_uses_target_weight_not_max_cash(self) -> None:
        """With multiple risky weights, trim sizing follows target_weight; redeploy via rebalance."""
        portfolio = Portfolio(("TQQQ", "FDL"), (0.9, 0.1), cash_source_ticker="CASH")
        portfolio.current_shares = np.array([10.0, 1.0, 0.0])

        portfolio.take_profit(
            "profit day",
            np.array([12.0, 10.0, 1.0]),
            ticker_index=0,
            target_weight=0.9,
            max_cash_ratio=0.1,
        )

        self.assertAlmostEqual(portfolio.current_shares[0], 9.75)
        self.assertAlmostEqual(portfolio.current_shares[1], 1.0)
        self.assertAlmostEqual(portfolio.current_shares[2], 3.0)
        self.assertAlmostEqual(portfolio.current_value(np.array([12.0, 10.0, 1.0])), 130.0, places=2)

    def test_take_profit_can_rebalance_inside_portfolio_method(self) -> None:
        """The take-profit activity can delegate to the portfolio rebalance path."""
        portfolio = Portfolio(("TQQQ", "FDL"), (0.9, 0.1), cash_source_ticker="CASH")
        portfolio.current_shares = np.array([10.0, 1.0, 0.0])

        portfolio.take_profit(
            "profit day",
            np.array([12.0, 10.0, 1.0]),
            ticker_index=0,
            target_weight=0.9,
            rebalance=True,
            weights=np.array([0.9, 0.1]),
        )

        self.assertAlmostEqual(portfolio.current_shares[0], 9.75)
        self.assertAlmostEqual(portfolio.current_shares[1], 1.3)
        self.assertAlmostEqual(portfolio.current_shares[2], 0.0)

    def test_take_profit_skips_rebalance_when_only_cash_can_absorb(self) -> None:
        """With one risky leg + cash sleeve, rebalance must not redeploy into the same leg."""
        portfolio = Portfolio(("TQQQ",), (1.0,), cash_source_ticker="CASH")
        portfolio.current_shares = np.array([10.0, 0.0])

        portfolio.take_profit(
            "profit day",
            np.array([12.0, 1.0]),
            ticker_index=0,
            target_weight=1.0,
            rebalance=True,
            weights=np.array([1.0]),
            max_cash_ratio=0.2,
        )

        self.assertAlmostEqual(portfolio.current_shares[0], 8.0)
        self.assertAlmostEqual(portfolio.current_shares[1], 24.0)
        self.assertAlmostEqual(portfolio.current_value(np.array([12.0, 1.0])), 120.0, places=2)

    def test_take_profit_activity_sorts_by_date_with_other_activities(self) -> None:
        """Mixed date inputs normalize before activity ordering."""
        activities = [
            Allocate(pd.Timestamp("2020-01-03"), 100.0),
            TakeProfit(np.datetime64("2020-01-02"), 0, 0.9, False),
            Rebalance(np.datetime64("2020-01-01")),
        ]

        self.assertEqual(
            [activity.date for activity in sorted(activities)],
            [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-03"),
            ],
        )

    def test_cash_source_sweeps_unallocated_into_shares(self) -> None:
        """Residual dollars after risky buys become shares of the configured cash ticker."""
        portfolio = Portfolio(("AAA",), (1.0,), cash_source_ticker="CASH")
        prices = np.array([100.0, 2.0])
        portfolio._sweep_to_cash(prices, 50.55)

        self.assertAlmostEqual(portfolio.current_shares[0], 0.0)
        self.assertAlmostEqual(portfolio.current_shares[1], 25.275)
        self.assertAlmostEqual(portfolio.current_value(prices), 50.55)

    def test_take_profit_applies_short_term_capital_gains_tax(self) -> None:
        """Lots held one year or less use the configured short-term gains rate."""
        portfolio = Portfolio(
            ("AAA",),
            (1.0,),
            cash_source_ticker="CASH",
            short_term_capital_gains_rate=0.35,
            long_term_capital_gains_rate=0.15,
        )
        buy_prices = np.array([100.0, 1.0])
        sell_prices = np.array([120.0, 1.0])
        portfolio.allocate(
            "buy day",
            buy_prices,
            1000.0,
            weights=None,
            date=pd.Timestamp("2020-01-01"),
        )

        portfolio.take_profit(
            "profit day",
            sell_prices,
            ticker_index=0,
            target_weight=0.9,
            date=pd.Timestamp("2020-06-01"),
        )

        final_value = portfolio.current_value(sell_prices)
        self.assertAlmostEqual(portfolio.current_shares[0] * sell_prices[0] / final_value, 0.9, places=3)
        self.assertGreater(portfolio.current_shares[1], 113.0)
        self.assertAlmostEqual(portfolio.total_tax_paid, round(portfolio.short_term_realized_gains * 0.35, 2))

    def test_take_profit_applies_net_investment_income_tax_to_realized_gains(self) -> None:
        """NIIT adds a 3.8% surtax to realized positive capital gains."""
        portfolio = Portfolio(("AAA",), (1.0,), cash_source_ticker="CASH", net_investment_income=True)
        buy_prices = np.array([100.0, 1.0])
        sell_prices = np.array([120.0, 1.0])
        portfolio.allocate(
            "buy day",
            buy_prices,
            1000.0,
            weights=None,
            date=pd.Timestamp("2020-01-01"),
        )

        portfolio.take_profit(
            "profit day",
            sell_prices,
            ticker_index=0,
            target_weight=0.9,
            date=pd.Timestamp("2020-06-01"),
        )

        final_value = portfolio.current_value(sell_prices)
        self.assertAlmostEqual(portfolio.current_shares[0] * sell_prices[0] / final_value, 0.9, places=3)
        self.assertAlmostEqual(portfolio.total_tax_paid, portfolio.net_investment_income_tax_paid)
        self.assertAlmostEqual(
            portfolio.total_tax_paid,
            round(portfolio.short_term_realized_gains * 0.038, 2),
        )

    def test_net_investment_income_tax_applies_to_taxable_distributions(self) -> None:
        """NIIT reduces distributions before they can be reinvested."""
        portfolio = Portfolio(("AAA",), (1.0,), net_investment_income=True)

        net_distribution = portfolio.tax_distribution(100.0)

        self.assertAlmostEqual(net_distribution, 96.2)
        self.assertAlmostEqual(portfolio.total_tax_paid, 3.8)
        self.assertAlmostEqual(portfolio.net_investment_income_tax_paid, 3.8)

    def test_distribution_taxable_as_defaults_to_zero(self) -> None:
        """Ticker-specific distribution tax character is opt-in per security."""
        portfolio = Portfolio(("AAA",), (1.0,))
        portfolio.allocate(
            "buy day",
            np.array([100.0]),
            1000.0,
            weights=None,
            date=pd.Timestamp("2020-01-01"),
        )

        net_distribution = portfolio.tax_distributions(
            np.array([10.0]),
            date=pd.Timestamp("2020-06-01"),
        )

        self.assertAlmostEqual(net_distribution, 100.0)
        self.assertAlmostEqual(portfolio.total_tax_paid, 0.0)
        self.assertAlmostEqual(portfolio.distribution_tax_paid, 0.0)

    def test_distribution_taxable_as_short_term_uses_portfolio_tax_rate(self) -> None:
        """Short-term taxable distribution percentages use the configured short-term rate."""
        portfolio = Portfolio(
            ("AAA",),
            (1.0,),
            short_term_capital_gains_rate=0.37,
            long_term_capital_gains_rate=0.15,
            distribution_taxable_as=((1.0, 0.0, 0.0),),
        )
        portfolio.allocate(
            "buy day",
            np.array([100.0]),
            1000.0,
            weights=None,
            date=pd.Timestamp("2020-01-01"),
        )

        net_distribution = portfolio.tax_distributions(
            np.array([10.0]),
            date=pd.Timestamp("2020-06-01"),
        )

        self.assertAlmostEqual(net_distribution, 63.0)
        self.assertAlmostEqual(portfolio.total_tax_paid, 37.0)
        self.assertAlmostEqual(portfolio.distribution_tax_paid, 37.0)

    def test_distribution_taxable_as_long_term_uses_portfolio_tax_rate(self) -> None:
        """Long-term taxable distribution percentages use the configured long-term rate."""
        portfolio = Portfolio(
            ("AAA",),
            (1.0,),
            short_term_capital_gains_rate=0.37,
            long_term_capital_gains_rate=0.15,
            distribution_taxable_as=((0.0, 1.0, 0.0),),
        )
        portfolio.allocate(
            "buy day",
            np.array([100.0]),
            1000.0,
            weights=None,
            date=pd.Timestamp("2020-01-01"),
        )

        net_distribution = portfolio.tax_distributions(
            np.array([10.0]),
            date=pd.Timestamp("2021-01-02"),
        )

        self.assertAlmostEqual(net_distribution, 85.0)
        self.assertAlmostEqual(portfolio.total_tax_paid, 15.0)
        self.assertAlmostEqual(portfolio.distribution_tax_paid, 15.0)

    def test_distribution_taxable_as_can_leave_part_of_distribution_tax_free(self) -> None:
        """Untyped distribution percentages are not taxed by capital gains rates."""
        portfolio = Portfolio(
            ("AAA",),
            (1.0,),
            short_term_capital_gains_rate=0.37,
            long_term_capital_gains_rate=0.15,
            distribution_taxable_as=((0.25, 0.50, 0.0),),
        )
        portfolio.allocate(
            "buy day",
            np.array([100.0]),
            1000.0,
            weights=None,
            date=pd.Timestamp("2020-01-01"),
        )

        net_distribution = portfolio.tax_distributions(
            np.array([10.0]),
            date=pd.Timestamp("2020-06-01"),
        )

        self.assertAlmostEqual(net_distribution, 83.25)
        self.assertAlmostEqual(portfolio.total_tax_paid, 16.75)
        self.assertAlmostEqual(portfolio.distribution_tax_paid, 16.75)

    def test_distribution_return_of_capital_lowers_cost_basis_without_tax(self) -> None:
        """Return of capital is received tax-free while reducing lot cost basis."""
        portfolio = Portfolio(
            ("AAA",),
            (1.0,),
            short_term_capital_gains_rate=0.37,
            long_term_capital_gains_rate=0.15,
            distribution_taxable_as=((0.0, 0.0, 1.0),),
        )
        portfolio.allocate(
            "buy day",
            np.array([100.0]),
            1000.0,
            weights=None,
            date=pd.Timestamp("2020-01-01"),
        )

        net_distribution = portfolio.tax_distributions(
            np.array([10.0]),
            date=pd.Timestamp("2020-06-01"),
        )

        self.assertAlmostEqual(net_distribution, 100.0)
        self.assertAlmostEqual(portfolio.total_tax_paid, 0.0)
        self.assertAlmostEqual(portfolio.distribution_tax_paid, 0.0)
        self.assertAlmostEqual(portfolio.tax_lots[0][0].cost_basis, 90.0)

    def test_distribution_return_of_capital_is_excluded_from_niit(self) -> None:
        """NIIT is applied only to the non-return-of-capital distribution amount."""
        portfolio = Portfolio(
            ("AAA",),
            (1.0,),
            net_investment_income=True,
            distribution_taxable_as=((0.0, 0.0, 0.5),),
        )
        portfolio.allocate(
            "buy day",
            np.array([100.0]),
            1000.0,
            weights=None,
            date=pd.Timestamp("2020-01-01"),
        )

        net_distribution = portfolio.tax_distributions(
            np.array([10.0]),
            date=pd.Timestamp("2020-06-01"),
        )

        self.assertAlmostEqual(net_distribution, 98.1)
        self.assertAlmostEqual(portfolio.total_tax_paid, 1.9)
        self.assertAlmostEqual(portfolio.net_investment_income_tax_paid, 1.9)
        self.assertAlmostEqual(portfolio.tax_lots[0][0].cost_basis, 95.0)

    def test_take_profit_applies_long_term_capital_gains_tax(self) -> None:
        """Lots held more than one year use the configured long-term gains rate."""
        portfolio = Portfolio(
            ("AAA",),
            (1.0,),
            cash_source_ticker="CASH",
            short_term_capital_gains_rate=0.35,
            long_term_capital_gains_rate=0.15,
        )
        buy_prices = np.array([100.0, 1.0])
        sell_prices = np.array([120.0, 1.0])
        portfolio.allocate(
            "buy day",
            buy_prices,
            1000.0,
            weights=None,
            date=pd.Timestamp("2020-01-01"),
        )

        portfolio.take_profit(
            "profit day",
            sell_prices,
            ticker_index=0,
            target_weight=0.9,
            date=pd.Timestamp("2021-01-02"),
        )

        final_value = portfolio.current_value(sell_prices)
        self.assertAlmostEqual(portfolio.current_shares[0] * sell_prices[0] / final_value, 0.9, places=3)
        self.assertGreater(portfolio.current_shares[1], 117.0)
        self.assertAlmostEqual(portfolio.total_tax_paid, round(portfolio.long_term_realized_gains * 0.15, 2))

    def test_rebalance_reinvests_sale_proceeds_after_capital_gains_tax(self) -> None:
        """Rebalances pay realized-gain tax before buying replacement holdings."""
        portfolio = Portfolio(
            ("AAA", "BBB"),
            (1.0, 0.0),
            short_term_capital_gains_rate=0.35,
            long_term_capital_gains_rate=0.15,
        )
        portfolio.allocate(
            "buy day",
            np.array([100.0, 100.0]),
            1000.0,
            weights=None,
            date=pd.Timestamp("2020-01-01"),
        )

        portfolio.rebalance(
            "risk off",
            np.array([120.0, 100.0]),
            weights=np.array([0.0, 1.0]),
            date=pd.Timestamp("2020-06-01"),
        )

        self.assertAlmostEqual(portfolio.current_shares[0], 0.0)
        self.assertAlmostEqual(portfolio.current_shares[1], 11.3)
        self.assertAlmostEqual(portfolio.total_tax_paid, 70.0)
        self.assertAlmostEqual(portfolio.current_value(np.array([120.0, 100.0])), 1130.0)

    def test_rebalance_sizes_partial_sales_against_after_tax_nav(self) -> None:
        """Partial rebalances sell enough to hit target weights after tax drag."""
        portfolio = Portfolio(
            ("AAA", "BBB"),
            (1.0, 0.0),
            short_term_capital_gains_rate=0.35,
            long_term_capital_gains_rate=0.15,
        )
        buy_prices = np.array([100.0, 100.0])
        rebalance_prices = np.array([120.0, 100.0])
        portfolio.allocate(
            "buy day",
            buy_prices,
            1000.0,
            weights=None,
            date=pd.Timestamp("2020-01-01"),
        )

        portfolio.rebalance(
            "half risk",
            rebalance_prices,
            weights=np.array([0.5, 0.5]),
            date=pd.Timestamp("2020-06-01"),
        )

        final_value = portfolio.current_value(rebalance_prices)
        self.assertAlmostEqual(portfolio.current_shares[0] * rebalance_prices[0] / final_value, 0.5, places=3)
        self.assertAlmostEqual(portfolio.current_shares[1] * rebalance_prices[1] / final_value, 0.5, places=3)
        self.assertGreater(portfolio.total_tax_paid, 35.0)


if __name__ == "__main__":
    unittest.main()

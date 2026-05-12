from dataclasses import dataclass

import numpy as np
import pandas as pd


NET_INVESTMENT_INCOME_TAX_RATE = 0.038


@dataclass
class TaxLot:
    """Shares acquired at a single date and cost basis."""

    shares: float
    cost_basis: float
    acquisition_date: pd.Timestamp | None


class Portfolio:
    """Stateful portfolio model used during backtest execution."""

    def __init__(
        self,
        tickers: tuple[str, ...],
        weights: tuple[float, ...],
        cash_source_ticker: str | None = None,
        track: bool = False,
        short_term_capital_gains_rate: float = 0.0,
        long_term_capital_gains_rate: float = 0.0,
        net_investment_income: bool = False,
        distribution_taxable_as: tuple[tuple[float, float, float], ...] | None = None,
        sell_on_rebalance: tuple[bool, ...] | None = None,
    ) -> None:
        """Initializes portfolio holdings, weights, and execution totals."""
        self.risky_count = len(tickers)
        self.cash_source_ticker = cash_source_ticker.strip() if cash_source_ticker else None
        if self.cash_source_ticker:
            self.tickers: tuple[str, ...] = (*tickers, self.cash_source_ticker)
            self.cash_index = self.risky_count
            self.weights: tuple[float, ...] = tuple(np.array(weights).round(decimals=4))
        else:
            self.tickers = tickers
            self.cash_index = None
            self.weights: tuple[float, ...] = tuple(np.array(weights).round(decimals=4))

        self.ticker_idx: dict[str, int] = {self.tickers[i]: i for i in range(len(self.tickers))}
        self.current_shares: np.ndarray = np.zeros(len(self.tickers))
        self.tax_lots: list[list[TaxLot]] = [[] for _ in self.tickers]
        self.short_term_capital_gains_rate = float(short_term_capital_gains_rate)
        self.long_term_capital_gains_rate = float(long_term_capital_gains_rate)
        self.net_investment_income = bool(net_investment_income)
        if sell_on_rebalance is None:
            sell_on_rebalance = tuple(True for _ in range(self.risky_count))
        if len(sell_on_rebalance) != self.risky_count:
            raise ValueError("sell_on_rebalance count must match risky tickers.")
        self.sell_on_rebalance = tuple(bool(value) for value in sell_on_rebalance)
        if self.cash_index is not None:
            self.sell_on_rebalance = (*self.sell_on_rebalance, True)
        if distribution_taxable_as is None:
            distribution_taxable_as = tuple((0.0, 0.0, 0.0) for _ in range(self.risky_count))
        if len(distribution_taxable_as) != self.risky_count:
            raise ValueError("Distribution taxable-as count must match risky tickers.")
        distribution_taxable_as_normalized = []
        for distribution_tax_character in distribution_taxable_as:
            if len(distribution_tax_character) == 2:
                short_percent, long_percent = distribution_tax_character
                return_of_capital_percent = 0.0
            elif len(distribution_tax_character) == 3:
                short_percent, long_percent, return_of_capital_percent = distribution_tax_character
            else:
                raise ValueError("Distribution taxable-as entries must have two or three percentages.")
            distribution_taxable_as_normalized.append(
                (float(short_percent), float(long_percent), float(return_of_capital_percent))
            )
        self.distribution_taxable_as = tuple(distribution_taxable_as_normalized)
        for short_percent, long_percent, return_of_capital_percent in self.distribution_taxable_as:
            if (
                short_percent < 0.0
                or long_percent < 0.0
                or return_of_capital_percent < 0.0
                or short_percent + long_percent + return_of_capital_percent > 1.0 + 1e-9
            ):
                raise ValueError("Distribution taxable-as percentages must be non-negative and sum to 1.0 or less.")
        if self.cash_index is not None:
            self.distribution_taxable_as = (*self.distribution_taxable_as, (0.0, 0.0, 0.0))
        self.total_tax_paid = 0.0
        self.net_investment_income_tax_paid = 0.0
        self.distribution_tax_paid = 0.0
        self.short_term_realized_gains = 0.0
        self.long_term_realized_gains = 0.0

        self.track = track
        self.total_new_capital = 0.0
        self.total_distribution = 0.0
        self.distribution_history = {}
        self.contribution_flows: list[tuple[object, float]] = []

    def current_value(self, price_data: np.ndarray) -> float:
        """Returns the current total value of holdings, including the cash sweep."""
        return float(np.sum(self.current_shares * price_data))

    def _cash_value(self, price_data: np.ndarray) -> float:
        """Returns the current value of the configured cash sweep."""
        if self.cash_index is None:
            return 0.0
        return float(round(self.current_shares[self.cash_index] * price_data[self.cash_index], 2))

    def record_distribution(self, date: object, amount: float) -> None:
        """Records portfolio-level distributions paid on a specific date."""
        self.distribution_history[date] = self.distribution_history.get(date, 0.0) + float(amount)

    def _tax_enabled(self) -> bool:
        """Returns True when realized capital gains should reduce portfolio value."""
        return (
            self.short_term_capital_gains_rate > 0.0
            or self.long_term_capital_gains_rate > 0.0
            or self.net_investment_income
        )

    def tax_distribution(self, amount: float, taxable_amount: float | None = None) -> float:
        """Applies portfolio-level NIIT to taxable distributions and returns after-tax amount."""
        gross = float(amount)
        niit_taxable = gross if taxable_amount is None else float(taxable_amount)
        if gross <= 0.0 or niit_taxable <= 0.0 or not self.net_investment_income:
            return gross
        tax_paid = float(round(niit_taxable * NET_INVESTMENT_INCOME_TAX_RATE, 2))
        self.net_investment_income_tax_paid += tax_paid
        self.total_tax_paid += tax_paid
        return max(0.0, gross - tax_paid)

    @staticmethod
    def _is_long_term_lot(lot_date: pd.Timestamp | None, tax_date: pd.Timestamp | None) -> bool:
        """Returns whether a lot is treated as long-term on the tax date."""
        return (
            tax_date is not None
            and lot_date is not None
            and tax_date > lot_date + pd.DateOffset(years=1)
        )

    def _ticker_distribution_tax(
        self,
        ticker_index: int,
        distribution_per_share: float,
    ) -> float:
        """Calculates ticker-specific distribution tax from configured tax character."""
        per_share = float(distribution_per_share)
        if per_share <= 0.0:
            return 0.0

        short_percent, long_percent, _ = self.distribution_taxable_as[ticker_index]
        if short_percent <= 0.0 and long_percent <= 0.0:
            return 0.0

        ticker_distribution = float(self.current_shares[ticker_index]) * per_share
        if ticker_distribution <= 0.0:
            return 0.0

        tax_due = ticker_distribution * short_percent * self.short_term_capital_gains_rate
        tax_due += ticker_distribution * long_percent * self.long_term_capital_gains_rate
        return float(round(tax_due, 2))

    def _apply_return_of_capital(
        self,
        ticker_index: int,
        distribution_per_share: float,
    ) -> float:
        """Lowers lot cost basis for return-of-capital distributions and returns the ROC amount."""
        per_share = float(distribution_per_share)
        if per_share <= 0.0:
            return 0.0

        _, _, return_of_capital_percent = self.distribution_taxable_as[ticker_index]
        if return_of_capital_percent <= 0.0:
            return 0.0

        shares = float(self.current_shares[ticker_index])
        if shares <= 0.0:
            return 0.0

        basis_reduction_per_share = per_share * return_of_capital_percent
        for lot in self.tax_lots[ticker_index]:
            lot.cost_basis = max(0.0, float(round(lot.cost_basis - basis_reduction_per_share, 10)))
        return shares * basis_reduction_per_share

    def tax_distributions(self, distribution_amounts: np.ndarray, date: object | None = None) -> float:
        """Applies ticker-specific distribution tax plus NIIT and returns total net payout."""
        amounts = np.asarray(distribution_amounts, dtype=float).ravel()
        gross = float(np.sum(self.current_shares * amounts))
        if gross <= 0.0:
            return gross

        ticker_tax = 0.0
        return_of_capital = 0.0
        for i, distribution_per_share in enumerate(amounts):
            ticker_tax += self._ticker_distribution_tax(i, float(distribution_per_share))
            return_of_capital += self._apply_return_of_capital(i, float(distribution_per_share))

        if ticker_tax > 0.0:
            self.distribution_tax_paid += ticker_tax
            self.total_tax_paid += ticker_tax

        net_after_niit = self.tax_distribution(gross, taxable_amount=max(0.0, gross - return_of_capital))
        return max(0.0, net_after_niit - ticker_tax)

    @staticmethod
    def _normalize_tax_date(date: object | None) -> pd.Timestamp | None:
        """Converts an execution date to a normalized Timestamp when available."""
        if date is None:
            return None
        return pd.Timestamp(date).normalize()

    def _add_tax_lot(
        self,
        ticker_index: int,
        shares: float,
        price: float,
        date: object | None,
    ) -> None:
        """Adds a FIFO tax lot for acquired shares."""
        if shares <= 0.0 or not self._tax_enabled():
            return
        self.tax_lots[ticker_index].append(
            TaxLot(
                shares=float(shares),
                cost_basis=float(price),
                acquisition_date=self._normalize_tax_date(date),
            )
        )

    def _realize_sale_tax(
        self,
        ticker_index: int,
        shares: float,
        price: float,
        date: object | None,
    ) -> float:
        """Consumes FIFO lots for sold shares and returns tax due on realized gains."""
        if shares <= 0.0:
            return 0.0

        sell_date = self._normalize_tax_date(date)
        remaining = min(float(shares), float(self.current_shares[ticker_index]))
        self.current_shares[ticker_index] -= remaining

        if not self._tax_enabled():
            return 0.0

        tax_due = 0.0
        niit_due = 0.0
        lots = self.tax_lots[ticker_index]
        tolerance = 1e-9

        while remaining > tolerance and lots:
            lot = lots[0]
            sold = min(remaining, lot.shares)
            gain = sold * (float(price) - lot.cost_basis)

            if gain > 0.0:
                is_long_term = self._is_long_term_lot(lot.acquisition_date, sell_date)
                if self.net_investment_income:
                    gain_niit = gain * NET_INVESTMENT_INCOME_TAX_RATE
                    niit_due += gain_niit
                    tax_due += gain_niit
                if is_long_term:
                    self.long_term_realized_gains += gain
                    tax_due += gain * self.long_term_capital_gains_rate
                else:
                    self.short_term_realized_gains += gain
                    tax_due += gain * self.short_term_capital_gains_rate

            lot.shares = round(lot.shares - sold, 10)
            remaining = round(remaining - sold, 10)
            if lot.shares <= tolerance:
                lots.pop(0)

        tax_paid = float(round(tax_due, 2))
        if self.net_investment_income:
            self.net_investment_income_tax_paid += float(round(niit_due, 2))
        self.total_tax_paid += tax_paid
        return tax_paid

    def _estimate_sale_tax(
        self,
        ticker_index: int,
        shares: float,
        price: float,
        date: object | None,
    ) -> float:
        """Estimates FIFO tax for a sale without changing shares or tax lots."""
        if shares <= 0.0 or not self._tax_enabled():
            return 0.0

        sell_date = self._normalize_tax_date(date)
        remaining = min(float(shares), float(self.current_shares[ticker_index]))
        tax_due = 0.0
        lots = self.tax_lots[ticker_index]
        tolerance = 1e-9

        for lot in lots:
            if remaining <= tolerance:
                break
            sold = min(remaining, lot.shares)
            gain = sold * (float(price) - lot.cost_basis)
            if gain > 0.0:
                if self.net_investment_income:
                    tax_due += gain * NET_INVESTMENT_INCOME_TAX_RATE
                if self._is_long_term_lot(lot.acquisition_date, sell_date):
                    tax_due += gain * self.long_term_capital_gains_rate
                else:
                    tax_due += gain * self.short_term_capital_gains_rate
            remaining = round(remaining - sold, 10)

        return float(round(tax_due, 2))

    def _buy_shares(
        self,
        ticker_index: int,
        shares: float,
        price: float,
        date: object | None,
    ) -> float:
        """Buys shares, records a tax lot, and returns dollars spent."""
        if shares <= 0.0:
            return 0.0
        self.current_shares[ticker_index] += shares
        self._add_tax_lot(ticker_index, shares, price, date)
        return float(round(shares * price, 2))

    def _sweep_to_cash(self, price_data: np.ndarray, amount: float, date: object | None = None) -> float:
        """Converts available dollars into shares of the cash instrument."""
        cash = float(amount)
        if cash <= 0.0:
            return 0.0
        if self.cash_index is None:
            return cash
        px = float(price_data[self.cash_index])
        if px <= 0.0 or np.isnan(px):
            return cash
        delta_shares = round(cash / px, 4)
        if delta_shares <= 0.0:
            return cash
        spent = self._buy_shares(self.cash_index, delta_shares, px, date)
        return max(0.0, cash - spent)

    def allocate(
        self,
        date_string: str,
        price_data: np.ndarray,
        amount: float,
        weights: tuple[float, ...],
        trace: bool = False,
        date: object | None = None,
    ) -> None:
        """Performs an allocation to all tickers on a specific date.

        If ``contribution_weights`` is set, that mix is used for this contribution only
        (rebalance target weights on the portfolio are unchanged).
        """

        # Check for weights provided
        if weights is None:
            weights = self.weights

        available_capital = float(amount)

        # Don't allocate anything less than $1 to risky assets.
        if available_capital < 1.0:
            self._sweep_to_cash(price_data, available_capital, date)
            return

        weights_arr = np.asarray(weights, dtype=float).ravel()
        risky_prices = price_data[: self.risky_count]

        # Get target ticker values (risky sleeve only)
        ticker_values = (weights_arr * available_capital).round(decimals=2)

        # Convert values to shares
        with np.errstate(divide="ignore", invalid="ignore"):
            ticker_share_deltas_risky = (ticker_values / risky_prices).round(decimals=4)
        ticker_share_deltas_risky = np.nan_to_num(ticker_share_deltas_risky, nan=0.0, posinf=0.0, neginf=0.0)

        spent_risky = 0.0
        for i, shares in enumerate(ticker_share_deltas_risky):
            spent_risky += self._buy_shares(i, float(shares), float(risky_prices[i]), date)

        self._sweep_to_cash(price_data, max(0.0, available_capital - spent_risky), date)

        if trace:
            parts = [
                f"${ticker_values[i]:,.2f} to {self.tickers[i]}"
                for i in range(self.risky_count)
                if abs(float(ticker_values[i])) >= 0.005
            ]
            print(f"Allocated ${np.sum(ticker_values):,.2f} -- " + ", ".join(parts) + f" on {date_string}")

    def rebalance(
        self,
        date_string: str,
        price_data: np.ndarray,
        weights: tuple[float, ...] | None = None,
        rebalance_indices: tuple[int, ...] | None = None,
        trace: bool = False,
        date: object | None = None,
    ) -> None:
        """Performs a rebalance on all tickers in a portfolio."""

        # Use existing weights if not provided; otherwise persist the new target
        # weights so future allocations follow the active rebalance leg.
        if weights is None:
            weights_arr = np.asarray(self.weights, dtype=float).ravel()
        else:
            weights_arr = np.asarray(weights, dtype=float).ravel()
            self.weights = tuple(weights_arr.round(decimals=4))

        # Get the current value of the portfolio, including cash available
        # for a full rebalance (cent-rounded to match other dollar math).
        current_portfolio_value = round(self.current_value(price_data), 2)

        # Get current ticker values
        current_ticker_values = self.current_shares * price_data

        can_sell = np.asarray(self.sell_on_rebalance, dtype=bool)
        rebalance_indices_array = None
        scoped_weights = None
        scoped_weight_total = None
        scoped_current_value = None
        if rebalance_indices is not None:
            rebalance_indices_array = np.array(rebalance_indices, dtype=int)
            scoped_weights = weights_arr[rebalance_indices_array]
            scoped_weight_total = np.sum(scoped_weights)
            if scoped_weight_total <= 0.0:
                return
            scoped_current_value = np.sum(current_ticker_values[rebalance_indices_array]).round(decimals=2)

        filtered_ticker_value_deltas = np.zeros(len(self.tickers), dtype=float)
        tax_estimate = 0.0
        for _ in range(8):
            effective_portfolio_value = max(0.0, current_portfolio_value - tax_estimate)

            if rebalance_indices is None:
                if self.cash_index is None:
                    target_ticker_values = weights_arr * effective_portfolio_value
                else:
                    if len(weights_arr) != self.risky_count:
                        raise ValueError(
                            "Target weights length must match risky tickers when cash_source is configured."
                        )
                    target_ticker_values = np.zeros(len(self.tickers), dtype=float)
                    target_ticker_values[: self.risky_count] = weights_arr * effective_portfolio_value
                    target_ticker_values[self.cash_index] = effective_portfolio_value - np.sum(
                        target_ticker_values[: self.risky_count]
                    )
            else:
                assert rebalance_indices_array is not None
                assert scoped_weights is not None
                assert scoped_weight_total is not None
                assert scoped_current_value is not None
                target_ticker_values = current_ticker_values.copy()
                effective_scoped_value = max(0.0, float(scoped_current_value) - tax_estimate)
                target_ticker_values[rebalance_indices_array] = (
                    scoped_weights / scoped_weight_total * effective_scoped_value
                )

            ticker_value_deltas = target_ticker_values - current_ticker_values
            filtered_ticker_value_deltas = np.where(
                (ticker_value_deltas >= 1.0) | (ticker_value_deltas <= -1.0),
                ticker_value_deltas,
                0.0,
            )
            filtered_ticker_value_deltas = np.where(
                (filtered_ticker_value_deltas < 0.0) & ~can_sell,
                0.0,
                filtered_ticker_value_deltas,
            )

            sell_values_for_tax = np.where(filtered_ticker_value_deltas < 0.0, filtered_ticker_value_deltas, 0.0)
            sell_share_deltas_for_tax = (sell_values_for_tax / price_data).round(decimals=3)
            next_tax_estimate = 0.0
            for i, share_delta in enumerate(sell_share_deltas_for_tax):
                shares_to_sell = abs(float(share_delta))
                if shares_to_sell <= 0.0:
                    continue
                next_tax_estimate += self._estimate_sale_tax(i, shares_to_sell, float(price_data[i]), date)
            next_tax_estimate = round(next_tax_estimate, 2)
            if abs(next_tax_estimate - tax_estimate) < 0.005:
                break
            tax_estimate = next_tax_estimate

        if np.all(filtered_ticker_value_deltas == 0.0):
            return

        sell_values = np.where(filtered_ticker_value_deltas < 0.0, filtered_ticker_value_deltas, 0.0)
        buy_values = np.where(filtered_ticker_value_deltas > 0.0, filtered_ticker_value_deltas, 0.0)

        sell_share_deltas = (sell_values / price_data).round(decimals=3)
        available_capital = 0.0
        for i, share_delta in enumerate(sell_share_deltas):
            shares_to_sell = abs(float(share_delta))
            if shares_to_sell <= 0.0:
                continue
            proceeds = round(shares_to_sell * float(price_data[i]), 2)
            tax_due = self._realize_sale_tax(i, shares_to_sell, float(price_data[i]), date)
            available_capital += max(0.0, proceeds - tax_due)

        required_buy_capital = float(np.sum(buy_values).round(decimals=2))
        if required_buy_capital > available_capital:
            scale = available_capital / required_buy_capital if required_buy_capital > 0.0 else 0.0
            buy_values = (buy_values * scale).round(decimals=2)

        buy_share_deltas = (buy_values / price_data).round(decimals=3)
        spent = 0.0
        for i, shares in enumerate(buy_share_deltas):
            spent += self._buy_shares(i, float(shares), float(price_data[i]), date)

        self._sweep_to_cash(price_data, max(0.0, available_capital - spent), date)

        if trace:
            print(
                "Rebalanced "
                + ", ".join(
                    f"{self.tickers[i]}: {'+' if filtered_ticker_value_deltas[i] > 0.0 else '-'}${abs(filtered_ticker_value_deltas[i]):,.2f}"
                    for i in range(len(self.tickers))
                )
                + f" on {date_string}"
            )

    def take_profit(
        self,
        date_string: str,
        price_data: np.ndarray,
        ticker_index: int,
        target_weight: float,
        rebalance: bool = False,
        weights: tuple[float, ...] | None = None,
        max_cash_ratio: float | None = None,
        trace: bool = False,
        date: object | None = None,
    ) -> None:
        """Takes profit by trimming one ticker and optionally rebalancing into other risks.

        When there are no additional risky holdings (only the active leg plus cash),
        ``max_cash_ratio`` caps portfolio cash after the sale and any trailing rebalance
        is skipped so proceeds are not reinvested immediately back into the same leg.

        When other risky positions exist, sizing follows ``target_weight`` on the active
        ticker and proceeds are redeployed via ``rebalance`` if requested.
        """

        leg_weights = np.asarray(weights if weights is not None else self.weights, dtype=float).ravel()
        has_additional_risk = False
        for i in range(self.risky_count):
            if i == ticker_index:
                continue
            if i >= len(leg_weights):
                break
            if float(leg_weights[i]) > 0.0:
                has_additional_risk = True
                break
            holding_value = float(self.current_shares[i] * price_data[i])
            if holding_value >= 1.0:
                has_additional_risk = True
                break

        cash_ratio_for_sizing = None if (max_cash_ratio is not None and has_additional_risk) else max_cash_ratio
        do_rebalance = rebalance and has_additional_risk

        portfolio_value = round(self.current_value(price_data), 2)
        current_cash = self._cash_value(price_data)
        current_value = (self.current_shares[ticker_index] * price_data[ticker_index]).round(decimals=2)
        price = float(price_data[ticker_index])
        share_delta = 0.0
        tax_estimate = 0.0

        if cash_ratio_for_sizing is None:
            target_value = current_value
            for _ in range(8):
                effective_portfolio_value = max(0.0, portfolio_value - tax_estimate)
                target_value = round(float(target_weight) * effective_portfolio_value, 2)
                excess_value = current_value - target_value
                if excess_value < 1.0:
                    share_delta = 0.0
                    break
                share_delta = round(excess_value / price, 3)
                share_delta = min(share_delta, self.current_shares[ticker_index])
                next_tax_estimate = self._estimate_sale_tax(ticker_index, share_delta, price, date)
                if abs(next_tax_estimate - tax_estimate) < 0.005:
                    break
                tax_estimate = next_tax_estimate
        else:
            target_value = current_value
            for _ in range(8):
                effective_portfolio_value = max(0.0, portfolio_value - tax_estimate)
                target_cash = round(float(cash_ratio_for_sizing) * effective_portfolio_value, 2)
                required_net_cash = max(0.0, target_cash - current_cash)
                sale_value = required_net_cash + tax_estimate
                if sale_value < 1.0:
                    share_delta = 0.0
                    break
                share_delta = round(sale_value / price, 3)
                share_delta = min(share_delta, self.current_shares[ticker_index])
                target_value = round(current_value - share_delta * price, 2)
                next_tax_estimate = self._estimate_sale_tax(ticker_index, share_delta, price, date)
                if abs(next_tax_estimate - tax_estimate) < 0.005:
                    break
                tax_estimate = next_tax_estimate

        if share_delta <= 0.0:
            if trace:
                print(f"No profit to take from {self.tickers[ticker_index]} on {date_string}")
            return

        proceeds = round(share_delta * price, 2)
        tax_due = self._realize_sale_tax(ticker_index, share_delta, price, date)
        net_proceeds = max(0.0, proceeds - tax_due)

        self._sweep_to_cash(price_data, net_proceeds, date)

        if trace:
            print(f"Took profit from {self.tickers[ticker_index]}: ${proceeds:,.2f} on {date_string}")
            print(f"Proceeds: ${proceeds:,.2f} Tax due: ${tax_due:,.2f} = Net ${proceeds - tax_due:,.2f}")

        if do_rebalance:
            self.rebalance(date_string, price_data, weights, trace=trace, date=date)

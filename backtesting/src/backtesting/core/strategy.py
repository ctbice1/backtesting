from abc import abstractmethod
import heapq

import pandas as pd
import numpy as np

from backtesting.core.types import Activity, Allocate, Distribute, Rebalance, TakeProfit
from backtesting.core.portfolio import Portfolio
from backtesting.core.types import Schedule, ScheduleFormat

class Strategy:
    """Base strategy with common allocation/distribution execution plumbing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes strategy scheduling and optional tracking settings."""
        allocation_schedule = kwargs.get("allocation_schedule")
        self._allocation_schedule: Schedule = allocation_schedule or Schedule(ScheduleFormat.DAYS, 365)
        self._initial_allocation = kwargs.get("initial_allocation", 0)
        self._yearly_allocation = kwargs.get("yearly_allocation", 0)

        self._start_date = kwargs.get("start_date", None)

        self._allocation_contribution_weights: tuple[float, ...] | None = kwargs.get(
            "allocation_weights"
        )
        self._distribution_contribution_weights: tuple[float, ...] | None = kwargs.get(
            "distribution_weights"
        )
        self._cash_management: str = str(kwargs.get("cash_management", "sweep")).strip().lower()

        self.activity_schedule: list[Activity] = []
        self.allocation_dates: dict = {}
        self.allocation_dates_sorted: tuple[object, ...] = ()

        self.track = kwargs.get("track", False)
        self.share_history = {}

        # Convert activity dates to a priority queue
        heapq.heapify(self.activity_schedule)

    def parameters(self) -> dict[str, object]:
        """Returns strategy parameters captured for reporting/output."""
        return {
            "allocation_schedule": self._allocation_schedule,
            "initial_allocation": self._initial_allocation,
            "yearly_allocation": self._yearly_allocation,
            "allocation_weights": self._allocation_contribution_weights,
            "distribution_weights": self._distribution_contribution_weights,
            "cash_management": self._cash_management,
        }

    @staticmethod
    def _split_even_usd(total_usd: float, parts: int) -> tuple[float, ...]:
        """Splits ``total_usd`` into ``parts`` amounts differing by at most one cent."""
        if parts <= 0:
            return ()
        cents = max(0, int(round(total_usd * 100)))
        q, r = divmod(cents, parts)
        return tuple((q + (1 if i < r else 0)) / 100.0 for i in range(parts))

    @staticmethod
    def _calendar_year_budget_fraction(
        year: int,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> float:
        """Fraction of ``year``'s calendar elapsed by the overlap of [window_start, window_end]."""

        year_start = pd.Timestamp(year=year, month=1, day=1)
        next_year_start = pd.Timestamp(year=year + 1, month=1, day=1)
        denom_days = (next_year_start - year_start).days
        if denom_days <= 0:
            return 0.0

        ws = pd.Timestamp(window_start).normalize()
        we = pd.Timestamp(window_end).normalize()
        overlap_start = max(ws, year_start)
        last_day_in_year = next_year_start - pd.Timedelta(days=1)
        overlap_end_inclusive = min(we, last_day_in_year)

        if overlap_end_inclusive < overlap_start:
            return 0.0

        overlap_days = (overlap_end_inclusive - overlap_start).days + 1
        return min(overlap_days / float(denom_days), 1.0)

    def _set_allocations(self, price_history_dates: np.ndarray[pd.Timestamp]) -> None:
        '''Configures the allocation amount.

        Periodic contributions exclude the lump-sum initial deposit. Within each calendar
        year, the annual budget equals ``yearly_allocation`` scaled by how much of that
        calendar year the backtest window covers (uniform calendar-day overlap from first
        to last simulated date). That budget is split evenly across periodic allocation
        dates in the year (amounts match ± one cent so whole cents sum exactly).
        '''
        if len(price_history_dates) == 0:
            return

        # Restrict allocations to the test window start.
        bounded_price_history_dates = price_history_dates
        if self._start_date is not None:
            start_date = pd.to_datetime(self._start_date)
            bounded_price_history_dates = price_history_dates[price_history_dates >= start_date]
            if len(bounded_price_history_dates) == 0:
                return

        allocation_dates = self._allocation_schedule.get_fixed_dates(bounded_price_history_dates)

        initial_allocation_date = bounded_price_history_dates[0]
        self.allocation_dates[initial_allocation_date] = self._initial_allocation

        if self._yearly_allocation <= 0:
            return

        window_start = pd.Timestamp(bounded_price_history_dates[0])
        window_end = pd.Timestamp(bounded_price_history_dates[-1])

        years = sorted({pd.Timestamp(d).year for d in allocation_dates})
        for year in years:
            year_budget = float(self._yearly_allocation) * self._calendar_year_budget_fraction(
                year, window_start, window_end
            )
            year_dates = sorted(
                d
                for d in allocation_dates
                if pd.Timestamp(d).year == year and not (
                    pd.Timestamp(d).normalize()
                    == pd.Timestamp(initial_allocation_date).normalize()
                )
            )
            if not year_dates:
                continue
            shares = Strategy._split_even_usd(year_budget, len(year_dates))
            for d, amount in zip(year_dates, shares, strict=True):
                self.allocation_dates[d] = amount

    def _set_passive_activity_schedule(
        self, price_history_dates: np.ndarray, distribution_history_dates: np.ndarray | None
    ) -> None:
        '''Sets allocation and distribution activities and enqueues them for execution.'''

        # Set allocation and rebalance dates
        price_dates = pd.to_datetime(list(price_history_dates))
        self._set_allocations(price_dates)

        # Add allocation dates and distribution dates
        for date in self.allocation_dates:
            heapq.heappush(
                self.activity_schedule,
                Allocate(
                    date,
                    self.allocation_dates[date],
                    self._allocation_contribution_weights,
                ),
            )

        if distribution_history_dates is not None:
            for date in pd.to_datetime([date for date in distribution_history_dates if date >= self._start_date]):
                heapq.heappush(
                    self.activity_schedule,
                    Distribute(
                        date,
                        self._distribution_contribution_weights,
                    ),
                )

        self.allocation_dates_sorted = tuple(sorted(self.allocation_dates.keys()))

        # Clear rebalance and allocation date dictionaries
        del self.allocation_dates

        return self

    def _effective_take_profit_max_cash_ratio(self, as_of: pd.Timestamp, base: float | None) -> float | None:
        """Scales configured max cash toward zero right after an allocation, up to ``base`` near the next.

        When no allocation calendar exists (unit tests, passive-only schedules), returns ``base``.
        """
        if base is None:
            return None
        dates_raw = getattr(self, "allocation_dates_sorted", ()) or ()
        if not dates_raw:
            return base
        dts = sorted(pd.Timestamp(d) for d in dates_raw)
        as_of_ts = pd.Timestamp(as_of)
        past = [d for d in dts if d <= as_of_ts]
        if not past:
            return base
        last_alloc = past[-1]
        if len(dts) >= 2:
            day_diffs = [(dts[i + 1] - dts[i]).days for i in range(len(dts) - 1)]
            positive_diffs = [x for x in day_diffs if x > 0]
            period = max(1, int(np.median(positive_diffs)) if positive_diffs else max(day_diffs))
        else:
            period = 30
        days_into = max(0, (as_of_ts.normalize() - last_alloc.normalize()).days)
        scale = min(1.0, days_into / period)
        return float(base * scale)

    def performance(
        self, price_history: np.ndarray, price_history_dates: np.ndarray, yearly: bool = False
    ) -> None:
        '''Print performance.'''
        _ = yearly

        if not self.share_history:
            return

        # Convert share history into a dataframe
        df = pd.DataFrame.from_dict(self.share_history, orient="index")

        # Create a new index for the share history from the price history dates
        share_history_start = min(self.share_history)
        date_index = pd.to_datetime(price_history_dates[price_history_dates >= share_history_start])

        # Reindex the share history and forward fill share counts
        df = df.reindex(date_index, method="ffill")

        # Restrict price history to share history dates
        price_history = price_history[price_history_dates >= share_history_start]

        # Multiply by the price history to get a value history of the portfolio
        df = df * price_history

        # print(df)
        # sys.exit()
        return

    def execute(
        self, historical_data: tuple, dates: tuple, portfolio: Portfolio, trace: bool = False
    ) -> tuple[Portfolio, dict[str, object]]:
        '''Set the activity schedule and execute the strategy.'''

        # Extract historical data and indices
        price_history, distribution_history = historical_data
        price_history_dates, distribution_history_dates = dates

        # Create passive activities
        self._set_passive_activity_schedule(price_history_dates, distribution_history_dates)

        # Execute strategy-specific procedure
        self.procedure(historical_data, dates, portfolio, trace)

        # Sort activities to form a priority queue
        self.activity_schedule = sorted(self.activity_schedule)

        # Execute activities from pipeline
        initial = True
        while self.activity_schedule:

            activity = heapq.heappop(self.activity_schedule)

            # Use a mask to get an index for price data
            price_date_index = (price_history_dates == activity.date)
            if not np.any(price_date_index):
                # Strategy schedules may include dates missing from the traded
                # portfolio universe; skip those activities safely.
                continue
            prices = price_history[price_date_index][0]

            # Date string
            date_string = activity.date.item().strftime("%A %B %d, %Y") if type(activity.date) == np.datetime64 else activity.date.strftime("%A %B %d, %Y")

            # Handle activities appropriately
            if isinstance(activity, Allocate):
                portfolio.total_new_capital += activity.amount
                portfolio.contribution_flows.append((activity.date, float(activity.amount)))
                portfolio.allocate(
                    date_string,
                    prices,
                    activity.amount,
                    activity.weights if initial is False else portfolio.weights,
                    trace=trace,
                    date=activity.date,
                )
                initial = False
            elif isinstance(activity, Rebalance):
                portfolio.rebalance(
                    date_string,
                    prices,
                    activity.weights,
                    activity.rebalance_indices,
                    trace=trace,
                    date=activity.date,
                )
            elif isinstance(activity, TakeProfit):
                max_cash_ratio = self._effective_take_profit_max_cash_ratio(
                    pd.Timestamp(activity.date),
                    activity.max_cash_ratio,
                )
                portfolio.take_profit(
                    date_string,
                    prices,
                    activity.ticker_index,
                    activity.target_weight,
                    activity.rebalance,
                    activity.weights,
                    max_cash_ratio,
                    trace=trace,
                    date=activity.date,
                )
            elif isinstance(activity, Distribute):
                if distribution_history_dates is None or distribution_history is None:
                    continue
                distribution_date_index = (distribution_history_dates == activity.date)
                if not np.any(distribution_date_index):
                    continue
                distribution_amounts = distribution_history[distribution_date_index][0]
                distribution_amount = np.sum(portfolio.current_shares * distribution_amounts)
                portfolio.total_distribution += distribution_amount
                portfolio.record_distribution(activity.date, distribution_amount)
                net_distribution_amount = portfolio.tax_distributions(distribution_amounts, activity.date)
                if self._cash_management == "distribute":
                    portfolio.allocate(
                        date_string,
                        prices,
                        net_distribution_amount,
                        activity.weights,
                        trace=trace,
                        date=activity.date,
                    )
                else:
                    cash_idx = portfolio.cash_index
                    if cash_idx is None:
                        portfolio.allocate(
                            date_string,
                            prices,
                            net_distribution_amount,
                            activity.weights,
                            trace=trace,
                            date=activity.date,
                        )
                    else:
                        cash_distribution = float(
                            portfolio.current_shares[cash_idx] * distribution_amounts[cash_idx]
                        )
                        risky_distribution = distribution_amount - cash_distribution
                        net_ratio = (
                            net_distribution_amount / distribution_amount
                            if distribution_amount > 0.0
                            else 1.0
                        )
                        cash_distribution *= net_ratio
                        risky_distribution *= net_ratio
                        if cash_distribution != 0.0:
                            portfolio._sweep_to_cash(prices, cash_distribution, activity.date)
                        portfolio.allocate(
                            date_string,
                            prices,
                            risky_distribution,
                            activity.weights,
                            trace=trace,
                            date=activity.date,
                        )

            if self.track:
                self.share_history[activity.date] = portfolio.current_shares.copy()

        if self.track:
            self.performance(price_history, price_history_dates)

        if trace:
            ticker_weights = set(zip(portfolio.tickers, portfolio.current_shares))
            print(f"Final share count: {', '.join(f"{ticker}: {round(weight, 3)}" for ticker, weight in ticker_weights)}")
            print(f"Final security values: {', '.join(f"{ticker}: ${value:,.2f}" for ticker, value in zip(portfolio.tickers, portfolio.current_shares * prices))}")

        return portfolio, self.parameters()

    @abstractmethod
    def procedure(
        self, historical_data: tuple, indices: tuple, portfolio: Portfolio, trace: bool
    ) -> None:
        """Schedules strategy-specific activity events before execution."""
        pass
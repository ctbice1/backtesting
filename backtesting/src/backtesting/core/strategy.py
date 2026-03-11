from abc import abstractmethod
import heapq
import sys

import pandas as pd
import numpy as np

from backtesting.core.activity import Activity, Allocate, Distribute, Rebalance
from backtesting.core.portfolio import Portfolio
from backtesting.core.schedule import Schedule, ScheduleFormat

class Strategy:
    def __init__(self, *args, **kwargs):
        allocation_schedule = kwargs.get("allocation_schedule")
        self._allocation_schedule: Schedule = allocation_schedule or Schedule(ScheduleFormat.DAYS, 365)
        self._initial_allocation = kwargs.get("initial_allocation", 0)
        self._yearly_allocation = kwargs.get("yearly_allocation", 0)

        self._start_date = kwargs.get("start_date", None)

        self.activity_schedule: list[Activity] = []
        self.allocation_dates: dict = {}
        self.rebalance_dates: list = []

        self.track = kwargs.get("track", False)
        self.share_history = {}

        # Convert activity dates to a priority queue
        heapq.heapify(self.activity_schedule)

    def parameters(self) -> dict:
        return {
            "allocation_schedule": self._allocation_schedule,
            "initial_allocation": self._initial_allocation,
            "yearly_allocation": self._yearly_allocation
        }

    def _set_allocations(self, price_history_dates: np.ndarray[pd.Timestamp]):
        '''Configures the allocation amount.'''
        if len(price_history_dates) == 0:
            return

        # Restrict allocations to the test window start.
        bounded_price_history_dates = price_history_dates
        if self._start_date is not None:
            start_date = pd.to_datetime(self._start_date)
            bounded_price_history_dates = price_history_dates[price_history_dates >= start_date]
            if len(bounded_price_history_dates) == 0:
                return

        # Generate a list of allocation dates
        allocation_dates = self._allocation_schedule.get_fixed_dates(bounded_price_history_dates)

        # Get number of allocation days in each year
        days_per_year = {
            year: len({date for date in allocation_dates if pd.to_datetime(date).year == year})
            for year in {pd.to_datetime(date).year for date in allocation_dates}
        }
        days_per_year = {year: days_per_year[year] for year in sorted(days_per_year.keys())}

        # Get number of allocation days in a full year
        days_in_full_year = None
        if self._allocation_schedule.fmt is ScheduleFormat.DAYS and self._allocation_schedule.value > 0:
            days_in_full_year = round(365.25 / self._allocation_schedule.value, 3)
        elif self._allocation_schedule.fmt is ScheduleFormat.WEEKLY:
            days_in_full_year = 52
        elif self._allocation_schedule.fmt is ScheduleFormat.MONTHLY:
            days_in_full_year = 12
        elif self._allocation_schedule.fmt is ScheduleFormat.YEARLY:
            days_in_full_year = 1
        else:
            print(f"Unable to determine periodic allocation, using initial allocation only.")
            self.allocation_dates[bounded_price_history_dates[0]] = self._initial_allocation
            return

        if days_in_full_year is None:
            print("Unable to determine allocation schedule.")
            sys.exit(-1)

        # Get a per-allocation amount for a full year
        per_allocation_amount = round(self._yearly_allocation / days_in_full_year, 2)
        remainder_allocation_amount = round(self._yearly_allocation - (per_allocation_amount * days_in_full_year), 2)

        # Set the initial allocation
        initial_allocation_date = bounded_price_history_dates[0]
        self.allocation_dates[initial_allocation_date] = self._initial_allocation

        # Evenly divide the yearly allocation by number of allocation days per year
        for year in days_per_year.keys():

            for date in allocation_dates:

                if pd.to_datetime(date).year == year and date != initial_allocation_date:
                    self.allocation_dates[date] = per_allocation_amount

    def _set_passive_activity_schedule(self, price_history_dates: tuple[np.ndarray], distribution_history_dates: tuple[np.ndarray]) -> None:
        '''Sets allocation and distribution activities and enqueues them for execution.'''

        # Set allocation and rebalance dates
        price_dates = pd.to_datetime(list(price_history_dates))
        self._set_allocations(price_dates)

        # Add allocation dates and distribution dates
        for date in self.allocation_dates:
            heapq.heappush(self.activity_schedule, Allocate(date, self.allocation_dates[date]))
        if distribution_history_dates is not None:
            for date in pd.to_datetime([date for date in distribution_history_dates if date >= self._start_date]):
                heapq.heappush(self.activity_schedule, Distribute(date))

        # Clear rebalance and allocation date dictionaries
        del self.allocation_dates

        return self

    def performance(self, price_history: np.ndarray, price_history_dates: np.ndarray, yearly: bool = False):
        '''Print performance.'''

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

        # Display initial and final values
        # begin_value = portfolio.historical_value_on(self._begin)
        # end_value = portfolio.historical_value_on(self._end)
        # print(f'Initial value: ${begin_value:,.2f}', self._begin)
        # print(f'Final value: ${end_value:,.2f}', self._end)

        if yearly:

            # Get beginning and end dates for each year covered
            dates = {}
            years = sorted({date.year for date in self._time_period})
            for year in years:
                dates[year] = (
                    min({day for day in self._time_period if day.year == year}),
                    max({day for day in self._time_period if day.year == year})
                )

            # Display yearly gain / loss
            # for year, (beginning, end) in dates.items():
            #     start_value = portfolio.historical_value_on(beginning)
            #     finish_value = portfolio.historical_value_on(end)
            #     print(f'Portfolio value difference in {year}:')
            #     print(f'  Beginning:\t${start_value:,.2f}')
            #     print(f'  End:\t\t${finish_value:,.2f}')
            #     print(f'  Gain/Loss:\t${finish_value - start_value:,.2f}')
            #     print(f'  Percent:\t{(finish_value - start_value) / start_value:.2%}')

        # Display compound annual growth rate
        cagr = ((end_value / begin_value) ** (1 / len({date.year for date in self._time_period})) - 1)
        print(f'Compound annual growth rate: {cagr:.2%}')

    def execute(self, historical_data: tuple, dates: tuple, portfolio: Portfolio, trace: bool = False, add_price_variance=False) -> Portfolio:
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
                portfolio.allocate(
                    date_string,
                    prices,
                    activity.amount,
                    details=trace,
                    add_price_variance=add_price_variance
                )
            elif isinstance(activity, Rebalance):
                portfolio.rebalance(
                    date_string,
                    prices,
                    activity.weights,
                    details=trace,
                    add_price_variance=add_price_variance
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
                portfolio.allocate(
                    date_string,
                    prices,
                    distribution_amount,
                    details=trace,
                    add_price_variance=add_price_variance
                )

            if self.track:
                self.share_history[activity.date] = portfolio.current_shares.copy()

        # print(f'Total allocated to {self.__doc__}: ${self._total_allocation:,.2f}')
        if self.track:
            self.performance(price_history, price_history_dates)

        if trace:
            ticker_weights = set(zip(portfolio.tickers, portfolio.current_shares))
            print(f"Final share count: {', '.join(f"{ticker}: {round(weight, 3)}" for ticker, weight in ticker_weights)}")

        return portfolio, self.parameters()

    @abstractmethod
    def procedure(self, historical_data: tuple, indices: tuple, portfolio: Portfolio, trace: bool):
        pass

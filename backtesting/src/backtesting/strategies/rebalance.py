import heapq
import sys

import numpy as np

from backtesting.core.types import Rebalance
from backtesting.indicators.basic import (
    SimpleMovingAverage,
    SimpleMovingStdDev,
    SimpleMovingVariance,
)
from backtesting.data import get_historical_data
from backtesting.core.portfolio import Portfolio
from backtesting.core.types import Schedule, ScheduleFormat
from backtesting.core.strategy import Strategy

class ScheduledRebalance(Strategy):
    '''Basic Rebalance Strategy'''
    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes a strategy that rebalances on a fixed schedule."""
        super().__init__(**kwargs)
        schedule = kwargs.get("schedule")
        if isinstance(schedule, Schedule):
            self.schedule = schedule
            return

        # Backwards compatibility for legacy strategy args:
        # schedule_format / schedule_value.
        schedule_format = kwargs.get("schedule_format")
        if schedule_format is None:
            print(
                "Missing rebalance schedule. Provide strategy.schedule.format "
                "or a legacy schedule_format."
            )
            sys.exit(-1)

        try:
            schedule_value = kwargs["schedule_value"] if "schedule_value" in kwargs else None
            self.schedule = Schedule(ScheduleFormat[schedule_format], schedule_value)
        except KeyError:
            print(f"Invalid ScheduleFormat: {schedule_format}")
            sys.exit(-1)

    def procedure(
        self, historical_data: tuple, indices: tuple, portfolio: Portfolio, trace: bool
    ) -> None:
        '''Rebalances over a fixed (scheduled) time period.'''

        # Add rebalance operations
        rebalance_dates = self.schedule.get_fixed_dates(indices[0])
        if not rebalance_dates:
            return
        for date in rebalance_dates:
            heapq.heappush(self.activity_schedule, Rebalance(date))

class SimpleMovingAverageRebalance(Strategy):
    '''Basic Rebalance Strategy'''
    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes SMA threshold rebalance parameters."""
        super().__init__(**kwargs)
        self.dxy_adjusted = kwargs.get("dxy_adjusted", False)
        self.target = kwargs.get("target", None)
        self.length = kwargs.get("length", 200)
        self.primary = kwargs.get("primary", None)
        self.alternate = kwargs.get("alternate", None)
        if self.target is None:
            print(f"Please provide a target ticker to use for the {__class__.__name__} strategy.")
            sys.exit(-1)

    def procedure(
        self, historical_data: tuple, indices: tuple, portfolio: Portfolio, trace: bool
    ) -> None:
        '''Rebalances over a fixed (scheduled) time period.'''

        # Get historical data for target
        try:
            (target_price_history, _) = get_historical_data((self.target,))
            target_price_dates, target_price_data = target_price_history
            target_price_data = target_price_data[:,0]

            # Restrict history to [start_date, stop_date]
            stop_date = indices[0][-1]
            start_date = indices[0][0]
            target_price_data = target_price_data[(target_price_dates <= stop_date)]
            target_price_dates = target_price_dates[(target_price_dates <= stop_date)]
            target_price_data = target_price_data[(target_price_dates >= start_date)]
            target_price_dates = target_price_dates[(target_price_dates >= start_date)]

            # Adjust the index price by the USD index
            if self.dxy_adjusted:
                (dxy_price_history, _) = get_historical_data(("DX-Y.NYB",))
                dxy_price_dates, dxy_price_data = dxy_price_history
                dxy_price_data = dxy_price_data[:,0] / 100

                dxy_price_data = dxy_price_data[(dxy_price_dates <= stop_date)]
                dxy_price_dates = dxy_price_dates[(dxy_price_dates <= stop_date)]
                dxy_price_data = dxy_price_data[(dxy_price_dates >= start_date)]
                dxy_price_dates = dxy_price_dates[(dxy_price_dates >= start_date)]

                # Adjust futures DXY price data to match the time frame for regular ticker data
                dates_to_remove = np.setdiff1d(dxy_price_dates, target_price_dates)
                dxy_price_data = dxy_price_data[~np.isin(dxy_price_dates, dates_to_remove)]
                dxy_price_dates = dxy_price_dates[~np.isin(dxy_price_dates, dates_to_remove)]
                new_dates = np.setdiff1d(target_price_dates, dxy_price_dates)
                new_dates_indices = np.searchsorted(dxy_price_dates, new_dates)
                dxy_price_dates = np.insert(dxy_price_dates, new_dates_indices, new_dates)

                # Forward fill the missing price data
                dxy_price_data = np.insert(dxy_price_data, new_dates_indices, np.full(len(new_dates), np.nan))
                mask = ~np.isnan(dxy_price_data)
                idx = np.where(mask, np.arange(len(dxy_price_data)), 0)
                np.maximum.accumulate(idx, out=idx)
                dxy_price_data = dxy_price_data[idx]

                # Adjust target index
                target_price_data = target_price_data / dxy_price_data

        except RuntimeError as e:
            print(e)
            sys.exit(-1)

        # Get SMA history for target
        target_sma = SimpleMovingAverage(self.length, target_price_data).history

        # Default to portfolio weights
        weights = portfolio.weights

        # Use an alternating max-weighting scheme if a primary and alternate were provided
        if self.primary:
            if self.alternate:

                # Fix weights at 100% and 0%, then alternate
                weights = np.array((1.0, 0.0))
                portfolio.weights = weights

                # Create rebalance activities by buying above the 200 day SMA and selling below
                for i, date in enumerate(target_price_dates):
                    current_target_sma = target_sma[i]
                    margin = 0.00 * current_target_sma
                    margin_above_sma = current_target_sma + margin
                    margin_below_sma = current_target_sma - margin
                    current_target_price = target_price_data[i]

                    # Choose the rebalance
                    if current_target_price >= margin_above_sma and weights[0] == 0.0:
                        weights = np.array((1.0, 0.0))
                        heapq.heappush(self.activity_schedule, Rebalance(date, weights))
                        # if trace:
                        #     print(f"Rebalanced into {self.primary} on {date.item().strftime("%A %B %d, %Y")}")
                    elif current_target_price <= margin_below_sma and weights[0] == 1.0:
                        weights = np.array((0.0, 1.0))
                        heapq.heappush(self.activity_schedule, Rebalance(date, weights))
                        # if trace:
                        #     print(f"Rebalanced into {self.alternate} on {date.item().strftime("%A %B %d, %Y")}")
            else:
                print("Please provide an alternate security ticker if using a primary.")
                sys.exit(-1)
        else:
            print("Only supporting primary and alternate rebalance for now.")
            sys.exit(-1)

class StdDevRebalance(SimpleMovingAverageRebalance):
    '''Volatility adjusted simple moving average rebalance Strategy'''
    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes SMA and deviation-window rebalance parameters."""
        super().__init__(**kwargs)
        self.dxy_adjusted = kwargs.get("dxy_adjusted", False)
        self.long_length = kwargs.get("long_length", 200)
        self.short_length = kwargs.get("short_length", 50)

    def procedure(
        self, historical_data: tuple, indices: tuple, portfolio: Portfolio, trace: bool
    ) -> None:
        '''Rebalances over a fixed (scheduled) time period.'''

        # Get historical data for target and volatility
        try:
            (target_price_history, _) = get_historical_data((self.target,))
            target_price_dates, target_price_data = target_price_history
            target_price_data = target_price_data[:,0]

            # Get short, long SMA and Standard Deviation history for target
            long_target_sma = SimpleMovingAverage(self.long_length, target_price_data).history
            long_target_sma_std_dev = SimpleMovingStdDev(self.long_length, target_price_data).history

            short_target_sma = SimpleMovingAverage(self.short_length, target_price_data).history
            short_target_sma_std_dev = SimpleMovingStdDev(self.short_length, target_price_data).history

            # Restrict history to [start_date, stop_date]
            stop_date = indices[0][-1]
            start_date = indices[0][0]

            bounds_mask = (target_price_dates >= start_date) & (target_price_dates <= stop_date)
            target_price_data = target_price_data[bounds_mask]
            long_target_sma = long_target_sma[bounds_mask]
            long_target_sma_std_dev = long_target_sma_std_dev[bounds_mask]
            short_target_sma = short_target_sma[bounds_mask]
            short_target_sma_std_dev = short_target_sma_std_dev[bounds_mask]
            target_price_dates = target_price_dates[bounds_mask]

            # Adjust the index price by the USD index
            if self.dxy_adjusted:
                (dxy_price_history, _) = get_historical_data(("DX-Y.NYB",))
                dxy_price_dates, dxy_price_data = dxy_price_history
                dxy_price_data = dxy_price_data[:,0] / 100

                dxy_price_data = dxy_price_data[(dxy_price_dates <= stop_date)]
                dxy_price_dates = dxy_price_dates[(dxy_price_dates <= stop_date)]
                dxy_price_data = dxy_price_data[(dxy_price_dates >= start_date)]
                dxy_price_dates = dxy_price_dates[(dxy_price_dates >= start_date)]

                # Adjust futures DXY price data to match the time frame for regular ticker data
                dates_to_remove = np.setdiff1d(dxy_price_dates, target_price_dates)
                dxy_price_data = dxy_price_data[~np.isin(dxy_price_dates, dates_to_remove)]
                dxy_price_dates = dxy_price_dates[~np.isin(dxy_price_dates, dates_to_remove)]
                new_dates = np.setdiff1d(target_price_dates, dxy_price_dates)
                new_dates_indices = np.searchsorted(dxy_price_dates, new_dates)
                dxy_price_dates = np.insert(dxy_price_dates, new_dates_indices, new_dates)

                # Forward fill the missing price data
                dxy_price_data = np.insert(dxy_price_data, new_dates_indices, np.full(len(new_dates), np.nan))
                mask = ~np.isnan(dxy_price_data)
                idx = np.where(mask, np.arange(len(dxy_price_data)), 0)
                np.maximum.accumulate(idx, out=idx)
                dxy_price_data = dxy_price_data[idx]

                # Adjust target index
                target_price_data = target_price_data / dxy_price_data

        except RuntimeError as e:
            print(e)
            sys.exit(-1)

        # Get Standard Deviations from Target SMA
        long_target_plus_2_std_dev = long_target_sma + (2 * short_target_sma_std_dev)
        long_target_minus_2_std_dev = long_target_sma - (2 * short_target_sma_std_dev)

        # Default to portfolio weights
        weights = portfolio.weights

        # Use an alternating max-weighting scheme if a primary and alternate were provided
        if self.primary:
            if self.alternate:

                # Fix weights at 100% and 0%, then alternate
                weights = np.array((1.0, 0.0))
                portfolio.weights = weights

                # Create rebalance activities by buying above the 200 day SMA and selling below
                for i, date in enumerate(target_price_dates):

                    # Choose the rebalance
                    current_price = target_price_data[i]
                    # Re-leverage if we drop below 2 standard deviations from the SMA, or rise above it again
                    if (
                        current_price <= long_target_minus_2_std_dev[i]
                        or current_price >= short_target_sma[i]
                    ) and weights[0] == 0.0:
                        weights = np.array((1.0, 0.0))
                        heapq.heappush(self.activity_schedule, Rebalance(date, weights))
                    # De-leverage if we exceed 2 standard deviations from the SMA, or drop below it again
                    elif (
                        current_price >= long_target_plus_2_std_dev[i]
                        or current_price <= short_target_sma[i]
                    ) and weights[0] == 1.0:
                        weights = np.array((0.0, 1.0))
                        heapq.heappush(self.activity_schedule, Rebalance(date, weights))
            else:
                print("Please provide an alternate security ticker if using a primary.")
                sys.exit(-1)

class SMACrossRebalance(SimpleMovingAverageRebalance):
    '''Volatility adjusted simple moving average rebalance Strategy'''
    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes dual-SMA crossover rebalance parameters."""
        super().__init__(**kwargs)
        self.dxy_adjusted = kwargs.get("dxy_adjusted", False)
        self.long_length = kwargs.get("long_length", None)
        self.short_length = kwargs.get("short_length", None)
        if self.long_length is None:
            print(f"Please provide a long SMA length to use for the {__class__.__name__} strategy.")
            sys.exit(-1)
        if self.short_length is None:
            print(f"Please provide a short SMA length to use for the {__class__.__name__} strategy.")
            sys.exit(-1)

    def procedure(
        self, historical_data: tuple, indices: tuple, portfolio: Portfolio, trace: bool
    ) -> None:
        '''Rebalances over a fixed (scheduled) time period.'''

        # Get historical data for target and volatility
        try:
            (target_price_history, _) = get_historical_data((self.target,))
            target_price_dates, target_price_data = target_price_history
            target_price_data = target_price_data[:,0]

            # Get short, long SMA and Standard Deviation history for target
            long_target_sma = SimpleMovingAverage(self.long_length, target_price_data).history

            short_target_sma = SimpleMovingAverage(self.short_length, target_price_data).history

            # Restrict history to [start_date, stop_date]
            stop_date = indices[0][-1]
            start_date = indices[0][0]

            bounds_mask = (target_price_dates >= start_date) & (target_price_dates <= stop_date)
            target_price_data = target_price_data[bounds_mask]
            long_target_sma = long_target_sma[bounds_mask]
            short_target_sma = short_target_sma[bounds_mask]
            target_price_dates = target_price_dates[bounds_mask]

            # Adjust the index price by the USD index
            if self.dxy_adjusted:
                (dxy_price_history, _) = get_historical_data(("DX-Y.NYB",))
                dxy_price_dates, dxy_price_data = dxy_price_history
                dxy_price_data = dxy_price_data[:,0] / 100

                dxy_price_data = dxy_price_data[(dxy_price_dates <= stop_date)]
                dxy_price_dates = dxy_price_dates[(dxy_price_dates <= stop_date)]
                dxy_price_data = dxy_price_data[(dxy_price_dates >= start_date)]
                dxy_price_dates = dxy_price_dates[(dxy_price_dates >= start_date)]

                # Adjust futures DXY price data to match the time frame for regular ticker data
                dates_to_remove = np.setdiff1d(dxy_price_dates, target_price_dates)
                dxy_price_data = dxy_price_data[~np.isin(dxy_price_dates, dates_to_remove)]
                dxy_price_dates = dxy_price_dates[~np.isin(dxy_price_dates, dates_to_remove)]
                new_dates = np.setdiff1d(target_price_dates, dxy_price_dates)
                new_dates_indices = np.searchsorted(dxy_price_dates, new_dates)
                dxy_price_dates = np.insert(dxy_price_dates, new_dates_indices, new_dates)

                # Forward fill the missing price data
                dxy_price_data = np.insert(dxy_price_data, new_dates_indices, np.full(len(new_dates), np.nan))
                mask = ~np.isnan(dxy_price_data)
                idx = np.where(mask, np.arange(len(dxy_price_data)), 0)
                np.maximum.accumulate(idx, out=idx)
                dxy_price_data = dxy_price_data[idx]

                # Adjust target index
                target_price_data = target_price_data / dxy_price_data

        except RuntimeError as e:
            print(e)
            sys.exit(-1)

        # Default to portfolio weights
        weights = portfolio.weights

        # Use an alternating max-weighting scheme if a primary and alternate were provided
        if self.primary:
            if self.alternate:

                # Fix weights at 100% and 0%, then alternate
                weights = np.array((1.0, 0.0))
                portfolio.weights = weights

                # Create rebalance activities by buying above the 200 day SMA and selling below
                for i, date in enumerate(target_price_dates):

                    # Re-leverage if SMAs cross above
                    if short_target_sma[i] > long_target_sma[i] and weights[0] == 0.0:
                        weights = np.array((1.0, 0.0))
                        heapq.heappush(self.activity_schedule, Rebalance(date, weights))
                    # De-leverage if SMAs cross below
                    elif short_target_sma[i] < long_target_sma[i] and weights[0] == 1.0:
                        weights = np.array((0.0, 1.0))
                        heapq.heappush(self.activity_schedule, Rebalance(date, weights))
            else:
                print("Please provide an alternate security ticker if using a primary.")
                sys.exit(-1)

class VolatilityAdjustedSMARebalance(SimpleMovingAverageRebalance):
    '''Volatility adjusted simple moving average rebalance Strategy'''
    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes volatility-gated SMA rebalance parameters."""
        super().__init__(**kwargs)
        self.dxy_adjusted = kwargs.get("dxy_adjusted", False)
        self.volatility_length = kwargs.get("volatility_length", self.length)
        self.volatility = kwargs.get("volatility", None)
        if self.volatility is None:
            print(f"Please provide a volatility ticker to use for the {__class__.__name__} strategy.")
            sys.exit(-1)

    def procedure(
        self, historical_data: tuple, indices: tuple, portfolio: Portfolio, trace: bool
    ) -> None:
        '''Rebalances over a fixed (scheduled) time period.'''

        # Get historical data for target and volatility
        try:
            (target_price_history, _) = get_historical_data((self.target,))
            target_price_dates, target_price_data = target_price_history
            target_price_data = target_price_data[:,0]

            # Get SMA and Standard Deviation history for target
            target_sma = SimpleMovingAverage(self.length, target_price_data).history
            (volatility_price_history, _) = get_historical_data((self.volatility,))
            volatility_price_dates, volatility_price_data = volatility_price_history
            volatility_price_data = volatility_price_data[:,0]

            # Get SMA and Standard Deviation history for volatility
            volatility_variance = SimpleMovingVariance(self.volatility_length, volatility_price_data).history

            # Restrict history to [start_date, stop_date]
            stop_date = indices[0][-1]
            start_date = indices[0][0]

            target_bounds_mask = (target_price_dates >= start_date) & (target_price_dates <= stop_date)
            target_price_data = target_price_data[target_bounds_mask]
            target_sma = target_sma[target_bounds_mask]
            target_price_dates = target_price_dates[target_bounds_mask]

            volatility_bounds_mask = (volatility_price_dates >= start_date) & (volatility_price_dates <= stop_date)
            volatility_price_data = volatility_price_data[volatility_bounds_mask]
            volatility_variance = volatility_variance[volatility_bounds_mask]
            volatility_price_dates = volatility_price_dates[volatility_bounds_mask]

            # Adjust the index price by the USD index
            if self.dxy_adjusted:
                (dxy_price_history, _) = get_historical_data(("DX-Y.NYB",))
                dxy_price_dates, dxy_price_data = dxy_price_history
                dxy_price_data = dxy_price_data[:,0] / 100

                dxy_price_data = dxy_price_data[(dxy_price_dates <= stop_date)]
                dxy_price_dates = dxy_price_dates[(dxy_price_dates <= stop_date)]
                dxy_price_data = dxy_price_data[(dxy_price_dates >= start_date)]
                dxy_price_dates = dxy_price_dates[(dxy_price_dates >= start_date)]

                # Adjust futures DXY price data to match the time frame for regular ticker data
                dates_to_remove = np.setdiff1d(dxy_price_dates, target_price_dates)
                dxy_price_data = dxy_price_data[~np.isin(dxy_price_dates, dates_to_remove)]
                dxy_price_dates = dxy_price_dates[~np.isin(dxy_price_dates, dates_to_remove)]
                new_dates = np.setdiff1d(target_price_dates, dxy_price_dates)
                new_dates_indices = np.searchsorted(dxy_price_dates, new_dates)
                dxy_price_dates = np.insert(dxy_price_dates, new_dates_indices, new_dates)

                # Forward fill the missing price data
                dxy_price_data = np.insert(dxy_price_data, new_dates_indices, np.full(len(new_dates), np.nan))
                mask = ~np.isnan(dxy_price_data)
                idx = np.where(mask, np.arange(len(dxy_price_data)), 0)
                np.maximum.accumulate(idx, out=idx)
                dxy_price_data = dxy_price_data[idx]

                # Adjust target index
                target_price_data = target_price_data / dxy_price_data

        except RuntimeError as e:
            print(e)
            sys.exit(-1)

        # Default to portfolio weights
        weights = portfolio.weights

        # Use an alternating max-weighting scheme if a primary and alternate were provided
        if self.primary:
            if self.alternate:

                # Fix weights at 100% and 0%, then alternate
                weights = np.array((1.0, 0.0))
                portfolio.weights = weights

                # Create rebalance activities by buying above the 200 day SMA and selling below
                for i, date in enumerate(target_price_dates):

                    # Choose the rebalance
                    current_volatility_variance = volatility_variance[i]
                    current_price = target_price_data[i]
                    # Re-leverage if we drop below the volatility SMA
                    if (current_volatility_variance < 20.0 and current_price >= target_sma[i]) and weights[0] == 0.0:
                        weights = np.array((1.0, 0.0))
                        heapq.heappush(self.activity_schedule, Rebalance(date, weights))
                    # De-leverage if we exceed 2 standard deviations from the Volatility SMA
                    elif current_volatility_variance > 20.0 and weights[0] == 1.0:
                        weights = np.array((0.0, 1.0))
                        heapq.heappush(self.activity_schedule, Rebalance(date, weights))
                        # if trace:
                        #     print(f"Rebalanced into {self.alternate} on {date.item().strftime("%A %B %d, %Y")}")
            else:
                print("Please provide an alternate security ticker if using a primary.")
                sys.exit(-1)

import heapq
import sys
from dataclasses import dataclass

import numpy as np

from backtesting.core.types import NewATH, Rebalance, TakeProfit, Trigger, ZScore
from backtesting.indicators.basic import (
    SimpleMovingAverage,
    SimpleMovingStdDev,
    SimpleMovingVariance,
    SimpleZScore,
)
from backtesting.data import get_historical_data
from backtesting.core.portfolio import Portfolio
from backtesting.core.types import Schedule, ScheduleFormat
from backtesting.core.strategy import Strategy


@dataclass(frozen=True)
class TakeProfitConfig:
    """Configuration for profit-taking rules on the active rebalance leg."""

    cooldown: int
    rebalance: bool
    trigger: Trigger
    max_cash_ratio: float | None


_TRIGGER_CLASS_KEYS = ("newath", "zscore")


def _build_take_profit_trigger(
    strategy_name: str,
    trigger_config: object,
    *,
    default_zscore_length: int,
) -> Trigger:
    """Builds a configured take-profit trigger from the strategy config.

    ``take_profit.trigger`` is either a trigger class name string or a single-key
    mapping ``{<classname>: {<property>: ...}}`` where ``classname`` is the
    concrete trigger class name in lowercase (e.g. ``zscore``).
    """
    if trigger_config is None:
        print(f"Please provide take_profit.trigger for the {strategy_name} strategy.")
        sys.exit(-1)
    if isinstance(trigger_config, bool):
        print(f"Invalid take_profit.trigger for the {strategy_name} strategy: {trigger_config}.")
        sys.exit(-1)

    class_key: str
    props: dict[str, object]
    if isinstance(trigger_config, str):
        class_key = trigger_config.strip().lower()
        props = {}
    elif isinstance(trigger_config, dict):
        if len(trigger_config) != 1:
            print(
                f"take_profit.trigger for the {strategy_name} strategy must declare exactly one "
                f"trigger class key ({', '.join(_TRIGGER_CLASS_KEYS)}). Got {trigger_config!r}."
            )
            sys.exit(-1)
        raw_key, raw_props = next(iter(trigger_config.items()))
        class_key = str(raw_key).strip().lower()
        if raw_props is None:
            props = {}
        elif isinstance(raw_props, dict):
            props = raw_props
        else:
            print(
                f"Invalid take_profit.trigger.{raw_key} for the {strategy_name} strategy: "
                f"expected a mapping of properties or null, got {type(raw_props).__name__}."
            )
            sys.exit(-1)
    else:
        print(
            f"Invalid take_profit.trigger for the {strategy_name} strategy: "
            f"expected a string or single-key mapping, got {type(trigger_config).__name__}."
        )
        sys.exit(-1)

    if class_key == "newath":
        if props:
            print(
                f"take_profit.trigger.newath for the {strategy_name} strategy does not accept "
                f"properties. Remove: {sorted(props.keys())}."
            )
            sys.exit(-1)
        return NewATH()

    if class_key == "zscore":
        distance = props.get("distance")
        if isinstance(distance, bool):
            print(
                f"Invalid take_profit.trigger.zscore.distance for the {strategy_name} strategy: {distance}. "
                "Must be numeric."
            )
            sys.exit(-1)
        try:
            distance_f = float(distance)
        except (TypeError, ValueError):
            print(
                f"Invalid take_profit.trigger.zscore.distance for the {strategy_name} strategy: {distance}. "
                "Must be numeric."
            )
            sys.exit(-1)
        if distance_f < 0.0:
            print(
                f"Invalid take_profit.trigger.zscore.distance for the {strategy_name} strategy: {distance_f}. "
                "Must be >= 0."
            )
            sys.exit(-1)

        rolling_len = props.get("length", None)
        if rolling_len is None:
            z_len = default_zscore_length
        else:
            if isinstance(rolling_len, bool):
                print(
                    f"Invalid take_profit.trigger.zscore.length for the {strategy_name} strategy: "
                    f"{rolling_len}. Must be a positive integer."
                )
                sys.exit(-1)
            try:
                z_len = int(rolling_len)
            except (TypeError, ValueError):
                print(
                    f"Invalid take_profit.trigger.zscore.length for the {strategy_name} strategy: "
                    f"{rolling_len}. Must be a positive integer."
                )
                sys.exit(-1)
            if z_len < 1:
                print(
                    f"Invalid take_profit.trigger.zscore.length for the {strategy_name} strategy: "
                    f"{z_len}. Must be a positive integer."
                )
                sys.exit(-1)

        unknown = set(props.keys()) - {"distance", "length"}
        if unknown:
            print(
                f"Unknown take_profit.trigger.zscore properties for the {strategy_name} strategy: "
                f"{sorted(unknown)}."
            )
            sys.exit(-1)

        return ZScore(distance_f, z_len)

    print(
        f"Unknown take_profit.trigger class key for the {strategy_name} strategy: {class_key!r}. "
        f"Expected one of: {', '.join(_TRIGGER_CLASS_KEYS)}."
    )
    sys.exit(-1)


def _parse_take_profit_config(
    strategy_name: str,
    take_profit_config: object,
    *,
    default_zscore_length: int,
) -> TakeProfitConfig | None:
    """Parses optional take-profit configuration."""
    if take_profit_config is None:
        return None
    if not isinstance(take_profit_config, dict):
        print(f"Invalid take_profit configuration for the {strategy_name} strategy: expected a mapping.")
        sys.exit(-1)

    cooldown = take_profit_config.get("cooldown", 0)
    if isinstance(cooldown, bool):
        print(f"Invalid take_profit.cooldown for the {strategy_name} strategy: {cooldown}. Must be a non-negative integer.")
        sys.exit(-1)
    try:
        cooldown = int(cooldown)
    except (TypeError, ValueError):
        print(f"Invalid take_profit.cooldown for the {strategy_name} strategy: {cooldown}. Must be a non-negative integer.")
        sys.exit(-1)
    if cooldown < 0:
        print(f"Invalid take_profit.cooldown for the {strategy_name} strategy: {cooldown}. Must be a non-negative integer.")
        sys.exit(-1)

    rebalance = take_profit_config.get("rebalance", True)
    if not isinstance(rebalance, bool):
        print(f"Invalid take_profit.rebalance for the {strategy_name} strategy: {rebalance}. Must be a boolean.")
        sys.exit(-1)

    max_cash_ratio = take_profit_config.get("max_cash_ratio")
    if max_cash_ratio is None and "cash_ratio" in take_profit_config:
        max_cash_ratio = take_profit_config.get("cash_ratio")
    if isinstance(max_cash_ratio, bool):
        print(
            f"Invalid take_profit.max_cash_ratio for the {strategy_name} strategy: {max_cash_ratio}. "
            "Must be between 0 and 1."
        )
        sys.exit(-1)
    if max_cash_ratio is not None:
        try:
            max_cash_ratio = float(max_cash_ratio)
        except (TypeError, ValueError):
            print(
                f"Invalid take_profit.max_cash_ratio for the {strategy_name} strategy: {max_cash_ratio}. "
                "Must be between 0 and 1."
            )
            sys.exit(-1)
        if max_cash_ratio < 0.0 or max_cash_ratio > 1.0:
            print(
                f"Invalid take_profit.max_cash_ratio for the {strategy_name} strategy: {max_cash_ratio}. "
                "Must be between 0 and 1."
            )
            sys.exit(-1)

    trigger = _build_take_profit_trigger(
        strategy_name,
        take_profit_config.get("trigger"),
        default_zscore_length=default_zscore_length,
    )
    return TakeProfitConfig(cooldown=cooldown, rebalance=rebalance, trigger=trigger, max_cash_ratio=max_cash_ratio)


def _parse_rebalance_leg(strategy_name: str, leg_name: str, leg_config: object) -> tuple[str | None, int, int]:
    """Parses a primary/alternate leg config into ticker and reentry cooldown."""
    if leg_config is None:
        return None, 0, 1
    if not isinstance(leg_config, dict):
        print(
            f"Invalid {leg_name} configuration for the {strategy_name} strategy: "
            "expected a mapping with ticker, min_reentry_days, and min_consecutive_reentry_days."
        )
        sys.exit(-1)

    ticker = leg_config.get("ticker")
    if isinstance(ticker, bool):
        print(f"Invalid {leg_name}.ticker for the {strategy_name} strategy: {ticker}. Must be a non-empty string.")
        sys.exit(-1)
    if ticker is not None:
        ticker = str(ticker).strip()
    if not ticker:
        print(f"Please provide {leg_name}.ticker for the {strategy_name} strategy.")
        sys.exit(-1)

    min_reentry_days = leg_config.get("min_reentry_days", 0)
    if isinstance(min_reentry_days, bool):
        print(
            f"Invalid {leg_name}.min_reentry_days for the {strategy_name} strategy: "
            f"{min_reentry_days}. Must be a non-negative integer."
        )
        sys.exit(-1)
    try:
        min_reentry_days = int(min_reentry_days)
    except (TypeError, ValueError):
        print(
            f"Invalid {leg_name}.min_reentry_days for the {strategy_name} strategy: "
            f"{min_reentry_days}. Must be a non-negative integer."
        )
        sys.exit(-1)
    if min_reentry_days < 0:
        print(
            f"Invalid {leg_name}.min_reentry_days for the {strategy_name} strategy: "
            f"{min_reentry_days}. Must be a non-negative integer."
        )
        sys.exit(-1)

    min_consecutive_reentry_days = leg_config.get("min_consecutive_reentry_days", 1)
    if isinstance(min_consecutive_reentry_days, bool):
        print(
            f"Invalid {leg_name}.min_consecutive_reentry_days for the {strategy_name} strategy: "
            f"{min_consecutive_reentry_days}. Must be a positive integer."
        )
        sys.exit(-1)
    try:
        min_consecutive_reentry_days = int(min_consecutive_reentry_days)
    except (TypeError, ValueError):
        print(
            f"Invalid {leg_name}.min_consecutive_reentry_days for the {strategy_name} strategy: "
            f"{min_consecutive_reentry_days}. Must be a positive integer."
        )
        sys.exit(-1)
    if min_consecutive_reentry_days < 1:
        print(
            f"Invalid {leg_name}.min_consecutive_reentry_days for the {strategy_name} strategy: "
            f"{min_consecutive_reentry_days}. Must be a positive integer."
        )
        sys.exit(-1)

    return ticker, min_reentry_days, min_consecutive_reentry_days


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

        self.target = kwargs.get("target", None)
        if self.target is None:
            print(f"Please provide a target ticker to use for the {__class__.__name__} strategy.")
            sys.exit(-1)

        length = kwargs.get("length", 200)
        if isinstance(length, bool):
            print(f"Invalid SMA length for the {__class__.__name__} strategy: {length}. Must be a positive integer.")
            sys.exit(-1)
        try:
            self.length = int(length)
        except (TypeError, ValueError):
            print(f"Invalid SMA length for the {__class__.__name__} strategy: {length}. Must be a positive integer.")
            sys.exit(-1)
        if self.length < 1:
            print(f"Invalid SMA length for the {__class__.__name__} strategy: {length}. Must be a positive integer.")
            sys.exit(-1)

        self.rebalance_on_rotation = kwargs.get("rebalance_on_rotation", True)
        if not isinstance(self.rebalance_on_rotation, bool):
            print(
                f"Invalid rebalance_on_rotation for the {__class__.__name__} strategy: "
                f"{self.rebalance_on_rotation}. Must be a boolean."
            )
            sys.exit(-1)

        (
            self.primary,
            self.primary_min_reentry_days,
            self.primary_min_consecutive_reentry_days,
        ) = _parse_rebalance_leg(
            __class__.__name__,
            "primary",
            kwargs.get("primary", None),
        )
        (
            self.alternate,
            self.alternate_min_reentry_days,
            self.alternate_min_consecutive_reentry_days,
        ) = _parse_rebalance_leg(
            __class__.__name__,
            "alternate",
            kwargs.get("alternate", None),
        )

        if self.primary is None:
            print("Only supporting primary and alternate rebalance for now.")
            sys.exit(-1)
        if self.alternate is None:
            print("Please provide an alternate security ticker if using a primary.")
            sys.exit(-1)

        self.take_profit = _parse_take_profit_config(
            __class__.__name__,
            kwargs.get("take_profit", None),
            default_zscore_length=self.length,
        )

    def _get_target_history(
        self, start_date: np.datetime64, stop_date: np.datetime64
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns target prices, SMA, and Z-score history restricted to the backtest window."""
        try:
            target_price_history, _ = get_historical_data((self.target,))
            target_price_dates, target_price_data = target_price_history
            target_price_data = target_price_data[:, 0]

            target_sma = SimpleMovingAverage(self.length, target_price_data).history
            z_window = (
                self.take_profit.trigger.length
                if self.take_profit is not None and isinstance(self.take_profit.trigger, ZScore)
                else self.length
            )
            target_z_score = SimpleZScore(z_window, target_price_data).history
            bounds_mask = (target_price_dates >= start_date) & (target_price_dates <= stop_date)

            return (
                target_price_dates[bounds_mask],
                target_price_data[bounds_mask],
                target_sma[bounds_mask],
                target_z_score[bounds_mask],
            )

        except RuntimeError as e:
            print(e)
            sys.exit(-1)

    def _get_sma_streaks(self, target_price_data: np.ndarray, target_sma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Counts consecutive closes above and below the configured target SMA."""
        target_difference = target_price_data - target_sma
        consecutive_above_sma = np.zeros(len(target_difference), dtype=int)
        consecutive_below_sma = np.zeros(len(target_difference), dtype=int)

        for i, difference in enumerate(target_difference):
            if difference > 0:
                consecutive_above_sma[i] = consecutive_above_sma[i - 1] + 1 if i > 0 else 1
            elif difference < 0:
                consecutive_below_sma[i] = consecutive_below_sma[i - 1] + 1 if i > 0 else 1

        return consecutive_above_sma, consecutive_below_sma

    def _get_rotation_weights(self, portfolio: Portfolio) -> tuple[int, int, np.ndarray, np.ndarray]:
        """Builds primary and alternate weight vectors from the portfolio baseline."""
        if self.primary not in portfolio.ticker_idx:
            print(f'Primary ticker "{self.primary}" is not in the portfolio.')
            sys.exit(-1)
        if self.alternate not in portfolio.ticker_idx:
            print(f'Alternate ticker "{self.alternate}" is not in the portfolio.')
            sys.exit(-1)

        primary_idx = portfolio.ticker_idx[self.primary]
        alternate_idx = portfolio.ticker_idx[self.alternate]
        baseline_weights = np.array(portfolio.weights, dtype=float)
        sleeve_weight = baseline_weights[primary_idx] + baseline_weights[alternate_idx]

        primary_weights = baseline_weights.copy()
        primary_weights[primary_idx] = sleeve_weight
        primary_weights[alternate_idx] = 0.0

        alternate_weights = baseline_weights.copy()
        alternate_weights[primary_idx] = 0.0
        alternate_weights[alternate_idx] = sleeve_weight

        return primary_idx, alternate_idx, primary_weights, alternate_weights

    def _schedule_take_profit(
        self,
        date: np.datetime64,
        price_index: int,
        target_bar_index: int,
        target_price_data: np.ndarray,
        target_sma: np.ndarray,
        target_z_score: np.ndarray,
        active_ticker_index: int,
        active_weights: np.ndarray,
        active_price_history: np.ndarray,
        last_take_profit_date: np.datetime64 | None,
        trace: bool,
    ) -> np.datetime64 | None:
        """Schedules a take-profit activity if the configured trigger fires."""
        if self.take_profit is None:
            return last_take_profit_date

        if last_take_profit_date is not None:
            days_since_take_profit = int((date - last_take_profit_date) / np.timedelta64(1, "D"))
            if days_since_take_profit < self.take_profit.cooldown:
                return last_take_profit_date

        trigger = self.take_profit.trigger
        if isinstance(trigger, ZScore):
            hit = trigger.hit(target_bar_index, target_price_data, target_z_score)
        else:
            hit = trigger.hit(price_index, active_price_history)
        if not hit:
            return last_take_profit_date

        heapq.heappush(
            self.activity_schedule,
            TakeProfit(
                date,
                active_ticker_index,
                float(active_weights[active_ticker_index]),
                self.take_profit.rebalance,
                active_weights,
                self.take_profit.max_cash_ratio,
            )
        )

        if trace:
            ticker = self.primary if active_ticker_index == self._primary_idx else self.alternate
            print(f"Take profit triggered for {ticker} on {date.item().strftime('%A %B %d, %Y')}")

        return date

    def procedure(
        self, historical_data: tuple, indices: tuple, portfolio: Portfolio, trace: bool
    ) -> None:
        '''Rebalances over a fixed (scheduled) time period.'''
        start_date = indices[0][0]
        stop_date = indices[0][-1]
        target_price_dates, target_price_data, target_sma, target_z_score = self._get_target_history(
            start_date,
            stop_date,
        )
        consecutive_above_sma, consecutive_below_sma = self._get_sma_streaks(target_price_data, target_sma)
        first_rebalance_idx = int(np.searchsorted(target_price_dates, start_date, side="left"))

        primary_idx, alternate_idx, primary_weights, alternate_weights = self._get_rotation_weights(portfolio)
        self._primary_idx = primary_idx

        # Start each run in the primary leg, preserving any non-rotation weights.
        portfolio.weights = primary_weights.round(decimals=4)
        primary_active = True

        # Full rebalances adjust every ticker. Partial rotations only trade the two
        # rotation tickers and leave the rest of the portfolio untouched.
        rotation_indices = None if self.rebalance_on_rotation else (primary_idx, alternate_idx)
        last_rebalance_date = None
        last_take_profit_date = None

        price_history, _ = historical_data
        price_history_dates = indices[0]
        if self.take_profit is not None:
            if price_history is None or price_history_dates is None:
                print("Take profit requires portfolio price history.")
                sys.exit(-1)

        for i in range(max(first_rebalance_idx, 1), len(target_price_dates)):
            if target_price_dates[i - 1] < start_date:
                continue

            date = target_price_dates[i]
            price_indices = np.where(price_history_dates == date)[0]
            price_index = int(price_indices[0]) if len(price_indices) else None

            days_since_rebalance = None
            if last_rebalance_date is not None:
                days_since_rebalance = int((date - last_rebalance_date) / np.timedelta64(1, "D"))

            if (
                consecutive_above_sma[i] >= self.primary_min_consecutive_reentry_days
                and not primary_active
                and (
                    days_since_rebalance is None
                    or days_since_rebalance >= self.primary_min_reentry_days
                )
            ):
                weights = primary_weights.round(decimals=4)
                heapq.heappush(self.activity_schedule, Rebalance(date, weights, rotation_indices))
                primary_active = True
                last_rebalance_date = date
                continue

            if (
                consecutive_below_sma[i] >= self.alternate_min_consecutive_reentry_days
                and primary_active
                and (
                    days_since_rebalance is None
                    or days_since_rebalance >= self.alternate_min_reentry_days
                )
            ):
                weights = alternate_weights.round(decimals=4)
                heapq.heappush(self.activity_schedule, Rebalance(date, weights, rotation_indices))
                primary_active = False
                last_rebalance_date = date
                continue

            if self.take_profit is None or price_index is None:
                continue

            active_idx = primary_idx if primary_active else alternate_idx
            active_weights = primary_weights if primary_active else alternate_weights
            last_take_profit_date = self._schedule_take_profit(
                date,
                price_index,
                i,
                target_price_data,
                target_sma,
                target_z_score,
                active_idx,
                active_weights.round(decimals=4),
                price_history[:, active_idx],
                last_take_profit_date,
                trace,
            )

class StdDevRebalance(SimpleMovingAverageRebalance):
    '''Volatility adjusted simple moving average rebalance Strategy'''
    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initializes SMA and deviation-window rebalance parameters."""
        super().__init__(**kwargs)
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

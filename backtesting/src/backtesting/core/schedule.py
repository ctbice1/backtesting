from enum import Enum, auto

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

WEEKDAY_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

class ScheduleFormat(Enum):
    """Enumeration of supported schedule cadence types."""

    DAYS = auto()
    WEEKDAY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    YEARLY = auto()

    def __lt__(self, other: object) -> bool:
        """Returns ordering by enum value for stable sorting."""
        if not isinstance(other, ScheduleFormat):
            return NotImplemented
        return self.value < other.value

    def __hash__(self) -> int:
        """Returns a stable hash for mapping/set usage."""
        return hash((self.name, self.value))

class Schedule:
    """Schedule definition used to generate recurring activity dates."""

    def __init__(self, fmt: ScheduleFormat, value: str | int | None = None) -> None:
        """Initializes a schedule with format and optional value."""
        self.fmt: ScheduleFormat = fmt
        self.value: str | int | None = value

    def __eq__(self, other: object) -> bool:
        """Returns whether two schedules are equivalent."""
        if not isinstance(other, Schedule):
            return NotImplemented
        return (self.fmt == other.fmt) and (self.value == other.value)

    def __lt__(self, other: object) -> bool:
        """Returns ordering for deterministic sorting of schedules."""
        if not isinstance(other, Schedule):
            return NotImplemented
        if self.fmt == other.fmt:
            return self.value < other.value
        return self.fmt < other.fmt

    def __hash__(self) -> int:
        """Returns a stable hash for mapping/set usage."""
        return hash((self.fmt, self.value))

    def __repr__(self) -> str:
        """Returns a concise schedule representation."""
        return f"{self.fmt}: {self.value}"

    def get_fixed_dates(
        self, date_range: np.ndarray[np.datetime64] | pd.DatetimeIndex
    ) -> list[pd.Timestamp]:
        """Returns a list of dates generated according to a fixed schedule."""

        beginning = date_range[0]
        end = date_range[-1]

        fixed_schedule_dates = []

        # Check type of allocation schedule
        if self.fmt is ScheduleFormat.DAYS:

            if self.value < 1:
                return []

            current_date = beginning
            while current_date < end:

                # Initial allocation date
                allocation_date = current_date + pd.Timedelta(days=self.value)

                # Get the next trading day if the rebalance date falls outside of a trading day
                while allocation_date not in date_range and allocation_date < end:
                    allocation_date += pd.Timedelta(days=1)

                # Update the current date
                current_date = allocation_date

                # Check for a rebalance date out of bounds
                if allocation_date > end:
                    break

                # Add the allocation date
                fixed_schedule_dates.append(allocation_date)

        elif self.fmt is ScheduleFormat.WEEKDAY:

            # Get the target weekday
            target_weekday = WEEKDAY_MAP[self.value]

            current_date = beginning
            while current_date < end:

                # Initial allocation date
                days_until_target = (target_weekday - current_date.weekday() + 7) % 7 if current_date.weekday() != target_weekday else 7
                allocation_date = current_date + pd.Timedelta(days=days_until_target)

                # Get the next trading day if the rebalance date falls outside of a trading day
                while allocation_date not in date_range and allocation_date < end:
                    allocation_date += pd.Timedelta(days=1)

                # Update the current date
                current_date = allocation_date

                # Check for a rebalance date out of bounds
                if allocation_date > end:
                    break

                # Add the allocation date
                fixed_schedule_dates.append(allocation_date)

        elif self.fmt is ScheduleFormat.WEEKLY:

            current_date = beginning
            while current_date < end:

                # Initial allocation date
                allocation_date = pd.Timestamp(current_date + relativedelta(weeks=+1))

                # Get the next trading day if the rebalance date falls outside of a trading day
                while allocation_date not in date_range and allocation_date < end:
                    allocation_date += pd.Timedelta(days=1)

                # Update the current date
                current_date = allocation_date

                # Check for a rebalance date out of bounds
                if allocation_date > end:
                    break

                # Add the allocation date
                fixed_schedule_dates.append(allocation_date)

        elif self.fmt is ScheduleFormat.MONTHLY:

            current_date = beginning
            while current_date < end:

                # Initial allocation date
                allocation_date = pd.Timestamp(current_date + relativedelta(months=1))

                # Get the next trading day if the rebalance date falls outside of a trading day
                while allocation_date not in date_range and allocation_date < end:
                    allocation_date += pd.Timedelta(days=1)

                # Update the current date
                current_date = allocation_date

                # Check for a rebalance date out of bounds
                if allocation_date > end:
                    break

                # Add the allocation date
                fixed_schedule_dates.append(allocation_date)

        elif self.fmt is ScheduleFormat.YEARLY:

            current_date = beginning
            while current_date < end:

                # Initial allocation date
                allocation_date = pd.Timestamp(current_date + relativedelta(years=+1))

                # Get the next trading day if the rebalance date falls outside of a trading day
                while allocation_date not in date_range and allocation_date < end:
                    allocation_date -= pd.Timedelta(days=1)

                # Update the current date
                current_date = allocation_date

                # Check for a rebalance date out of bounds
                if allocation_date > end:
                    break

                # Add the allocation date
                fixed_schedule_dates.append(allocation_date)

        return fixed_schedule_dates
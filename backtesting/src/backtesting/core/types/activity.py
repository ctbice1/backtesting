from typing import Self

import pandas as pd


class Activity:
    """Base scheduled activity with date ordering and priority tie-breakers."""

    def __init__(self, priority: int, date: pd.Timestamp) -> None:
        """Initializes an activity with a processing priority and date."""
        self.priority: int = priority
        self.date: pd.Timestamp = pd.Timestamp(date)

    def __eq__(self, other: Self) -> bool:
        """Returns whether two activities occur on the same date."""
        return self.date == other.date

    def __lt__(self, other: Self) -> bool:
        """Returns ordering by date, then by priority when dates match."""
        if self == other:
            return self.priority < other.priority
        return self.date < other.date

    def __gt__(self, other: Self) -> bool:
        """Returns reverse ordering by date, then by priority when dates match."""
        if self == other:
            return self.priority > other.priority
        return self.date > other.date

class TakeProfit(Activity):
    """Active-position profit taking activity."""

    def __init__(
        self,
        date: pd.Timestamp,
        ticker_index: int,
        target_weight: float,
        rebalance: bool,
        weights: tuple | None = None,
        max_cash_ratio: float | None = None,
    ) -> None:
        """Creates a take-profit activity for one ticker."""
        super().__init__(1, date)
        self.ticker_index = ticker_index
        self.target_weight = target_weight
        self.rebalance = rebalance
        self.weights = weights
        self.max_cash_ratio = max_cash_ratio

    def __repr__(self) -> str:
        """Returns a readable representation for logs/debug output."""
        return f"Take profit on\t{self.date}"

class Distribute(Activity):
    """Dividend/distribution reinvestment activity."""

    def __init__(self, date: pd.Timestamp, weights: tuple | None = None) -> None:
        """Creates a distribution activity for the given date."""
        super().__init__(2, date)
        self.weights = weights

    def __repr__(self) -> str:
        """Returns a readable representation for logs/debug output."""
        return f"Distribution on\t{self.date}"

class Allocate(Activity):
    """Capital allocation activity."""

    def __init__(self, date: pd.Timestamp, amount: float, weights: tuple | None = None) -> None:
        """Creates an allocation activity with a target amount."""
        super().__init__(3, date)
        self.amount = amount
        self.weights = weights

    def __repr__(self) -> str:
        """Returns a readable representation for logs/debug output."""
        return f"Allocation on\t{self.date}: ${self.amount:,}"

class Rebalance(Activity):
    """Portfolio rebalance activity."""

    def __init__(
        self,
        date: pd.Timestamp,
        weights: tuple | None = None,
        rebalance_indices: tuple[int, ...] | None = None,
    ) -> None:
        """Creates a rebalance activity with optional target weights."""
        super().__init__(4, date)
        self.weights = weights
        self.rebalance_indices = rebalance_indices

    def __repr__(self) -> str:
        """Returns a readable representation for logs/debug output."""
        return f"Rebalance on\t{self.date}"

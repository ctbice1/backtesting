from typing import Self

import pandas as pd


class Activity:
    def __init__(self, priority: int, date: pd.Timestamp):
        self.priority: int = priority
        self.date: pd.Timestamp = date

    def __eq__(self, other: Self):
        return self.date == other.date

    def __lt__(self, other: Self):
        if self == other:
            return self.priority < other.priority
        return self.date < other.date

    def __gt__(self, other: Self):
        if self == other:
            return self.priority > other.priority
        return self.date > other.date

class Distribute(Activity):
    def __init__(self, date: pd.Timestamp):
        super().__init__(2, date)

    def __repr__(self):
        return f"Distribution on\t{self.date}"

class Allocate(Activity):
    def __init__(self, date: pd.Timestamp, amount: float):
        super().__init__(3, date)
        self.amount = amount

    def __repr__(self):
        return f"Allocation on\t{self.date}: ${self.amount:,}"

class Rebalance(Activity):
    def __init__(self, date: pd.Timestamp, weights: tuple|None = None):
        super().__init__(4, date)
        self.weights = weights

    def __repr__(self):
        return f"Rebalance on\t{self.date}"

class ForwardFill(Activity):
    def __init__(self, date: pd.Timestamp):
        super().__init__(5, date)

    def __repr__(self):
        return f"Forward fill on\t{self.date}"
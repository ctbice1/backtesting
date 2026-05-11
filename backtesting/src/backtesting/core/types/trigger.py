"""Trigger definitions used by strategy-specific rules."""

from abc import ABC, abstractmethod

import numpy as np


class Trigger(ABC):
    """Base class for price-based strategy triggers."""

    @abstractmethod
    def hit(
        self,
        index: int,
        prices: np.ndarray,
        z_score: np.ndarray | None = None,
    ) -> bool:
        """Returns whether the trigger condition is met at ``index``."""
        pass


class NewATH(Trigger):
    """Triggers when the current price reaches a new all-time high."""

    def hit(
        self,
        index: int,
        prices: np.ndarray,
        z_score: np.ndarray | None = None,
    ) -> bool:
        """Returns true when price is above all earlier prices."""
        if index <= 0 or np.isnan(prices[index]):
            return False
        previous_high = np.nanmax(prices[:index])
        return prices[index] > previous_high


class ZScore(Trigger):
    """Triggers when target price Z-score vs its SMA reaches a configured threshold."""

    def __init__(self, distance: float, length: int) -> None:
        """Initializes Z-score threshold and rolling window length for the target series."""
        self.distance = distance
        self.length = length

    def hit(
        self,
        index: int,
        prices: np.ndarray,
        z_score: np.ndarray | None = None,
    ) -> bool:
        """Returns true when price is at least ``distance`` standard deviations above the SMA."""
        if z_score is None or np.isnan(z_score[index]):
            return False
        return z_score[index] >= self.distance

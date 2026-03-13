"""Date normalization and history slicing helpers."""

import numpy as np


def coerce_date(value: str | None, fallback: np.datetime64) -> np.datetime64:
    """Converts a date-like config value to day precision datetime64."""
    if value is None:
        return np.datetime64(fallback, "D")
    return np.datetime64(value, "D")


def bound_to_available_dates(
    date: np.datetime64, available_dates: np.ndarray, upper_bound: bool
) -> np.datetime64:
    """Bounds a date to the available range and nearest existing index date."""
    if upper_bound:
        bounded = min(date, available_dates[-1])
        index = np.searchsorted(available_dates, bounded, side="right") - 1
    else:
        bounded = max(date, available_dates[0])
        index = np.searchsorted(available_dates, bounded, side="left")
    return available_dates[index]


def slice_history(
    dates: np.ndarray, history: np.ndarray, start_date: np.datetime64, stop_date: np.datetime64
) -> tuple[np.ndarray, np.ndarray]:
    """Slices history and dates to the inclusive date range."""
    mask = (dates >= start_date) & (dates <= stop_date)
    return dates[mask], history[mask]

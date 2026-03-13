import numpy as np


def _validated_window(length: int) -> int:
    """Returns a validated rolling-window length."""
    if not isinstance(length, int) or length < 1:
        raise ValueError(f"Indicator length must be a positive integer, got: {length!r}")
    return length


class SimpleMovingAverage:
    """Simple moving average indicator."""

    def __init__(self, length: int, prices: np.ndarray) -> None:
        """Builds an SMA history over the provided price series."""
        self.length = _validated_window(length)

        # Generate a historical moving average
        self.history = np.zeros_like(prices, dtype=float)
        if len(prices) == 0:
            return

        # Generate first N - 1 averages
        cumsum = np.cumsum(prices[: self.length - 1])
        window_sizes = np.arange(1, self.length)
        for i in range(min(self.length - 1, len(prices))):
            self.history[i] = cumsum[i] / window_sizes[i]

        # Generate remaining N length averages
        if len(prices) >= self.length:
            self.history[self.length - 1 :] = np.convolve(
                prices,
                np.ones(self.length) / self.length,
                "valid",
            )

    def __repr__(self) -> str:
        """Returns a readable indicator label."""
        return f"{self.length} day simple moving average"

class SimpleMovingStdDev:
    """Rolling sample standard deviation indicator."""

    def __init__(self, length: int, prices: np.ndarray) -> None:
        """Builds a rolling standard-deviation history for prices."""
        self.length = _validated_window(length)

        # Generate a historical standard deviation
        self.history = np.zeros_like(prices, dtype=float)
        if len(prices) == 0:
            return

        # first standard deviation is always 0:
        # - avoids divide by zero when using Bessel's correction for sample variance
        self.history[0] = 0.0

        if self.length == 1:
            return

        # Get remaining N - 1 values where N = length
        for i in range(1, min(self.length - 1, len(prices))):
            self.history[i] = np.std(prices[:i+1], ddof=1)

        # Create kernel
        kernel = np.ones(self.length)

        # Calculate Welford variance
        sum_x = np.convolve(prices, kernel, "valid")
        sum_x_sq = np.convolve(prices**2, kernel, "valid")
        welford_var = sum_x_sq / (self.length) - (sum_x / (self.length)) ** 2

        # Apply Bessel's correction
        welford_var *= (self.length / (self.length - 1))

        # Calculate the standard deviation
        std_dev = np.sqrt(welford_var)

        # Add to the historical data set
        self.history[self.length - 1:] = std_dev

class SimpleMovingVariance:
    """Rolling sample variance indicator."""

    def __init__(self, length: int, prices: np.ndarray) -> None:
        """Builds a rolling variance history for prices."""
        self.length = _validated_window(length)

        # Generate a historical standard deviation
        self.history = np.zeros_like(prices, dtype=float)
        if len(prices) == 0:
            return

        # first standard deviation is always 0:
        # - avoids divide by zero when using Bessel's correction for sample variance
        self.history[0] = 0.0

        if self.length == 1:
            return

        # Get remaining N - 1 values where N = length
        for i in range(1, min(self.length - 1, len(prices))):
            self.history[i] = np.var(prices[:i+1], ddof=1)

        # Create kernel
        kernel = np.ones(self.length)

        # Calculate Welford variance
        sum_x = np.convolve(prices, kernel, "valid")
        sum_x_sq = np.convolve(prices**2, kernel, "valid")
        welford_var = sum_x_sq / (self.length) - (sum_x / (self.length)) ** 2

        # Apply Bessel's correction
        welford_var *= (self.length / (self.length - 1))

        # Add to the historical data set
        self.history[self.length - 1:] = welford_var
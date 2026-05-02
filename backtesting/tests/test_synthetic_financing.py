"""Tests for synthetic financing drag in market_data._build_synthetic_open_close."""

import unittest

import numpy as np
import pandas as pd

from backtesting.data.market_data import _build_synthetic_open_close


def _make_source_history(opens: list[float], closes: list[float]) -> pd.DataFrame:
    """Builds a small underlying history frame with daily Open/Close columns."""
    dates = pd.date_range("2020-01-02", periods=len(opens), freq="B").normalize()
    return pd.DataFrame({"Open": opens, "Close": closes}, index=dates)


def _make_financing_series(index: pd.DatetimeIndex, annual_percent: float) -> pd.Series:
    """Builds a constant-rate financing series in annual percent units."""
    return pd.Series(annual_percent, index=index, dtype=float)


class SyntheticFinancingTests(unittest.TestCase):

    def test_one_x_multiplier_has_no_financing_drag(self) -> None:
        """A 1x synthetic has zero borrowed exposure and ignores funding rates."""
        source = _make_source_history([100.0, 101.0, 102.0], [101.0, 102.0, 103.5])
        rates = _make_financing_series(source.index, annual_percent=5.0)

        without_financing_open, without_financing_close = _build_synthetic_open_close(
            source, daily_return_multiplier=1.0, expense_ratio=0.0
        )
        with_financing_open, with_financing_close = _build_synthetic_open_close(
            source,
            daily_return_multiplier=1.0,
            expense_ratio=0.0,
            financing_rate_history=rates,
        )

        np.testing.assert_allclose(with_financing_open.to_numpy(), without_financing_open.to_numpy())
        np.testing.assert_allclose(with_financing_close.to_numpy(), without_financing_close.to_numpy())

    def test_three_x_multiplier_applies_two_times_funding_drag(self) -> None:
        """A 3x synthetic borrows 2x NAV and pays daily funding on that exposure."""
        source = _make_source_history([100.0, 101.0, 102.0], [101.0, 102.0, 103.5])
        annual_percent = 3.65
        rates = _make_financing_series(source.index, annual_percent=annual_percent)
        daily_rate = (annual_percent / 100.0) / 365.0
        borrowed_size = 2.0

        synthetic_open, synthetic_close = _build_synthetic_open_close(
            source,
            daily_return_multiplier=3.0,
            expense_ratio=0.0,
            financing_rate_history=rates,
        )

        source_open = source["Open"].to_numpy(dtype=float)
        source_close = source["Close"].to_numpy(dtype=float)

        expected_open = np.empty_like(source_open)
        expected_close = np.empty_like(source_close)
        expected_open[0] = 1.0
        first_return = expected_open[0] * (3.0 * ((source_close[0] / source_open[0]) - 1.0))
        first_drag = expected_open[0] * daily_rate * borrowed_size
        expected_close[0] = expected_open[0] + first_return - first_drag
        for i in range(1, source_open.size):
            expected_open[i] = expected_close[i - 1]
            day_return = expected_open[i] * (3.0 * ((source_close[i] / source_close[i - 1]) - 1.0))
            day_drag = expected_open[i] * daily_rate * borrowed_size
            expected_close[i] = expected_open[i] + day_return - day_drag

        np.testing.assert_allclose(synthetic_open.to_numpy(), expected_open)
        np.testing.assert_allclose(synthetic_close.to_numpy(), expected_close)

    def test_financing_rate_alignment_uses_forward_fill(self) -> None:
        """Sparse rate observations are forward-filled across each underlying date."""
        source = _make_source_history([100.0, 101.0, 102.0, 103.0], [101.0, 102.0, 103.0, 104.0])
        sparse_rates = pd.Series(
            [3.65, 7.30],
            index=pd.DatetimeIndex([source.index[0], source.index[2]]),
            dtype=float,
        )

        _, synthetic_close_sparse = _build_synthetic_open_close(
            source,
            daily_return_multiplier=2.0,
            expense_ratio=0.0,
            financing_rate_history=sparse_rates,
        )

        explicit_rates = pd.Series(
            [3.65, 3.65, 7.30, 7.30],
            index=source.index,
            dtype=float,
        )
        _, synthetic_close_explicit = _build_synthetic_open_close(
            source,
            daily_return_multiplier=2.0,
            expense_ratio=0.0,
            financing_rate_history=explicit_rates,
        )

        np.testing.assert_allclose(
            synthetic_close_sparse.to_numpy(),
            synthetic_close_explicit.to_numpy(),
        )

    def test_financing_drag_reduces_close_versus_no_financing(self) -> None:
        """Positive funding rates always reduce leveraged synthetic closes."""
        source = _make_source_history([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
        rates = _make_financing_series(source.index, annual_percent=5.0)

        _, baseline_close = _build_synthetic_open_close(
            source, daily_return_multiplier=2.0, expense_ratio=0.0
        )
        _, financed_close = _build_synthetic_open_close(
            source,
            daily_return_multiplier=2.0,
            expense_ratio=0.0,
            financing_rate_history=rates,
        )

        self.assertTrue(np.all(financed_close.to_numpy() < baseline_close.to_numpy()))


if __name__ == "__main__":
    unittest.main()

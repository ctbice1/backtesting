"""Historical market data loading and cache management."""

import io
import os
import pickle
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf


_FRED_CSV_URL_TEMPLATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
_FRED_DATE_COLUMN_CANDIDATES = ("observation_date", "DATE", "date")


def _load_raw_ticker_history(ticker: str, tickers_dir: str) -> pd.DataFrame:
    """Loads cached ticker data from disk, downloading if needed."""
    ticker_file = os.path.join(tickers_dir, f"{ticker}.pkl")
    if not os.path.exists(ticker_file):
        download_historical_data(ticker, tickers_dir)

    try:
        raw = pd.read_pickle(ticker_file)
    except (EOFError, pickle.UnpicklingError, ValueError):
        download_historical_data(ticker, tickers_dir)
        raw = pd.read_pickle(ticker_file)

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(1, axis=1)

    missing_columns = {"Open", "Close", "Dividends"} - set(raw.columns)
    if missing_columns:
        missing_columns_str = ", ".join(sorted(missing_columns))
        raise RuntimeError(
            f"Downloaded data for {ticker} is missing required columns: {missing_columns_str}"
        )

    columns_to_keep = ["Open", "Close", "Dividends"]
    if "Capital Gains" in raw.columns:
        columns_to_keep.append("Capital Gains")
    return raw[columns_to_keep].copy()


def _fred_cache_dir(tickers_dir: str) -> str:
    """Returns the FRED rate cache directory, creating it if needed."""
    fred_dir = os.path.join(tickers_dir, "fred")
    if not os.path.exists(fred_dir):
        os.mkdir(fred_dir)
    return fred_dir


def _fred_cache_path(series_id: str, tickers_dir: str) -> str:
    """Returns the cache file path for a FRED series."""
    return os.path.join(_fred_cache_dir(tickers_dir), f"{series_id}.pkl")


def _atomic_write_pickle(payload: object, target_file: str, target_dir: str) -> None:
    """Writes a pickle to disk via temp file + os.replace, retrying on Windows file locks."""
    with tempfile.NamedTemporaryFile(dir=target_dir, suffix=".tmp", delete=False) as tmp_file:
        temp_path = tmp_file.name
    try:
        pd.to_pickle(payload, temp_path)
        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                os.replace(temp_path, target_file)
                break
            except PermissionError:
                if os.path.exists(target_file):
                    try:
                        pd.read_pickle(target_file)
                        break
                    except (EOFError, pickle.UnpicklingError, ValueError, PermissionError):
                        pass
                if attempt == max_attempts - 1:
                    raise
                time.sleep(0.15 * (attempt + 1))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def download_fred_series(series_id: str, tickers_dir: str) -> None:
    """Downloads and caches a daily FRED time series as a pandas Series of floats."""
    url = _FRED_CSV_URL_TEMPLATE.format(series_id=series_id)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            csv_bytes = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Unable to download FRED series {series_id}: {exc}") from exc

    raw = pd.read_csv(io.BytesIO(csv_bytes), na_values=["."])
    if raw.empty:
        raise RuntimeError(f"FRED series {series_id} returned no data.")

    date_column = next(
        (candidate for candidate in _FRED_DATE_COLUMN_CANDIDATES if candidate in raw.columns),
        None,
    )
    if date_column is None or series_id not in raw.columns:
        raise RuntimeError(
            f"FRED CSV for {series_id} is missing expected columns; got {list(raw.columns)}."
        )

    raw[date_column] = pd.to_datetime(raw[date_column], errors="coerce")
    raw = raw.dropna(subset=[date_column])
    series = raw.set_index(date_column)[series_id].astype(float).dropna()
    series.index = series.index.normalize()
    series = series.sort_index()
    if series.empty:
        raise RuntimeError(f"FRED series {series_id} has no usable observations.")

    target_file = _fred_cache_path(series_id, tickers_dir)
    _atomic_write_pickle(series, target_file, _fred_cache_dir(tickers_dir))


def _load_fred_series(series_id: str, tickers_dir: str) -> pd.Series:
    """Loads a cached FRED series, downloading on first use or cache corruption."""
    target_file = _fred_cache_path(series_id, tickers_dir)
    if not os.path.exists(target_file):
        download_fred_series(series_id, tickers_dir)

    try:
        series = pd.read_pickle(target_file)
    except (EOFError, pickle.UnpicklingError, ValueError):
        download_fred_series(series_id, tickers_dir)
        series = pd.read_pickle(target_file)

    if not isinstance(series, pd.Series):
        raise RuntimeError(f"Cached FRED series {series_id} has unexpected type: {type(series).__name__}")
    return series


def _build_synthetic_open_close(
    source_history: pd.DataFrame,
    daily_return_multiplier: float,
    expense_ratio: float,
    financing_rate_history: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Builds synthetic open/close prices from an underlying ticker.

    Synthetic closes are built from daily percentage returns:
      synthetic_close = previous_synthetic_close
                      * (1 + underlying_daily_return * multiplier - daily_expense_rate)

    Synthetic opens preserve the leveraged overnight percentage move:
      synthetic_open = previous_synthetic_close
                     * (1 + underlying_overnight_return * multiplier)

    When ``financing_rate_history`` is provided (annual percent rates indexed by date),
    a daily borrowing-cost drag is subtracted from each close. The drag scales with
    the borrowed exposure of the synthetic position:
      effective_borrowing_size = max(daily_return_multiplier - 1.0, 0.0)
      daily_financing_rate = annual_rate_percent / 100 / 365
      financing_drag = synthetic_open * daily_financing_rate * effective_borrowing_size
    """
    source_open = source_history["Open"].to_numpy(dtype=float, copy=False)
    source_close = source_history["Close"].to_numpy(dtype=float, copy=False)
    daily_expense_rate = expense_ratio / 252.0

    effective_borrowing_size = max(daily_return_multiplier - 1.0, 0.0)
    if financing_rate_history is not None and effective_borrowing_size > 0.0:
        aligned_rates = (
            financing_rate_history.reindex(source_history.index).ffill().bfill()
        )
        if aligned_rates.isna().any():
            daily_financing_rates = np.zeros(source_open.size, dtype=float)
        else:
            daily_financing_rates = (aligned_rates.to_numpy(dtype=float) / 100.0) / 365.0
    else:
        daily_financing_rates = np.zeros(source_open.size, dtype=float)

    synthetic_open = np.empty_like(source_open)
    synthetic_close = np.empty_like(source_close)

    if source_open.size == 0:
        return (
            pd.Series(dtype=float, index=source_history.index),
            pd.Series(dtype=float, index=source_history.index),
        )

    first_synthetic_open = 1
    synthetic_open[0] = first_synthetic_open
    first_daily_return = first_synthetic_open * (daily_return_multiplier * ((source_close[0] / source_open[0]) - 1.0))
    first_financing_drag = first_synthetic_open * daily_financing_rates[0] * effective_borrowing_size
    synthetic_close[0] = (
        first_synthetic_open
        + first_daily_return
        - (daily_expense_rate * np.abs(first_daily_return))
        - first_financing_drag
    )

    for i in range(1, synthetic_open.size):
        synthetic_open[i] = synthetic_close[i-1]

        daily_return = synthetic_open[i] * (daily_return_multiplier * ((source_close[i] / source_close[i-1]) - 1.0))
        financing_drag = synthetic_open[i] * daily_financing_rates[i] * effective_borrowing_size
        synthetic_close[i] = (
            synthetic_open[i]
            + daily_return
            - (daily_expense_rate * np.abs(daily_return))
            - financing_drag
        )

    return (
        pd.Series(synthetic_open, index=source_history.index, dtype=float),
        pd.Series(synthetic_close, index=source_history.index, dtype=float),
    )


def get_historical_data(
    securities: tuple[str, ...],
    synthetic_securities: dict[str, dict[str, object]] | None = None,
    include_open_prices: bool = False,
    financing_config: dict[str, object] | None = None,
) -> tuple:
    """Loads, cleans, and returns close history (+ optional open history)."""
    ticker_list = tuple(securities)
    synthetic_map = synthetic_securities or {}
    close_history = pd.DataFrame()
    open_history = pd.DataFrame()
    distribution_history = pd.DataFrame()

    tickers_dir = os.path.join(os.getcwd(), "tickers")
    if not os.path.exists(tickers_dir):
        os.mkdir(tickers_dir)

    financing_rate_history: pd.Series | None = None
    if financing_config and bool(financing_config.get("enabled", True)) and synthetic_map:
        needs_financing = any(
            float(spec.get("daily_return_multiplier", 1.0)) > 1.0
            for spec in synthetic_map.values()
        )
        if needs_financing:
            series_id = str(financing_config.get("series_id", "DFF")).strip() or "DFF"
            financing_rate_history = _load_fred_series(series_id, tickers_dir)

    for ticker in ticker_list:
        security_spec = synthetic_map.get(ticker)
        if security_spec is not None:
            underlying_ticker = str(security_spec["underlying_ticker"]).strip()
            daily_return_multiplier = float(security_spec["daily_return_multiplier"])
            expense_ratio = float(security_spec.get("expense_ratio", 0.0))
            source_history = _load_raw_ticker_history(underlying_ticker, tickers_dir)

            synthetic_open, synthetic_close = _build_synthetic_open_close(
                source_history,
                daily_return_multiplier,
                expense_ratio,
                financing_rate_history=financing_rate_history,
            )
            open_series = synthetic_open.rename(ticker)
            close_series = synthetic_close.rename(ticker)
            distribution_data = pd.DataFrame(columns=[ticker])
        else:
            source_history = _load_raw_ticker_history(ticker, tickers_dir)
            open_series = source_history["Open"].rename(ticker)
            close_series = source_history["Close"].rename(ticker)

            distributions = source_history["Dividends"].copy()
            if "Capital Gains" in source_history.columns:
                distributions = distributions + source_history["Capital Gains"]
            distribution_data = distributions.loc[distributions > 0.0].to_frame(name=ticker)

        open_history = open_history.join(open_series.to_frame(), how="outer")
        close_history = close_history.join(close_series.to_frame(), how="outer")
        distribution_history = distribution_history.join(distribution_data, how="outer")

    distribution_history = distribution_history.fillna(0.0)
    close_history = close_history.dropna()
    open_history = open_history.reindex(close_history.index)

    if not distribution_history.empty:
        distribution_history = distribution_history[close_history.index[0] :]
        distribution_history_dates = distribution_history.index.normalize().to_numpy(dtype="datetime64[D]")
        distribution_history = distribution_history.to_numpy()
        distribution_data = (distribution_history_dates, distribution_history)
    else:
        distribution_data = (None, None)

    price_history_dates = close_history.index.normalize().to_numpy(dtype="datetime64[D]")
    price_history = close_history.to_numpy()
    price_data = (price_history_dates, price_history)

    if not include_open_prices:
        return price_data, distribution_data

    open_prices = open_history.to_numpy()
    open_data = (price_history_dates, open_prices)
    return price_data, distribution_data, open_data


def download_historical_data(ticker: str, tickers_dir: str) -> None:
    """Downloads ticker historical data."""
    data = yf.download(tickers=[ticker], period="max", interval="1d", actions=True, auto_adjust=False)
    if len(data) == 0:
        raise RuntimeError(f"Unable to download data for {ticker}")

    target_file = os.path.join(tickers_dir, f"{ticker}.pkl")
    _atomic_write_pickle(data, target_file, tickers_dir)

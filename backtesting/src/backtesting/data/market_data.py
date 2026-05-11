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

# FRED series IDs used as synthetic cash (yield → NAV + distributions); not loaded via Yahoo.
FRED_CASH_SERIES_IDS = frozenset({"TB3MS"})


def is_fred_cash_series(series_id: str) -> bool:
    """Returns True if ``series_id`` names a FRED-backed synthetic cash instrument."""
    sid = str(series_id).strip().upper()
    return sid in FRED_CASH_SERIES_IDS


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


def fred_annual_yield_percent_aligned(
    series_id: str,
    trading_index: pd.DatetimeIndex,
    tickers_dir: str,
) -> pd.Series:
    """
    Aligns a FRED annualized yield series (quoted in percent per annum) to ``trading_index``.

    Monthly series such as TB3MS are forward-filled (then back-filled at the start).
    """
    monthly = _load_fred_series(series_id.upper(), tickers_dir)
    aligned = monthly.reindex(trading_index.normalize()).ffill().bfill()
    return aligned.astype(float).rename(series_id)


def _build_fred_cash_open_close_distributions(
    series_id: str,
    trading_index: pd.DatetimeIndex,
    tickers_dir: str,
    column_name: str,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Builds synthetic cash OHLC and dividend-per-share history from a FRED yield series.

    TB3MS-style inputs are quoted as annualized percent (e.g. 5.0 = 5%). Each trading day
    accrues simple interest at ``yield/100/365``. On the last trading day of each calendar month,
    accrued interest since the month's opening level is paid as a dividend per share and the
    end-of-day close resets to that month's opening level (stable NAV around par), matching a
    distributing money-market style cash sleeve.
    """
    y_daily_pct = fred_annual_yield_percent_aligned(series_id, trading_index, tickers_dir)
    dates = list(trading_index)
    n = len(dates)
    if n == 0:
        empty = pd.Series(dtype=float, index=trading_index)
        return empty, empty, pd.DataFrame(columns=[column_name])

    opens = np.zeros(n, dtype=float)
    closes = np.zeros(n, dtype=float)
    dist = np.zeros(n, dtype=float)

    close_prev = 100.0
    month_open_nav = 100.0

    y_vals = y_daily_pct.to_numpy(dtype=float, copy=False)

    for i in range(n):
        dt = dates[i]
        if i > 0 and dt.month != dates[i - 1].month:
            month_open_nav = close_prev

        open_i = close_prev
        r = float(y_vals[i]) / 100.0 / 365.0
        cum_close = open_i * (1.0 + r)

        last_of_month = (i == n - 1) or (dates[i + 1].month != dt.month)
        if last_of_month:
            div_ps = cum_close - month_open_nav
            if div_ps > 0.0:
                dist[i] = div_ps
            close_i = month_open_nav
            close_prev = close_i
        else:
            close_i = cum_close
            close_prev = close_i

        opens[i] = open_i
        closes[i] = close_i

    idx = trading_index
    open_series = pd.Series(opens, index=idx, dtype=float).rename(column_name)
    close_series = pd.Series(closes, index=idx, dtype=float).rename(column_name)
    distribution_series = pd.Series(dist, index=idx, dtype=float)
    distribution_data = distribution_series.loc[distribution_series > 0.0].to_frame(name=column_name)
    return open_series, close_series, distribution_data


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

    fred_cash_names = [t for t in ticker_list if is_fred_cash_series(t)]
    equity_list = [t for t in ticker_list if not is_fred_cash_series(t)]
    if len(equity_list) == 0:
        raise RuntimeError(
            "At least one non-FRED security is required in addition to FRED cash series "
            f"(got only {ticker_list})."
        )

    for ticker in equity_list:
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

    master_index = close_history.index
    for fred_name in fred_cash_names:
        fid = str(fred_name).strip().upper()
        open_c, close_c, dist_c = _build_fred_cash_open_close_distributions(
            fid,
            pd.DatetimeIndex(master_index),
            tickers_dir,
            column_name=fred_name,
        )
        open_history = open_history.join(open_c.to_frame(), how="left")
        close_history = close_history.join(close_c.to_frame(), how="left")
        # Outer join so FRED cash payout dates appear even when no equity dividend row
        # exists that day; left join would drop TB3MS rows off the equity index (and an
        # all-synthetic distribution frame can be empty, wiping cash distributions).
        distribution_history = distribution_history.join(dist_c, how="outer")

    distribution_history = distribution_history.fillna(0.0)

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

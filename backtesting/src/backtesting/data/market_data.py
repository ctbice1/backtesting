"""Historical market data loading and cache management."""

import os
import pickle
import tempfile
import time

import pandas as pd
import yfinance as yf


def get_historical_data(securities: tuple[str]) -> tuple:
    """Loads, cleans, and returns historical data."""
    price_history = pd.DataFrame()
    distribution_history = pd.DataFrame()

    tickers_dir = os.path.join(os.getcwd(), "tickers")
    if not os.path.exists(tickers_dir):
        os.mkdir(tickers_dir)

    for ticker in securities:
        ticker_file = os.path.join(tickers_dir, f"{ticker}.pkl")
        if not os.path.exists(ticker_file):
            download_historical_data(ticker, tickers_dir)

        try:
            raw = pd.read_pickle(ticker_file)
        except (EOFError, pickle.UnpicklingError, ValueError):
            download_historical_data(ticker, tickers_dir)
            raw = pd.read_pickle(ticker_file)

        raw = raw.droplevel(1, axis=1)
        raw = raw.drop(columns=["Open", "High", "Low", "Volume", "Stock Splits", "Adj Close"])
        distribution_data = raw.drop(columns=["Close"])
        price_data = raw.drop(columns=["Dividends"])

        distribution_data = distribution_data.rename(columns={"Dividends": "Distributions"})

        if "Capital Gains" in raw.columns:
            price_data = price_data.drop(columns=["Capital Gains"])
            distribution_data["Distributions"] = distribution_data["Distributions"] + distribution_data["Capital Gains"]
            distribution_data = distribution_data.drop(columns=["Capital Gains"])

        distribution_data = distribution_data.loc[distribution_data["Distributions"] > 0.0]
        distribution_data = distribution_data.rename(columns={"Distributions": ticker})
        price_data = price_data.rename(columns={"Close": ticker})

        distribution_history = distribution_history.join(distribution_data, how="outer")
        price_history = price_history.join(price_data, how="outer")

    distribution_history = distribution_history.fillna(0.0)
    price_history = price_history.dropna()

    if not distribution_history.empty:
        distribution_history = distribution_history[price_history.index[0] :]
        distribution_history_dates = distribution_history.index.normalize().to_numpy(dtype="datetime64[D]")
        distribution_history = distribution_history.to_numpy()
        distribution_data = (distribution_history_dates, distribution_history)
    else:
        distribution_data = (None, None)

    price_history_dates = price_history.index.normalize().to_numpy(dtype="datetime64[D]")
    price_history = price_history.to_numpy()
    price_data = (price_history_dates, price_history)

    return price_data, distribution_data


def download_historical_data(ticker: str, tickers_dir: str) -> None:
    """Downloads ticker historical data."""
    data = yf.download(tickers=[ticker], period="max", interval="1d", actions=True, auto_adjust=False)
    if len(data) == 0:
        raise RuntimeError(f"Unable to download data for {ticker}")

    target_file = os.path.join(tickers_dir, f"{ticker}.pkl")
    with tempfile.NamedTemporaryFile(dir=tickers_dir, suffix=".tmp", delete=False) as tmp_file:
        temp_path = tmp_file.name
    try:
        pd.to_pickle(data, temp_path)
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

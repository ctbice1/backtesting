"""Performance metrics and statistical analysis helpers."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def portfolio_value_history(
    share_history: Mapping[Any, np.ndarray],
    price_history: np.ndarray,
    price_history_dates: np.ndarray,
) -> pd.DataFrame:
    """Builds a daily portfolio value history from tracked share snapshots."""
    if not share_history:
        return pd.DataFrame()

    share_df = pd.DataFrame.from_dict(share_history, orient="index")
    share_history_start = min(share_history)
    date_index = pd.to_datetime(price_history_dates[price_history_dates >= share_history_start])
    if len(date_index) == 0:
        return pd.DataFrame()

    share_df = share_df.reindex(date_index, method="ffill")
    bounded_price_history = price_history[price_history_dates >= share_history_start]
    if len(bounded_price_history) == 0:
        return pd.DataFrame()

    return share_df * bounded_price_history


def parameter_ranking_stats(results: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Aggregates ranking statistics for a simulation parameter."""
    if parameter not in results.columns:
        raise KeyError(
            f'Parameter "{parameter}" not found. Available columns: {", ".join(results.columns)}'
        )
    if "amount" not in results.columns:
        raise KeyError('Missing "amount" column in simulation results; cannot rank parameters.')

    stats_df = results[[parameter, "amount", "rank"]].dropna(subset=[parameter, "amount"])
    if stats_df.empty:
        return pd.DataFrame()

    return (
        stats_df.groupby(parameter, dropna=False)
        .agg(
            observations=("amount", "size"),
            best_final_value=("amount", "max"),
            avg_final_value=("amount", "mean"),
            median_final_value=("amount", "median"),
            worst_final_value=("amount", "min"),
            std_final_value=("amount", "std"),
            best_observed_rank=("rank", "min"),
        )
        .sort_values(["best_final_value", "avg_final_value"], ascending=False)
        .reset_index()
    )


def print_parameter_ranking_stats(results: pd.DataFrame, parameter: str) -> None:
    """Prints statistical ranking analysis for a simulation parameter."""
    try:
        grouped = parameter_ranking_stats(results, parameter)
    except KeyError as error:
        print(error)
        return

    if grouped.empty:
        print(f'No usable rows found for parameter "{parameter}".')
        return

    print(f"\n=== Parameter analysis: {parameter} ===")
    print(grouped.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    best_row = grouped.iloc[0]
    print(
        f'\nBest "{parameter}" by max final portfolio value: {best_row[parameter]} '
        f'(max ${best_row["best_final_value"]:,.2f}, mean ${best_row["avg_final_value"]:,.2f})'
    )
